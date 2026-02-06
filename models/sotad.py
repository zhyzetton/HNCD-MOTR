# Copyright (c) Zhang Haoyu. All Rights Reserved.
"""
SO-TAD 事故检测模型
HNCD 替代 VAE 作为特征提取器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
import models.helper as hf
from utils.nested_tensor import NestedTensor
import torch.nn.utils as utils

from models import build_hncd


class Residual_Block(nn.Module):
    """
    3D 残差块（so-tad原始实现）
    """
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(input_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel)
        )

    def forward(self, x0):
        x1 = self.layer(x0)
        return x0 + x1

class Residual_Block_SN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_Block_SN, self).__init__()
        # 这里的 Conv3d 加上 spectral_norm
        self.conv1 = utils.spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.PReLU()
        self.conv2 = utils.spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += res
        out = self.relu(out)
        return out

class Generator(nn.Module):
    """
    TCFF-GAN 生成器

    输入：
    - t0, t1, t2: (B, C, H, W) - 场景引入特征 [f_{t-3}, f_{t-2}, f_{t-1}]
    - d01, d12: (B, C, H, W) - 场景差分特征 [d_{3-2}, d_{2-1}]

    输出：
    - output: (B, C, H, W) - 预测特征 f̂_t
    """
    def __init__(self, input_channel, ch, output_channel, upscale_factor=4, kernel_size=3, stride=1, bias=False, padding=1):
        super().__init__()
        self.ps = nn.PixelShuffle(upscale_factor)

        self.conv_in_1 = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch*2, kernel_size, stride=[1,2,2], bias=False, padding=1),
            nn.BatchNorm3d(ch*2),
            nn.PReLU(),
        )

        self.conv_in_2 = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch * 2, kernel_size, stride=[1, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            nn.Conv3d(ch * 2, ch*2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch*2),
            nn.PReLU(),
        )

        self.conv_in_3 = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch * 2, kernel_size, stride=[1, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            nn.Conv3d(ch * 2, ch*2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch*2),
            nn.PReLU(),
        )

        self.pipeline_1 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
        )

        self.pipeline_2 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
        )

        self.pipeline_3 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
        )

        self.conv_in_4 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride=[1, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
        )

        self.conv_in_5 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride=[2, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride=[2, 1, 1], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
        )

        self.conv_in_6 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride=[2, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride=[2, 1, 1], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride=[2, 1, 1], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
        )

        self.act_fnc = nn.PReLU()

        self.conv_in_7 = nn.Sequential(
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride, bias=False, padding=1),
        )

    def forward(self, t0, t1, t2, d01, d12):
        """
        Args:
            t0, t1, t2: (B, C, H, W) - 场景引入特征
            d01, d12: (B, C, H, W) - 场景差分特征

        Returns:
            output: (B, C, H, W) - 预测特征
        """
        # [B, C, H, W] -> [B, C/(r^2), H*r, W*r] -> [B, C/(r^2), 1, H*r, W*r]
        t0 = torch.unsqueeze(self.ps(t0), 2)
        t1 = torch.unsqueeze(self.ps(t1), 2)
        t2 = torch.unsqueeze(self.ps(t2), 2)
        d01 = torch.unsqueeze(self.ps(d01), 2)
        d12 = torch.unsqueeze(self.ps(d12), 2)

        # 构造三路输入（与原始so-tad完全一致）
        x0 = t0  # [B, C', 1, H, W]
        x1 = torch.cat((t0, d01, t1), dim=2)  # [B, C', 3, H, W]
        x2 = torch.cat((t0, d01, t1, d12, t2), dim=2)  # [B, C', 5, H, W]

        # 三路处理
        x0 = self.conv_in_1(x0)
        x00 = self.pipeline_1(x0)
        x00 = self.act_fnc(x0 + x00)
        x00 = self.conv_in_4(x00)

        x1 = self.conv_in_2(x1)
        x11 = self.pipeline_2(x1)
        x11 = self.act_fnc(x1 + x11)
        x11 = self.conv_in_5(x11)

        x2 = self.conv_in_3(x2)
        x22 = self.pipeline_3(x2)
        x22 = self.act_fnc(x2 + x22)
        x22 = self.conv_in_6(x22)

        # 融合
        output = self.act_fnc(x00 + x11 + x22)
        output = self.conv_in_7(output)
        output = torch.squeeze(output, 2)

        return output


class Discriminator(nn.Module):
    def __init__(self, input_channel, ch, output_channel, upscale_factor=4,kernel_size=3, stride=1, bias=False, padding=1):
        super().__init__()
        self.ps = nn.PixelShuffle(upscale_factor)

        self.convs_3d = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch * 2, kernel_size, stride=[2,2,2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            # Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride=[2,2,2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
        )

        self.convs_2d = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 8, kernel_size, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(ch * 8),
            nn.PReLU(),
            nn.Conv2d(ch * 8, ch * 16, kernel_size, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(ch * 16),
            nn.PReLU(),
            nn.Conv2d(ch * 16, ch * 16, kernel_size, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(ch * 16),
            nn.PReLU(),
            nn.Conv2d(ch * 16, ch * 32, kernel_size, stride=2, bias=False, padding=1),
        )

        self.act_fnc = nn.PReLU()

        self.linears = nn.Sequential(
            nn.Linear(ch * 32 * 5 * 4, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, t0, t1, t2, t3):
        t0 = torch.unsqueeze(self.ps(t0), 2)
        t1 = torch.unsqueeze(self.ps(t1), 2)
        t2 = torch.unsqueeze(self.ps(t2), 2)
        t3 = torch.unsqueeze(self.ps(t3), 2)

        x0 = torch.concatenate((t0, t1, t2, t3), dim=2)
        x1 = self.convs_3d(x0)
        x1 = self.act_fnc(x1)
        x1 = torch.squeeze(x1, 2)
        x1 = self.convs_2d(x1)
        x1 = x1.view(-1,512*5*4)
        output = self.linears(x1)


        return output

class DiscriminatorGAP(nn.Module):
    def __init__(self, input_channel, ch, output_channel,
                 upscale_factor=4, kernel_size=3, stride=1,
                 bias=False, padding=1):
        super().__init__()

        # PixelShuffle 上采样层 (保持不变)
        self.ps = nn.PixelShuffle(upscale_factor)

        # ------ 3D Conv 部分 (保持不变) ------
        self.convs_3d = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch * 2, kernel_size, stride=[2,2,2], padding=1, bias=False),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride=[2,2,2], padding=1, bias=False),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
        )

        # ------ 2D Conv 部分 (保持不变) ------
        final_channels = ch * 32
        
        self.convs_2d = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 8, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.PReLU(),
            nn.Conv2d(ch * 8, ch * 16, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 16),
            nn.PReLU(),
            nn.Conv2d(ch * 16, ch * 16, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 16),
            nn.PReLU(),
            nn.Conv2d(ch * 16, final_channels, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.PReLU(),
        )

        # ------ 结构优化核心：全局平均池化 + 轻量级分类头 ------
        
        # 全局平均池化: 将 [B, C, H, W] -> [B, C, 1, 1]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 轻量级全连接层
        # 输入维度固定为 final_channels
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, 512), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # 可选：加个 Dropout 进一步防止过拟合
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, t0, t1, t2, t3):
        # 1. PixelShuffle + 增加深度维度
        t0 = torch.unsqueeze(self.ps(t0), 2)
        t1 = torch.unsqueeze(self.ps(t1), 2)
        t2 = torch.unsqueeze(self.ps(t2), 2)
        t3 = torch.unsqueeze(self.ps(t3), 2)

        # 拼接
        x = torch.cat([t0, t1, t2, t3], dim=2)

        # 卷积特征提取
        x = self.convs_3d(x)
        
        # 压缩深度维度 [B, C, D, H, W] -> [B, C, H, W]
        # 注意：经过 3D conv 后，D 维度应该是 1，所以可以直接 squeeze
        x = torch.squeeze(x, 2)

        x = self.convs_2d(x)
        
        # ------ 结构优化部分 ------
        
        # 全局平均池化
        # x.shape: [B, 512, H, W] -> [B, 512, 1, 1]
        x = self.avg_pool(x)
        
        # x.shape: [B, 512, 1, 1] -> [B, 512]
        x = torch.flatten(x, 1)
        
        # 8. 分类
        out = self.classifier(x)
        
        return out
    
class DiscriminatorStandard(nn.Module):
    def __init__(self, input_channel, ch, output_channel,
                 upscale_factor=4, kernel_size=3, stride=1,
                 bias=False, padding=1, fc_dim=10240):
        super().__init__()

        self.ps = nn.PixelShuffle(upscale_factor)

        self.convs_3d = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch * 2, kernel_size, stride=[2,2,2], padding=1, bias=False),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride=[2,2,2], padding=1, bias=False),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
        )

        self.convs_2d = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 8, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.PReLU(),
            nn.Conv2d(ch * 8, ch * 16, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 16),
            nn.PReLU(),
            nn.Conv2d(ch * 16, ch * 16, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 16),
            nn.PReLU(),
            nn.Conv2d(ch * 16, ch * 32, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 32),
            nn.PReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, t0, t1, t2, t3):
        t0 = torch.unsqueeze(self.ps(t0), 2)
        t1 = torch.unsqueeze(self.ps(t1), 2)
        t2 = torch.unsqueeze(self.ps(t2), 2)
        t3 = torch.unsqueeze(self.ps(t3), 2)

        x = torch.cat([t0, t1, t2, t3], dim=2)
        x = self.convs_3d(x)
        x = torch.squeeze(x, 2)
        x = self.convs_2d(x)
        x = torch.flatten(x, start_dim=1)
        
        return self.fc(x)


class DiscriminatorRP(nn.Module):
    def __init__(self, input_channel, ch, output_channel,
                 upscale_factor=4, kernel_size=3, stride=1,
                 bias=False, padding=1, fc_dim=10240):
        super().__init__()

        self.ps = nn.PixelShuffle(upscale_factor)

        self.convs_3d = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch * 2, kernel_size, stride=[2,2,2], padding=1, bias=False),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride=[2,2,2], padding=1, bias=False),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
        )

        self.convs_2d = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 8, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.PReLU(),
            nn.Conv2d(ch * 8, ch * 16, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 16),
            nn.PReLU(),
            nn.Conv2d(ch * 16, ch * 16, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 16),
            nn.PReLU(),
            nn.Conv2d(ch * 16, ch * 32, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 32),
            nn.PReLU(),
        )

        # ------ 随机投影头 (Random Projection Head) ------
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重，并冻结
        self._init_and_freeze_fc()

    def _init_and_freeze_fc(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                # 使用正交初始化 (保证随机投影的质量)
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
                # 冻结参数
                m.weight.requires_grad = False
                if m.bias is not None:
                    m.bias.requires_grad = False

    def forward(self, t0, t1, t2, t3):
        t0 = torch.unsqueeze(self.ps(t0), 2)
        t1 = torch.unsqueeze(self.ps(t1), 2)
        t2 = torch.unsqueeze(self.ps(t2), 2)
        t3 = torch.unsqueeze(self.ps(t3), 2)

        x = torch.cat([t0, t1, t2, t3], dim=2)
        x = self.convs_3d(x)
        x = torch.squeeze(x, 2)
        x = self.convs_2d(x)
        x = torch.flatten(x, start_dim=1)
        
        # 此时 fc 是固定的，这迫使前面的 convs 学习适配 fc 的特征
        return self.fc(x)



class HNDFeatureExtractorWithEnc(nn.Module):
    """
    使用 HND Backbone + Transformer Encoder 替代 SO-TAD 的 VAE。
    将 HND 多尺度特征 (3层 + 扩展的第4层) 映射为 (B,256,77,57) latent，
    供 TCFF-GAN 使用。
    """

    def __init__(self, hnd_model: nn.Module, freeze: bool = True):
        super().__init__()

        self.hnd_model = hnd_model
        self.freeze = freeze

        # SO-TAD latent 目标尺寸
        self.target_hw = (77, 57)

        # 冻结 HND（backbone + encoder + transformer）
        if self.freeze:
            for p in self.hnd_model.parameters():
                p.requires_grad = False

        # 多尺度融合 + 平滑，使 latent 更像 VAE 的分布，GAN 更稳定
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 * self.hnd_model.n_feature_levels, 512, 3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ELU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ELU(inplace=True),
        )

        self.latent_smooth = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        """
        输入:  x: (B,3,640,480)
        输出:  latent: (B,256,77,57)
        """

        B = x.size(0)

        # ================= 1. Backbone + 扩展到 n_feature_levels =================
        with torch.no_grad() if self.freeze else torch.enable_grad():
            # 构造 NestedTensor
            mask = torch.zeros((B, x.shape[2], x.shape[3]),
                               dtype=torch.bool, device=x.device)
            nested_input = NestedTensor(x, mask)

            backbone_features, pos = self.hnd_model.backbone(nested_input)
            # backbone_features: 原始3层 NestedTensor 列表
            # pos: 对应位置编码列表（同样3层）

            srcs, masks = [], []

            # ① 用 feature_projs 对原始3层做通道投影
            for lvl, feat in enumerate(backbone_features):
                src, m = feat.decompose()                     # src:(B,C,H,W), m:(B,H,W)
                src = self.hnd_model.feature_projs[lvl](src)  # → (B,256,H,W)
                srcs.append(src)
                masks.append(m)

            # ② 若 n_feature_levels > backbone 的输出层数，则扩展额外层
            #    这一段就是你原来自己写的「扩一层」逻辑，我给你完整保留并整合
            if self.hnd_model.n_feature_levels > len(srcs):
                srcs_len = len(srcs)
                for lvl in range(srcs_len, self.hnd_model.n_feature_levels):
                    if lvl == srcs_len:
                        # 第一个额外层用最后一层 backbone 特征
                        src = self.hnd_model.feature_projs[lvl](backbone_features[-1].tensors)
                    else:
                        # 后续额外层基于上一层 src 再做一次 proj
                        src = self.hnd_model.feature_projs[lvl](srcs[-1])

                    # 生成对应尺度的 mask
                    m = nested_input.masks
                    m = F.interpolate(m[None].float(),
                                      size=src.shape[-2:])[0].to(torch.bool)

                    # 生成该层位置编码并追加到 pos 列表
                    pos_embed = self.hnd_model.backbone.position_embedding(
                        NestedTensor(src, m)
                    )
                    pos.append(pos_embed.to(src.device))

                    srcs.append(src)
                    masks.append(m)

            # 此时：
            # srcs: n_feature_levels 个 (B,256,H_l,W_l)
            # masks: n_feature_levels 个 (B,H_l,W_l)
            # pos: n_feature_levels 个 (B,256,H_l,W_l)

            # ================= 2. 展平准备送入 Transformer Encoder =================
            src_flatten, mask_flatten, pos_flatten = [], [], []
            spatial_shapes = []

            for lvl, (src, m, pos_embed) in enumerate(zip(srcs, masks, pos)):
                B, C, H, W = src.shape
                spatial_shapes.append([H, W])

                # (B,C,H,W) → (B,HW,C)
                src_l = src.flatten(2).transpose(1, 2)
                m_l = m.flatten(1)  # (B,HW)
                pos_l = pos_embed.flatten(2).transpose(1, 2)

                # 加上 level_embed
                pos_l = pos_l + self.hnd_model.transformer.level_embed[lvl].view(1, 1, -1)

                src_flatten.append(src_l)
                mask_flatten.append(m_l)
                pos_flatten.append(pos_l)

            src_flatten = torch.cat(src_flatten, dim=1)     # (B,ΣHW,256)
            mask_flatten = torch.cat(mask_flatten, dim=1)   # (B,ΣHW)
            pos_flatten = torch.cat(pos_flatten, dim=1)     # (B,ΣHW,256)

            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=x.device
            )   # (n_levels,2)

            level_start_index = torch.cat((
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1]
            ))  # (n_levels,)

            valid_ratios = torch.stack(
                [self.hnd_model.transformer.get_valid_ratio(m) for m in masks],
                dim=1
            )   # (B,n_levels,2)

            # ================= 3. Transformer Encoder =================
            memory = self.hnd_model.transformer.encoder(
                src=src_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                pos=pos_flatten,
                padding_mask=mask_flatten
            )   # (B,ΣHW,256)

        # ================= 4. memory → 还原回多尺度 feature maps =================
        feats = []
        index_start = 0
        for (H, W) in spatial_shapes:
            HW = H * W
            mem_l = memory[:, index_start:index_start + HW]     # (B,HW,256)
            mem_l = mem_l.transpose(1, 2).reshape(B, 256, H, W) # (B,256,H,W)
            feats.append(mem_l)
            index_start += HW

        # ================= 5. 上采样到 (77,57) 并多尺度融合 =================
        up_feats = []
        for f in feats:
            up = F.interpolate(f, size=self.target_hw,
                               mode='bilinear', align_corners=False)
            up_feats.append(up)

        fused = torch.cat(up_feats, dim=1)  # (B,256*n_levels,77,57)

        # ================= 6. 融合卷积 + 平滑，使 latent 更友好给 GAN =================
        latent = self.fusion_conv(fused)     # (B,256,77,57)
        latent = self.latent_smooth(latent)  # (B,256,77,57)

        # 训练阶段加一点小噪声，提高对抗训练稳定性
        if self.training:
            latent = latent + 0.01 * torch.randn_like(latent)

        return latent

def build_extracotr(config: dict):
    hnd_model = build_hncd(config)
    if config.get("WITH_ENC", True):
        extractor = HNDFeatureExtractorWithEnc(hnd_model)

    return extractor

def get_discriminator(config: dict):
    version = config.get('NETD_VERSION', 'standard')
    netD_dict = {
        'original': Discriminator,
        'standard': DiscriminatorStandard,
        'gap': DiscriminatorGAP,
        'rp': DiscriminatorRP
    }
    return netD_dict[version]