"""
异常检测器 - 基于 SO-TAD 的交通异常检测
使用 HNCD backbone+encoder 作为特征提取器
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
from collections import deque

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from models.sotad import Generator, build_extracotr
from utils.utils import yaml_to_dict
from utils.nested_tensor import NestedTensor
import torchvision.transforms.functional as TF


class AnomalyDetector:
    """
    交通异常检测器

    基于 SO-TAD 方法：
    - 使用 HNCD backbone+encoder 提取特征
    - 使用 Generator 预测下一帧特征
    - 通过预测误差检测异常
    """

    def __init__(
        self,
        config_path: str = None,
        extractor_weight: str = None,
        generator_weight: str = None,
        device: str = 'cuda',
        threshold: float = 0.1,
        window_size: int = 4  # t0, t1, t2, t3
    ):
        """
        初始化异常检测器

        Args:
            config_path: 配置文件路径
            extractor_weight: 特征提取器权重路径（HNCD 模型）
            generator_weight: 生成器权重路径
            device: 运行设备
            threshold: 异常阈值
            window_size: 时间窗口大小
        """
        self.device = device
        self.threshold = threshold
        self.window_size = window_size

        # 设置路径
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = config_path or os.path.join(backend_dir, 'configs', 'anomaly.yaml')
        self.weights_dir = os.path.join(backend_dir, 'weights')
        self.extractor_weight = extractor_weight or os.path.join(self.weights_dir, 'aicity22.pth')
        self.generator_weight = generator_weight or os.path.join(self.weights_dir, 'anomaly_netG.pt')

        # 图像预处理参数
        self.image_size = (640, 480)  # SO-TAD 使用的尺寸
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # 模型
        self.extractor = None
        self.generator = None
        self.config = None

        # 特征缓存（滑动窗口）
        self.feature_buffer: deque = deque(maxlen=window_size)

        # 历史分数
        self.score_history: List[float] = []

        self._load_models()

    def _load_models(self):
        """加载模型"""
        # 加载配置
        if os.path.exists(self.config_path):
            self.config = yaml_to_dict(self.config_path)
        else:
            # 使用默认配置
            default_config = os.path.join(PROJECT_ROOT, 'configs', 'train_accident_sotad.yaml')
            if os.path.exists(default_config):
                self.config = yaml_to_dict(default_config)
            else:
                # 最小必要配置
                self.config = {
                    'DEVICE': self.device,
                    'AVAILABLE_GPUS': None,
                    'USE_DAB': True,
                    'BACKBONE': 'resnet50',
                    'HIDDEN_DIM': 256,
                    'FFN_DIM': 2048,
                    'NUM_FEATURE_LEVELS': 4,
                    'NUM_HEADS': 8,
                    'NUM_ENC_POINTS': 4,
                    'NUM_DEC_POINTS': 4,
                    'NUM_ENC_LAYERS': 6,
                    'NUM_DEC_LAYERS': 6,
                    'DROPOUT': 0.0,
                    'NUM_DET_QUERIES': 10,
                    'DATASET': 'AICity22',
                    'WITH_ENC': True,
                    'DET_DB': True,
                    'NUM_CDN_GROUP': 0,
                    'ID_NOISE_RATIO': 0.3,
                    'BOX_NOISE_SCALE': 0.4,
                    'CDN_K': 5,
                    'USE_TINY_NOISE': True,
                    'USE_CHECKPOINT': False,
                    'CHECKPOINT_LEVEL': 0,
                    'VISUALIZE': False,
                    'ACTIVATION': 'ReLU',
                }

        self.config['DEVICE'] = self.device
        self.config['AVAILABLE_GPUS'] = None

        # 构建特征提取器
        print("Loading feature extractor...")
        self.extractor = build_extracotr(self.config).to(self.device)

        # 加载特征提取器权重（从 HNCD 模型）
        if os.path.exists(self.extractor_weight):
            checkpoint = torch.load(self.extractor_weight, map_location=self.device)
            # 提取 backbone 和 encoder 相关的权重
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # 过滤并加载 hnd_model 的权重
            hnd_state = {}
            for k, v in state_dict.items():
                if k.startswith('hnd_model.'):
                    hnd_state[k.replace('hnd_model.', '')] = v
                elif not k.startswith('fusion_conv') and not k.startswith('latent_smooth'):
                    # 尝试直接匹配
                    hnd_state[k] = v

            if hnd_state:
                self.extractor.hnd_model.load_state_dict(hnd_state, strict=False)
            print(f"Loaded extractor weights from {self.extractor_weight}")
        else:
            print(f"Warning: Extractor weight not found: {self.extractor_weight}")

        self.extractor.eval()

        # 构建生成器
        print("Loading generator...")
        self.generator = Generator(16, 64, 16).to(self.device)

        # 加载生成器权重
        if os.path.exists(self.generator_weight):
            state_dict = torch.load(self.generator_weight, map_location=self.device)
            # 处理 DDP 包装的权重
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state[k.replace('module.', '')] = v
                else:
                    new_state[k] = v
            self.generator.load_state_dict(new_state)
            print(f"Loaded generator weights from {self.generator_weight}")
        else:
            print(f"Warning: Generator weight not found: {self.generator_weight}")

        self.generator.eval()

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        预处理帧图像

        Args:
            frame: BGR 格式的 numpy 数组 (H, W, C)

        Returns:
            预处理后的 tensor (1, C, H, W)
        """
        import cv2

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 缩放到固定尺寸
        frame_resized = cv2.resize(frame_rgb, self.image_size)

        # 转换为 tensor 并归一化
        frame_tensor = TF.normalize(TF.to_tensor(frame_resized), self.mean, self.std)

        return frame_tensor.unsqueeze(0)  # (1, C, H, W)

    @torch.no_grad()
    def extract_features(self, frame: np.ndarray) -> torch.Tensor:
        """
        提取帧特征

        Args:
            frame: BGR 格式的帧

        Returns:
            特征张量 (1, 256, 77, 57)
        """
        frame_tensor = self.preprocess_frame(frame).to(self.device)
        features = self.extractor(frame_tensor)
        return features

    def add_frame(self, frame: np.ndarray) -> Optional[float]:
        """
        添加帧并计算异常分数

        Args:
            frame: BGR 格式的帧

        Returns:
            异常分数（如果缓冲区已满），否则返回 None
        """
        # 提取特征
        features = self.extract_features(frame)
        self.feature_buffer.append(features)

        # 如果缓冲区未满，返回 None
        if len(self.feature_buffer) < self.window_size:
            return None

        # 计算异常分数
        score = self._compute_anomaly_score()
        self.score_history.append(score)

        return score

    @torch.no_grad()
    def _compute_anomaly_score(self) -> float:
        """
        计算异常分数

        使用前三帧预测第四帧，然后计算预测误差

        Returns:
            异常分数（余弦距离）
        """
        # 获取四帧特征
        t0, t1, t2, t3 = list(self.feature_buffer)

        # 计算差分特征
        d01 = F.normalize(t1 - t0, dim=1)
        d12 = F.normalize(t2 - t1, dim=1)

        # 生成预测特征
        predicted = self.generator(t0, t1, t2, d01, d12)

        # 计算余弦距离
        p = predicted.view(1, -1)
        x = t3.view(1, -1)
        cos_sim = F.cosine_similarity(p, x, dim=1)
        cos_dist = 1 - cos_sim.item()

        return cos_dist

    def is_anomaly(self, score: float = None) -> bool:
        """
        判断是否为异常

        Args:
            score: 异常分数（如果为 None，使用最新分数）

        Returns:
            是否为异常
        """
        if score is None:
            if len(self.score_history) == 0:
                return False
            score = self.score_history[-1]

        return score > self.threshold

    def get_anomaly_level(self, score: float = None) -> str:
        """
        获取异常级别

        Args:
            score: 异常分数

        Returns:
            异常级别字符串
        """
        if score is None:
            if len(self.score_history) == 0:
                return "unknown"
            score = self.score_history[-1]

        if score < self.threshold * 0.5:
            return "normal"
        elif score < self.threshold:
            return "warning"
        elif score < self.threshold * 1.5:
            return "anomaly"
        else:
            return "critical"

    def reset(self):
        """重置检测器状态"""
        self.feature_buffer.clear()
        self.score_history.clear()

    def get_statistics(self) -> dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        if len(self.score_history) == 0:
            return {
                'total_frames': 0,
                'anomaly_count': 0,
                'anomaly_ratio': 0,
                'mean_score': 0,
                'max_score': 0,
                'min_score': 0
            }

        scores = np.array(self.score_history)
        anomaly_count = np.sum(scores > self.threshold)

        return {
            'total_frames': len(self.score_history),
            'anomaly_count': int(anomaly_count),
            'anomaly_ratio': float(anomaly_count / len(self.score_history)),
            'mean_score': float(np.mean(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores))
        }


class AnomalyDetectorLite:
    """
    轻量级异常检测器 - 不使用深度学习模型

    基于简单的帧差分和运动检测，用于在没有 GPU 或权重文件时的备用方案
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.prev_frame = None
        self.score_history: List[float] = []

    def add_frame(self, frame: np.ndarray) -> Optional[float]:
        """添加帧并计算异常分数"""
        import cv2

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return None

        # 计算帧差
        diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray

        # 计算运动强度
        motion_score = np.mean(diff) / 255.0

        # 归一化到 0-1 范围
        score = min(motion_score * 5, 1.0)  # 放大并限制
        self.score_history.append(score)

        return score

    def is_anomaly(self, score: float = None) -> bool:
        if score is None:
            if len(self.score_history) == 0:
                return False
            score = self.score_history[-1]
        return score > self.threshold

    def reset(self):
        self.prev_frame = None
        self.score_history.clear()

    def get_statistics(self) -> dict:
        if len(self.score_history) == 0:
            return {
                'total_frames': 0,
                'anomaly_count': 0,
                'anomaly_ratio': 0,
                'mean_score': 0,
                'max_score': 0,
                'min_score': 0
            }

        scores = np.array(self.score_history)
        anomaly_count = np.sum(scores > self.threshold)

        return {
            'total_frames': len(self.score_history),
            'anomaly_count': int(anomaly_count),
            'anomaly_ratio': float(anomaly_count / len(self.score_history)),
            'mean_score': float(np.mean(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores))
        }
