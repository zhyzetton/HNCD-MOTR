"""
HNCD Tracker - 基于 MeMOTR 的跟踪器封装
支持逐帧在线跟踪，使用 YOLOv8 提供的 proposals
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from models.utils import load_checkpoint, get_model
from models.runtime_tracker import RuntimeTracker
from structures.track_instances import TrackInstances
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.box_ops import box_cxcywh_to_xyxy
from utils.utils import yaml_to_dict
import torchvision.transforms.functional as TF


class HNCDTracker:
    """
    HNCD MOT 跟踪器封装类

    支持：
    - 加载不同场景的预训练权重
    - 逐帧在线跟踪
    - 使用实时 YOLOv8 检测结果作为 proposals
    """

    # 场景配置映射
    SCENE_CONFIGS = {
        'dance': {
            'config': 'dancetrack.yaml',
            'weight': 'dancetrack.pth',
            'num_classes': 1,
        },
        'sports': {
            'config': 'sportsmot.yaml',
            'weight': 'sportsmot.pth',
            'num_classes': 1,
        },
        'traffic': {
            'config': 'aicity22.yaml',
            'weight': 'aicity22.pth',
            'num_classes': 1,
        }
    }

    def __init__(
        self,
        scene: str = 'dance',
        config_dir: str = None,
        weights_dir: str = None,
        device: str = 'cuda',
        det_score_thresh: float = 0.5,
        track_score_thresh: float = 0.5,
        result_score_thresh: float = 0.5,
        miss_tolerance: int = 30,
    ):
        """
        初始化跟踪器

        Args:
            scene: 场景类型 ('dance', 'sports', 'traffic')
            config_dir: 配置文件目录
            weights_dir: 权重文件目录
            device: 运行设备
            det_score_thresh: 检测分数阈值
            track_score_thresh: 跟踪分数阈值
            result_score_thresh: 结果分数阈值
            miss_tolerance: 丢失容忍帧数
        """
        self.scene = scene
        self.device = device
        self.det_score_thresh = det_score_thresh
        self.track_score_thresh = track_score_thresh
        self.result_score_thresh = result_score_thresh
        self.miss_tolerance = miss_tolerance

        # 设置目录
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = config_dir or os.path.join(backend_dir, 'configs')
        self.weights_dir = weights_dir or os.path.join(backend_dir, 'weights')

        # 图像预处理参数
        self.image_height = 800
        self.image_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # 加载模型
        self.model = None
        self.config = None
        self.runtime_tracker = None
        self._load_model()

        # 跟踪状态
        self.tracks: List[TrackInstances] = None
        self.frame_count = 0

    def _load_model(self):
        """加载模型和配置"""
        scene_cfg = self.SCENE_CONFIGS.get(self.scene)
        if scene_cfg is None:
            raise ValueError(f"Unknown scene: {self.scene}. Available: {list(self.SCENE_CONFIGS.keys())}")

        # 加载配置
        config_path = os.path.join(self.config_dir, scene_cfg['config'])
        if not os.path.exists(config_path):
            # 使用默认配置
            config_path = os.path.join(PROJECT_ROOT, 'configs', 'train_dancetrack.yaml')

        self.config = yaml_to_dict(config_path)

        # 设置必要的配置参数
        self.config['DEVICE'] = self.device
        self.config['AVAILABLE_GPUS'] = None  # 禁用分布式
        self.config['USE_DAB'] = self.config.get('USE_DAB', True)
        self.config['DET_DB'] = True  # 启用 proposal 模式
        self.config['NUM_DET_QUERIES'] = self.config.get('NUM_DET_QUERIES', 10)
        self.config['HIDDEN_DIM'] = self.config.get('HIDDEN_DIM', 256)
        self.config['NUM_CDN_GROUP'] = 0  # 推理时不使用 CDN

        # 构建模型
        self.model = build_model(self.config)

        # 加载权重
        weight_path = os.path.join(self.weights_dir, scene_cfg['weight'])
        if os.path.exists(weight_path):
            load_checkpoint(self.model, weight_path)
            print(f"Loaded weights from {weight_path}")
        else:
            print(f"Warning: Weight file not found: {weight_path}")

        self.model.eval()

        # 初始化 RuntimeTracker
        self.runtime_tracker = RuntimeTracker(
            det_score_thresh=self.det_score_thresh,
            track_score_thresh=self.track_score_thresh,
            miss_tolerance=self.miss_tolerance,
            use_motion=False,
            use_dab=self.config.get('USE_DAB', True),
        )

    def reset(self):
        """重置跟踪状态"""
        self.tracks = [TrackInstances(
            hidden_dim=get_model(self.model).hidden_dim,
            num_classes=get_model(self.model).num_classes,
            use_dab=self.config.get('USE_DAB', True)
        ).to(self.device)]
        self.frame_count = 0
        # 重置 RuntimeTracker 的 ID 计数器
        self.runtime_tracker.max_obj_id = 0
        self.runtime_tracker.motions = {}

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        预处理帧图像

        Args:
            frame: BGR 格式的 numpy 数组 (H, W, C)

        Returns:
            processed_frame: 预处理后的 tensor
            original_size: 原始图像尺寸 (h, w)
        """
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = frame_rgb.shape[:2]

        # 缩放
        scale = self.image_height / min(ori_h, ori_w)
        if max(ori_h, ori_w) * scale > self.image_width:
            scale = self.image_width / max(ori_h, ori_w)

        target_h = int(ori_h * scale)
        target_w = int(ori_w * scale)
        frame_resized = cv2.resize(frame_rgb, (target_w, target_h))

        # 转换为 tensor 并归一化
        frame_tensor = TF.normalize(TF.to_tensor(frame_resized), self.mean, self.std)

        return frame_tensor, (ori_h, ori_w)

    def prepare_proposals(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        ori_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        将 YOLOv8 检测结果转换为 proposals 格式

        Args:
            boxes: xyxy 格式的边界框 (N, 4)
            scores: 检测分数 (N,)
            ori_size: 原始图像尺寸 (h, w)

        Returns:
            proposals: (N, 5) 格式为 [cx, cy, w, h, score] (归一化坐标)
        """
        if len(boxes) == 0:
            return torch.zeros((0, 5))

        ori_h, ori_w = ori_size

        proposals = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / ori_w
            cy = (y1 + y2) / 2 / ori_h
            w = (x2 - x1) / ori_w
            h = (y2 - y1) / ori_h
            proposals.append([cx, cy, w, h, score])

        return torch.as_tensor(proposals, dtype=torch.float32)

    @torch.no_grad()
    def track_frame(
        self,
        frame: np.ndarray,
        proposals: torch.Tensor
    ) -> Dict:
        """
        处理单帧图像进行跟踪

        Args:
            frame: BGR 格式的原始帧
            proposals: 检测结果 proposals (N, 5)

        Returns:
            结果字典包含:
            - boxes: 跟踪框 (N, 4) xyxy 格式，像素坐标
            - ids: 跟踪 ID (N,)
            - scores: 跟踪分数 (N,)
            - labels: 类别标签 (N,)
        """
        if self.tracks is None:
            self.reset()

        # 预处理帧
        frame_tensor, ori_size = self.preprocess_frame(frame)
        ori_h, ori_w = ori_size

        # 构建 NestedTensor
        nested_frame = tensor_list_to_nested_tensor([frame_tensor]).to(self.device)

        # 将 proposals 移到设备上
        proposals = proposals.to(self.device)

        # 模型推理
        res = self.model(
            frame=nested_frame,
            tracks=self.tracks,
            proposals=[proposals]
        )

        # RuntimeTracker 更新
        previous_tracks, new_tracks = self.runtime_tracker.update(
            model_outputs=res,
            tracks=self.tracks
        )

        # 后处理更新 tracks
        self.tracks = get_model(self.model).postprocess_single_frame(
            previous_tracks, new_tracks, None
        )

        # 提取结果
        tracks_result = self.tracks[0].to(torch.device("cpu"))

        # 计算面积并过滤
        tracks_result.area = tracks_result.boxes[:, 2] * ori_w * tracks_result.boxes[:, 3] * ori_h

        # 按分数过滤
        keep = torch.max(tracks_result.scores, dim=-1).values > self.result_score_thresh
        tracks_result = tracks_result[keep]

        # 按面积过滤
        if len(tracks_result) > 0:
            keep = tracks_result.area > 100
            tracks_result = tracks_result[keep]

        # 转换为 xyxy 像素坐标
        if len(tracks_result) > 0:
            boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
            boxes = boxes * torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float)
        else:
            boxes = torch.zeros((0, 4))

        self.frame_count += 1

        return {
            'boxes': boxes.numpy(),
            'ids': tracks_result.ids.numpy() if len(tracks_result) > 0 else np.array([]),
            'scores': torch.max(tracks_result.scores, dim=-1).values.numpy() if len(tracks_result) > 0 else np.array([]),
            'labels': tracks_result.labels.numpy() if len(tracks_result) > 0 else np.array([]),
            'frame_idx': self.frame_count
        }

    def get_track_embeddings(self) -> Optional[np.ndarray]:
        """获取当前跟踪的 embeddings（用于异常检测）"""
        if self.tracks is None or len(self.tracks[0]) == 0:
            return None
        return self.tracks[0].output_embed.cpu().numpy()


class TrackerManager:
    """
    跟踪器管理器 - 预加载多个场景的跟踪器
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.trackers: Dict[str, HNCDTracker] = {}

    def get_tracker(self, scene: str) -> HNCDTracker:
        """获取指定场景的跟踪器（懒加载）"""
        if scene not in self.trackers:
            print(f"Loading tracker for scene: {scene}")
            self.trackers[scene] = HNCDTracker(scene=scene, device=self.device)
        return self.trackers[scene]

    def preload_all(self):
        """预加载所有场景的跟踪器"""
        for scene in HNCDTracker.SCENE_CONFIGS.keys():
            self.get_tracker(scene)
