"""
可视化工具类 - 跟踪结果绘制
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import colorsys


class TrackVisualizer:
    """跟踪结果可视化器"""

    def __init__(
        self,
        line_thickness: int = 2,
        font_scale: float = 0.6,
        show_trajectory: bool = True,
        trajectory_length: int = 30
    ):
        """
        初始化可视化器

        Args:
            line_thickness: 线条粗细
            font_scale: 字体大小
            show_trajectory: 是否显示轨迹
            trajectory_length: 轨迹长度（帧数）
        """
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.show_trajectory = show_trajectory
        self.trajectory_length = trajectory_length

        # 颜色映射缓存
        self.color_cache: Dict[int, Tuple[int, int, int]] = {}

        # 轨迹历史
        self.trajectories: Dict[int, List[Tuple[int, int]]] = {}

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        为每个 track_id 生成唯一的颜色

        Args:
            track_id: 跟踪 ID

        Returns:
            BGR 颜色元组
        """
        if track_id not in self.color_cache:
            # 使用 HSV 颜色空间生成鲜艳的颜色
            hue = (track_id * 0.618033988749895) % 1.0  # 黄金分割法
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            self.color_cache[track_id] = bgr
        return self.color_cache[track_id]

    def update_trajectory(self, track_id: int, center: Tuple[int, int]):
        """更新轨迹历史"""
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []

        self.trajectories[track_id].append(center)

        # 限制轨迹长度
        if len(self.trajectories[track_id]) > self.trajectory_length:
            self.trajectories[track_id] = self.trajectories[track_id][-self.trajectory_length:]

    def draw_box(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        track_id: int,
        score: float = None,
        label: str = None
    ) -> np.ndarray:
        """
        绘制单个跟踪框

        Args:
            frame: 原始帧
            box: 边界框 [x1, y1, x2, y2]
            track_id: 跟踪 ID
            score: 跟踪分数
            label: 类别标签

        Returns:
            绘制后的帧
        """
        x1, y1, x2, y2 = map(int, box)
        color = self.get_color(track_id)

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)

        # 绘制 ID 标签
        text = f"ID:{track_id}"
        if score is not None:
            text += f" {score:.2f}"
        if label is not None:
            text = f"{label} " + text

        # 计算文本背景
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.line_thickness
        )

        # 绘制文本背景
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        # 绘制文本
        cv2.putText(
            frame,
            text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            self.line_thickness
        )

        # 更新并绘制轨迹
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.update_trajectory(track_id, center)

        if self.show_trajectory and track_id in self.trajectories:
            trajectory = self.trajectories[track_id]
            for i in range(1, len(trajectory)):
                # 渐变透明度
                alpha = i / len(trajectory)
                thickness = max(1, int(self.line_thickness * alpha))
                cv2.line(
                    frame,
                    trajectory[i - 1],
                    trajectory[i],
                    color,
                    thickness
                )

        return frame

    def draw_results(
        self,
        frame: np.ndarray,
        results: Dict,
        copy: bool = True
    ) -> np.ndarray:
        """
        绘制所有跟踪结果

        Args:
            frame: 原始帧
            results: 跟踪结果字典，包含 boxes, ids, scores, labels
            copy: 是否复制帧

        Returns:
            绘制后的帧
        """
        if copy:
            frame = frame.copy()

        boxes = results.get('boxes', [])
        ids = results.get('ids', [])
        scores = results.get('scores', [])
        labels = results.get('labels', [])

        for i in range(len(boxes)):
            box = boxes[i]
            track_id = int(ids[i]) if i < len(ids) else i
            score = float(scores[i]) if i < len(scores) else None
            label = str(labels[i]) if i < len(labels) else None

            self.draw_box(frame, box, track_id, score, label)

        return frame

    def draw_info(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float = None,
        extra_info: Dict = None
    ) -> np.ndarray:
        """
        绘制帧信息

        Args:
            frame: 原始帧
            frame_idx: 帧索引
            fps: 处理帧率
            extra_info: 额外信息

        Returns:
            绘制后的帧
        """
        y_offset = 30
        x_offset = 10

        # 帧号
        text = f"Frame: {frame_idx}"
        cv2.putText(frame, text, (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), 2)
        y_offset += 30

        # FPS
        if fps is not None:
            text = f"FPS: {fps:.1f}"
            cv2.putText(frame, text, (x_offset, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), 2)
            y_offset += 30

        # 额外信息
        if extra_info:
            for key, value in extra_info.items():
                text = f"{key}: {value}"
                cv2.putText(frame, text, (x_offset, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), 2)
                y_offset += 30

        return frame

    def reset(self):
        """重置可视化器状态"""
        self.trajectories.clear()


class AnomalyVisualizer:
    """异常检测结果可视化器"""

    def __init__(self, threshold: float = 0.5):
        """
        初始化异常可视化器

        Args:
            threshold: 异常阈值
        """
        self.threshold = threshold
        self.score_history: List[float] = []

    def add_score(self, score: float):
        """添加异常分数"""
        self.score_history.append(score)

    def draw_anomaly_indicator(
        self,
        frame: np.ndarray,
        score: float
    ) -> np.ndarray:
        """
        绘制异常指示器

        Args:
            frame: 原始帧
            score: 异常分数

        Returns:
            绘制后的帧
        """
        h, w = frame.shape[:2]

        # 判断是否为异常
        is_anomaly = score > self.threshold

        # 绘制异常状态
        if is_anomaly:
            # 红色边框警告
            cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 0, 255), 5)

            # 异常文本
            text = "ANOMALY DETECTED!"
            cv2.putText(frame, text, (w // 4, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            # 正常状态
            text = f"Normal - Score: {score:.3f}"
            cv2.putText(frame, text, (w - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def draw_score_chart(
        self,
        frame: np.ndarray,
        chart_height: int = 100,
        chart_width: int = 300
    ) -> np.ndarray:
        """
        绘制异常分数时间曲线

        Args:
            frame: 原始帧
            chart_height: 图表高度
            chart_width: 图表宽度

        Returns:
            绘制后的帧
        """
        if len(self.score_history) < 2:
            return frame

        h, w = frame.shape[:2]

        # 图表位置（右下角）
        x_start = w - chart_width - 20
        y_start = h - chart_height - 20

        # 绘制背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start),
                      (x_start + chart_width, y_start + chart_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # 绘制阈值线
        threshold_y = int(y_start + chart_height * (1 - self.threshold))
        cv2.line(frame, (x_start, threshold_y),
                 (x_start + chart_width, threshold_y), (0, 255, 255), 1)

        # 绘制分数曲线
        scores = self.score_history[-chart_width:]
        points = []
        for i, score in enumerate(scores):
            x = x_start + i * chart_width // len(scores)
            y = int(y_start + chart_height * (1 - min(score, 1.0)))
            points.append((x, y))

        if len(points) > 1:
            for i in range(1, len(points)):
                color = (0, 0, 255) if scores[i] > self.threshold else (0, 255, 0)
                cv2.line(frame, points[i - 1], points[i], color, 2)

        return frame

    def reset(self):
        """重置可视化器状态"""
        self.score_history.clear()
