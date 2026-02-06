"""
视频处理管线 - 整合检测、跟踪、异常检测
"""
import os
import time
import subprocess
import numpy as np
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

from .detector import ProposalGenerator
from .tracker import HNCDTracker
from .anomaly_detector import AnomalyDetector, AnomalyDetectorLite
from helpers.video_utils import VideoReader, VideoWriter
from helpers.visualization import TrackVisualizer, AnomalyVisualizer
from helpers.task_manager import TaskManager, TaskStatus


@dataclass
class ProcessingConfig:
    """处理配置"""
    scene: str = 'dance'                    # 场景类型
    enable_anomaly: bool = False            # 启用异常检测
    det_conf_thresh: float = 0.5            # 检测置信度阈值
    track_score_thresh: float = 0.5         # 跟踪分数阈值
    result_score_thresh: float = 0.5        # 结果分数阈值
    miss_tolerance: int = 30                # 丢失容忍帧数
    anomaly_threshold: float = 0.1          # 异常阈值
    show_trajectory: bool = True            # 显示轨迹
    device: str = 'cuda'                    # 设备


class VideoProcessor:
    """
    视频处理器 - 完整的 MOT + 异常检测管线
    """

    def __init__(
        self,
        config: ProcessingConfig = None,
        detector: ProposalGenerator = None,
        tracker: HNCDTracker = None,
        anomaly_detector: AnomalyDetector = None,
    ):
        """
        初始化视频处理器

        Args:
            config: 处理配置
            detector: 检测器（可选，不传则自动创建）
            tracker: 跟踪器（可选，不传则自动创建）
            anomaly_detector: 异常检测器（可选）
        """
        self.config = config or ProcessingConfig()

        # 检测器
        self.detector = detector or ProposalGenerator(device=self.config.device)

        # 跟踪器
        self.tracker = tracker

        # 异常检测器
        self.anomaly_detector = anomaly_detector

        # 可视化器
        self.track_visualizer = TrackVisualizer(
            show_trajectory=self.config.show_trajectory
        )
        self.anomaly_visualizer = AnomalyVisualizer(
            threshold=self.config.anomaly_threshold
        ) if self.config.enable_anomaly else None

    def _init_tracker(self):
        """初始化跟踪器"""
        if self.tracker is None:
            self.tracker = HNCDTracker(
                scene=self.config.scene,
                device=self.config.device,
                det_score_thresh=self.config.det_conf_thresh,
                track_score_thresh=self.config.track_score_thresh,
                result_score_thresh=self.config.result_score_thresh,
                miss_tolerance=self.config.miss_tolerance,
            )

    def _init_anomaly_detector(self):
        """初始化异常检测器"""
        if self.config.enable_anomaly and self.anomaly_detector is None:
            try:
                self.anomaly_detector = AnomalyDetector(
                    device=self.config.device,
                    threshold=self.config.anomaly_threshold,
                )
            except Exception as e:
                print(f"Warning: Failed to load AnomalyDetector, using lite version: {e}")
                self.anomaly_detector = AnomalyDetectorLite(
                    threshold=self.config.anomaly_threshold
                )

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Callable[[int, int, float, Optional[float]], None] = None,
        task_manager: TaskManager = None,
        task_id: str = None,
    ) -> Dict[str, Any]:
        """
        处理视频

        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            progress_callback: 进度回调函数 (processed_frames, total_frames, fps, anomaly_score)
            task_manager: 任务管理器
            task_id: 任务 ID

        Returns:
            处理结果字典
        """
        # 初始化模型
        self._init_tracker()
        if self.config.enable_anomaly:
            self._init_anomaly_detector()

        # 重置状态
        self.tracker.reset()
        self.track_visualizer.reset()
        if self.anomaly_detector:
            self.anomaly_detector.reset()
        if self.anomaly_visualizer:
            self.anomaly_visualizer.reset()

        # 打开视频
        reader = VideoReader(input_path)
        total_frames = reader.frame_count

        # 更新任务状态
        if task_manager and task_id:
            task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                total_frames=total_frames
            )

        # 创建输出视频写入器
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 先用 mp4v 编码写入临时文件
        temp_output_path = output_path.replace('.mp4', '_temp.mp4')
        writer = VideoWriter(
            temp_output_path,
            fps=reader.fps,
            width=reader.width,
            height=reader.height,
            codec='mp4v'
        )

        # 处理统计
        start_time = time.time()
        frame_times = []
        all_results = []
        anomaly_scores = []
        anomaly_timestamps = []

        try:
            for frame_idx, frame in reader:
                frame_start = time.time()

                # 1. 检测
                boxes, scores, classes = self.detector.detect(
                    frame,
                    conf_thres=self.config.det_conf_thresh
                )

                # 2. 准备 proposals
                proposals = self.tracker.prepare_proposals(
                    boxes, scores, (frame.shape[0], frame.shape[1])
                )

                # 3. 跟踪
                track_results = self.tracker.track_frame(frame, proposals)
                all_results.append(track_results)

                # 4. 异常检测（如果启用）
                anomaly_score = None
                if self.config.enable_anomaly and self.anomaly_detector:
                    anomaly_score = self.anomaly_detector.add_frame(frame)
                    if anomaly_score is not None:
                        anomaly_scores.append(anomaly_score)
                        anomaly_timestamps.append(frame_idx / reader.fps)

                        # 更新任务异常数据
                        if task_manager and task_id:
                            task_manager.update_task(
                                task_id,
                                anomaly_score=anomaly_score,
                                anomaly_timestamp=frame_idx / reader.fps
                            )

                # 5. 可视化
                vis_frame = self.track_visualizer.draw_results(frame, track_results)

                # 添加帧信息
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                current_fps = 1.0 / frame_time if frame_time > 0 else 0

                vis_frame = self.track_visualizer.draw_info(
                    vis_frame,
                    frame_idx=frame_idx,
                    fps=current_fps,
                    extra_info={'Tracks': len(track_results['ids'])}
                )

                # 异常可视化
                if self.anomaly_visualizer and anomaly_score is not None:
                    self.anomaly_visualizer.add_score(anomaly_score)
                    vis_frame = self.anomaly_visualizer.draw_anomaly_indicator(
                        vis_frame, anomaly_score
                    )
                    vis_frame = self.anomaly_visualizer.draw_score_chart(vis_frame)

                # 6. 写入输出
                writer.write_frame(vis_frame)

                # 7. 进度回调
                if progress_callback:
                    progress_callback(frame_idx + 1, total_frames, current_fps, anomaly_score)

                # 更新任务进度
                if task_manager and task_id:
                    task_manager.update_task(
                        task_id,
                        processed_frames=frame_idx + 1,
                        current_fps=current_fps
                    )

        except Exception as e:
            if task_manager and task_id:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error_message=str(e)
                )
            raise

        finally:
            reader.close()
            writer.close()

        # 使用 ffmpeg 将 mp4v 转换为 H.264 (浏览器兼容)
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([
                ffmpeg_path, '-y', '-i', temp_output_path,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-movflags', 'faststart',
                output_path
            ], check=True, capture_output=True)
            # 删除临时文件
            os.remove(temp_output_path)
        except Exception as e:
            # 如果 ffmpeg 失败，使用原始文件
            print(f"Warning: ffmpeg conversion failed: {e}")
            if os.path.exists(temp_output_path):
                os.rename(temp_output_path, output_path)

        # 计算统计信息
        total_time = time.time() - start_time
        avg_fps = total_frames / total_time if total_time > 0 else 0

        result = {
            'total_frames': total_frames,
            'total_time': total_time,
            'average_fps': avg_fps,
            'output_path': output_path,
            'track_count': len(set(
                id for r in all_results for id in r['ids']
            )),
        }

        # 添加异常检测统计
        if self.config.enable_anomaly and self.anomaly_detector:
            anomaly_stats = self.anomaly_detector.get_statistics()
            result['anomaly_statistics'] = anomaly_stats
            result['anomaly_scores'] = anomaly_scores
            result['anomaly_timestamps'] = anomaly_timestamps

        # 更新任务完成状态
        if task_manager and task_id:
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                result=result
            )

        return result


def process_video_task(
    input_path: str,
    output_path: str,
    scene: str,
    enable_anomaly: bool = False,
    task_manager: TaskManager = None,
    task_id: str = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    便捷函数：处理视频任务

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        scene: 场景类型
        enable_anomaly: 是否启用异常检测
        task_manager: 任务管理器
        task_id: 任务 ID
        device: 设备

    Returns:
        处理结果
    """
    config = ProcessingConfig(
        scene=scene,
        enable_anomaly=enable_anomaly,
        device=device
    )

    processor = VideoProcessor(config=config)

    return processor.process_video(
        input_path=input_path,
        output_path=output_path,
        task_manager=task_manager,
        task_id=task_id
    )
