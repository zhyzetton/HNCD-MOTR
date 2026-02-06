"""
视频处理工具类
"""
import cv2
import os
import numpy as np
from typing import Generator, Tuple, Optional


class VideoReader:
    """视频读取器 - 支持逐帧读取"""

    def __init__(self, video_path: str):
        """
        初始化视频读取器

        Args:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        self.cap = None
        self._open()

    def _open(self):
        """打开视频文件"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

    @property
    def fps(self) -> float:
        """获取视频帧率"""
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        """获取视频宽度"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """获取视频高度"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        """获取总帧数"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration(self) -> float:
        """获取视频时长（秒）"""
        return self.frame_count / self.fps if self.fps > 0 else 0

    def read_frame(self) -> Optional[np.ndarray]:
        """读取下一帧"""
        ret, frame = self.cap.read()
        return frame if ret else None

    def seek(self, frame_idx: int):
        """跳转到指定帧"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """迭代所有帧"""
        frame_idx = 0
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1

    def close(self):
        """关闭视频"""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


class VideoWriter:
    """视频写入器"""

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = 'mp4v'
    ):
        """
        初始化视频写入器

        Args:
            output_path: 输出文件路径
            fps: 帧率
            width: 视频宽度
            height: 视频高度
            codec: 视频编码器
        """
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.writer = None
        self._open()

    def _open(self):
        """打开视频写入器"""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {self.output_path}")

    def write_frame(self, frame: np.ndarray):
        """写入一帧"""
        # 如果尺寸不匹配，进行缩放
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)

    def close(self):
        """关闭视频写入器"""
        if self.writer is not None:
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1) -> int:
    """
    从视频中提取帧并保存为图片

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        frame_interval: 帧间隔（每隔多少帧提取一帧）

    Returns:
        提取的帧数
    """
    os.makedirs(output_dir, exist_ok=True)

    with VideoReader(video_path) as reader:
        count = 0
        for idx, frame in reader:
            if idx % frame_interval == 0:
                output_path = os.path.join(output_dir, f"{idx:06d}.jpg")
                cv2.imwrite(output_path, frame)
                count += 1

    return count


def get_video_info(video_path: str) -> dict:
    """
    获取视频信息

    Args:
        video_path: 视频文件路径

    Returns:
        视频信息字典
    """
    with VideoReader(video_path) as reader:
        return {
            'fps': reader.fps,
            'width': reader.width,
            'height': reader.height,
            'frame_count': reader.frame_count,
            'duration': reader.duration
        }
