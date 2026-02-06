"""
任务管理器 - 管理视频处理任务状态
"""
import uuid
import time
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from threading import Lock


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"          # 等待处理
    PROCESSING = "processing"    # 处理中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 已取消


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    scene: str = ""
    video_path: str = ""
    output_path: str = ""
    enable_anomaly: bool = False

    # 进度信息
    total_frames: int = 0
    processed_frames: int = 0
    current_fps: float = 0.0

    # 时间信息
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # 结果信息
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # 异常检测结果
    anomaly_scores: list = field(default_factory=list)
    anomaly_timestamps: list = field(default_factory=list)

    @property
    def progress(self) -> float:
        """计算进度百分比"""
        if self.total_frames == 0:
            return 0.0
        return (self.processed_frames / self.total_frames) * 100

    @property
    def elapsed_time(self) -> float:
        """已用时间（秒）"""
        if self.started_at is None:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    @property
    def estimated_remaining(self) -> float:
        """预估剩余时间（秒）"""
        if self.processed_frames == 0 or self.current_fps == 0:
            return 0.0
        remaining_frames = self.total_frames - self.processed_frames
        return remaining_frames / self.current_fps

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'scene': self.scene,
            'video_path': self.video_path,
            'output_path': self.output_path,
            'enable_anomaly': self.enable_anomaly,
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'progress': round(self.progress, 2),
            'current_fps': round(self.current_fps, 2),
            'elapsed_time': round(self.elapsed_time, 2),
            'estimated_remaining': round(self.estimated_remaining, 2),
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'error_message': self.error_message,
            'has_anomaly_data': len(self.anomaly_scores) > 0,
        }


class TaskManager:
    """
    任务管理器 - 线程安全的任务状态管理

    可以扩展为使用 Redis 进行持久化存储
    """

    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = Lock()

    def create_task(
        self,
        scene: str,
        video_path: str,
        output_path: str,
        enable_anomaly: bool = False,
        task_id: str = None
    ) -> str:
        """
        创建新任务

        Args:
            scene: 场景类型
            video_path: 输入视频路径
            output_path: 输出视频路径
            enable_anomaly: 是否启用异常检测
            task_id: 指定任务 ID（可选，不传则自动生成）

        Returns:
            任务 ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        task = TaskInfo(
            task_id=task_id,
            scene=scene,
            video_path=video_path,
            output_path=output_path,
            enable_anomaly=enable_anomaly
        )

        with self._lock:
            self._tasks[task_id] = task

        return task_id

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(
        self,
        task_id: str,
        status: TaskStatus = None,
        total_frames: int = None,
        processed_frames: int = None,
        current_fps: float = None,
        error_message: str = None,
        result: Dict = None,
        anomaly_score: float = None,
        anomaly_timestamp: float = None
    ):
        """
        更新任务状态

        Args:
            task_id: 任务 ID
            status: 新状态
            total_frames: 总帧数
            processed_frames: 已处理帧数
            current_fps: 当前处理帧率
            error_message: 错误信息
            result: 结果数据
            anomaly_score: 异常分数
            anomaly_timestamp: 异常时间戳
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

            if status is not None:
                task.status = status
                if status == TaskStatus.PROCESSING and task.started_at is None:
                    task.started_at = time.time()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = time.time()

            if total_frames is not None:
                task.total_frames = total_frames

            if processed_frames is not None:
                task.processed_frames = processed_frames

            if current_fps is not None:
                task.current_fps = current_fps

            if error_message is not None:
                task.error_message = error_message

            if result is not None:
                task.result = result

            if anomaly_score is not None:
                task.anomaly_scores.append(anomaly_score)

            if anomaly_timestamp is not None:
                task.anomaly_timestamps.append(anomaly_timestamp)

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False

    def list_tasks(self, status: TaskStatus = None) -> list:
        """
        列出任务

        Args:
            status: 筛选状态（可选）

        Returns:
            任务列表
        """
        with self._lock:
            tasks = list(self._tasks.values())
            if status is not None:
                tasks = [t for t in tasks if t.status == status]
            return [t.to_dict() for t in tasks]

    def cleanup_old_tasks(self, max_age_hours: float = 24):
        """
        清理旧任务

        Args:
            max_age_hours: 最大保留时间（小时）
        """
        max_age_seconds = max_age_hours * 3600
        current_time = time.time()

        with self._lock:
            tasks_to_delete = []
            for task_id, task in self._tasks.items():
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    if current_time - task.created_at > max_age_seconds:
                        tasks_to_delete.append(task_id)

            for task_id in tasks_to_delete:
                del self._tasks[task_id]


# 全局任务管理器实例
task_manager = TaskManager()
