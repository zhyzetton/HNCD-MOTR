"""
WebSocket 连接管理 - 实时进度推送
"""
import asyncio
import json
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect

from helpers.task_manager import task_manager, TaskStatus


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        # task_id -> set of websocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, task_id: str):
        """
        接受新的 WebSocket 连接

        Args:
            websocket: WebSocket 连接
            task_id: 任务 ID
        """
        await websocket.accept()

        async with self._lock:
            if task_id not in self.active_connections:
                self.active_connections[task_id] = set()
            self.active_connections[task_id].add(websocket)

    async def disconnect(self, websocket: WebSocket, task_id: str):
        """
        断开 WebSocket 连接

        Args:
            websocket: WebSocket 连接
            task_id: 任务 ID
        """
        async with self._lock:
            if task_id in self.active_connections:
                self.active_connections[task_id].discard(websocket)
                if not self.active_connections[task_id]:
                    del self.active_connections[task_id]

    async def send_progress(self, task_id: str, data: dict):
        """
        向指定任务的所有连接发送进度更新

        Args:
            task_id: 任务 ID
            data: 进度数据
        """
        async with self._lock:
            connections = self.active_connections.get(task_id, set()).copy()

        dead_connections = []
        for connection in connections:
            try:
                await connection.send_json(data)
            except Exception:
                dead_connections.append(connection)

        # 清理断开的连接
        for connection in dead_connections:
            await self.disconnect(connection, task_id)

    async def broadcast(self, message: dict):
        """
        向所有连接广播消息

        Args:
            message: 消息内容
        """
        async with self._lock:
            all_connections = [
                conn
                for conns in self.active_connections.values()
                for conn in conns
            ]

        for connection in all_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


# 全局连接管理器
connection_manager = ConnectionManager()


async def progress_monitor(websocket: WebSocket, task_id: str):
    """
    监控任务进度并推送更新

    Args:
        websocket: WebSocket 连接
        task_id: 任务 ID
    """
    await connection_manager.connect(websocket, task_id)

    try:
        last_progress = -1

        while True:
            # 获取任务状态
            task = task_manager.get_task(task_id)

            if task is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Task not found"
                })
                break

            # 发送状态更新
            current_progress = task.processed_frames

            # 只在有变化时发送
            if current_progress != last_progress or task.status in [
                TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED
            ]:
                await websocket.send_json({
                    "type": "progress",
                    "task_id": task_id,
                    "status": task.status.value,
                    "progress": round(task.progress, 2),
                    "processed_frames": task.processed_frames,
                    "total_frames": task.total_frames,
                    "current_fps": round(task.current_fps, 2),
                    "elapsed_time": round(task.elapsed_time, 2),
                    "estimated_remaining": round(task.estimated_remaining, 2),
                    "latest_anomaly_score": task.anomaly_scores[-1] if task.anomaly_scores else None
                })
                last_progress = current_progress

            # 如果任务完成或失败，发送最终状态并退出
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task.status == TaskStatus.COMPLETED:
                    await websocket.send_json({
                        "type": "completed",
                        "task_id": task_id,
                        "result": task.result,
                        "output_url": f"/static/results/{task_id}_result.mp4"
                    })
                elif task.status == TaskStatus.FAILED:
                    await websocket.send_json({
                        "type": "failed",
                        "task_id": task_id,
                        "error": task.error_message
                    })
                break

            # 等待一段时间再检查
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect(websocket, task_id)


async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket 端点处理函数

    Args:
        websocket: WebSocket 连接
        task_id: 任务 ID
    """
    await progress_monitor(websocket, task_id)
