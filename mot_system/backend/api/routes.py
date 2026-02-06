"""
REST API 路由
"""
import os
import uuid
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from helpers.task_manager import task_manager, TaskStatus
from helpers.video_utils import get_video_info
from core.pipeline import process_video_task

router = APIRouter(prefix="/api", tags=["api"])

# 获取静态文件目录
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BACKEND_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BACKEND_DIR, "static", "results")

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def run_processing_task(
    task_id: str,
    input_path: str,
    output_path: str,
    scene: str,
    enable_anomaly: bool,
    device: str = 'cuda'
):
    """后台处理任务"""
    try:
        process_video_task(
            input_path=input_path,
            output_path=output_path,
            scene=scene,
            enable_anomaly=enable_anomaly,
            task_manager=task_manager,
            task_id=task_id,
            device=device
        )
    except Exception as e:
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            error_message=str(e)
        )


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scene: str = Form(...),
    enable_anomaly: bool = Form(False),
    device: str = Form('cuda')
):
    """
    上传视频并开始处理

    Args:
        file: 上传的视频文件
        scene: 场景类型 (dance, sports, traffic)
        enable_anomaly: 是否启用异常检测（仅 traffic 场景有效）
        device: 计算设备

    Returns:
        任务信息
    """
    # 验证场景
    valid_scenes = ['dance', 'sports', 'traffic']
    if scene not in valid_scenes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scene: {scene}. Must be one of {valid_scenes}"
        )

    # 只有交通场景支持异常检测
    if enable_anomaly and scene != 'traffic':
        enable_anomaly = False

    # 生成文件名
    file_ext = os.path.splitext(file.filename)[1] or '.mp4'
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}{file_ext}"
    output_filename = f"{file_id}_result.mp4"

    input_path = os.path.join(UPLOAD_DIR, input_filename)
    output_path = os.path.join(RESULT_DIR, output_filename)

    # 保存上传文件
    try:
        content = await file.read()
        with open(input_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # 获取视频信息
    try:
        video_info = get_video_info(input_path)
    except Exception as e:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {e}")

    # 创建任务（使用 file_id 作为 task_id，保持一致）
    task_id = task_manager.create_task(
        scene=scene,
        video_path=input_path,
        output_path=output_path,
        enable_anomaly=enable_anomaly,
        task_id=file_id
    )

    # 更新任务的总帧数
    task_manager.update_task(task_id, total_frames=video_info['frame_count'])

    # 在后台启动处理任务
    background_tasks.add_task(
        run_processing_task,
        task_id,
        input_path,
        output_path,
        scene,
        enable_anomaly,
        device
    )

    return {
        "task_id": task_id,
        "status": "pending",
        "video_info": video_info,
        "scene": scene,
        "enable_anomaly": enable_anomaly
    }


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态

    Args:
        task_id: 任务 ID

    Returns:
        任务状态信息
    """
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return task.to_dict()


@router.get("/task/{task_id}/result")
async def get_task_result(task_id: str):
    """
    获取任务结果

    Args:
        task_id: 任务 ID

    Returns:
        结果视频文件
    """
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed. Current status: {task.status.value}"
        )

    if not os.path.exists(task.output_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        task.output_path,
        media_type="video/mp4",
        filename=os.path.basename(task.output_path)
    )


@router.get("/task/{task_id}/anomaly")
async def get_anomaly_data(task_id: str):
    """
    获取异常检测数据

    Args:
        task_id: 任务 ID

    Returns:
        异常检测数据
    """
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.enable_anomaly:
        raise HTTPException(
            status_code=400,
            detail="Anomaly detection was not enabled for this task"
        )

    return {
        "task_id": task_id,
        "scores": task.anomaly_scores,
        "timestamps": task.anomaly_timestamps,
        "statistics": task.result.get('anomaly_statistics') if task.result else None
    }


@router.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """
    删除任务及其相关文件

    Args:
        task_id: 任务 ID

    Returns:
        删除结果
    """
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # 删除相关文件
    if os.path.exists(task.video_path):
        os.remove(task.video_path)
    if os.path.exists(task.output_path):
        os.remove(task.output_path)

    # 删除任务记录
    task_manager.delete_task(task_id)

    return {"message": "Task deleted successfully"}


@router.get("/tasks")
async def list_tasks(status: Optional[str] = None):
    """
    列出所有任务

    Args:
        status: 筛选状态（可选）

    Returns:
        任务列表
    """
    filter_status = None
    if status:
        try:
            filter_status = TaskStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}"
            )

    tasks = task_manager.list_tasks(status=filter_status)
    return {"tasks": tasks}


@router.get("/scenes")
async def get_available_scenes():
    """
    获取可用场景列表

    Returns:
        场景列表及其描述
    """
    return {
        "scenes": [
            {
                "id": "dance",
                "name": "舞蹈场景",
                "description": "适用于舞蹈、群体运动等场景的多目标跟踪",
                "supports_anomaly": False
            },
            {
                "id": "sports",
                "name": "运动场景",
                "description": "适用于体育比赛、运动训练等场景的多目标跟踪",
                "supports_anomaly": False
            },
            {
                "id": "traffic",
                "name": "交通场景",
                "description": "适用于交通监控场景，支持车辆跟踪和异常检测",
                "supports_anomaly": True
            }
        ]
    }


@router.post("/cleanup")
async def cleanup_old_tasks(max_age_hours: float = 24):
    """
    清理旧任务

    Args:
        max_age_hours: 最大保留时间（小时）

    Returns:
        清理结果
    """
    task_manager.cleanup_old_tasks(max_age_hours)
    return {"message": f"Cleaned up tasks older than {max_age_hours} hours"}
