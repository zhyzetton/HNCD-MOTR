"""
MOT System Backend - 多目标跟踪与异常检测系统后端

FastAPI 应用主入口
"""
import os
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from api.websocket import websocket_endpoint

# 创建 FastAPI 应用
app = FastAPI(
    title="MOT System API",
    description="多目标跟踪与交通异常检测系统 API",
    version="1.0.0",
)

# CORS 配置 - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件目录
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BACKEND_DIR, "static")

# 确保静态文件目录存在
os.makedirs(os.path.join(STATIC_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "results"), exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 注册 API 路由
app.include_router(api_router)


# WebSocket 端点
@app.websocket("/ws/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket 进度推送端点

    Args:
        websocket: WebSocket 连接
        task_id: 任务 ID
    """
    await websocket_endpoint(websocket, task_id)


# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "MOT System Backend"}


# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "MOT System API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /api/upload",
            "task_status": "GET /api/task/{task_id}",
            "task_result": "GET /api/task/{task_id}/result",
            "websocket": "WS /ws/{task_id}",
            "scenes": "GET /api/scenes"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
