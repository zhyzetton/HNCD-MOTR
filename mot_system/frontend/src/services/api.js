import axios from 'axios';

const API_BASE = '/api';

// API 服务
export const api = {
  // 获取可用场景
  async getScenes() {
    const response = await axios.get(`${API_BASE}/scenes`);
    return response.data.scenes;
  },

  // 上传视频
  async uploadVideo(file, scene, enableAnomaly = false, device = 'cuda') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('scene', scene);
    formData.append('enable_anomaly', enableAnomaly);
    formData.append('device', device);

    const response = await axios.post(`${API_BASE}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // 获取任务状态
  async getTaskStatus(taskId) {
    const response = await axios.get(`${API_BASE}/task/${taskId}`);
    return response.data;
  },

  // 获取任务列表
  async getTasks(status = null) {
    const params = status ? { status } : {};
    const response = await axios.get(`${API_BASE}/tasks`, { params });
    return response.data.tasks;
  },

  // 获取异常数据
  async getAnomalyData(taskId) {
    const response = await axios.get(`${API_BASE}/task/${taskId}/anomaly`);
    return response.data;
  },

  // 删除任务
  async deleteTask(taskId) {
    const response = await axios.delete(`${API_BASE}/task/${taskId}`);
    return response.data;
  },

  // 获取结果视频 URL
  getResultVideoUrl(taskId) {
    return `/static/results/${taskId}_result.mp4`;
  },
};

// WebSocket 连接
export function createWebSocket(taskId, onMessage, onError, onClose) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const ws = new WebSocket(`${protocol}//${host}/ws/${taskId}`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    if (onError) onError(error);
  };

  ws.onclose = () => {
    if (onClose) onClose();
  };

  return ws;
}
