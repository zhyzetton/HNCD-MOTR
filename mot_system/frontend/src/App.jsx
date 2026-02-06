import React, { useState, useEffect, useCallback } from 'react';
import SceneSelector from './components/SceneSelector';
import VideoUpload from './components/VideoUpload';
import ProgressBar from './components/ProgressBar';
import VideoPlayer from './components/VideoPlayer';
import AnomalyChart from './components/AnomalyChart';
import { api, createWebSocket } from './services/api';
import './App.css';

// 默认场景配置
const defaultScenes = [
  {
    id: 'dance',
    name: '舞蹈场景',
    description: '适用于舞蹈、群体运动等场景的多目标跟踪',
    supports_anomaly: false,
  },
  {
    id: 'sports',
    name: '运动场景',
    description: '适用于体育比赛、运动训练等场景的多目标跟踪',
    supports_anomaly: false,
  },
  {
    id: 'traffic',
    name: '交通场景',
    description: '适用于交通监控场景，支持车辆跟踪和异常检测',
    supports_anomaly: true,
  },
];

function App() {
  // 状态
  const [scenes, setScenes] = useState(defaultScenes);
  const [selectedScene, setSelectedScene] = useState('dance');
  const [selectedFile, setSelectedFile] = useState(null);
  const [enableAnomaly, setEnableAnomaly] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // 获取场景列表
  useEffect(() => {
    api.getScenes()
      .then(setScenes)
      .catch(() => {
        // 使用默认场景
      });
  }, []);

  // WebSocket 连接
  useEffect(() => {
    if (!taskId) return;

    const ws = createWebSocket(
      taskId,
      (data) => {
        setProgress(data);

        if (data.type === 'completed') {
          setResult(data);
        } else if (data.type === 'failed') {
          setError(data.error || '处理失败');
        }
      },
      (err) => {
        console.error('WebSocket error:', err);
      },
      () => {
        // WebSocket closed
      }
    );

    return () => {
      ws.close();
    };
  }, [taskId]);

  // 获取当前选中场景的配置
  const currentScene = scenes.find((s) => s.id === selectedScene);

  // 处理场景选择
  const handleSceneSelect = (sceneId) => {
    setSelectedScene(sceneId);
    const scene = scenes.find((s) => s.id === sceneId);
    if (!scene?.supports_anomaly) {
      setEnableAnomaly(false);
    }
  };

  // 处理文件选择
  const handleFileSelect = (file) => {
    setSelectedFile(file);
    // 重置状态
    setTaskId(null);
    setProgress(null);
    setResult(null);
    setError(null);
  };

  // 开始处理
  const handleStartProcessing = async () => {
    if (!selectedFile || !selectedScene) return;

    setIsUploading(true);
    setError(null);

    try {
      const response = await api.uploadVideo(
        selectedFile,
        selectedScene,
        enableAnomaly
      );

      setTaskId(response.task_id);
    } catch (err) {
      setError(err.response?.data?.detail || '上传失败');
    } finally {
      setIsUploading(false);
    }
  };

  // 重新开始
  const handleReset = () => {
    setSelectedFile(null);
    setTaskId(null);
    setProgress(null);
    setResult(null);
    setError(null);
  };

  // 计算状态
  const isProcessing = taskId && progress && progress.status === 'processing';
  const isCompleted = progress?.status === 'completed' || result;
  const canStart = selectedFile && selectedScene && !isUploading && !isProcessing;

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="container">
          <div className="header-content">
            <div className="logo">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 1v6m0 6v10"/>
                <path d="M1 12h6m6 0h10"/>
              </svg>
              <span>MOT System</span>
            </div>
            <nav className="nav">
              <span className="nav-item active">视频处理</span>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="container">
          <div className="page-header">
            <h1>多目标跟踪与异常检测</h1>
            <p className="page-description">
              上传视频，选择场景，系统将自动进行多目标跟踪。交通场景支持异常检测功能。
            </p>
          </div>

          {/* 未完成时显示配置面板 */}
          {!isCompleted && (
            <>
              {/* Scene Selector */}
              <SceneSelector
                scenes={scenes}
                selectedScene={selectedScene}
                onSelect={handleSceneSelect}
              />

              {/* Video Upload */}
              <VideoUpload
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                disabled={isProcessing}
              />

              {/* Anomaly Detection Toggle */}
              {currentScene?.supports_anomaly && (
                <div className="anomaly-toggle">
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={enableAnomaly}
                      onChange={(e) => setEnableAnomaly(e.target.checked)}
                      disabled={isProcessing}
                    />
                    <span className="toggle-switch"></span>
                    <span className="toggle-text">启用异常检测</span>
                  </label>
                  <p className="toggle-hint">
                    开启后将同时分析交通异常行为，处理时间可能增加
                  </p>
                </div>
              )}

              {/* Start Button */}
              <div className="action-bar">
                <button
                  className="btn btn-primary btn-lg"
                  onClick={handleStartProcessing}
                  disabled={!canStart}
                >
                  {isUploading ? (
                    <>
                      <span className="spinner"></span>
                      上传中...
                    </>
                  ) : (
                    <>
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="20" height="20">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                      </svg>
                      开始处理
                    </>
                  )}
                </button>
              </div>
            </>
          )}

          {/* Error Message */}
          {error && (
            <div className="error-message">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="20" height="20">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
              </svg>
              <span>{error}</span>
            </div>
          )}

          {/* Progress Bar */}
          {(isProcessing || progress) && !isCompleted && (
            <ProgressBar
              progress={progress?.progress}
              status={progress?.status}
              processedFrames={progress?.processed_frames}
              totalFrames={progress?.total_frames}
              fps={progress?.current_fps}
              elapsedTime={progress?.elapsed_time}
              estimatedRemaining={progress?.estimated_remaining}
            />
          )}

          {/* Results */}
          {isCompleted && (
            <>
              <div className="result-header">
                <h2>处理完成</h2>
                <button className="btn btn-secondary" onClick={handleReset}>
                  处理新视频
                </button>
              </div>

              {/* Video Player */}
              <VideoPlayer
                videoUrl={result?.output_url || api.getResultVideoUrl(taskId)}
                title="跟踪结果视频"
              />

              {/* Anomaly Chart */}
              {enableAnomaly && result?.result?.anomaly_scores && (
                <AnomalyChart
                  scores={result.result.anomaly_scores}
                  timestamps={result.result.anomaly_timestamps}
                  threshold={0.1}
                />
              )}

              {/* Statistics */}
              {result?.result && (
                <div className="result-stats">
                  <h3>处理统计</h3>
                  <div className="stats-grid">
                    <div className="stat-card">
                      <span className="stat-value">{result.result.total_frames}</span>
                      <span className="stat-label">总帧数</span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-value">{result.result.track_count}</span>
                      <span className="stat-label">跟踪目标</span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-value">{result.result.average_fps?.toFixed(1)}</span>
                      <span className="stat-label">平均FPS</span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-value">{result.result.total_time?.toFixed(1)}s</span>
                      <span className="stat-label">处理时间</span>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="container">
          <p>MOT System - 多目标跟踪与异常检测系统</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
