import React from 'react';
import './ProgressBar.css';

export default function ProgressBar({ progress, status, processedFrames, totalFrames, fps, elapsedTime, estimatedRemaining }) {
  const formatTime = (seconds) => {
    if (!seconds || seconds <= 0) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusBadge = () => {
    switch (status) {
      case 'pending':
        return <span className="badge badge-info">等待中</span>;
      case 'processing':
        return <span className="badge badge-warning">处理中</span>;
      case 'completed':
        return <span className="badge badge-success">已完成</span>;
      case 'failed':
        return <span className="badge badge-error">失败</span>;
      default:
        return null;
    }
  };

  return (
    <div className="progress-panel">
      <div className="progress-header">
        <h3>处理进度</h3>
        {getStatusBadge()}
      </div>

      <div className="progress-container">
        <div
          className="progress-bar"
          style={{ width: `${Math.min(progress || 0, 100)}%` }}
        />
      </div>

      <div className="progress-stats">
        <div className="stat-item">
          <span className="stat-label">进度</span>
          <span className="stat-value">{(progress || 0).toFixed(1)}%</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">帧数</span>
          <span className="stat-value">{processedFrames || 0} / {totalFrames || 0}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">处理速度</span>
          <span className="stat-value">{(fps || 0).toFixed(1)} FPS</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">已用时间</span>
          <span className="stat-value">{formatTime(elapsedTime)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">预计剩余</span>
          <span className="stat-value">{formatTime(estimatedRemaining)}</span>
        </div>
      </div>
    </div>
  );
}
