import React from 'react';
import './VideoPlayer.css';

export default function VideoPlayer({ videoUrl, title }) {
  if (!videoUrl) {
    return null;
  }

  return (
    <div className="video-player">
      <h3 className="player-title">{title || '处理结果'}</h3>
      <div className="player-container">
        <video
          src={videoUrl}
          controls
          className="video-element"
        >
          您的浏览器不支持视频播放
        </video>
      </div>
      <div className="player-actions">
        <a
          href={videoUrl}
          download
          className="btn btn-primary"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          下载视频
        </a>
      </div>
    </div>
  );
}
