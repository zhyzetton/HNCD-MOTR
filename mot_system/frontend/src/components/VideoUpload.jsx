import React, { useRef } from 'react';
import { useFileDrop } from '../hooks/useWebSocket';
import './VideoUpload.css';

export default function VideoUpload({ onFileSelect, selectedFile, disabled }) {
  const fileInputRef = useRef(null);

  const handleFileDrop = (file) => {
    if (!disabled) {
      onFileSelect(file);
    }
  };

  const { isDragging, dragHandlers } = useFileDrop(handleFileDrop);

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="video-upload">
      <h3 className="upload-title">上传视频</h3>

      <div
        className={`upload-zone ${isDragging ? 'dragging' : ''} ${disabled ? 'disabled' : ''}`}
        onClick={handleClick}
        {...dragHandlers}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          disabled={disabled}
        />

        {selectedFile ? (
          <div className="file-preview">
            <div className="file-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="23 7 16 12 23 17 23 7"/>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
              </svg>
            </div>
            <div className="file-info">
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">{formatFileSize(selectedFile.size)}</span>
            </div>
            <button
              className="btn btn-secondary remove-btn"
              onClick={(e) => {
                e.stopPropagation();
                onFileSelect(null);
              }}
            >
              移除
            </button>
          </div>
        ) : (
          <div className="upload-placeholder">
            <div className="upload-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <p className="upload-text">
              拖拽视频文件到此处，或点击选择
            </p>
            <p className="upload-hint">
              支持 MP4、AVI、MOV 等常见格式
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
