import { useState, useEffect, useCallback, useRef } from 'react';
import { createWebSocket } from '../services/api';

/**
 * WebSocket Hook - 用于实时进度更新
 */
export function useWebSocket(taskId) {
  const [progress, setProgress] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  const connect = useCallback(() => {
    if (!taskId || wsRef.current) return;

    wsRef.current = createWebSocket(
      taskId,
      // onMessage
      (data) => {
        setProgress(data);
        setStatus(data.status);

        if (data.type === 'error') {
          setError(data.message);
        }

        if (data.type === 'completed' || data.type === 'failed') {
          // 任务完成后关闭连接
          if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
          }
        }
      },
      // onError
      (err) => {
        setError('WebSocket connection error');
        setIsConnected(false);
      },
      // onClose
      () => {
        setIsConnected(false);
        wsRef.current = null;
      }
    );

    wsRef.current.onopen = () => {
      setIsConnected(true);
    };
  }, [taskId]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  useEffect(() => {
    if (taskId) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [taskId, connect, disconnect]);

  return {
    progress,
    status,
    error,
    isConnected,
    connect,
    disconnect,
  };
}

/**
 * 文件拖放 Hook
 */
export function useFileDrop(onFileDrop) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('video/')) {
        onFileDrop(file);
      }
    }
  }, [onFileDrop]);

  return {
    isDragging,
    dragHandlers: {
      onDragEnter: handleDragEnter,
      onDragLeave: handleDragLeave,
      onDragOver: handleDragOver,
      onDrop: handleDrop,
    },
  };
}
