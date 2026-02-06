import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import './AnomalyChart.css';

export default function AnomalyChart({ scores, timestamps, threshold = 0.1 }) {
  if (!scores || scores.length === 0) {
    return null;
  }

  // 准备图表数据
  const data = scores.map((score, index) => ({
    time: timestamps ? timestamps[index].toFixed(1) : index,
    score: score,
    isAnomaly: score > threshold,
  }));

  // 统计信息
  const anomalyCount = scores.filter(s => s > threshold).length;
  const maxScore = Math.max(...scores);
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;

  return (
    <div className="anomaly-chart">
      <div className="chart-header">
        <h3>异常检测分析</h3>
        <div className="chart-stats">
          <div className="chart-stat">
            <span className="stat-label">异常帧数</span>
            <span className="stat-value anomaly-count">{anomalyCount}</span>
          </div>
          <div className="chart-stat">
            <span className="stat-label">最大分数</span>
            <span className="stat-value">{maxScore.toFixed(3)}</span>
          </div>
          <div className="chart-stat">
            <span className="stat-label">平均分数</span>
            <span className="stat-value">{avgScore.toFixed(3)}</span>
          </div>
        </div>
      </div>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="time"
              stroke="#666"
              tick={{ fill: '#666', fontSize: 12 }}
              label={{ value: '时间 (秒)', position: 'bottom', fill: '#666' }}
            />
            <YAxis
              stroke="#666"
              tick={{ fill: '#666', fontSize: 12 }}
              domain={[0, 1]}
              label={{ value: '异常分数', angle: -90, position: 'insideLeft', fill: '#666' }}
            />
            <Tooltip
              contentStyle={{
                background: '#1e1e1e',
                border: '1px solid #333',
                borderRadius: '8px',
              }}
              labelStyle={{ color: '#a0a0a0' }}
              formatter={(value, name) => [value.toFixed(4), '异常分数']}
              labelFormatter={(label) => `时间: ${label}s`}
            />
            <ReferenceLine
              y={threshold}
              stroke="#ef4444"
              strokeDasharray="5 5"
              label={{ value: '阈值', fill: '#ef4444', fontSize: 12 }}
            />
            <Line
              type="monotone"
              dataKey="score"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#3b82f6' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {anomalyCount > 0 && (
        <div className="anomaly-warning">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="20" height="20">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          <span>检测到 {anomalyCount} 帧异常，请关注相关时间段的视频内容</span>
        </div>
      )}
    </div>
  );
}
