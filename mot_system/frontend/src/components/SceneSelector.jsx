import React from 'react';
import './SceneSelector.css';

const sceneIcons = {
  dance: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="5" r="3"/>
      <path d="M12 8v4m0 0l-3 8m3-8l3 8m-6-6h6"/>
    </svg>
  ),
  sports: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10"/>
      <path d="M2 12h20"/>
    </svg>
  ),
  traffic: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="11" width="18" height="10" rx="2"/>
      <circle cx="7.5" cy="18.5" r="1.5"/>
      <circle cx="16.5" cy="18.5" r="1.5"/>
      <path d="M5 11V6a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v5"/>
    </svg>
  ),
};

export default function SceneSelector({ scenes, selectedScene, onSelect }) {
  return (
    <div className="scene-selector">
      <h3 className="scene-selector-title">选择场景</h3>
      <div className="scene-cards">
        {scenes.map((scene) => (
          <div
            key={scene.id}
            className={`scene-card ${selectedScene === scene.id ? 'selected' : ''}`}
            onClick={() => onSelect(scene.id)}
          >
            <div className="scene-icon">
              {sceneIcons[scene.id]}
            </div>
            <div className="scene-info">
              <h4 className="scene-name">{scene.name}</h4>
              <p className="scene-description">{scene.description}</p>
              {scene.supports_anomaly && (
                <span className="badge badge-info">支持异常检测</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
