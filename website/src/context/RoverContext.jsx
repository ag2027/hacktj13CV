import React, { createContext, useContext, useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

const RoverContext = createContext();
const MAX_ANOMALY_POINTS = 100;

function toNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function extractPointFromDetection(detection) {
  if (!detection || typeof detection !== 'object') {
    return null;
  }

  const bbox = detection.bbox || {};
  const x = toNumber(bbox.x);
  const y = toNumber(bbox.y);
  const w = toNumber(bbox.w);
  const h = toNumber(bbox.h);

  return {
    x,
    y,
    w,
    h,
    cx: Math.round(x + w / 2),
    cy: Math.round(y + h / 2),
  };
}

export function RoverProvider({ children }) {
  const isDev = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const apiBase = isDev ? 'http://127.0.0.1:8090' : '';
  const wsBase = isDev ? 'ws://127.0.0.1:8090' : `ws://${window.location.host}`;

  const { data: feedData, status: feedStatus } = useWebSocket(`${wsBase}/ws/feed`);
  const { data: cvStatsData, status: cvStatsStatus } = useWebSocket(`${wsBase}/ws/cv_stats`);

  const [systemStatus, setSystemStatus] = useState({
    camera: { status: 'disconnected' },
    cv: { status: 'disconnected' },
    qml: { status: 'unknown' },
    droidcam_stream_url: ''
  });
  const [anomalyPoints, setAnomalyPoints] = useState([]);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${apiBase}/api/status`);
        if (res.ok) {
          const data = await res.json();
          setSystemStatus(data);
        }
      } catch (err) {
        // ignore polling errors and keep last known status
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [apiBase]);

  useEffect(() => {
    if (!feedData?.detections?.length) {
      return;
    }

    const now = Date.now();
    const nextPoints = feedData.detections
      .map((detection, index) => {
        const point = extractPointFromDetection(detection);
        if (!point) {
          return null;
        }

        return {
          id: `${now}-${index}-${Math.random().toString(36).slice(2, 8)}`,
          timestamp: now,
          timeText: new Date(now).toLocaleTimeString(),
          type: detection.threat_type || detection.label || 'unknown',
          severity: detection.severity || 'unknown',
          score: toNumber(detection.danger_score, toNumber(detection.confidence, 0)),
          ...point,
        };
      })
      .filter(Boolean);

    if (!nextPoints.length) {
      return;
    }

    setAnomalyPoints(prev => [...nextPoints, ...prev].slice(0, MAX_ANOMALY_POINTS));
  }, [feedData]);

  const droidCamStreamUrl = (
    systemStatus?.droidcam_stream_url ||
    import.meta.env.VITE_DROIDCAM_STREAM_URL ||
    ''
  ).trim();

  const value = {
    apiBase,
    wsBase,
    feedData,
    feedStatus,
    cvStatsData,
    cvStatsStatus,
    systemStatus,
    anomalyPoints,
    droidCamStreamUrl,
  };

  return (
    <RoverContext.Provider value={value}>
      {children}
    </RoverContext.Provider>
  );
}

export function useRover() {
  return useContext(RoverContext);
}
