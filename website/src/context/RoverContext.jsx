import React, { createContext, useContext, useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

const RoverContext = createContext();
const MAX_ANOMALY_POINTS = 100;
const MAX_ULTRA_EVENTS = 80;

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

function extractUltrasonicEvent(feedData) {
  if (!feedData || typeof feedData !== 'object') return null;

  const now = Date.now();
  const distanceCm =
    toNumber(feedData.ultrasound_distance_cm, NaN) ||
    toNumber(feedData.distance_cm, NaN) ||
    toNumber(feedData.dist_cm, NaN) ||
    toNumber(feedData.ultrasonic?.distance_cm, NaN);

  const thresholdCm =
    toNumber(feedData.ultra_threshold_cm, NaN) ||
    toNumber(feedData.ultrasonic?.threshold_cm, NaN) ||
    10;

  const explicitTrigger =
    Boolean(feedData.ultrasound_triggered_avoidance) ||
    Boolean(feedData.ultrasonic?.triggered) ||
    Boolean(feedData.obstacle_detected);

  const inferredTrigger = Number.isFinite(distanceCm) && distanceCm > 0 && distanceCm <= thresholdCm;
  const triggered = explicitTrigger || inferredTrigger;
  if (!triggered) return null;

  const pose = feedData.robot_pose || feedData.pose || null;
  return {
    id: `ultra-${now}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp: now,
    timeText: new Date(now).toLocaleTimeString(),
    type: 'Ultrasonic Obstacle',
    severity: (Number.isFinite(distanceCm) && distanceCm <= thresholdCm) ? 'critical' : 'warning',
    score: Number.isFinite(distanceCm) ? Math.max(0, Math.min(1, (thresholdCm - distanceCm + 1) / (thresholdCm + 1))) : 0.7,
    distanceCm: Number.isFinite(distanceCm) ? distanceCm : null,
    thresholdCm,
    pose,
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
  const [ultrasonicAnomalies, setUltrasonicAnomalies] = useState([]);

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

  useEffect(() => {
    const event = extractUltrasonicEvent(feedData);
    if (!event) return;

    setUltrasonicAnomalies(prev => {
      const last = prev[0];
      if (last && last.distanceCm === event.distanceCm && (event.timestamp - last.timestamp) < 1000) {
        return prev;
      }
      return [event, ...prev].slice(0, MAX_ULTRA_EVENTS);
    });
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
    ultrasonicAnomalies,
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
