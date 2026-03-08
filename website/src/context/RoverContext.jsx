import React, { createContext, useContext, useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

const RoverContext = createContext();

export function RoverProvider({ children }) {
  // Use current window host for API, fallback to localhost:8090 if dev
  const isDev = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const apiBase = isDev ? 'http://127.0.0.1:8090' : '';
  const wsBase = isDev ? 'ws://127.0.0.1:8090' : `ws://${window.location.host}`;

  const { data: feedData, status: feedStatus } = useWebSocket(`${wsBase}/ws/feed`);
  const { data: cvStatsData, status: cvStatsStatus } = useWebSocket(`${wsBase}/ws/cv_stats`);

  const [systemStatus, setSystemStatus] = useState({
    camera: { status: 'disconnected' },
    cv: { status: 'disconnected' },
    qml: { status: 'unknown' }
  });

  // Poll status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${apiBase}/api/status`);
        if (res.ok) {
          const data = await res.json();
          setSystemStatus(data);
        }
      } catch (err) {
        // console.error('Failed to fetch status', err);
      }
    };
    
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [apiBase]);

  const value = {
    apiBase,
    wsBase,
    feedData,
    feedStatus,
    cvStatsData,
    cvStatsStatus,
    systemStatus
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
