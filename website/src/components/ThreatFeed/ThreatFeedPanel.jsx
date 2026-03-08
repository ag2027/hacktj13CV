import React, { useState, useEffect } from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { useRover } from '../../context/RoverContext';

export function ThreatFeedPanel() {
  const { feedData, anomalyPoints } = useRover();
  const [threats, setThreats] = useState([]);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    if (feedData && feedData.detections) {
      const newThreats = feedData.detections;

      if (newThreats.length > 0) {
        setThreats(prev => {
          const added = newThreats.map(t => ({
            id: Math.random().toString(36).substring(7),
            type: t.label || t.threat_type || t.type || 'Unknown Threat',
            severity: (t.danger_score || 0) > 0.8 ? 'critical' : 'warning',
            time: new Date().toLocaleTimeString(),
            score: (t.danger_score || t.confidence || 0).toFixed(2)
          }));

          const combined = [...added, ...prev];
          return combined.slice(0, 50);
        });
      }
    }
  }, [feedData]);

  const filteredThreats = threats.filter(t => {
    if (filter === 'all') return true;
    return t.severity === filter;
  });

  return (
    <Card style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', height: '520px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>Threat Log</h2>
        <Badge variant="critical">{threats.filter(t => t.severity === 'critical').length} alerts</Badge>
      </div>

      <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
        <Badge
          variant={filter === 'all' ? 'safe' : 'secondary'}
          style={{ cursor: 'pointer', opacity: filter === 'all' ? 1 : 0.5 }}
          onClick={() => setFilter('all')}
        >
          All
        </Badge>
        <Badge
          variant={filter === 'critical' ? 'critical' : 'secondary'}
          style={{ cursor: 'pointer', opacity: filter === 'critical' ? 1 : 0.5 }}
          onClick={() => setFilter('critical')}
        >
          Critical
        </Badge>
        <Badge
          variant={filter === 'warning' ? 'warning' : 'secondary'}
          style={{ cursor: 'pointer', opacity: filter === 'warning' ? 1 : 0.5 }}
          onClick={() => setFilter('warning')}
        >
          Warning
        </Badge>
      </div>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-2)',
        marginTop: 'var(--space-2)',
        maxHeight: '220px',
        overflowY: 'auto'
      }}>
        {filteredThreats.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: 'var(--space-4)' }}>
            No threats detected
          </div>
        ) : (
          filteredThreats.map(threat => (
            <div key={threat.id} style={{
              padding: 'var(--space-3)',
              background: 'rgba(255,255,255,0.03)',
              borderRadius: '6px',
              borderLeft: `4px solid ${
                threat.severity === 'critical' ? 'var(--critical)' :
                threat.severity === 'warning' ? 'var(--warning)' : 'var(--safe)'
              }`
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-1)' }}>
                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{threat.type}</span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{threat.time}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Score: {threat.score}</span>
                <Badge variant={threat.severity}>
                  {threat.severity.toUpperCase()}
                </Badge>
              </div>
            </div>
          ))
        )}
      </div>

      <div style={{ marginTop: 'var(--space-2)', paddingTop: 'var(--space-2)', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0, fontSize: '1rem' }}>Specific Anomaly Points</h3>
          <Badge variant="secondary">{anomalyPoints.length}</Badge>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)', marginTop: 'var(--space-2)', maxHeight: '180px', overflowY: 'auto' }}>
          {anomalyPoints.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              Waiting for anomaly coordinates...
            </div>
          ) : (
            anomalyPoints.map(point => (
              <div key={point.id} style={{ padding: 'var(--space-2)', background: 'rgba(255,255,255,0.03)', borderRadius: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                  <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{point.type}</span>
                  <span style={{ color: 'var(--text-muted)' }}>{point.timeText}</span>
                </div>
                <div style={{ marginTop: '4px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  center=({point.cx}, {point.cy}) px | bbox=({point.x}, {point.y}, {point.w}, {point.h}) | score={point.score.toFixed(2)}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </Card>
  );
}
