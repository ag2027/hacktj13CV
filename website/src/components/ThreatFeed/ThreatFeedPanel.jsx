import React, { useState, useEffect } from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { useRover } from '../../context/RoverContext';

export function ThreatFeedPanel() {
  const { feedData } = useRover();
  const [threats, setThreats] = useState([]);
  const [filter, setFilter] = useState('all'); // 'all', 'critical', 'warning'

  useEffect(() => {
    if (feedData && feedData.detections) {
      const newThreats = feedData.detections;
      
      // We could append to history, or just show current frame threats.
      // For this implementation, we accumulate unique distinct threats or just show a rolling log.
      // Let's create a rolling log.
      if (newThreats.length > 0) {
        setThreats(prev => {
          const added = newThreats.map(t => ({
            id: Math.random().toString(36).substring(7),
            type: t.label || t.type || 'Unknown Threat',
            severity: (t.danger_score || 0) > 0.8 ? 'critical' : 'warning',
            time: new Date().toLocaleTimeString(),
            score: (t.danger_score || t.confidence || 0).toFixed(2)
          }));
          
          const combined = [...added, ...prev];
          // Keep last 50
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
    <Card style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', height: '400px' }}>
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
        flex: 1,
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
    </Card>
  );
}
