import React, { useState, useEffect } from 'react';
import { Card } from './components/ui/Card';
import { Badge } from './components/ui/Badge';
import { Button } from './components/ui/Button';
import { StatusDot } from './components/ui/StatusDot';
import { LiveFeedPanel } from './components/LiveFeed/LiveFeedPanel';
import { ThreatFeedPanel } from './components/ThreatFeed/ThreatFeedPanel';
import { SystemStatsPanel } from './components/Stats/SystemStatsPanel';
import { useRover } from './context/RoverContext';

function App() {
  const { systemStatus, feedStatus, cvStatsStatus } = useRover();

  return (
    <div style={{ padding: 'var(--space-4)', maxWidth: '1600px', margin: '0 auto', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        paddingBottom: 'var(--space-4)',
        borderBottom: '1px solid rgba(255,255,255,0.1)',
        marginBottom: 'var(--space-4)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)' }}>
          <h1 style={{ color: 'var(--accent)', margin: 0, letterSpacing: '2px', fontWeight: 700 }}>SENTINEL</h1>
          <Badge variant="safe">OBSERVATION MODE</Badge>
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-6)', alignItems: 'center' }}>
          <StatusDot status={systemStatus?.camera?.status || "disconnected"} label="Camera" />
          <StatusDot status={systemStatus?.cv?.status || "disconnected"} label="CV Engine" />
          <StatusDot status={systemStatus?.qml?.status || "unknown"} label="QML Simulator" />
          <div style={{ borderLeft: '1px solid rgba(255,255,255,0.2)', height: '24px', margin: '0 var(--space-2)' }}></div>
          <StatusDot status={feedStatus === 'connected' ? 'connected' : 'disconnected'} label="WS Feed" />
        </div>
      </header>

      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 350px', 
        gap: 'var(--space-4)',
        flex: 1,
        minHeight: 0
      }}>
        {/* Main Feed Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', minHeight: 0 }}>
          <LiveFeedPanel />
        </div>

        {/* Info Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', overflowY: 'auto', minHeight: 0 }}>
          <ThreatFeedPanel />
          <SystemStatsPanel />
        </div>
      </div>
    </div>
  );
}

export default App;