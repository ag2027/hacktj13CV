import React from 'react';
import { Card } from './components/ui/Card';
import { Badge } from './components/ui/Badge';
import { Button } from './components/ui/Button';
import { StatusDot } from './components/ui/StatusDot';
import { Sparkline } from './components/ui/Sparkline';

function App() {
  return (
    <div style={{ padding: 'var(--space-4)' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)' }}>
        <h1 style={{ color: 'var(--accent)', margin: 0, letterSpacing: '1px' }}>SENTINEL</h1>
        <div style={{ display: 'flex', gap: 'var(--space-4)' }}>
          <StatusDot status="connected" label="Camera" />
          <StatusDot status="connecting" label="Arduino" />
          <StatusDot status="disconnected" label="CV Engine" />
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 'var(--space-4)' }}>
        <Card>
          <h2>Live Feed Placeholder</h2>
          <div style={{ background: '#000', height: '200px', borderRadius: '4px', marginBottom: 'var(--space-3)' }}></div>
          <Button>Toggle Overlay</Button>
        </Card>

        <Card>
          <h2>Threat Log</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Weapon Detected</span>
              <Badge variant="critical">CRITICAL</Badge>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Pathway Blocked</span>
              <Badge variant="warning">WARNING</Badge>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Area Scanned</span>
              <Badge variant="safe">SAFE</Badge>
            </div>
          </div>
        </Card>

        <Card>
          <h2>Telemetry</h2>
          <div style={{ marginBottom: 'var(--space-4)' }}>
            <span style={{ color: 'var(--text-muted)' }}>Distance Sensor</span>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 'var(--space-2)', marginTop: 'var(--space-2)' }}>
              <span style={{ fontSize: '2rem', fontFamily: 'var(--font-mono)' }}>45.2</span>
              <span style={{ paddingBottom: '0.5rem' }}>cm</span>
              <div style={{ marginLeft: 'auto' }}>
                <Sparkline data={[12, 14, 18, 15, 22, 30, 45, 42, 45, 45]} width={100} height={30} />
              </div>
            </div>
          </div>
          
          <Button variant="danger" style={{ width: '100%' }}>EMERGENCY STOP</Button>
        </Card>
      </div>
    </div>
  );
}

export default App;