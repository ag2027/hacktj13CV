import React from 'react';
import { Card } from '../ui/Card';
import { useRover } from '../../context/RoverContext';

export function LiveFeedPanel() {
  const { droidCamStreamUrl } = useRover();

  return (
    <Card style={{ flex: 1, padding: 0, overflow: 'hidden' }}>
      {droidCamStreamUrl ? (
        <img
          src={droidCamStreamUrl}
          alt="DroidCam live feed"
          style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block', background: '#000' }}
        />
      ) : (
        <div
          style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#000',
            color: 'var(--text-muted)',
            fontSize: '0.95rem'
          }}
        >
          DroidCam stream not configured
        </div>
      )}
    </Card>
  );
}
