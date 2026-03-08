import React, { useRef, useEffect, useState } from 'react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { useRover } from '../../context/RoverContext';
import { Badge } from '../ui/Badge';

export function LiveFeedPanel() {
  const [showOverlay, setShowOverlay] = useState(true);
  const canvasRef = useRef(null);
  const { feedData, feedStatus, droidCamStreamUrl } = useRover();
  const useDroidCamPrimary = Boolean(droidCamStreamUrl);

  useEffect(() => {
    if (useDroidCamPrimary) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (feedData && feedData.image_jpeg_base64) {
      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
        const x = (canvas.width / 2) - (img.width / 2) * scale;
        const y = (canvas.height / 2) - (img.height / 2) * scale;
        const w = img.width * scale;
        const h = img.height * scale;

        ctx.drawImage(img, x, y, w, h);

        if (showOverlay && feedData.detections) {
          // Optional overlay rendering hook.
        }
      };
      img.src = `data:image/jpeg;base64,${feedData.image_jpeg_base64}`;
    } else {
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 1;
      for (let i = 0; i < canvas.width; i += 50) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke();
      }
      for (let i = 0; i < canvas.height; i += 50) {
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(canvas.width, i); ctx.stroke();
      }

      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.font = '16px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting video stream...', canvas.width / 2, canvas.height / 2);
    }
  }, [feedData, showOverlay, useDroidCamPrimary]);

  return (
    <Card style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: 'var(--space-3)', alignItems: 'center' }}>
          <h2 style={{ margin: 0 }}>Live Operation Feed</h2>
          {feedStatus === 'connected' ? (
            <Badge variant="safe">LIVE</Badge>
          ) : (
            <Badge variant="critical">OFFLINE</Badge>
          )}
          <Badge variant={useDroidCamPrimary ? 'safe' : 'secondary'}>
            {useDroidCamPrimary ? 'DROIDCAM' : 'PIPELINE'}
          </Badge>
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
          <Button onClick={() => setShowOverlay(!showOverlay)} variant="secondary" disabled={useDroidCamPrimary}>
            {showOverlay ? 'Hide Overlay' : 'Show Overlay'}
          </Button>
        </div>
      </div>

      <div style={{
        flex: 1,
        backgroundColor: '#000',
        borderRadius: '8px',
        overflow: 'hidden',
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: '1px solid rgba(255,255,255,0.1)'
      }}>
        {useDroidCamPrimary ? (
          <img
            src={droidCamStreamUrl}
            alt="DroidCam live feed"
            style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
          />
        ) : (
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        )}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
        <span>Frame latency: {feedData?.pipeline_time_s ? (feedData.pipeline_time_s * 1000).toFixed(0) : '0'}ms</span>
        <span>{useDroidCamPrimary ? 'Source: DroidCam URL' : 'Source: Gateway pipeline'}</span>
      </div>
    </Card>
  );
}
