import React, { useState, useEffect } from 'react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Sparkline } from '../ui/Sparkline';
import { useRover } from '../../context/RoverContext';
import { Badge } from '../ui/Badge';

export function SystemStatsPanel() {
  const { cvStatsData, feedData, apiBase } = useRover();
  const [latencyHistory, setLatencyHistory] = useState(Array(10).fill(0));
  const [dispatchCount, setDispatchCount] = useState(0);

  useEffect(() => {
    if (cvStatsData && cvStatsData.pipeline_latency_ms) {
      setLatencyHistory(prev => {
        const next = [...prev, cvStatsData.pipeline_latency_ms];
        return next.slice(-10);
      });
    }
  }, [cvStatsData]);

  const handleManualDispatch = async () => {
    if (!feedData?.confirmed_threats?.length) {
      alert("No confirmed threats to dispatch.");
      return;
    }
    
    try {
      const payload = {
        threats: feedData.confirmed_threats,
        image_base64: feedData.image_jpeg_base64,
        session_id: "manual-dispatch-" + Date.now()
      };
      
      const res = await fetch(`${apiBase}/api/terac`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      if (res.ok) {
        setDispatchCount(c => c + 1);
      }
    } catch (err) {
      console.error("Dispatch failed", err);
    }
  };

  const latency = cvStatsData?.pipeline_latency_ms || 0;
  const groqCalls = cvStatsData?.groq_calls || 0;
  const groqCap = cvStatsData?.groq_cap || 100;
  const fillPercent = Math.min((groqCalls / groqCap) * 100, 100);

  return (
    <Card style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <h2 style={{ margin: 0 }}>System Pipeline</h2>
      </div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
        
        {/* Pipeline Latency */}
        <div>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>CV Processing Latency</span>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 'var(--space-2)', marginTop: 'var(--space-1)' }}>
            <span style={{ fontSize: '1.5rem', fontFamily: 'var(--font-mono)' }}>{latency}</span>
            <span style={{ paddingBottom: '0.2rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>ms</span>
            <div style={{ marginLeft: 'auto' }}>
              <Sparkline data={latencyHistory} width={80} height={20} />
            </div>
          </div>
        </div>

        {/* API Usage */}
        <div>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Groq LLM Calls</span>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 'var(--space-2)', marginTop: 'var(--space-1)' }}>
            <span style={{ fontSize: '1.5rem', fontFamily: 'var(--font-mono)' }}>{groqCalls}</span>
            <span style={{ paddingBottom: '0.2rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>/ {groqCap} req limit</span>
          </div>
          <div style={{ 
            height: '4px', 
            background: 'rgba(255,255,255,0.1)', 
            borderRadius: '2px', 
            marginTop: 'var(--space-2)',
            overflow: 'hidden'
          }}>
            <div style={{ height: '100%', width: `${fillPercent}%`, background: fillPercent > 90 ? 'var(--critical)' : 'var(--accent)' }} />
          </div>
        </div>
        
        {/* CV Frames / Sec */}
        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          <span>Total Frames: {cvStatsData?.frames_processed || 0}</span>
          <span>FPS: {latency > 0 ? (1000 / latency).toFixed(1) : 0}</span>
        </div>

        {/* Terac Dispatch */}
        <div style={{ 
          marginTop: 'var(--space-2)', 
          paddingTop: 'var(--space-3)', 
          borderTop: '1px solid rgba(255,255,255,0.1)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-3)'
        }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <h3 style={{ margin: 0, fontSize: '1rem' }}>Terac API Uplink</h3>
              <Badge variant="safe">Active</Badge>
            </div>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{dispatchCount} pending dispatch records</span>
          </div>
          <Button onClick={handleManualDispatch} style={{ width: '100%', justifyContent: 'center' }}>
            Force Manual Dispatch
          </Button>
        </div>

      </div>
    </Card>
  );
}
