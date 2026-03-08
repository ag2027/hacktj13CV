import React from 'react';

export function StatusDot({ status = 'disconnected', label }) {
  // status: 'connected' (green), 'connecting' (amber), 'disconnected' (red)
  const colors = {
    connected: 'var(--safe)',
    connecting: 'var(--warning)',
    disconnected: 'var(--critical)'
  };
  
  const color = colors[status] || colors.disconnected;
  const isPulsing = status === 'connecting' || status === 'disconnected';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontFamily: 'var(--font-mono), monospace' }}>
      <div 
        style={{ 
          width: '12px', 
          height: '12px', 
          borderRadius: '50%', 
          backgroundColor: color,
          boxShadow: `0 0 12px ${color}, inset 0 0 4px rgba(255,255,255,0.8)`,
          border: `1px solid rgba(255,255,255,0.3)`,
          animation: isPulsing ? 'statusPulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none'
        }} 
      />
      {label && (
        <span style={{ 
          fontSize: '0.8rem', 
          color: color, 
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
          fontWeight: '600',
          textShadow: `0 0 10px ${color}`
        }}>
          {label}
        </span>
      )}
      <style>
        {`
          @keyframes statusPulse {
            0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 12px ${color}, inset 0 0 4px rgba(255,255,255,0.8); }
            50% { opacity: 0.6; transform: scale(0.9); box-shadow: 0 0 4px ${color}, inset 0 0 2px rgba(255,255,255,0.5); }
          }
        `}
      </style>
    </div>
  );
}