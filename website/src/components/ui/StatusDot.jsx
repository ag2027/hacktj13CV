import React from 'react';

export function StatusDot({ status = 'disconnected', label }) {
  // status: 'connected' (green), 'connecting' (amber), 'disconnected' (red)
  const colors = {
    connected: 'var(--safe)',
    connecting: 'var(--warning)',
    disconnected: 'var(--critical)'
  };
  
  const color = colors[status] || colors.disconnected;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <div 
        style={{ 
          width: '10px', 
          height: '10px', 
          borderRadius: '50%', 
          backgroundColor: color,
          boxShadow: status === 'connecting' ? `0 0 8px ${color}` : 'none',
          animation: status === 'connecting' ? 'pulse 1.5s infinite' : 'none'
        }} 
      />
      {label && (
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          {label}
        </span>
      )}
      <style>
        {`
          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
          }
        `}
      </style>
    </div>
  );
}