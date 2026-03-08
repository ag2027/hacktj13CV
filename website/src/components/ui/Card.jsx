import React from 'react';

export function Card({ children, className = '', style }) {
  return (
    <div 
      className={`card ${className}`}
      style={{
        background: 'rgba(11, 18, 32, 0.85)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(11, 110, 253, 0.2)',
        borderRadius: '6px', /* sharper corners for tactical look */
        padding: '1.25rem',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
        color: 'var(--text-primary)',
        position: 'relative',
        overflow: 'hidden',
        ...style
      }}
    >
      {/* Decorative tactical corner line */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '2px',
        background: 'linear-gradient(90deg, var(--accent) 0%, transparent 100%)',
        opacity: 0.5
      }} />
      {children}
    </div>
  );
}