import React from 'react';

export function Card({ children, className = '', style }) {
  return (
    <div 
      className={`card ${className}`}
      style={{
        background: 'var(--card-bg)',
        border: '1px solid var(--card-border)',
        borderRadius: 'var(--radius-card)',
        padding: 'var(--space-4)',
        boxShadow: '0 6px 24px rgba(2,6,23,0.6)',
        color: 'var(--text-primary)',
        ...style
      }}
    >
      {children}
    </div>
  );
}