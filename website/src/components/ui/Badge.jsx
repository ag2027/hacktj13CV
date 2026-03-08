import React from 'react';

export function Badge({ children, variant = 'info', className = '' }) {
  const variants = {
    critical: { bg: 'rgba(239, 68, 68, 0.15)', color: 'var(--critical)', border: 'var(--critical)', shadow: 'rgba(239, 68, 68, 0.4)' },
    warning: { bg: 'rgba(245, 158, 11, 0.15)', color: 'var(--warning)', border: 'var(--warning)', shadow: 'rgba(245, 158, 11, 0.4)' },
    safe: { bg: 'rgba(34, 197, 94, 0.15)', color: 'var(--safe)', border: 'var(--safe)', shadow: 'rgba(34, 197, 94, 0.4)' },
    info: { bg: 'rgba(11, 110, 253, 0.15)', color: 'var(--accent)', border: 'var(--accent)', shadow: 'rgba(11, 110, 253, 0.4)' }
  };

  const style = variants[variant] || variants.info;

  return (
    <span 
      className={`badge ${className}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '3px 10px',
        borderRadius: '3px', /* blockier look */
        fontSize: '0.7rem',
        fontFamily: 'var(--font-mono), monospace',
        fontWeight: '700',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        backgroundColor: style.bg,
        color: style.color,
        border: `1px solid ${style.border}`,
        boxShadow: `0 0 10px ${style.shadow}, inset 0 0 8px ${style.bg}`,
        textShadow: `0 0 8px ${style.color}`
      }}
    >
      {children}
    </span>
  );
}