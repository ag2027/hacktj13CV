import React from 'react';

export function Badge({ children, variant = 'info', className = '' }) {
  const variants = {
    critical: { bg: 'rgba(239, 68, 68, 0.2)', color: 'var(--critical)', border: 'var(--critical)' },
    warning: { bg: 'rgba(245, 158, 11, 0.2)', color: 'var(--warning)', border: 'var(--warning)' },
    safe: { bg: 'rgba(34, 197, 94, 0.2)', color: 'var(--safe)', border: 'var(--safe)' },
    info: { bg: 'rgba(11, 110, 253, 0.2)', color: 'var(--accent)', border: 'var(--accent)' }
  };

  const style = variants[variant] || variants.info;

  return (
    <span 
      className={`badge ${className}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '2px 8px',
        borderRadius: 'var(--radius-badge)',
        fontSize: '0.75rem',
        fontWeight: '600',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        backgroundColor: style.bg,
        color: style.color,
        border: `1px solid ${style.border}`
      }}
    >
      {children}
    </span>
  );
}