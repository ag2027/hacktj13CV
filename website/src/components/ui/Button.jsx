import React from 'react';

export function Button({ 
  children, 
  variant = 'primary', 
  onClick, 
  className = '', 
  disabled = false,
  style = {} 
}) {
  const baseStyle = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '8px 16px',
    borderRadius: '4px', /* sharper, more tactical corners */
    fontWeight: '700',
    fontSize: '0.85rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    cursor: disabled ? 'not-allowed' : 'pointer',
    border: '1px solid transparent',
    transition: 'all 0.2s ease',
    opacity: disabled ? 0.6 : 1,
    fontFamily: 'var(--font-mono), monospace',
    boxShadow: '0 2px 10px rgba(0,0,0,0.3)',
    ...style
  };

  const variants = {
    primary: {
      background: 'rgba(11, 110, 253, 0.15)',
      color: 'var(--accent)',
      borderColor: 'var(--accent)',
      boxShadow: '0 0 10px rgba(11, 110, 253, 0.2), inset 0 0 5px rgba(11, 110, 253, 0.1)',
    },
    danger: {
      background: 'rgba(239, 68, 68, 0.15)',
      color: 'var(--critical)',
      borderColor: 'var(--critical)',
      boxShadow: '0 0 10px rgba(239, 68, 68, 0.2), inset 0 0 5px rgba(239, 68, 68, 0.1)',
    },
    outline: {
      background: 'transparent',
      border: '1px solid rgba(255, 255, 255, 0.15)',
      color: 'var(--text-muted)',
    }
  };

  const currentStyle = { ...baseStyle, ...(variants[variant] || variants.primary) };

  return (
    <button 
      className={className} 
      style={currentStyle} 
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}