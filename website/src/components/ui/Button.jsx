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
    borderRadius: '8px',
    fontWeight: '600',
    cursor: disabled ? 'not-allowed' : 'pointer',
    border: 'none',
    transition: 'opacity 0.2s',
    opacity: disabled ? 0.6 : 1,
    fontFamily: 'inherit',
    ...style
  };

  const variants = {
    primary: {
      background: 'var(--accent)',
      color: '#fff',
    },
    danger: {
      background: 'var(--critical)',
      color: '#fff',
    },
    outline: {
      background: 'transparent',
      border: '1px solid rgba(255,255,255,0.2)',
      color: 'var(--text-primary)',
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