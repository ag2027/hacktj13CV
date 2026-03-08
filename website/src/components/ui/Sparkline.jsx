import React from 'react';

export function Sparkline({ data = [], width = 100, height = 30, color = 'var(--accent)' }) {
  if (!data || data.length === 0) {
    return <svg width={width} height={height}><path d="" /></svg>;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1; // Prevent division by zero

  const points = data.map((val, i) => {
    const x = (i / (data.length - 1 || 1)) * width;
    const y = height - ((val - min) / range) * height;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2"
        points={points}
      />
    </svg>
  );
}