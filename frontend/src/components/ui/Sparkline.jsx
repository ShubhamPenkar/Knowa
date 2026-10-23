import React from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

/**
 * Lightweight SVG sparkline for KPI tiles and data tables.
 * Takes an array of numeric values and renders a smooth area / line curve.
 */
export function Sparkline({
  data = [10, 15, 12, 18, 22, 20, 26],
  width = 72,
  height = 24,
  color = 'teal', // 'teal' | 'coral' | 'amber' | 'muted'
  showTrend = false,
  className = '',
}) {
  if (!data || data.length < 2) return null

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1

  const padding = 2
  const effectiveWidth = width - padding * 2
  const effectiveHeight = height - padding * 2

  const points = data.map((val, idx) => {
    const x = padding + (idx / (data.length - 1)) * effectiveWidth
    const y = height - padding - ((val - min) / range) * effectiveHeight
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })

  const pathD = `M ${points.join(' L ')}`
  const areaD = `M ${padding},${height} L ${points.join(' L ')} L ${width - padding},${height} Z`

  const colorConfig = {
    teal: {
      stroke: 'var(--teal)',
      fill: 'rgba(0, 200, 180, 0.15)',
      trendText: 'text-teal',
    },
    coral: {
      stroke: 'var(--coral)',
      fill: 'rgba(255, 90, 31, 0.15)',
      trendText: 'text-coral',
    },
    amber: {
      stroke: 'var(--amber)',
      fill: 'rgba(245, 184, 56, 0.15)',
      trendText: 'text-[#f5b838]',
    },
    muted: {
      stroke: 'var(--muted)',
      fill: 'rgba(154, 148, 138, 0.12)',
      trendText: 'text-muted',
    },
  }[color] || {
    stroke: 'var(--teal)',
    fill: 'rgba(0, 200, 180, 0.15)',
    trendText: 'text-teal',
  }

  const isUp = data[data.length - 1] > data[0]
  const isDown = data[data.length - 1] < data[0]

  return (
    <div className={`inline-flex items-center gap-1.5 ${className}`}>
      <svg
        width={width}
        height={height}
        className="overflow-visible shrink-0"
        aria-hidden="true"
      >
        <path d={areaD} fill={colorConfig.fill} />
        <path
          d={pathD}
          fill="none"
          stroke={colorConfig.stroke}
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      {showTrend && (
        <span className={`text-[10px] font-medium flex items-center ${colorConfig.trendText}`}>
          {isUp ? (
            <TrendingUp size={11} className="mr-0.5" />
          ) : isDown ? (
            <TrendingDown size={11} className="mr-0.5" />
          ) : (
            <Minus size={11} className="mr-0.5" />
          )}
        </span>
      )}
    </div>
  )
}

export default Sparkline
