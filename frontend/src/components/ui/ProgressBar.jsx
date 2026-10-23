import React from 'react'

/**
 * Visual multi-segment or single progress bar for capacity, accuracy, and portfolio distribution.
 */
export function ProgressBar({
  value = 0,
  max = 100,
  color = 'teal', // 'teal' | 'coral' | 'amber' | 'blue'
  size = 'md', // 'xs' | 'sm' | 'md' | 'lg'
  showLabel = false,
  className = '',
}) {
  const pct = Math.min(Math.max((value / (max || 1)) * 100, 0), 100)

  const heightClass = {
    xs: 'h-1',
    sm: 'h-1.5',
    md: 'h-2',
    lg: 'h-3',
  }[size] || 'h-2'

  const colorClass = {
    teal: 'bg-teal',
    coral: 'bg-coral',
    amber: 'bg-[#f5b838]',
    blue: 'bg-[var(--blue)]',
  }[color] || 'bg-teal'

  return (
    <div className={`w-full ${className}`}>
      {showLabel && (
        <div className="flex justify-between text-[11px] text-muted mb-1 tabular-nums">
          <span>{value}</span>
          <span>{Math.round(pct)}%</span>
        </div>
      )}
      <div className={`w-full bg-mist/60 rounded-full overflow-hidden ${heightClass}`}>
        <div
          className={`${colorClass} ${heightClass} rounded-full transition-all duration-300`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

/**
 * Stacked Distribution Bar for Portfolio Status (e.g., Overdue, Due now, Upcoming, Closed)
 */
export function StackedBar({ segments = [], height = 'h-2.5', className = '' }) {
  const total = segments.reduce((sum, s) => sum + (Number(s.value) || 0), 0)
  if (total === 0) return null

  return (
    <div className={`w-full ${className}`}>
      <div className={`w-full bg-mist/40 rounded-full overflow-hidden flex ${height} shadow-inner`}>
        {segments.map((seg, idx) => {
          const val = Number(seg.value) || 0
          if (val === 0) return null
          const pct = (val / total) * 100
          return (
            <div
              key={idx}
              className={`${seg.colorClass} transition-all duration-300 first:rounded-l-full last:rounded-r-full`}
              style={{ width: `${pct}%` }}
              title={`${seg.label}: ${val} (${Math.round(pct)}%)`}
            />
          )
        })}
      </div>
      <div className="flex flex-wrap items-center gap-4 mt-2 text-xs text-muted">
        {segments.map((seg, idx) => {
          const val = Number(seg.value) || 0
          return (
            <div key={idx} className="flex items-center gap-1.5">
              <span className={`h-2 w-2 rounded-full ${seg.colorClass}`} />
              <span className="text-ink font-medium">{val}</span>
              <span className="text-muted">{seg.label}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default ProgressBar
