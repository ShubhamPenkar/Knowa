import React from 'react'

export function Skeleton({ className = '' }) {
  return (
    <div
      className={`rounded skeleton-shimmer ${className}`}
      aria-hidden="true"
    />
  )
}

export function SkeletonCard({ lines = 3, className = '' }) {
  return (
    <div
      className={`bg-surface border border-mist rounded-control p-5 flex flex-col gap-3 ${className}`}
      aria-hidden="true"
    >
      <div className="flex items-center justify-between gap-4">
        <Skeleton className="h-5 w-1/3" />
        <Skeleton className="h-5 w-16 rounded-full" />
      </div>
      <Skeleton className="h-4 w-4/5" />
      {lines > 2 && <Skeleton className="h-4 w-2/3" />}
      <div className="mt-2 pt-3 border-t border-mist/40 flex items-center justify-between">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-20" />
      </div>
    </div>
  )
}

export function SkeletonTable({ rows = 5, cols = 4, className = '' }) {
  return (
    <div className={`border border-mist rounded-control overflow-hidden bg-surface ${className}`} aria-hidden="true">
      <div className="border-b border-mist bg-mist/30 px-4 py-3 flex gap-4">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={i} className="h-4 flex-1" />
        ))}
      </div>
      <div className="divide-y divide-mist/40 p-1">
        {Array.from({ length: rows }).map((_, r) => (
          <div key={r} className="px-4 py-3 flex gap-4 items-center">
            {Array.from({ length: cols }).map((_, c) => (
              <Skeleton
                key={c}
                className={`h-4 flex-1 ${c === 0 ? 'w-1/4' : ''}`}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

export function SkeletonStat({ className = '' }) {
  return (
    <div className={`bg-surface border border-mist rounded-control p-5 flex flex-col gap-2 ${className}`} aria-hidden="true">
      <Skeleton className="h-3.5 w-24" />
      <Skeleton className="h-8 w-16" />
      <Skeleton className="h-3 w-32 mt-1" />
    </div>
  )
}

export default Skeleton
