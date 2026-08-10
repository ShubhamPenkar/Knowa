import React from 'react'

/** Lightweight loading indicator matching App / page loaders. */
export default function Spinner({ className = '', label = 'Loading' }) {
  return (
    <div
      className={`h-6 w-6 border-2 border-teal border-t-transparent rounded-full animate-spin ${className}`}
      role="status"
      aria-label={label}
    />
  )
}
