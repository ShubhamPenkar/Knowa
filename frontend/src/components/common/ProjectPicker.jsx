import React from 'react'

/**
 * Compact project selector for top-level hubs (Cases, What-if).
 */
export default function ProjectPicker({
  projects = [],
  value,
  onChange,
  id = 'hub-project',
  label = 'Project',
  emptyLabel = 'No ready projects yet',
  className = '',
}) {
  if (!projects.length) {
    return (
      <p className={`text-sm text-[var(--muted)] ${className}`}>{emptyLabel}</p>
    )
  }

  return (
    <div className={className}>
      <label htmlFor={id} className="block text-xs font-medium text-[var(--muted)] mb-1.5">
        {label}
      </label>
      <select
        id={id}
        value={value || ''}
        onChange={(e) => onChange?.(e.target.value)}
        className="input max-w-md"
      >
        {projects.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>
    </div>
  )
}
