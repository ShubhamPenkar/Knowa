import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

/**
 * Compact org pulse — overdue follow-ups, don't-act queue, model health.
 * @param {object} props
 * @param {string} [props.projectId]
 * @param {'default'|'follow-ups'} [props.variant] — on Follow-ups, skip the redundant due cell
 */
export default function OrgHealthStrip({ projectId, variant = 'default' }) {
  const { token } = useAuth()
  const [health, setHealth] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!token) return
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('/api/projects/org-health', {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (!res.ok) {
          if (!cancelled) setError('Could not load org health')
          return
        }
        const data = await res.json()
        if (!cancelled) {
          setHealth(data)
          setError('')
        }
      } catch (err) {
        if (!cancelled) setError(err?.message || 'Could not load org health')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [token])

  if (error && !health) {
    return null
  }
  if (!health) {
    return (
      <div className="mb-8 border border-mist px-4 py-4">
        <div className="skeleton h-3 w-24 mb-3" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="skeleton h-14" />
          ))}
        </div>
      </div>
    )
  }

  const c = health.counts || {}
  const pid = projectId || health.primary_project_id

  const cells = [
    {
      label: 'Follow-ups due',
      value: c.due_attention ?? 0,
      accent: (c.due_attention || 0) > 0 ? 'text-coral' : 'text-ink',
      to: '/follow-ups',
      cta: 'Open',
      hideOn: ['follow-ups'],
    },
    {
      label: "Logged don't act",
      value: c.soft_cases ?? 0,
      accent: (c.soft_cases || 0) > 0 ? 'text-coral' : 'text-ink',
      to: pid ? `/cases?project=${pid}&filter=dont-act` : '/cases?filter=dont-act',
      cta: 'Open',
    },
    {
      label: 'Ready projects',
      value: c.ready_projects ?? 0,
      accent: 'text-teal',
      to: '/projects',
      cta: 'Manage',
    },
    {
      label: 'Rough guides',
      value: c.rough_models ?? 0,
      accent: (c.rough_models || 0) > 0 ? 'text-coral' : 'text-[var(--muted)]',
      to: '/monitoring',
      cta: 'Monitor',
    },
  ].filter((cell) => !(cell.hideOn || []).includes(variant))

  return (
    <section className="mb-8 border border-mist" aria-label="Organization health">
      <div className="px-4 py-3 border-b border-mist flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-[11px] uppercase tracking-wide text-teal font-semibold">Today</p>
          <p className="text-sm text-ink mt-0.5 max-w-2xl">{health.plain_summary}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {variant !== 'follow-ups' && (
            <Link to="/follow-ups" className="btn-ghost text-xs py-1.5">
              Follow-ups
            </Link>
          )}
          <Link
            to={pid ? `/cases?project=${pid}&filter=dont-act` : '/cases?filter=dont-act'}
            className="btn-ghost text-xs py-1.5"
          >
            Don&apos;t-act queue
          </Link>
          <Link to="/monitoring" className="btn-ghost text-xs py-1.5">
            Monitoring
          </Link>
        </div>
      </div>
      <div
        className={`grid grid-cols-2 gap-px bg-mist ${
          cells.length >= 4 ? 'md:grid-cols-4' : 'md:grid-cols-3'
        }`}
      >
        {cells.map((cell) => (
          <Link
            key={cell.label}
            to={cell.to}
            className="bg-paper px-4 py-4 hover:bg-mist/30 transition-colors block"
          >
            <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
              {cell.label}
            </div>
            <div
              className={`mt-1 font-display text-2xl font-semibold tabular-nums ${cell.accent}`}
            >
              {cell.value}
            </div>
            <span className="text-xs text-teal mt-1 inline-block">{cell.cta} →</span>
          </Link>
        ))}
      </div>
    </section>
  )
}
