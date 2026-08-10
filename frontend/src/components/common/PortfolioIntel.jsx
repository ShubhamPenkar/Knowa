import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import Spinner from './Spinner'

/**
 * ROI-by-action + capacity strip for the Follow-ups board.
 */
export default function PortfolioIntel({ projectId }) {
  const { token } = useAuth()
  const [intel, setIntel] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!token) return
    let cancelled = false
    ;(async () => {
      setLoading(true)
      setError('')
      try {
        const q = projectId ? `?project_id=${encodeURIComponent(projectId)}` : ''
        const res = await fetch(`/api/projects/decisions/portfolio-intel${q}`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (!res.ok) {
          if (!cancelled) {
            setIntel(null)
            setError('Could not load portfolio intelligence')
          }
          return
        }
        const data = await res.json()
        if (!cancelled) {
          setIntel(data)
          setError('')
        }
      } catch (err) {
        if (!cancelled) {
          setIntel(null)
          setError(err?.message || 'Could not load portfolio intelligence')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [token, projectId])

  if (error && !intel) {
    return null
  }

  const actions = intel?.actions || []
  const alerts = intel?.capacity_alerts || []
  const counts = intel?.counts || {}

  return (
    <section className="mb-10 border border-mist" aria-label="Portfolio intelligence">
      <div className="px-5 py-4 border-b border-mist flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-display text-lg font-semibold text-ink">Action ROI & capacity</h2>
          <p className="text-sm text-[var(--muted)] mt-1 max-w-2xl">
            {loading && !intel
              ? 'Loading expected lift vs outcomes…'
              : intel?.plain_summary ||
                'See which actions pay off — and when you’re overcommitted.'}
          </p>
        </div>
        <Link to="/cases" className="btn-ghost text-sm shrink-0">
          Open cases
        </Link>
      </div>

      {loading && !intel && (
        <div className="px-5 py-8 flex items-center gap-3 text-sm text-[var(--muted)]">
          <Spinner className="h-4 w-4" /> Reading ledger…
        </div>
      )}

      {intel && actions.length === 0 && (
        <div className="px-5 py-6 text-sm text-[var(--muted)]">
          No committed actions yet. Save a follow-up from a case — ROI and capacity show up here.
        </div>
      )}

      {intel && actions.length > 0 && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-mist border-b border-mist">
            {[
              {
                label: 'Actions tracked',
                value: counts.actions ?? 0,
                accent: 'text-ink',
              },
              {
                label: 'Open follow-ups',
                value: counts.open ?? 0,
                accent: 'text-ink',
              },
              {
                label: 'Outcomes logged',
                value: counts.with_outcomes ?? 0,
                accent: 'text-teal',
              },
              {
                label: 'Over capacity',
                value: counts.over_capacity ?? 0,
                accent: (counts.over_capacity || 0) > 0 ? 'text-coral' : 'text-[var(--muted)]',
              },
            ].map((cell) => (
              <div key={cell.label} className="bg-paper px-4 py-4">
                <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                  {cell.label}
                </div>
                <div
                  className={`mt-1 font-display text-2xl font-semibold tabular-nums ${cell.accent}`}
                >
                  {cell.value}
                </div>
              </div>
            ))}
          </div>

          {alerts.length > 0 && (
            <ul className="border-b border-mist divide-y divide-mist">
              {alerts.map((a) => (
                <li
                  key={a.action_code}
                  className="px-5 py-3 text-sm bg-coral-soft/30 text-ink"
                >
                  <span className="font-medium">Capacity — </span>
                  {a.plain}
                </li>
              ))}
            </ul>
          )}

          <ul className="divide-y divide-mist">
            {actions.slice(0, 8).map((a) => {
              const favPct =
                a.favorable_rate != null ? Math.round(a.favorable_rate * 100) : null
              const lift =
                a.avg_expected_lift_pp != null
                  ? `${a.avg_expected_lift_pp > 0 ? '+' : ''}${a.avg_expected_lift_pp} pp`
                  : '—'
              return (
                <li
                  key={a.action_code}
                  className="px-5 py-4 flex flex-wrap items-start justify-between gap-3 text-sm"
                >
                  <div className="min-w-0 max-w-xl">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-medium text-ink">{a.action_name}</span>
                      {a.over_capacity && (
                        <span className="badge bg-coral-soft border border-coral/30 text-ink">
                          Over capacity
                        </span>
                      )}
                      {a.evidence === 'thin' && (
                        <span className="badge bg-mist text-[var(--muted)]">Thin evidence</span>
                      )}
                      {a.evidence === 'reliable' && (
                        <span className="badge bg-teal-soft/40 text-ink">Enough to read</span>
                      )}
                    </div>
                    <p className="text-xs text-[var(--muted)] mt-1.5 leading-relaxed">
                      {a.autopsy_strip}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-4 text-xs text-[var(--muted)] tabular-nums shrink-0">
                    <div>
                      <div className="uppercase tracking-wide">Open</div>
                      <div className="text-ink text-sm mt-0.5">
                        {a.open_n}
                        <span className="text-[var(--muted)]"> / {a.capacity_open}</span>
                      </div>
                    </div>
                    <div>
                      <div className="uppercase tracking-wide">Favorable</div>
                      <div className="text-ink text-sm mt-0.5">
                        {favPct != null ? `${favPct}%` : '—'}
                        {a.outcome_n ? (
                          <span className="text-[var(--muted)]"> · n={a.outcome_n}</span>
                        ) : null}
                      </div>
                    </div>
                    <div>
                      <div className="uppercase tracking-wide">Expected</div>
                      <div className="text-ink text-sm mt-0.5">{lift}</div>
                    </div>
                  </div>
                </li>
              )
            })}
          </ul>
        </>
      )}
    </section>
  )
}
