import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import ProjectPicker from '../components/common/ProjectPicker'
import Spinner from '../components/common/Spinner'
import { getLastProjectId, setLastProjectId } from '../lib/lastProject'

function pct(v) {
  if (v == null || Number.isNaN(Number(v))) return '—'
  return `${Math.round(Number(v) * 100)}%`
}

function scoreLabel(project) {
  const ready = project?.status === 'ready' || project?.status === 'trained'
  const m = project?.active_model
  if (!m) {
    if (ready) {
      return {
        label: 'Ready',
        detail: 'Select to inspect ranking quality and learning.',
        tone: 'teal',
      }
    }
    return { label: 'Not prepared', detail: 'Train this project to see performance.', tone: 'muted' }
  }
  if (project.problem_type === 'regression') {
    const r2 = m.r2_score != null ? Number(m.r2_score) : null
    if (r2 == null) return { label: 'Ready', detail: 'Regression model is live.', tone: 'teal' }
    if (r2 >= 0.5) return { label: 'Strong guide', detail: `R² ${r2.toFixed(2)} — patterns look reliable.`, tone: 'teal' }
    if (r2 >= 0.2) return { label: 'Useful guide', detail: `R² ${r2.toFixed(2)} — directionally helpful.`, tone: 'ink' }
    return { label: 'Rough guide', detail: `R² ${r2.toFixed(2)} — use as a hint only.`, tone: 'coral' }
  }
  const auc = m.auc_roc != null ? Number(m.auc_roc) : null
  const acc = m.accuracy != null ? Number(m.accuracy) : null
  const score = auc ?? acc
  if (score == null) return { label: 'Ready', detail: 'Model is live.', tone: 'teal' }
  if (score >= 0.8) return { label: 'Strong guide', detail: 'Good at ranking who needs attention.', tone: 'teal' }
  if (score >= 0.7) return { label: 'Solid guide', detail: 'Solid enough for a daily priority list.', tone: 'teal' }
  if (score >= 0.6) return { label: 'Useful guide', detail: 'Helpful direction — confirm before big spends.', tone: 'ink' }
  return { label: 'Rough guide', detail: 'Treat as a first filter; add human judgment.', tone: 'coral' }
}

/**
 * Top-level Monitoring — model quality + learning without opening project settings.
 */
export default function Monitoring() {
  const { token } = useAuth()
  const [searchParams, setSearchParams] = useSearchParams()
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)
  const [projectId, setProjectId] = useState('')
  const [project, setProject] = useState(null)
  const [spotCheck, setSpotCheck] = useState(null)
  const [feedback, setFeedback] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [spotLoading, setSpotLoading] = useState(false)

  const readyProjects = useMemo(
    () => projects.filter((p) => p.status === 'ready' || p.status === 'trained'),
    [projects]
  )

  useEffect(() => {
    if (!token) return
    ;(async () => {
      try {
        const res = await fetch('/api/projects', {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.ok) {
          const data = await res.json()
          setProjects(data)
          const ready = data.filter((p) => p.status === 'ready' || p.status === 'trained')
          const fromQuery = searchParams.get('project')
          const last = getLastProjectId()
          const pick =
            ready.find((p) => p.id === fromQuery)?.id ||
            ready.find((p) => p.id === last)?.id ||
            ready[0]?.id ||
            ''
          setProjectId(pick)

          // Enrich cards with active_model metrics (list endpoint omits them)
          if (ready.length) {
            const details = await Promise.all(
              ready.slice(0, 12).map(async (p) => {
                try {
                  const r = await fetch(`/api/projects/${p.id}`, {
                    headers: { Authorization: `Bearer ${token}` },
                  })
                  return r.ok ? await r.json() : p
                } catch {
                  return p
                }
              })
            )
            const byId = Object.fromEntries(details.map((d) => [d.id, d]))
            setProjects((prev) => prev.map((p) => byId[p.id] || p))
          }
        }
      } catch (err) {
        console.error(err)
      }
      setLoading(false)
    })()
  }, [token])

  const loadDetail = async (id) => {
    if (!token || !id) {
      setProject(null)
      setSpotCheck(null)
      setFeedback(null)
      return
    }
    setDetailLoading(true)
    setLastProjectId(id)
    try {
      const [pRes, sRes, fRes] = await Promise.all([
        fetch(`/api/projects/${id}`, { headers: { Authorization: `Bearer ${token}` } }),
        fetch(`/api/projects/${id}/spot-check?limit=50`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch(`/api/projects/${id}/feedback-summary`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ])
      setProject(pRes.ok ? await pRes.json() : null)
      setSpotCheck(sRes.ok ? await sRes.json() : null)
      setFeedback(fRes.ok ? await fRes.json() : null)
    } catch {
      setProject(null)
      setSpotCheck(null)
      setFeedback(null)
    }
    setDetailLoading(false)
  }

  useEffect(() => {
    loadDetail(projectId)
  }, [token, projectId])

  const handleProjectChange = (id) => {
    setProjectId(id)
    const next = new URLSearchParams(searchParams)
    if (id) next.set('project', id)
    else next.delete('project')
    setSearchParams(next, { replace: true })
  }

  const refreshSpot = async () => {
    if (!projectId) return
    setSpotLoading(true)
    try {
      const res = await fetch(`/api/projects/${projectId}/spot-check?limit=50`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) setSpotCheck(await res.json())
    } catch {
      /* ignore */
    }
    setSpotLoading(false)
  }

  const health = scoreLabel(project)
  const m = project?.active_model
  const isReg = project?.problem_type === 'regression'

  if (loading) {
    return (
      <div className="page flex justify-center items-center min-h-[40vh]">
        <Spinner />
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Quality</p>
          <h1 className="page-title">Monitoring</h1>
          <p className="page-sub max-w-xl">
            How well guidance is ranking and matching real outcomes — without opening project setup.
          </p>
        </div>
        {projectId && (
          <div className="flex flex-wrap gap-2">
            <Link to={`/cases?project=${projectId}`} className="btn-secondary text-sm">
              Open cases
            </Link>
            <Link to={`/projects/${projectId}`} className="btn-ghost text-sm">
              Project settings
            </Link>
          </div>
        )}
      </div>

      {readyProjects.length === 0 ? (
        <div className="empty-state">
          <h3 className="font-display text-xl font-semibold text-ink mb-2">Nothing to monitor yet</h3>
          <p className="text-sm text-[var(--muted)] mb-6 max-w-md mx-auto">
            Prepare a project first — then model quality and learning show up here.
          </p>
          <Link to="/projects" className="btn-primary">
            Go to projects
          </Link>
        </div>
      ) : (
        <>
          {/* Org overview cards */}
          <section className="mb-10">
            <h2 className="font-display text-lg font-semibold text-ink mb-3">All ready projects</h2>
            <ul className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {readyProjects.map((p) => {
                const h = scoreLabel(p)
                const selected = p.id === projectId
                return (
                  <li key={p.id}>
                    <button
                      type="button"
                      onClick={() => handleProjectChange(p.id)}
                      className={`w-full text-left border px-4 py-4 rounded-control transition-colors ${
                        selected
                          ? 'border-teal bg-teal-soft/20'
                          : 'border-mist hover:border-teal/40'
                      }`}
                    >
                      <div className="font-medium text-ink truncate">{p.name}</div>
                      <div
                        className={`mt-2 text-sm font-medium ${
                          h.tone === 'coral'
                            ? 'text-coral'
                            : h.tone === 'teal'
                              ? 'text-teal'
                              : 'text-ink'
                        }`}
                      >
                        {h.label}
                      </div>
                      <p className="mt-1 text-xs text-[var(--muted)] leading-relaxed">{h.detail}</p>
                    </button>
                  </li>
                )
              })}
            </ul>
          </section>

          <div className="mb-6">
            <ProjectPicker
              projects={readyProjects}
              value={projectId}
              onChange={handleProjectChange}
              id="monitoring-project"
              label="Inspect project"
            />
          </div>

          {detailLoading ? (
            <div className="flex items-center gap-3 text-sm text-[var(--muted)] py-10">
              <Spinner className="h-4 w-4" /> Loading performance…
            </div>
          ) : !project ? (
            <p className="text-sm text-[var(--muted)]">Could not load this project.</p>
          ) : (
            <div className="space-y-10">
              <section>
                <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
                  <div>
                    <h2 className="font-display text-lg font-semibold text-ink">{project.name}</h2>
                    <p className="text-sm text-[var(--muted)] mt-1">
                      <span className="text-ink font-medium">{health.label}</span>
                      {' — '}
                      {health.detail}
                    </p>
                  </div>
                  {m?.trained_at && (
                    <p className="text-xs text-[var(--muted)]">
                      Last prepared {new Date(m.trained_at).toLocaleDateString()}
                      {m.version != null ? ` · v${m.version}` : ''}
                    </p>
                  )}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-mist border border-mist">
                  {isReg ? (
                    <>
                      <MetricCell label="R²" value={m?.r2_score != null ? Number(m.r2_score).toFixed(2) : '—'} />
                      <MetricCell label="MAE" value={m?.mae != null ? Number(m.mae).toFixed(2) : '—'} />
                      <MetricCell label="RMSE" value={m?.rmse != null ? Number(m.rmse).toFixed(2) : '—'} />
                      <MetricCell label="Guidance" value={health.label} />
                    </>
                  ) : (
                    <>
                      <MetricCell label="Ranking quality (AUC)" value={pct(m?.auc_roc)} accent="text-teal" />
                      <MetricCell label="Accuracy" value={pct(m?.accuracy)} />
                      <MetricCell label="Precision" value={pct(m?.precision)} />
                      <MetricCell label="Recall" value={pct(m?.recall)} />
                    </>
                  )}
                </div>
                <p className="mt-3 text-xs text-[var(--muted)] max-w-2xl leading-relaxed">
                  These are training holdout scores — a guide for whether the ranking is trustworthy,
                  not a guarantee on every future case.
                </p>
              </section>

              {!isReg && (
                <section className="border border-mist">
                  <div className="px-5 py-4 border-b border-mist flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h2 className="font-display text-lg font-semibold text-ink">
                        Does the ranking help?
                      </h2>
                      <p className="text-sm text-[var(--muted)] mt-1">
                        Spot-check against known outcomes in your data
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={refreshSpot}
                      disabled={spotLoading}
                      className="btn-ghost text-sm"
                    >
                      {spotLoading ? 'Checking…' : 'Refresh'}
                    </button>
                  </div>
                  <div className="px-5 py-5">
                    {spotCheck?.supported === false && (
                      <p className="text-sm text-[var(--muted)]">{spotCheck.message}</p>
                    )}
                    {spotCheck && spotCheck.n === 0 && (
                      <p className="text-sm text-[var(--muted)]">
                        {spotCheck.message || 'Not enough known outcomes to check yet.'}
                      </p>
                    )}
                    {spotCheck && spotCheck.n > 0 && (
                      <>
                        <p className="text-sm text-ink max-w-2xl leading-relaxed mb-4">
                          {spotCheck.plain_summary}
                        </p>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-px bg-mist border border-mist">
                          <MetricCell
                            label="Matched known outcomes"
                            value={pct(spotCheck.agree_rate)}
                            accent="text-ink"
                          />
                          <MetricCell
                            label="Right when flagged high"
                            value={
                              spotCheck.high_risk_precision != null
                                ? pct(spotCheck.high_risk_precision)
                                : '—'
                            }
                          />
                          <MetricCell
                            label="Right when flagged low"
                            value={
                              spotCheck.low_risk_true_negative_rate != null
                                ? pct(spotCheck.low_risk_true_negative_rate)
                                : '—'
                            }
                          />
                        </div>
                      </>
                    )}
                    {!spotCheck && (
                      <p className="text-sm text-[var(--muted)]">Spot-check unavailable.</p>
                    )}
                  </div>
                </section>
              )}

              <section className="border border-mist">
                <div className="px-5 py-4 border-b border-mist">
                  <h2 className="font-display text-lg font-semibold text-ink">
                    Outcomes your team recorded
                  </h2>
                  <p className="text-sm text-[var(--muted)] mt-1">
                    Learning from logged results and follow-up check-ins
                  </p>
                </div>
                <div className="px-5 py-5 space-y-4">
                  {feedback ? (
                    <>
                      <p className="text-sm text-[var(--muted)] leading-relaxed max-w-2xl">
                        {feedback.plain_summary}
                      </p>
                      {feedback.learning?.plain && (
                        <p className="text-sm text-teal leading-relaxed max-w-2xl">
                          {feedback.learning.plain}
                        </p>
                      )}
                      <div className="grid sm:grid-cols-3 gap-px bg-mist border border-mist">
                        <MetricCell
                          label="Recorded"
                          value={String(feedback.with_feedback || 0)}
                          hint={`of ${feedback.total_predictions || 0} cases reviewed`}
                        />
                        <MetricCell
                          label="Estimate matched"
                          value={
                            feedback.model_match_rate != null
                              ? pct(feedback.model_match_rate)
                              : '—'
                          }
                        />
                        <MetricCell
                          label="Reshaping rankings"
                          value={String(feedback.learning?.actions_reshaping_rankings ?? 0)}
                          hint="actions with 3+ outcomes"
                        />
                      </div>
                      {Array.isArray(feedback.action_effectiveness_ranked) &&
                        feedback.action_effectiveness_ranked.length > 0 && (
                          <ul className="mt-2 space-y-2">
                            {feedback.action_effectiveness_ranked.slice(0, 6).map((a) => (
                              <li
                                key={a.action_code}
                                className="flex flex-wrap items-baseline justify-between gap-2 text-sm"
                              >
                                <span className="text-ink font-medium">
                                  {a.action_name || a.action_code}
                                </span>
                                <span className="tabular-nums text-[var(--muted)] text-right">
                                  {a.success_n}/{a.n} favorable
                                  {a.reliable ? ' · reshapes ranking' : ' · small sample'}
                                </span>
                              </li>
                            ))}
                          </ul>
                        )}
                    </>
                  ) : (
                    <p className="text-sm text-[var(--muted)]">
                      No outcome log yet. Log results from Cases or follow-up check-ins.
                    </p>
                  )}
                </div>
              </section>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function MetricCell({ label, value, accent = 'text-ink', hint }) {
  return (
    <div className="bg-paper px-4 py-4">
      <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">{label}</div>
      <div className={`mt-1 font-display text-2xl font-semibold tabular-nums ${accent}`}>
        {value}
      </div>
      {hint && <p className="text-xs text-[var(--muted)] mt-1">{hint}</p>}
    </div>
  )
}
