import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Activity,
  ShieldCheck,
  CheckCircle2,
  BarChart3,
  RefreshCw,
  Sliders,
  Sparkles,
  TrendingUp,
  Briefcase,
  Settings,
  ArrowRight,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { useViewMode } from '../context/ViewModeContext'
import ProjectPicker from '../components/common/ProjectPicker'
import Spinner from '../components/common/Spinner'
import { getLastProjectId, setLastProjectId } from '../lib/lastProject'
import { Badge } from '../components/ui/Badge'
import { ProgressBar } from '../components/ui/ProgressBar'
import { EmptyState } from '../components/ui/EmptyState'
import { SkeletonCard, SkeletonStat } from '../components/ui/Skeleton'

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
  const { plainEnglish, term } = useViewMode()
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
          <p className="page-kicker">Operational Trust</p>
          <h1 className="page-title">Model Health & Reliability</h1>
          <p className="page-sub max-w-xl">
            Evaluate whether prediction guidance remains reliable over time, verify ranking quality against known cases, and inspect feedback learning from recorded business outcomes.
          </p>
        </div>
        {projectId && (
          <div className="flex flex-wrap gap-2">
            <Link to={`/cases?project=${projectId}`} className="btn-secondary text-xs inline-flex items-center gap-1.5">
              <Briefcase size={13} />
              <span>Open cases</span>
            </Link>
            <Link to={`/projects/${projectId}`} className="btn-ghost text-xs inline-flex items-center gap-1.5">
              <Settings size={13} />
              <span>Project settings</span>
            </Link>
          </div>
        )}
      </div>

      {readyProjects.length === 0 ? (
        <EmptyState
          icon={Activity}
          title="Nothing to monitor yet"
          description="Prepare a project first — then model quality, operational drift, and feedback learning show up here."
          actionLabel="Go to Projects"
          actionHref="/projects"
        />
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
                      <MetricCell
                        label={plainEnglish ? 'Pattern Reliability (R²)' : 'R² Score'}
                        value={m?.r2_score != null ? Number(m.r2_score).toFixed(2) : '—'}
                      />
                      <MetricCell
                        label={plainEnglish ? 'Avg Prediction Error (MAE)' : 'Mean Absolute Error (MAE)'}
                        value={m?.mae != null ? Number(m.mae).toFixed(2) : '—'}
                      />
                      <MetricCell
                        label={plainEnglish ? 'Severe Miss Penalty (RMSE)' : 'Root Mean Sq Error (RMSE)'}
                        value={m?.rmse != null ? Number(m.rmse).toFixed(2) : '—'}
                      />
                      <MetricCell
                        label={plainEnglish ? 'Usability Guidance' : 'Readiness Status'}
                        value={health.label}
                      />
                    </>
                  ) : (
                    <>
                      <MetricCell
                        label={plainEnglish ? 'Ranking Reliability (AUC)' : 'ROC-AUC Score'}
                        value={pct(m?.auc_roc)}
                        accent="text-teal"
                        progress={m?.auc_roc}
                        color="teal"
                      />
                      <MetricCell
                        label={plainEnglish ? 'Overall Match Rate' : 'Accuracy Score'}
                        value={pct(m?.accuracy)}
                        progress={m?.accuracy}
                        color="teal"
                      />
                      <MetricCell
                        label={plainEnglish ? 'False Alarm Filter (Precision)' : 'Precision (PPV)'}
                        value={pct(m?.precision)}
                        progress={m?.precision}
                        color="teal"
                      />
                      <MetricCell
                        label={plainEnglish ? 'Catch Rate (Recall)' : 'Recall / Sensitivity'}
                        value={pct(m?.recall)}
                        progress={m?.recall}
                        color="teal"
                      />
                    </>
                  )}
                </div>
                <p className="mt-3 text-xs text-[var(--muted)] max-w-2xl leading-relaxed">
                  {plainEnglish
                    ? 'These scores measure how well the AI performed on real test data it had never seen before.'
                    : 'Training holdout validation metrics evaluated on an out-of-sample test partition.'}
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
                            label={plainEnglish ? 'Matched Known Outcomes' : 'Validation Concordance Rate'}
                            value={pct(spotCheck.agree_rate)}
                            accent="text-ink"
                          />
                          <MetricCell
                            label={plainEnglish ? 'Right When Flagged High' : 'Precision @ High Risk Cutoff'}
                            value={
                              spotCheck.high_risk_precision != null
                                ? pct(spotCheck.high_risk_precision)
                                : '—'
                            }
                          />
                          <MetricCell
                            label={plainEnglish ? 'Right When Flagged Low' : 'Specificity @ Low Risk Cutoff'}
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
                          <ul className="mt-3 space-y-2.5">
                            {feedback.action_effectiveness_ranked.slice(0, 6).map((a) => {
                              const rate = a.n > 0 ? (a.success_n / a.n) * 100 : 0
                              return (
                                <li
                                  key={a.action_code}
                                  className="p-3 bg-paper/60 border border-mist/60 rounded-control"
                                >
                                  <div className="flex flex-wrap items-center justify-between gap-2 text-sm mb-1.5">
                                    <span className="text-ink font-semibold flex items-center gap-2">
                                      {a.action_name || a.action_code}
                                      {a.reliable && (
                                        <Badge variant="safe" size="xs">
                                          Reshapes ranking
                                        </Badge>
                                      )}
                                    </span>
                                    <span className="tabular-nums text-muted text-xs">
                                      <span className="text-teal font-semibold">{a.success_n}</span>/{a.n} favorable ({Math.round(rate)}%)
                                    </span>
                                  </div>
                                  <ProgressBar value={rate} max={100} color="teal" size="xs" />
                                </li>
                              )
                            })}
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

function MetricCell({ label, value, accent = 'text-ink', hint, progress, color = 'teal' }) {
  return (
    <div className="bg-paper px-4 py-4 flex flex-col justify-between">
      <div>
        <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] font-medium">{label}</div>
        <div className={`mt-1 font-display text-2xl font-semibold tabular-nums ${accent}`}>
          {value}
        </div>
      </div>
      {progress != null && !Number.isNaN(Number(progress)) && (
        <div className="mt-2.5">
          <ProgressBar value={Number(progress) * 100} max={100} color={color} size="xs" />
        </div>
      )}
      {hint && <p className="text-xs text-[var(--muted)] mt-1">{hint}</p>}
    </div>
  )
}
