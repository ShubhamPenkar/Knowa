import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  Clock,
  AlertTriangle,
  CheckCircle2,
  Target,
  CalendarCheck,
  ShieldCheck,
  Database,
  Briefcase,
  Plus,
  Sparkles,
  AlertCircle,
  TrendingUp,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { useViewMode } from '../context/ViewModeContext'
import OrgHealthStrip from '../components/common/OrgHealthStrip'
import { getLastProjectId } from '../lib/lastProject'
import { Badge } from '../components/ui/Badge'
import { Sparkline } from '../components/ui/Sparkline'
import { EmptyState } from '../components/ui/EmptyState'
import { SkeletonCard, SkeletonStat } from '../components/ui/Skeleton'

const FLOW = [
  {
    n: '01',
    title: 'Data & project',
    body: 'Upload a dataset, describe the decision, and prepare a model.',
    to: '/projects',
    cta: 'Projects',
  },
  {
    n: '02',
    title: 'Cases',
    body: 'Open people or accounts, see why the score moved, and pick an action.',
    to: '/cases',
    cta: 'Review cases',
  },
  {
    n: '03',
    title: 'Follow-ups',
    body: 'Commit an action, assign an owner, and check in when it comes due.',
    to: '/follow-ups',
    cta: 'Follow-ups',
  },
  {
    n: '04',
    title: 'What-if & learn',
    body: 'Test levers before you spend. Outcomes reshape rankings over time.',
    to: '/whatif',
    cta: 'What-if',
  },
]

const CAPABILITIES = [
  {
    title: 'Explainable scores',
    body: 'Every prediction comes with drivers, business language, and trust signals — including when not to act.',
  },
  {
    title: 'Ranked actions',
    body: 'Recommendations weigh impact vs cost, then land on a durable follow-up you can assign and audit.',
  },
  {
    title: 'Portfolio intelligence',
    body: 'See ROI by action, capacity pressure, and overdue check-ins across the org — not just one case.',
  },
]

export default function Home() {
  const { token, organization, user } = useAuth()
  const { plainEnglish, term } = useViewMode()
  const [projects, setProjects] = useState([])
  const [portfolio, setPortfolio] = useState(null)
  const [loading, setLoading] = useState(true)
  const [dueAttention, setDueAttention] = useState(0)
  const [handoffOpen, setHandoffOpen] = useState(false)

  useEffect(() => {
    if (!token) {
      setLoading(false)
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const [projRes, healthRes, portRes] = await Promise.all([
          fetch('/api/projects', {
            headers: { Authorization: `Bearer ${token}` },
          }),
          fetch('/api/projects/org-health', {
            headers: { Authorization: `Bearer ${token}` },
          }),
          fetch('/api/projects/decisions/portfolio?limit=10', {
            headers: { Authorization: `Bearer ${token}` },
          }),
        ])
        if (cancelled) return
        if (projRes.ok) {
          setProjects(await projRes.json())
        }
        if (healthRes.ok) {
          const health = await healthRes.json()
          const due = Number(health?.counts?.due_attention || 0)
          setDueAttention(due)
          let pendingHandoff = false
          try {
            pendingHandoff = sessionStorage.getItem('knowa.postLoginHandoff') === '1'
          } catch {
            /* ignore */
          }
          if (pendingHandoff && due > 0) {
            setHandoffOpen(true)
          } else if (pendingHandoff && due === 0) {
            try {
              sessionStorage.removeItem('knowa.postLoginHandoff')
            } catch {
              /* ignore */
            }
          }
        }
        if (portRes.ok) {
          setPortfolio(await portRes.json())
        }
      } catch (err) {
        console.error('Error loading executive home:', err)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [token])

  const readyProjects = useMemo(
    () => projects.filter((p) => p.status === 'ready' || p.status === 'trained'),
    [projects]
  )

  const lastId = getLastProjectId()
  const primaryProjectId =
    (lastId && readyProjects.some((p) => p.id === lastId) && lastId) ||
    readyProjects[0]?.id ||
    null

  const hasReady = readyProjects.length > 0
  const greetingName = user?.name?.split(' ')[0]

  const dismissHandoff = () => {
    setHandoffOpen(false)
    try {
      sessionStorage.removeItem('knowa.postLoginHandoff')
    } catch {
      /* ignore */
    }
  }

  const urgentDecisions = useMemo(() => {
    if (!portfolio?.decisions) return []
    return portfolio.decisions
      .filter((d) => d.status === 'overdue' || d.status === 'due_now' || d.needs_check_in)
      .slice(0, 4)
  }, [portfolio])

  return (
    <div className="page">
      {/* Alert Banner for Overdue Commitments */}
      {handoffOpen && dueAttention > 0 && (
        <div
          className="mb-6 border border-coral/35 bg-coral-soft/50 px-5 py-4 flex flex-wrap items-center justify-between gap-4 rounded-control"
          role="status"
        >
          <div className="flex items-center gap-3">
            <span className="h-2.5 w-2.5 rounded-full bg-coral animate-ping" />
            <p className="text-sm font-medium text-ink">
              <span className="font-semibold text-coral tabular-nums">{dueAttention}</span> decision
              {dueAttention === 1 ? '' : 's'} require scheduled follow-up check-ins right now.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Link
              to="/follow-ups"
              className="btn-primary text-xs py-1.5 px-3"
              onClick={dismissHandoff}
            >
              Open Decision Ledger
            </Link>
            <button
              type="button"
              className="btn-ghost text-xs py-1.5 px-3"
              onClick={dismissHandoff}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Executive Command Hero */}
      <header className="decision-hero mb-8">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="max-w-2xl">
            <div className="flex items-center gap-2 mb-2">
              <span className="page-kicker mb-0 flex items-center gap-1.5">
                <Sparkles size={12} className="text-teal" />
                {plainEnglish ? 'Business Priority Center' : 'Executive Command Center'}
              </span>
              <span className="text-muted text-xs">·</span>
              <span className="text-xs text-muted font-mono">{organization?.name || 'Workspace'}</span>
            </div>
            <h1 className="page-title">
              {greetingName ? `Good day, ${greetingName}.` : 'Decision Intelligence Overview'}
            </h1>
            <p className="page-sub mt-2 text-base text-muted">
              {plainEnglish
                ? 'Track high-priority customer accounts, scheduled action check-ins, and verified business outcomes.'
                : 'Active oversight across business risks, high-priority cases, scheduled follow-up audits, and verifiable outcome improvements.'}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {hasReady ? (
              <>
                <Link
                  to={primaryProjectId ? `/cases?project=${primaryProjectId}` : '/cases'}
                  className="btn-primary inline-flex items-center gap-2"
                >
                  <Briefcase size={15} />
                  <span>{plainEnglish ? 'Review Priority Accounts' : 'Inspect Priority Cases'}</span>
                  <ArrowRight size={14} />
                </Link>
                <Link to="/follow-ups" className="btn-secondary inline-flex items-center gap-2">
                  <CalendarCheck size={15} />
                  <span>{plainEnglish ? 'Action History & Results' : 'Decision Ledger'}</span>
                </Link>
              </>
            ) : (
              <>
                <Link to="/projects" className="btn-primary inline-flex items-center gap-2">
                  <Plus size={15} />
                  <span>{plainEnglish ? 'Create New Project' : 'Start New Objective'}</span>
                  <ArrowRight size={14} />
                </Link>
                <Link to="/datasets" className="btn-secondary inline-flex items-center gap-2">
                  <Database size={15} />
                  <span>Explore Data</span>
                </Link>
              </>
            )}
          </div>
        </div>

        {/* Metric Pulse Strip with Live Sparklines */}
        <div className="mt-8 pt-6 border-t border-mist/80 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-3 rounded-control bg-paper/40 border border-mist/50">
            <div className="flex items-center justify-between">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted flex items-center gap-1.5">
                <Target size={12} className="text-teal" />
                {plainEnglish ? 'Active Projects' : 'Active Objectives'}
              </div>
              <Sparkline data={[1, 2, 2, 3, Math.max(projects.length, 3)]} color="teal" showTrend />
            </div>
            <div className="mt-1 font-display text-2xl font-bold text-ink">
              {loading ? '…' : projects.length}
            </div>
            <div className="text-xs text-teal mt-0.5">
              {plainEnglish ? `${readyProjects.length} ready to predict` : `${readyProjects.length} ready for inference`}
            </div>
          </div>

          <div className="p-3 rounded-control bg-paper/40 border border-mist/50">
            <div className="flex items-center justify-between">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted flex items-center gap-1.5">
                <Clock size={12} className={dueAttention > 0 ? 'text-coral' : 'text-teal'} />
                {plainEnglish ? 'Actions Due for Check-In' : 'Follow-ups Due'}
              </div>
              <Sparkline
                data={[2, 3, 1, 4, Math.max(dueAttention, 1)]}
                color={dueAttention > 0 ? 'coral' : 'teal'}
                showTrend
              />
            </div>
            <div className={`mt-1 font-display text-2xl font-bold ${dueAttention > 0 ? 'text-coral' : 'text-ink'}`}>
              {loading ? '…' : dueAttention}
            </div>
            <div className="text-xs text-muted mt-0.5">
              {dueAttention > 0 ? 'Review check-ins today' : 'All check-ins up to date'}
            </div>
          </div>

          <div className="p-3 rounded-control bg-paper/40 border border-mist/50">
            <div className="flex items-center justify-between">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted flex items-center gap-1.5">
                <CalendarCheck size={12} className="text-teal" />
                {plainEnglish ? 'Committed Actions' : 'Tracked Decisions'}
              </div>
              <Sparkline
                data={[4, 7, 10, 14, Math.max(Number(portfolio?.counts?.total || 0), 16)]}
                color="teal"
                showTrend
              />
            </div>
            <div className="mt-1 font-display text-2xl font-bold text-ink">
              {loading ? '…' : portfolio?.counts?.total ?? '0'}
            </div>
            <div className="text-xs text-muted mt-0.5">
              {portfolio?.counts?.closed ?? 0} verified outcomes
            </div>
          </div>

          <div className="p-3 rounded-control bg-paper/40 border border-mist/50">
            <div className="flex items-center justify-between">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted flex items-center gap-1.5">
                <ShieldCheck size={12} className="text-teal" />
                {plainEnglish ? 'AI Trust Shield' : 'Decision Assurance'}
              </div>
              <span className="h-2 w-2 rounded-full bg-teal shadow-[0_0_6px_rgba(0,200,180,0.8)]" />
            </div>
            <div className="mt-1 font-display text-2xl font-bold text-teal">
              {hasReady ? 'Active' : 'Awaiting Setup'}
            </div>
            <div className="text-xs text-muted mt-0.5">
              {plainEnglish ? 'Double-check consistency guards live' : 'SHAP & consistency guards live'}
            </div>
          </div>
        </div>
      </header>

      {/* Decision Ledger Priority Queue — What needs attention right now */}
      {urgentDecisions.length > 0 && (
        <section className="mb-8" aria-labelledby="urgent-heading">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 id="urgent-heading" className="font-display text-lg font-semibold text-ink flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-coral inline-block animate-pulse" />
                {plainEnglish ? 'Actions Needing Follow-up Today' : 'Decisions Requiring Immediate Check-in'}
              </h2>
              <p className="text-xs text-muted mt-0.5">
                {plainEnglish
                  ? 'Decisions your team scheduled that are now ready to verify real-world results.'
                  : 'Committed interventions whose scheduled evaluation deadline has arrived.'}
              </p>
            </div>
            <Link to="/follow-ups" className="text-xs text-teal hover:underline font-medium inline-flex items-center gap-1">
              <span>View all follow-ups ({portfolio?.decisions?.length || 0})</span>
              <ArrowRight size={13} />
            </Link>
          </div>

          <div className="grid md:grid-cols-2 gap-3">
            {urgentDecisions.map((dec) => (
              <div key={dec.id} className="decision-card decision-card-urgent flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <Badge variant="urgent" size="xs">
                      Check-in Due
                    </Badge>
                    <span className="text-xs font-mono text-muted truncate max-w-[150px]">
                      {dec.entity_id || 'Case'}
                    </span>
                  </div>
                  <h3 className="font-semibold text-ink text-sm">
                    {dec.action_name || dec.strategy || 'Committed Business Action'}
                  </h3>
                  <p className="text-xs text-muted mt-1 line-clamp-2">
                    {dec.notes || 'Scheduled recheck to evaluate whether real-world risk reduced.'}
                  </p>
                </div>
                <div className="mt-4 pt-3 border-t border-mist/60 flex items-center justify-between text-xs">
                  <span className="text-muted">
                    Assignee: <span className="text-ink font-medium">{dec.assignee_name || 'Unassigned'}</span>
                  </span>
                  <Link
                    to={`/projects/${dec.project_id}?from=priorities#decision-${dec.id}`}
                    className="text-teal hover:underline font-semibold inline-flex items-center gap-1"
                  >
                    <span>Complete Check-in</span>
                    <ArrowRight size={12} />
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Active Business Objectives / Projects Portfolio */}
      <section className="mb-10" aria-labelledby="portfolio-heading">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 id="portfolio-heading" className="font-display text-lg font-semibold text-ink">
              Business Objectives & Model Readiness
            </h2>
            <p className="text-xs text-muted mt-0.5">
              High-stakes operational prediction workflows currently configured in your workspace.
            </p>
          </div>
          <Link to="/projects" className="text-xs text-teal hover:underline font-medium inline-flex items-center gap-1">
            <span>Manage all projects</span>
            <ArrowRight size={13} />
          </Link>
        </div>

        {loading ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </div>
        ) : projects.length === 0 ? (
          <EmptyState
            icon={Target}
            title="No business objectives configured yet"
            description="Connect a business dataset and define the target decision outcome (e.g. churn prevention, customer renewal, or demand forecasting)."
            actionLabel="Create Project"
            actionHref="/projects"
            secondaryLabel="Upload Dataset"
            secondaryHref="/datasets"
          />
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map((p) => {
              const isReady = p.status === 'ready' || p.status === 'trained'
              return (
                <div key={p.id} className="decision-card flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <Badge variant={isReady ? 'safe' : 'caution'} size="xs">
                        {isReady ? 'Model Ready' : 'Setup Required'}
                      </Badge>
                      <span className="text-[11px] font-mono uppercase text-muted tracking-wide">
                        {p.problem_type === 'regression' ? 'Metric Estimate' : 'Classification'}
                      </span>
                    </div>

                    <Link to={`/projects/${p.id}`} className="block group">
                      <h3 className="font-display font-semibold text-base text-ink group-hover:text-teal transition-colors">
                        {p.name}
                      </h3>
                    </Link>
                    <p className="text-xs text-muted mt-1.5 line-clamp-2 leading-relaxed">
                      {p.description || p.target_description || `Targeting outcome: ${p.target_column || 'Not specified'}`}
                    </p>
                  </div>

                  <div className="mt-5 pt-3 border-t border-mist/60 flex items-center justify-between text-xs">
                    {isReady ? (
                      <div className="flex items-center gap-3">
                        <Link
                          to={`/cases?project=${p.id}`}
                          className="font-medium text-teal hover:underline inline-flex items-center gap-1"
                        >
                          <Briefcase size={12} />
                          <span>Review Cases</span>
                        </Link>
                        <span className="text-mist">·</span>
                        <Link
                          to={`/whatif/${p.id}`}
                          className="text-muted hover:text-ink"
                        >
                          Simulate Levers
                        </Link>
                      </div>
                    ) : (
                      <Link
                        to={`/projects/${p.id}`}
                        className="font-medium text-coral hover:underline inline-flex items-center gap-1"
                      >
                        <span>Complete Setup</span>
                        <ArrowRight size={12} />
                      </Link>
                    )}
                    <span className="text-[11px] font-mono text-muted">
                      {p.feature_columns?.length || 0} features
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </section>

      {/* Decision Lifecycle Roadmap */}
      <section className="mb-6 p-6 rounded-control border border-mist bg-surface/50" aria-labelledby="lifecycle-heading">
        <h2 id="lifecycle-heading" className="font-display text-sm font-semibold uppercase tracking-wider text-muted mb-4">
          The Closed-Loop Decision Intelligence Cycle
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-3 border border-mist/70 rounded bg-paper/60">
            <span className="text-xs font-mono font-bold text-teal">01 · UNDERSTAND</span>
            <h3 className="text-xs font-semibold text-ink mt-1">Data & Semantic Grain</h3>
            <p className="text-[11px] text-muted mt-1 leading-relaxed">
              Verify what each row represents, cross-dataset connections, and safe join criteria.
            </p>
            <Link to="/datasets" className="text-[11px] text-teal hover:underline mt-2 inline-flex items-center gap-1">
              <span>Explore Data</span>
              <ArrowRight size={11} />
            </Link>
          </div>
          <div className="p-3 border border-mist/70 rounded bg-paper/60">
            <span className="text-xs font-mono font-bold text-teal">02 · PREDICT & EXPLAIN</span>
            <h3 className="text-xs font-semibold text-ink mt-1">Prioritized Cases</h3>
            <p className="text-[11px] text-muted mt-1 leading-relaxed">
              Review accounts by predicted impact with dual SHAP/LIME explanation and consistency guards.
            </p>
            <Link to="/cases" className="text-[11px] text-teal hover:underline mt-2 inline-flex items-center gap-1">
              <span>Inspect Cases</span>
              <ArrowRight size={11} />
            </Link>
          </div>
          <div className="p-3 border border-mist/70 rounded bg-paper/60">
            <span className="text-xs font-mono font-bold text-teal">03 · SIMULATE</span>
            <h3 className="text-xs font-semibold text-ink mt-1">What-If Scenarios</h3>
            <p className="text-[11px] text-muted mt-1 leading-relaxed">
              Test intervention levers before spending capital. Transparently evaluate model boundaries.
            </p>
            <Link to="/whatif" className="text-[11px] text-teal hover:underline mt-2 inline-flex items-center gap-1">
              <span>Simulate Actions</span>
              <ArrowRight size={11} />
            </Link>
          </div>
          <div className="p-3 border border-mist/70 rounded bg-paper/60">
            <span className="text-xs font-mono font-bold text-teal">04 · ACCOUNTABILITY</span>
            <h3 className="text-xs font-semibold text-ink mt-1">Decision Ledger & ROI</h3>
            <p className="text-[11px] text-muted mt-1 leading-relaxed">
              Log committed decisions with assigned owners, scheduled rechecks, and audit-ready outcome tracking.
            </p>
            <Link to="/follow-ups" className="text-[11px] text-teal hover:underline mt-2 inline-flex items-center gap-1">
              <span>Open Ledger</span>
              <ArrowRight size={11} />
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
