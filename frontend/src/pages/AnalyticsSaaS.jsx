import { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  Download,
  RefreshCw,
  UserCheck,
  Clock,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Calendar,
  Sparkles,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../components/ui/Toast';
import { StackedBar } from '../components/ui/ProgressBar';
import { Badge } from '../components/ui/Badge';
import Spinner from '../components/common/Spinner';
import OrgHealthStrip from '../components/common/OrgHealthStrip';
import PortfolioIntel from '../components/common/PortfolioIntel';

/** Business labels — never show ML jargon to decision users. */
function humanizeLabel(name) {
  if (!name) return 'this outcome';
  return String(name)
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function attentionLabel(risk) {
  const r = String(risk || '').toLowerCase();
  if (r === 'critical' || r === 'high') return 'Act now';
  if (r === 'medium') return 'Watch closely';
  return 'On track';
}

function attentionTone(risk) {
  const r = String(risk || '').toLowerCase();
  if (r === 'critical' || r === 'high') return 'coral';
  if (r === 'medium') return 'muted';
  return 'teal';
}

function PortfolioStatSkeleton() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-mist border-b border-mist">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="bg-paper px-4 py-4">
          <div className="skeleton h-3 w-16 mb-3" />
          <div className="skeleton h-8 w-10" />
        </div>
      ))}
    </div>
  );
}

/**
 * Business briefing: org follow-ups first, then project case focus.
 */
export default function AnalyticsSaaS() {
  const { token, user } = useAuth();
  const toast = useToast();
  const [projects, setProjects] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedProject, setSelectedProject] = useState(null);
  const [projectDetails, setProjectDetails] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const [portfolioError, setPortfolioError] = useState('');
  const [sweeping, setSweeping] = useState(false);
  const [sweepNote, setSweepNote] = useState('');
  const [mineOnly, setMineOnly] = useState(false);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    if (token) {
      fetchProjects();
    }
  }, [token]);

  useEffect(() => {
    if (token) {
      fetchPortfolio();
    }
  }, [token, mineOnly]);

  const fetchPortfolio = async () => {
    setPortfolioLoading(true);
    setPortfolioError('');
    try {
      const q = new URLSearchParams({ limit: '80' });
      if (mineOnly) q.set('mine', 'true');
      const res = await fetch(`/api/projects/decisions/portfolio?${q}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setPortfolio(await res.json());
      } else {
        setPortfolio(null);
        const data = await res.json().catch(() => ({}));
        const detail = data.detail;
        setPortfolioError(
          typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
              : `Could not load follow-ups (${res.status})`
        );
      }
    } catch (err) {
      console.error(err);
      setPortfolio(null);
      setPortfolioError(err?.message || 'Could not load follow-ups');
    }
    setPortfolioLoading(false);
  };

  const exportDecisions = async (fmt) => {
    setExporting(true);
    try {
      const q = new URLSearchParams({ format: fmt, include_activity: 'true', limit: '500' });
      if (mineOnly && user?.id) q.set('assignee_user_id', user.id);
      const res = await fetch(`/api/projects/decisions/export?${q}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const msg = typeof data.detail === 'string' ? data.detail : 'Could not export decisions';
        setSweepNote(msg);
        toast.error(msg);
        return;
      }
      if (fmt === 'csv') {
        const text = await res.text();
        const blob = new Blob([text], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'knowa-decisions.csv';
        a.click();
        URL.revokeObjectURL(url);
      } else {
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'knowa-decisions.json';
        a.click();
        URL.revokeObjectURL(url);
      }
      const note = `Exported decisions as ${fmt.toUpperCase()}.`;
      setSweepNote(note);
      toast.success(note);
    } catch (err) {
      const msg = err?.message || 'Could not export decisions';
      setSweepNote(msg);
      toast.error(msg);
    } finally {
      setExporting(false);
    }
  };

  const runRecheckSweep = async () => {
    setSweeping(true);
    setSweepNote('');
    try {
      const res = await fetch('/api/projects/decisions/recheck-sweep?limit=200', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg =
          typeof data.detail === 'string'
            ? data.detail
            : 'Could not mark overdue follow-ups';
        setSweepNote(msg);
        toast.error(msg);
      } else {
        const msg = data.plain_summary || `Flagged ${data.flagged || 0} follow-up(s).`;
        setSweepNote(msg);
        toast.success(msg);
        await fetchPortfolio();
      }
    } catch (err) {
      const msg = err?.message || 'Could not mark overdue follow-ups';
      setSweepNote(msg);
      toast.error(msg);
    }
    setSweeping(false);
  };

  const fetchProjects = async () => {
    try {
      const res = await fetch('/api/projects', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setProjects(data);
        const trained = data.find((p) => p.status === 'trained' || p.status === 'ready');
        if (trained) {
          setSelectedProject(trained.id);
          fetchProjectDetails(trained.id);
          fetchPredictions(trained.id);
        }
      }
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const fetchProjectDetails = async (projectId) => {
    try {
      const res = await fetch(`/api/projects/${projectId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setProjectDetails(await res.json());
    } catch (err) {
      console.error(err);
    }
  };

  const fetchPredictions = async (projectId) => {
    try {
      const res = await fetch(`/api/projects/${projectId}/predictions?limit=100`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setPredictions(await res.json());
    } catch (err) {
      console.error(err);
    }
  };

  const handleProjectChange = (projectId) => {
    setSelectedProject(projectId);
    setProjectDetails(null);
    if (projectId) {
      fetchProjectDetails(projectId);
      fetchPredictions(projectId);
    } else {
      setPredictions([]);
    }
  };

  const outcomeName = humanizeLabel(
    projectDetails?.target_description || projectDetails?.target_column || 'outcome'
  );

  const readyProjects = projects.filter((p) => p.status === 'trained' || p.status === 'ready');

  const stats = useMemo(() => {
    const needsAction = predictions.filter(
      (p) => p.risk_level === 'high' || p.risk_level === 'critical'
    );
    const watch = predictions.filter((p) => p.risk_level === 'medium');
    const stable = predictions.filter((p) => p.risk_level === 'low' || !p.risk_level);
    const uncertain = predictions.filter((p) => p.low_confidence);
    return {
      needsAction: needsAction.length,
      watch: watch.length,
      stable: stable.length,
      uncertain: uncertain.length,
      total: predictions.length,
    };
  }, [predictions]);

  const topDrivers = useMemo(() => {
    const fi = projectDetails?.active_model?.feature_importance;
    if (!fi || typeof fi !== 'object') return [];
    return Object.entries(fi)
      .sort((a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])))
      .slice(0, 5)
      .map(([name, score]) => ({
        name: humanizeLabel(name),
        strength: Math.min(100, Math.round(Math.abs(Number(score)) * 100)),
      }));
  }, [projectDetails]);

  const priorityQueue = useMemo(() => {
    return [...predictions]
      .filter((p) => p.risk_level === 'high' || p.risk_level === 'critical' || p.risk_level === 'medium')
      .sort((a, b) => {
        const rank = { critical: 0, high: 1, medium: 2, low: 3 };
        const r =
          (rank[String(a.risk_level).toLowerCase()] ?? 9) -
          (rank[String(b.risk_level).toLowerCase()] ?? 9);
        if (r !== 0) return r;
        return (b.probability || 0) - (a.probability || 0);
      })
      .slice(0, 8);
  }, [predictions]);

  if (loading) {
    return (
      <div className="page flex justify-center items-center min-h-[40vh]">
        <Spinner />
      </div>
    );
  }

  const portfolioRows = portfolio
    ? [
        ...(portfolio.overdue || []).map((d) => ({ ...d, _bucket: 'Overdue' })),
        ...(portfolio.due_now || []).map((d) => ({ ...d, _bucket: 'Due now' })),
        ...(portfolio.upcoming || []).slice(0, 6).map((d) => ({
          ...d,
          _bucket: 'Upcoming',
        })),
      ]
    : [];

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Decision Accountability</p>
          <h1 className="page-title">Decision Ledger & Follow-up Board</h1>
          <p className="page-sub max-w-xl">
            Audit committed operational actions, enforce review deadlines with assigned owners, and verify whether predicted improvements occurred in reality.
          </p>
        </div>
        {readyProjects.length > 0 && selectedProject && (
          <div className="flex flex-wrap gap-2">
            <Link to={`/cases?project=${selectedProject}`} className="btn-primary">
              Open cases
            </Link>
            <Link to={`/whatif/${selectedProject}`} className="btn-secondary">
              What-if
            </Link>
          </div>
        )}
      </div>

      <OrgHealthStrip projectId={selectedProject} variant="follow-ups" />

      <PortfolioIntel />

      {/* Org follow-up board — always first */}
      <section className="mb-10 border border-mist">
        <div className="px-5 py-4 border-b border-mist flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="font-display text-lg font-semibold text-ink">Follow-ups due</h2>
            <p className="text-sm text-[var(--muted)] mt-1 max-w-2xl">
              Across all projects — actions you saved from cases. Check in when they come due.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 shrink-0 items-center">
            {user?.id && (
              <button
                type="button"
                onClick={() => setMineOnly((v) => !v)}
                className={`text-xs px-3 py-1.5 border rounded-control inline-flex items-center gap-1.5 ${
                  mineOnly
                    ? 'border-teal bg-teal-soft/30 text-ink'
                    : 'border-mist text-[var(--muted)] hover:text-ink'
                }`}
              >
                <UserCheck size={13} />
                <span>{mineOnly ? 'All follow-ups' : 'My follow-ups'}</span>
              </button>
            )}
            <button
              type="button"
              onClick={() => exportDecisions('csv')}
              disabled={exporting || portfolioLoading}
              className="btn-ghost text-xs inline-flex items-center gap-1.5"
            >
              <Download size={13} />
              <span>{exporting ? 'Exporting…' : 'Export CSV'}</span>
            </button>
            <button
              type="button"
              onClick={() => exportDecisions('json')}
              disabled={exporting || portfolioLoading}
              className="btn-ghost text-xs inline-flex items-center gap-1.5"
            >
              <Download size={13} />
              <span>Export JSON</span>
            </button>
            <button
              type="button"
              onClick={runRecheckSweep}
              disabled={sweeping || portfolioLoading}
              className="btn-ghost text-xs inline-flex items-center gap-1.5"
              title="Mark due scheduled follow-ups ready for check-in"
            >
              <Clock size={13} />
              <span>{sweeping ? 'Updating…' : 'Mark overdue'}</span>
            </button>
            <button
              type="button"
              onClick={fetchPortfolio}
              disabled={portfolioLoading}
              className="btn-ghost text-xs inline-flex items-center gap-1.5"
            >
              <RefreshCw size={13} className={portfolioLoading ? 'animate-spin' : ''} />
              <span>{portfolioLoading ? 'Refreshing…' : 'Refresh'}</span>
            </button>
          </div>
        </div>
        {sweepNote && (
          <div className="px-5 py-2 border-b border-mist text-xs text-[var(--muted)]">
            {sweepNote}
          </div>
        )}
        {portfolioError && !portfolioLoading && (
          <div className="px-5 py-4 text-sm text-coral">{portfolioError}</div>
        )}
        {portfolioLoading && !portfolio && <PortfolioStatSkeleton />}
        {portfolio && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-mist border-b border-mist">
              {[
                {
                  key: 'overdue',
                  label: 'Overdue',
                  value: portfolio.counts?.overdue ?? 0,
                  accent: 'text-coral',
                },
                {
                  key: 'due_now',
                  label: 'Due now',
                  value: portfolio.counts?.due_now ?? 0,
                  accent: 'text-ink',
                },
                {
                  key: 'upcoming',
                  label: 'Upcoming',
                  value: portfolio.counts?.upcoming ?? 0,
                  accent: 'text-teal',
                },
                {
                  key: 'closed_recent',
                  label: 'Closed recently',
                  value: portfolio.counts?.closed_recent ?? 0,
                  accent: 'text-[var(--muted)]',
                },
              ].map((cell) => (
                <div key={cell.key} className="bg-paper px-4 py-4">
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

            {/* Visual Portfolio Distribution Bar */}
            <div className="px-5 py-3.5 bg-surface/50 border-b border-mist">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted mb-2">
                Decision Pipeline Distribution
              </div>
              <StackedBar
                segments={[
                  { label: 'Overdue', value: portfolio.counts?.overdue ?? 0, colorClass: 'bg-coral' },
                  { label: 'Due now', value: portfolio.counts?.due_now ?? 0, colorClass: 'bg-[#f5b838]' },
                  { label: 'Upcoming', value: portfolio.counts?.upcoming ?? 0, colorClass: 'bg-teal' },
                  { label: 'Closed', value: portfolio.counts?.closed_recent ?? 0, colorClass: 'bg-mist' },
                ]}
              />
            </div>

            {portfolioRows.length === 0 ? (
              <div className="px-5 py-6 text-sm text-[var(--muted)]">
                No open follow-ups yet. Open a case, pick an action, and save a follow-up —
                it will show up here when a check-in is due.
              </div>
            ) : (
              <ul className="divide-y divide-mist">
                {portfolioRows.slice(0, 12).map((d) => (
                  <li
                    key={d.id}
                    className="px-5 py-3.5 flex flex-wrap items-start justify-between gap-3 text-sm hover:bg-mist/20 transition-colors"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-ink">{d.action_name}</span>
                        <Badge
                          variant={
                            d._bucket === 'Overdue'
                              ? 'urgent'
                              : d._bucket === 'Due now'
                                ? 'caution'
                                : 'safe'
                          }
                          size="xs"
                        >
                          {d._bucket}
                        </Badge>
                      </div>
                      <p className="text-xs text-[var(--muted)] mt-1">
                        {d.project_name ? `${d.project_name}` : ''}
                        {d.assignee?.name
                          ? ` · Owner: ${d.assignee.name}`
                          : ' · Unassigned'}
                        {d.recheck_date ? ` · Target: ${new Date(d.recheck_date).toLocaleDateString()}` : ''}
                      </p>
                    </div>
                    <Link
                      to={`/projects/${d.project_id}?from=ledger#decision-${d.id}`}
                      className="text-xs text-teal hover:underline font-semibold inline-flex items-center gap-1 shrink-0 mt-1"
                    >
                      <span>Inspect</span>
                      <ArrowRight size={12} />
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
        {!portfolioLoading && !portfolio && !portfolioError && (
          <div className="px-5 py-6 text-sm text-[var(--muted)]">
            Follow-ups will appear here once you save actions from cases.
          </div>
        )}
      </section>

      {readyProjects.length === 0 ? (
        <div className="empty-state">
          <h3 className="font-display text-xl font-semibold text-ink mb-2">No ready projects yet</h3>
          <p className="text-[var(--muted)] mb-6 text-sm max-w-md mx-auto">
            Connect a dataset, set up a project, and prepare it — then case focus fills in below.
          </p>
          <div className="flex flex-wrap gap-2 justify-center">
            <Link to="/projects" className="btn-primary">
              Go to projects
            </Link>
            <Link to="/cases" className="btn-secondary">
              Cases
            </Link>
          </div>
        </div>
      ) : (
        <>
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 className="font-display text-lg font-semibold text-ink">Today&apos;s focus</h2>
              <p className="text-sm text-[var(--muted)] mt-1">
                Project-scoped only — pick which project&apos;s recent cases to review.
              </p>
            </div>
            <div className="min-w-[14rem]">
              <label htmlFor="brief-project" className="block text-xs font-medium text-[var(--muted)] mb-1.5">
                Project for case focus
              </label>
              <select
                id="brief-project"
                value={selectedProject || ''}
                onChange={(e) => handleProjectChange(e.target.value)}
                className="input"
              >
                {readyProjects.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <section className="mb-10">
            <p className="text-sm text-[var(--muted)] mb-4">
              Based on recent assessments for {outcomeName.toLowerCase()}
              {stats.total ? ` (${stats.total} reviewed)` : ''}.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-mist border border-mist">
              {[
                {
                  label: 'Need action',
                  value: stats.needsAction,
                  hint: 'High likelihood — intervene soon',
                  accent: 'text-coral',
                },
                {
                  label: 'Watch closely',
                  value: stats.watch,
                  hint: 'Elevated — stay close',
                  accent: 'text-ink',
                },
                {
                  label: 'On track',
                  value: stats.stable,
                  hint: 'Lower risk — maintain course',
                  accent: 'text-teal',
                },
                {
                  label: 'Uncertain calls',
                  value: stats.uncertain,
                  hint: 'Review before acting hard',
                  accent: 'text-[var(--muted)]',
                },
              ].map((cell) => (
                <div key={cell.label} className="bg-paper px-4 py-5">
                  <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                    {cell.label}
                  </div>
                  <div className={`mt-1 font-display text-3xl font-semibold tabular-nums ${cell.accent}`}>
                    {stats.total ? cell.value : '—'}
                  </div>
                  <p className="mt-2 text-xs text-[var(--muted)] leading-snug">{cell.hint}</p>
                </div>
              ))}
            </div>
            {stats.total === 0 && (
              <p className="mt-4 text-sm text-[var(--muted)]">
                No recent cases yet.{' '}
                <Link to={`/cases?project=${selectedProject}`} className="text-teal hover:underline">
                  Review people or accounts
                </Link>{' '}
                to fill this brief.
              </p>
            )}
          </section>

          <div className="grid lg:grid-cols-5 gap-10 mb-10">
            <section className="lg:col-span-3 min-w-0">
              <h2 className="font-display text-lg font-semibold text-ink mb-1">Who to prioritize</h2>
              <p className="text-sm text-[var(--muted)] mb-4">
                Sorted by urgency for {outcomeName.toLowerCase()}.
              </p>
              {priorityQueue.length === 0 ? (
                <div className="border border-mist px-5 py-8 text-sm text-[var(--muted)]">
                  {stats.total === 0
                    ? 'Assess a few rows on the project page to build a queue.'
                    : 'No elevated cases in recent assessments — stack looks healthy.'}
                </div>
              ) : (
                <ol className="border-t border-mist divide-y divide-mist">
                  {priorityQueue.map((pred, idx) => {
                    const tone = attentionTone(pred.risk_level);
                    const pct =
                      pred.probability != null
                        ? Math.round(Number(pred.probability) * 100)
                        : null;
                    const caseHref = pred.id
                      ? `/cases?project=${selectedProject}&prediction=${pred.id}`
                      : `/cases?project=${selectedProject}`;
                    return (
                      <li key={pred.id || idx} className="py-4 flex flex-wrap items-start gap-4">
                        <span className="text-[var(--muted)] tabular-nums w-6 shrink-0 pt-0.5">
                          {idx + 1}.
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <span
                              className={`badge border ${
                                tone === 'coral'
                                  ? 'bg-coral-soft border-coral/30'
                                  : tone === 'teal'
                                    ? 'bg-teal-soft/50 border-teal/20'
                                    : 'bg-mist border-mist'
                              }`}
                            >
                              {attentionLabel(pred.risk_level)}
                            </span>
                            {pred.low_confidence && (
                              <span className="text-xs text-coral">Double-check before big moves</span>
                            )}
                          </div>
                          <p className="mt-1.5 text-sm text-ink">
                            {pct != null ? (
                              <>
                                About <strong className="font-semibold tabular-nums">{pct}%</strong> likelihood
                                of {outcomeName.toLowerCase()}
                              </>
                            ) : (
                              'Elevated case'
                            )}
                            {pred.entity_id ? (
                              <span className="text-[var(--muted)]"> · {pred.entity_id}</span>
                            ) : null}
                          </p>
                          {pred.created_at && (
                            <p className="text-xs text-[var(--muted)] mt-1">
                              Reviewed {new Date(pred.created_at).toLocaleDateString()}
                            </p>
                          )}
                        </div>
                        <Link
                          to={caseHref}
                          className="shrink-0 text-xs px-3 py-1.5 border border-mist hover:border-teal text-ink rounded-control self-center"
                        >
                          Open case
                        </Link>
                      </li>
                    );
                  })}
                </ol>
              )}
              {selectedProject && (
                <Link
                  to={`/cases?project=${selectedProject}`}
                  className="inline-flex mt-5 text-sm font-medium text-teal hover:underline"
                >
                  Open Cases →
                </Link>
              )}
            </section>

            <section className="lg:col-span-2">
              <h2 className="font-display text-lg font-semibold text-ink mb-1">
                What usually shows up in explanations
              </h2>
              <p className="text-sm text-[var(--muted)] mb-4">
                Patterns that most often push {outcomeName.toLowerCase()} up or down for this project.
              </p>
              {topDrivers.length === 0 ? (
                <p className="text-sm text-[var(--muted)] border border-mist px-4 py-6">
                  Patterns appear after the project is prepared. Retrain if you recently changed the data.
                </p>
              ) : (
                <ul className="space-y-4">
                  {topDrivers.map((d) => (
                    <li key={d.name}>
                      <div className="flex justify-between gap-3 text-sm mb-1.5">
                        <span className="text-ink font-medium">{d.name}</span>
                        <span className="text-[var(--muted)] tabular-nums shrink-0">
                          {d.strength > 60 ? 'Strong' : d.strength > 30 ? 'Moderate' : 'Mild'}
                        </span>
                      </div>
                      <div className="h-1 bg-mist rounded-[1px] overflow-hidden">
                        <div
                          className="h-full bg-teal"
                          style={{ width: `${Math.max(d.strength, 8)}%` }}
                        />
                      </div>
                    </li>
                  ))}
                </ul>
              )}
              <div className="mt-8 border-l-2 border-teal pl-4">
                <p className="text-sm text-ink leading-relaxed">
                  These are common explanation factors across the model — useful for review, but not
                  a guarantee that changing them will move a specific person&apos;s score. Confirm
                  with a what-if on that case.
                </p>
              </div>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
