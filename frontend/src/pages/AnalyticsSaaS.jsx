import { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import Spinner from '../components/common/Spinner';

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
  const { token } = useAuth();
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

  useEffect(() => {
    if (token) {
      fetchProjects();
      fetchPortfolio();
    }
  }, [token]);

  const fetchPortfolio = async () => {
    setPortfolioLoading(true);
    setPortfolioError('');
    try {
      const res = await fetch('/api/projects/decisions/portfolio?limit=80', {
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
        setSweepNote(
          typeof data.detail === 'string'
            ? data.detail
            : 'Could not mark overdue follow-ups'
        );
      } else {
        setSweepNote(data.plain_summary || `Flagged ${data.flagged || 0} follow-up(s).`);
        await fetchPortfolio();
      }
    } catch (err) {
      setSweepNote(err?.message || 'Could not mark overdue follow-ups');
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
          <p className="page-kicker">Org board</p>
          <h1 className="page-title">Follow-ups</h1>
          <p className="page-sub max-w-xl">
            Check in on saved actions across your org. Use Cases for new reviews; What-if for scenarios.
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

      {/* Org follow-up board — always first */}
      <section className="mb-10 border border-mist">
        <div className="px-5 py-4 border-b border-mist flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="font-display text-lg font-semibold text-ink">Follow-ups due</h2>
            <p className="text-sm text-[var(--muted)] mt-1 max-w-2xl">
              Across all projects — actions you saved from cases. Check in when they come due.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 shrink-0">
            <button
              type="button"
              onClick={runRecheckSweep}
              disabled={sweeping || portfolioLoading}
              className="btn-ghost text-sm"
              title="Mark due scheduled follow-ups ready for check-in"
            >
              {sweeping ? 'Updating…' : 'Mark overdue for check-in'}
            </button>
            <button
              type="button"
              onClick={fetchPortfolio}
              disabled={portfolioLoading}
              className="btn-ghost text-sm"
            >
              {portfolioLoading ? 'Refreshing…' : 'Refresh'}
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
                    className="px-5 py-3 flex flex-wrap items-start justify-between gap-3 text-sm"
                  >
                    <div className="min-w-0">
                      <div className="font-medium text-ink">{d.action_name}</div>
                      <p className="text-xs text-[var(--muted)] mt-1">
                        <span
                          className={
                            d._bucket === 'Overdue'
                              ? 'text-coral'
                              : d._bucket === 'Due now'
                                ? 'text-ink'
                                : ''
                          }
                        >
                          {d._bucket}
                        </span>
                        {d.project_name ? ` · ${d.project_name}` : ''}
                        {d.recheck_at
                          ? ` · check back ${String(d.recheck_at).slice(0, 10)}`
                          : ''}
                        {d.impact_hint ? ` · ${d.impact_hint}` : ''}
                      </p>
                    </div>
                    <Link
                      to={`/projects/${d.project_id}?decision=${d.id}&checkin=1&from=follow-ups${
                        d.prediction_id ? `&prediction=${d.prediction_id}` : ''
                      }`}
                      className="shrink-0 text-xs px-3 py-1.5 border border-mist hover:border-teal text-ink rounded-control"
                    >
                      {d._bucket === 'Overdue' || d._bucket === 'Due now'
                        ? 'Update follow-up'
                        : 'Open follow-up'}
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
