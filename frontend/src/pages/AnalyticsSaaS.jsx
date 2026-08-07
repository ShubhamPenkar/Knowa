import { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

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

/**
 * Business briefing: who needs attention, what tends to drive risk, what to do.
 * No Accuracy / Feature Importance / Prediction Distribution.
 */
export default function AnalyticsSaaS() {
  const { token } = useAuth();
  const [projects, setProjects] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedProject, setSelectedProject] = useState(null);
  const [projectDetails, setProjectDetails] = useState(null);

  useEffect(() => {
    if (token) fetchProjects();
  }, [token]);

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
        <div className="h-6 w-6 border-2 border-teal border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Decision brief</p>
          <h1 className="page-title">Priorities</h1>
          <p className="page-sub max-w-xl">
            Who needs attention for {outcomeName.toLowerCase()}, what usually drives it, and where to act first.
          </p>
        </div>
      </div>

      {readyProjects.length === 0 ? (
        <div className="empty-state">
          <h3 className="font-display text-xl font-semibold text-ink mb-2">Nothing ready yet</h3>
          <p className="text-[var(--muted)] mb-6 text-sm max-w-md mx-auto">
            Connect a dataset, set up a project, and prepare it — then this brief fills with who to focus on.
          </p>
          <Link to="/projects" className="btn-primary">
            Go to projects
          </Link>
        </div>
      ) : (
        <>
          <div className="mb-8">
            <label htmlFor="brief-project" className="block text-sm font-medium text-ink mb-2">
              Looking at
            </label>
            <select
              id="brief-project"
              value={selectedProject || ''}
              onChange={(e) => handleProjectChange(e.target.value)}
              className="input max-w-md"
            >
              {readyProjects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* What matters today */}
          <section className="mb-10">
            <h2 className="font-display text-lg font-semibold text-ink mb-1">Today&apos;s focus</h2>
            <p className="text-sm text-[var(--muted)] mb-4">
              Based on recent assessments in this project
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
                <Link to={`/projects/${selectedProject}`} className="text-teal hover:underline">
                  Review people or accounts
                </Link>{' '}
                to fill this brief.
              </p>
            )}
          </section>

          <div className="grid lg:grid-cols-5 gap-10 mb-10">
            {/* Priority queue */}
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
                      </li>
                    );
                  })}
                </ol>
              )}
              {selectedProject && (
                <Link
                  to={`/projects/${selectedProject}`}
                  className="inline-flex mt-5 text-sm font-medium text-teal hover:underline"
                >
                  Open full case list →
                </Link>
              )}
            </section>

            {/* What drives this */}
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

          {/* Simple next steps */}
          <section className="border-t border-mist pt-8">
            <h2 className="font-display text-lg font-semibold text-ink mb-3">Suggested next steps</h2>
            <ol className="space-y-3 text-sm text-ink max-w-2xl">
              <li className="flex gap-3">
                <span className="text-teal font-semibold tabular-nums">1</span>
                <span>
                  Start with the <strong>Need action</strong> queue — reach out or intervene before the risk solidifies.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="text-teal font-semibold tabular-nums">2</span>
                <span>
                  For any case marked uncertain, open the project, pick that row, and read{' '}
                  <em>why</em> the call is soft before committing budget.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="text-teal font-semibold tabular-nums">3</span>
                <span>
                  Use <strong>Try a scenario</strong> on a project to test “what if we improved X?” before you act in the real world.
                </span>
              </li>
            </ol>
            <div className="mt-6 flex flex-wrap gap-3">
              {selectedProject && (
                <>
                  <Link to={`/projects/${selectedProject}`} className="btn-primary">
                    Review cases
                  </Link>
                  <Link to={`/projects/${selectedProject}/whatif`} className="btn-secondary">
                    Try a scenario
                  </Link>
                </>
              )}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
