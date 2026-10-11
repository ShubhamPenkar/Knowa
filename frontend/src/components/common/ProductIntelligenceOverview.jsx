import { useEffect, useState } from 'react';
import Spinner from './Spinner';

export default function ProductIntelligenceOverview({ token, initialDatasetId = null }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [showUnavailable, setShowUnavailable] = useState(false);

  const fetchOverview = async (forceRefresh = false) => {
    if (!token) return;
    if (forceRefresh) setRefreshing(true);
    else setLoading(true);
    setError('');

    try {
      const url = initialDatasetId
        ? `/api/datasets/${initialDatasetId}/intelligence${forceRefresh ? '?refresh=true' : ''}`
        : `/api/datasets/intelligence/overview${forceRefresh ? '?refresh=true' : ''}`;

      const res = await fetch(url, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to load Product Intelligence');
      }

      const result = await res.json();
      setData(result);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Error loading Product Intelligence');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchOverview();
  }, [token, initialDatasetId]);

  if (loading && !data) {
    return (
      <div className="p-8 text-center border border-mist rounded-control bg-surface">
        <Spinner className="h-6 w-6 mx-auto mb-3 text-teal" />
        <p className="text-sm font-medium text-ink">Analyzing semantic structures & relationships…</p>
        <p className="text-xs text-[var(--muted)] mt-1">
          Evaluating column roles, distributions, dataset grain, and join safety.
        </p>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="p-6 border border-coral/30 bg-coral-soft rounded-control">
        <h4 className="font-display text-sm font-semibold text-coral">Product Intelligence Unavailable</h4>
        <p className="text-xs text-ink/80 mt-1">{error}</p>
        <button
          type="button"
          onClick={() => fetchOverview(true)}
          className="btn-secondary text-xs mt-3"
        >
          Retry Analysis
        </button>
      </div>
    );
  }

  const readiness = data?.readiness || { total_datasets: 0, ready_datasets: 0, total_rows: 0, dataset_summaries: [] };
  const metrics = data?.metrics || [];
  const unavailableMetrics = data?.unavailable_metrics || [];
  const insights = data?.insights || [];
  const opportunities = data?.opportunities || [];
  const qualityWarnings = data?.data_quality_warnings || [];
  const safetyWarnings = data?.safety_warnings || [];

  if (readiness.total_datasets === 0) {
    return (
      <div className="p-8 text-center border border-mist rounded-control bg-surface">
        <h3 className="font-display text-lg font-semibold text-ink">No Datasets Available</h3>
        <p className="text-sm text-[var(--muted)] mt-1">
          Upload at least one dataset to generate dynamic product intelligence, metric summaries, and insights.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-page-in">
      {/* Top Banner & Readiness Summary */}
      <div className="p-5 border border-mist rounded-control bg-surface flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="font-display text-lg font-semibold text-ink">Dynamic Product Intelligence</h2>
            <span className="badge bg-teal-soft text-teal border border-teal/20 text-xs">
              {readiness.ready_datasets} of {readiness.total_datasets} datasets ready
            </span>
          </div>
          <p className="text-xs text-[var(--muted)] mt-1 max-w-2xl">
            Synthesized across {readiness.total_rows.toLocaleString()} records using inferred semantic roles and validated relationships. Never assumes rigid industry schemas or unproven causation.
          </p>
        </div>
        <button
          type="button"
          onClick={() => fetchOverview(true)}
          disabled={refreshing}
          className="btn-secondary text-xs px-3 py-1.5 flex items-center gap-1.5"
        >
          {refreshing && <Spinner className="h-3.5 w-3.5 text-teal" />}
          {refreshing ? 'Refreshing…' : 'Refresh Intelligence'}
        </button>
      </div>

      {/* Safety & Quality Alerts */}
      {(safetyWarnings.length > 0 || qualityWarnings.length > 0) && (
        <div className="space-y-2">
          {safetyWarnings.map((sw, idx) => (
            <div key={`sw-${idx}`} className="p-3.5 border border-coral/30 bg-coral-soft rounded-control flex items-start gap-3 text-xs">
              <span className="text-coral font-bold shrink-0">⚠️ Safety Notice:</span>
              <div className="flex-1">
                <p className="text-ink font-medium">{sw.message}</p>
                {sw.affected_measures?.length > 0 && (
                  <p className="text-[var(--muted)] mt-0.5">
                    Measures safeguarded from double counting: {sw.affected_measures.join(', ')}
                  </p>
                )}
              </div>
            </div>
          ))}
          {qualityWarnings.slice(0, 3).map((qw, idx) => (
            <div key={`qw-${idx}`} className="p-3 border border-mist bg-paper rounded-control flex items-start gap-3 text-xs">
              <span className="text-amber-500 font-semibold shrink-0">ℹ️ Data Quality:</span>
              <div className="flex-1">
                <span className="text-ink font-medium">{qw.message}</span>
                <span className="text-[var(--muted)] ml-2">— {qw.impact}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Key Data-Driven Metrics */}
      <section>
        <div className="flex justify-between items-center mb-3">
          <h3 className="font-display text-sm font-semibold uppercase tracking-wider text-[var(--muted)]">
            Discovered Metrics & Distributions ({metrics.length})
          </h3>
          {unavailableMetrics.length > 0 && (
            <button
              type="button"
              onClick={() => setShowUnavailable(!showUnavailable)}
              className="text-xs text-teal hover:underline"
            >
              {showUnavailable ? 'Hide Unsupported Metrics' : `View Unsupported Metrics (${unavailableMetrics.length})`}
            </button>
          )}
        </div>

        {showUnavailable && unavailableMetrics.length > 0 && (
          <div className="mb-4 p-4 border border-mist bg-paper rounded-control space-y-2 text-xs">
            <h4 className="font-semibold text-ink">Unsupported Metrics & Evidence Rationale</h4>
            <ul className="divide-y divide-mist">
              {unavailableMetrics.map((um, i) => (
                <li key={i} className="py-2 flex justify-between gap-4">
                  <span className="font-medium text-ink">{um.label}</span>
                  <span className="text-[var(--muted)] text-right">{um.reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {metrics.length === 0 ? (
          <div className="p-5 border border-mist rounded-control text-xs text-[var(--muted)] text-center">
            No active measures or categorical dimensions discovered for computation.
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {metrics.map((m) => (
              <div key={m.id} className="p-4 border border-mist rounded-control bg-surface flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-start gap-2 mb-2">
                    <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded bg-mist text-ink/70">
                      {m.category.replace('_', ' ')}
                    </span>
                    <span className="text-[10px] text-[var(--muted)] truncate max-w-[120px]">
                      {m.dataset_name}
                    </span>
                  </div>
                  <h4 className="font-display text-sm font-semibold text-ink mb-1">{m.label}</h4>

                  {/* Metric Value or Breakdown */}
                  {m.summary_stats?.breakdown ? (
                    <div className="mt-3 space-y-1.5">
                      {m.summary_stats.breakdown.slice(0, 4).map((g, gi) => (
                        <div key={gi} className="text-xs">
                          <div className="flex justify-between text-[11px] mb-0.5">
                            <span className="text-ink font-medium truncate max-w-[140px]">{g.group}</span>
                            <span className="text-[var(--muted)]">{g.share_pct}%</span>
                          </div>
                          <div className="w-full bg-mist h-1 rounded-full overflow-hidden">
                            <div className="bg-teal h-1 rounded-full" style={{ width: `${Math.min(100, g.share_pct)}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : m.summary_stats?.periods ? (
                    <div className="mt-2">
                      <div className="flex items-baseline gap-2">
                        <span className="text-xl font-display font-semibold text-ink">
                          {m.value !== null ? `${m.value > 0 ? '+' : ''}${m.value}%` : '—'}
                        </span>
                        <span className={`text-xs font-medium ${m.summary_stats.direction === 'upward' ? 'text-teal' : m.summary_stats.direction === 'downward' ? 'text-coral' : 'text-[var(--muted)]'}`}>
                          {m.summary_stats.direction}
                        </span>
                      </div>
                      <p className="text-[11px] text-[var(--muted)] mt-1">
                        Across {m.summary_stats.period_count} {m.summary_stats.period_type} periods ({m.summary_stats.start_period} to {m.summary_stats.end_period})
                      </p>
                    </div>
                  ) : (
                    <div className="mt-2">
                      <span className="text-2xl font-display font-semibold text-ink">
                        {typeof m.value === 'number' ? m.value.toLocaleString() : (m.value ?? '—')}
                      </span>
                      {m.summary_stats?.mean !== undefined && (
                        <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-mist text-[11px] text-[var(--muted)]">
                          <div>Mean: {m.summary_stats.mean?.toLocaleString()}</div>
                          <div>Median: {m.summary_stats.median?.toLocaleString()}</div>
                          <div>Min: {m.summary_stats.min?.toLocaleString()}</div>
                          <div>Max: {m.summary_stats.max?.toLocaleString()}</div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Evidence & Caveats */}
                <div className="mt-3 pt-3 border-t border-mist space-y-1">
                  {m.evidence?.map((ev, ei) => (
                    <p key={ei} className="text-[10px] text-[var(--muted)] leading-tight">
                      • {ev}
                    </p>
                  ))}
                  {m.caveats?.map((cav, ci) => (
                    <p key={ci} className="text-[10px] text-amber-600 dark:text-amber-400 leading-tight">
                      * {cav}
                    </p>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Explainable Insights & Opportunities Side-by-Side */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Insights Column */}
        <section>
          <h3 className="font-display text-sm font-semibold uppercase tracking-wider text-[var(--muted)] mb-3">
            Explainable Insights ({insights.length})
          </h3>
          {insights.length === 0 ? (
            <div className="p-4 border border-mist rounded-control text-xs text-[var(--muted)]">
              Insufficient variance or patterns to formulate high-confidence observations.
            </div>
          ) : (
            <div className="space-y-3">
              {insights.map((ins) => (
                <div key={ins.id} className="p-4 border border-mist rounded-control bg-surface">
                  <div className="flex justify-between items-start gap-2 mb-1.5">
                    <h4 className="font-display text-sm font-semibold text-ink">{ins.title}</h4>
                    <span className="badge text-[10px] bg-teal-soft text-teal border border-teal/20" title="Heuristic rule confidence, not calibrated probability">
                      Heuristic: {(ins.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p className="text-xs text-ink/90 font-medium mb-2">{ins.finding}</p>
                  <p className="text-xs text-[var(--muted)] mb-3 leading-relaxed">{ins.why_it_matters}</p>

                  <div className="p-2.5 rounded bg-paper border border-mist text-[11px] space-y-1">
                    <p className="text-[10px] uppercase font-semibold text-[var(--muted)]">Evidence & Limitations:</p>
                    {ins.evidence?.map((ev, i) => (
                      <p key={i} className="text-[var(--muted)]">• {ev}</p>
                    ))}
                    {ins.limitations?.map((lim, i) => (
                      <p key={i} className="text-amber-600 dark:text-amber-400 italic">⚠️ {lim}</p>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Opportunities Column */}
        <section>
          <h3 className="font-display text-sm font-semibold uppercase tracking-wider text-[var(--muted)] mb-3">
            Investigation Opportunities ({opportunities.length})
          </h3>
          {opportunities.length === 0 ? (
            <div className="p-4 border border-mist rounded-control text-xs text-[var(--muted)]">
              No immediate investigative candidates surfaced from current metric evidence.
            </div>
          ) : (
            <div className="space-y-3">
              {opportunities.map((opp) => (
                <div key={opp.id} className="p-4 border border-teal/30 bg-teal-soft/10 rounded-control">
                  <div className="flex justify-between items-start gap-2 mb-1.5">
                    <h4 className="font-display text-sm font-semibold text-ink">{opp.title}</h4>
                    <span className="badge text-[10px] bg-mist text-ink/80">
                      Investigation Candidate
                    </span>
                  </div>
                  <p className="text-xs text-ink/90 mb-2 leading-relaxed">{opp.candidate_area}</p>

                  <div className="space-y-1 text-[11px] mt-2 pt-2 border-t border-teal/20">
                    <p className="text-[10px] uppercase font-semibold text-[var(--muted)]">Missing Evidence Needed Before Action:</p>
                    {opp.missing_evidence?.map((me, i) => (
                      <p key={i} className="text-[var(--muted)]">• {me}</p>
                    ))}
                    {opp.uncertainty_notes?.map((un, i) => (
                      <p key={i} className="text-teal font-medium mt-1">ℹ️ {un}</p>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
