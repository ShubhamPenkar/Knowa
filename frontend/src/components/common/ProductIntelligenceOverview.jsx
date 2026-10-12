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

  const [expandedPairs, setExpandedPairs] = useState({});
  const [validatingRelId, setValidatingRelId] = useState(null);
  const [validationError, setValidationError] = useState('');

  const togglePair = (pairKey) => {
    setExpandedPairs((prev) => ({ ...prev, [pairKey]: !prev[pairKey] }));
  };

  const handleValidateRelationship = async (rel) => {
    if (!token || !rel) return;
    setValidatingRelId(rel.id);
    setValidationError('');
    try {
      const res = await fetch('/api/datasets/relationships/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          source_dataset_id: rel.source_dataset_id,
          source_column: rel.source_column,
          target_dataset_id: rel.target_dataset_id,
          target_column: rel.target_column,
        }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to validate relationship join');
      }
      await fetchOverview(true);
    } catch (err) {
      console.error(err);
      setValidationError(err.message || 'Error validating relationship');
    } finally {
      setValidatingRelId(null);
    }
  };

  const readiness = data?.readiness || { total_datasets: 0, ready_datasets: 0, total_rows: 0, dataset_summaries: [] };
  const metrics = data?.metrics || [];
  const unavailableMetrics = data?.unavailable_metrics || [];
  const insights = data?.insights || [];
  const opportunities = data?.opportunities || [];
  const qualityWarnings = data?.data_quality_warnings || [];
  const safetyWarnings = data?.safety_warnings || [];
  const relationships = data?.relationships || [];

  // Group relationships by dataset pair
  const pairGroups = relationships.reduce((acc, rel) => {
    const ids = [rel.source_dataset_id, rel.target_dataset_id].sort();
    const pairKey = `${ids[0]}__${ids[1]}`;
    if (!acc[pairKey]) {
      acc[pairKey] = {
        pairKey,
        sourceName: rel.source_dataset_name,
        targetName: rel.target_dataset_name,
        pairLabel: `${rel.source_dataset_name} ↔ ${rel.target_dataset_name}`,
        relationships: [],
        counts: { candidate: 0, needs_review: 0, validated: 0, rejected: 0 },
        hasDoubleCounting: false,
      };
    }
    acc[pairKey].relationships.push(rel);
    const status = rel.validation_status;
    if (status === 'validated') acc[pairKey].counts.validated += 1;
    else if (status === 'rejected') acc[pairKey].counts.rejected += 1;
    else if (status === 'needs_review' || status === 'ambiguous') acc[pairKey].counts.needs_review += 1;
    else acc[pairKey].counts.candidate += 1;

    if (rel.double_counting_risk) {
      acc[pairKey].hasDoubleCounting = true;
    }
    return acc;
  }, {});

  const pairGroupsList = Object.values(pairGroups);

  const overallCounts = relationships.reduce(
    (acc, r) => {
      const status = r.validation_status;
      if (status === 'validated') acc.validated += 1;
      else if (status === 'rejected') acc.rejected += 1;
      else if (status === 'needs_review' || status === 'ambiguous') acc.needs_review += 1;
      else acc.candidate += 1;
      return acc;
    },
    { candidate: 0, needs_review: 0, validated: 0, rejected: 0 }
  );

  // Critical double-counting risk measures
  const doubleCountingWarnings = safetyWarnings.filter(
    (sw) => sw.risk === 'double_counting_risk' || sw.warning_type === 'fan_out'
  );
  const allSafeguardedMeasures = Array.from(
    new Set(doubleCountingWarnings.flatMap((w) => w.affected_measures || []))
  );

  const crossDatasetUnavailable = unavailableMetrics.find((um) => um.category === 'cross_dataset');
  const hasCrossDatasetMetric = metrics.some((m) => m.category === 'cross_dataset');

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

      {/* Prominent Data-Integrity & Double-Counting Safeguard Alert */}
      {allSafeguardedMeasures.length > 0 && (
        <div className="p-3.5 border border-coral/40 bg-coral-soft rounded-control flex items-start gap-3 text-xs">
          <span className="text-coral font-bold shrink-0">🛡️ Double-Counting Safeguards Active:</span>
          <div className="flex-1">
            <p className="text-ink font-medium">
              Fan-out join guards are active to prevent measure inflation across independent grains.
            </p>
            <p className="text-[var(--muted)] mt-0.5">
              Measures safeguarded from double counting: {allSafeguardedMeasures.join(', ')}
            </p>
          </div>
        </div>
      )}

      {/* Compact Relationship Discovery & Review Section */}
      {(relationships.length > 0 || readiness.total_datasets > 1) && (
        <section className="border border-mist rounded-control bg-surface p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-display text-sm font-semibold text-ink">
                  Cross-Dataset Relationship Discovery & Join Safety
                </h3>
                <span className="text-xs text-[var(--muted)]">
                  ({relationships.length} pair{relationships.length !== 1 ? 's' : ''} evaluated)
                </span>
              </div>
              <p className="text-xs text-[var(--muted)] mt-0.5 max-w-2xl">
                {hasCrossDatasetMetric
                  ? 'Unified cross-dataset metrics active using validated, grain-safe joins.'
                  : crossDatasetUnavailable
                  ? crossDatasetUnavailable.reason
                  : 'Candidate relationships require explicit validation before analytical use to safeguard against fan-out inflation.'}
              </p>
            </div>

            {/* Overall status count badges */}
            <div className="flex flex-wrap items-center gap-2 text-xs">
              {overallCounts.candidate > 0 && (
                <span className="badge bg-teal-soft text-teal border border-teal/20" title="High-confidence candidates awaiting validation">
                  {overallCounts.candidate} Candidate{overallCounts.candidate !== 1 ? 's' : ''}
                </span>
              )}
              {overallCounts.needs_review > 0 && (
                <span className="badge bg-amber-500/10 text-amber-700 dark:text-amber-300 border border-amber-500/30" title="Ambiguous or partial overlap pairs requiring confirmation">
                  {overallCounts.needs_review} Needs Review
                </span>
              )}
              {overallCounts.validated > 0 && (
                <span className="badge bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border border-emerald-500/30" title="Explicitly validated for analytical use">
                  {overallCounts.validated} Validated
                </span>
              )}
              {overallCounts.rejected > 0 && (
                <span className="badge bg-mist text-ink/70 border border-mist" title="Rejected join candidates">
                  {overallCounts.rejected} Rejected
                </span>
              )}
            </div>
          </div>

          {validationError && (
            <div className="p-2.5 rounded bg-coral-soft border border-coral/30 text-xs text-coral">
              {validationError}
            </div>
          )}

          {/* Grouped by Dataset Pair */}
          {pairGroupsList.length === 0 ? (
            <p className="text-xs text-[var(--muted)] italic">
              No plausible cross-dataset key relationships discovered between registered datasets.
            </p>
          ) : (
            <div className="space-y-3">
              {pairGroupsList.map((group) => {
                const isExpanded = !!expandedPairs[group.pairKey];
                return (
                  <div key={group.pairKey} className="border border-mist rounded-control bg-paper overflow-hidden">
                    {/* Clickable Header */}
                    <div
                      onClick={() => togglePair(group.pairKey)}
                      className="p-3.5 flex flex-wrap items-center justify-between gap-3 cursor-pointer hover:bg-mist/30 transition-colors"
                    >
                      <div className="flex items-center gap-2.5">
                        <span className="font-semibold text-xs text-ink">{group.pairLabel}</span>
                        <div className="flex items-center gap-1.5 text-[11px]">
                          {group.counts.validated > 0 && (
                            <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-700 font-medium">
                              {group.counts.validated} validated
                            </span>
                          )}
                          {group.counts.candidate > 0 && (
                            <span className="px-1.5 py-0.5 rounded bg-teal-soft text-teal font-medium">
                              {group.counts.candidate} candidate{group.counts.candidate !== 1 ? 's' : ''}
                            </span>
                          )}
                          {group.counts.needs_review > 0 && (
                            <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-700 font-medium">
                              {group.counts.needs_review} needs review
                            </span>
                          )}
                          {group.counts.rejected > 0 && (
                            <span className="px-1.5 py-0.5 rounded bg-mist text-ink/70">
                              {group.counts.rejected} rejected
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {group.hasDoubleCounting && (
                          <span className="text-[10px] text-amber-600 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20 font-medium">
                            Fan-out Guarded
                          </span>
                        )}
                        <button
                          type="button"
                          className="text-xs font-medium text-teal hover:underline flex items-center gap-1"
                        >
                          {isExpanded ? 'Hide Details ▲' : `Review Candidates (${group.relationships.length}) ▼`}
                        </button>
                      </div>
                    </div>

                    {/* Expandable Candidates List */}
                    {isExpanded && (
                      <div className="p-3.5 border-t border-mist space-y-3 bg-surface">
                        {group.relationships.map((rel) => {
                          const isVal = rel.validation_status === 'validated';
                          const isRej = rel.validation_status === 'rejected';
                          const isReview = rel.validation_status === 'needs_review' || rel.validation_status === 'ambiguous';
                          const isValidating = validatingRelId === rel.id;

                          return (
                            <div key={rel.id} className="p-3 border border-mist rounded bg-paper space-y-2">
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="flex items-center gap-2 text-xs font-mono">
                                  <span className="font-semibold text-ink">{rel.source_dataset_name}.{rel.source_column}</span>
                                  <span className="text-[var(--muted)]">➔</span>
                                  <span className="font-semibold text-ink">{rel.target_dataset_name}.{rel.target_column}</span>
                                </div>

                                <div className="flex items-center gap-1.5">
                                  <span
                                    className={`badge text-[10px] ${
                                      isVal
                                        ? 'bg-emerald-500/10 text-emerald-700 border border-emerald-500/30'
                                        : isRej
                                        ? 'bg-mist text-ink/70 border border-mist'
                                        : isReview
                                        ? 'bg-amber-500/10 text-amber-700 border border-amber-500/30'
                                        : 'bg-teal-soft text-teal border border-teal/20'
                                    }`}
                                  >
                                    {isVal ? 'Validated' : isRej ? 'Rejected' : isReview ? 'Needs Review' : 'Candidate'}
                                  </span>
                                  <span className="badge text-[10px] bg-mist text-ink/70">
                                    {rel.cardinality.replace('_', ' ')}
                                  </span>
                                  <span className="badge text-[10px] bg-mist text-ink/70" title="Rule-based heuristic score, not calibrated probability">
                                    Heuristic: {(rel.confidence * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </div>

                              {/* Metrics & Overlap Grid */}
                              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-[11px] text-[var(--muted)] border-t border-mist">
                                <div>Src Uniqueness: {(rel.source_uniqueness_ratio * 100).toFixed(0)}%</div>
                                <div>Tgt Uniqueness: {(rel.target_uniqueness_ratio * 100).toFixed(0)}%</div>
                                <div>Containment: {(rel.containment_ratio * 100).toFixed(1)}%</div>
                                <div>Distinct Overlap: {rel.overlapping_values_count.toLocaleString()}</div>
                              </div>

                              {/* Notice / Reason Callouts */}
                              {rel.is_ambiguous && (
                                <div className="p-2 bg-amber-500/10 border border-amber-500/20 rounded text-[11px] text-amber-700 dark:text-amber-300">
                                  ⚠️ <strong>Ambiguity Notice:</strong> Multiple candidate join keys in target share comparable confidence. Review required to confirm intended join.
                                </div>
                              )}
                              {rel.review_reason && !rel.is_ambiguous && (
                                <div className="p-2 bg-amber-500/10 border border-amber-500/20 rounded text-[11px] text-amber-700 dark:text-amber-300">
                                  ℹ️ <strong>Needs Review:</strong> {rel.review_reason}
                                </div>
                              )}
                              {rel.rejection_reason && (
                                <div className="p-2 bg-coral-soft border border-coral/30 rounded text-[11px] text-coral font-medium">
                                  ✕ <strong>Rejected:</strong> {rel.rejection_reason}
                                </div>
                              )}

                              {/* Evidence list */}
                              {rel.evidence?.length > 0 && (
                                <div className="space-y-0.5 text-[11px] text-[var(--muted)] pt-1">
                                  {rel.evidence.map((ev, ei) => (
                                    <p key={ei}>• {ev}</p>
                                  ))}
                                </div>
                              )}

                              {/* Warnings list */}
                              {rel.warnings?.length > 0 && (
                                <div className="space-y-0.5 text-[11px] text-amber-600 dark:text-amber-400">
                                  {rel.warnings.map((w, wi) => (
                                    <p key={wi}>⚠️ {w}</p>
                                  ))}
                                </div>
                              )}

                              {/* Review & Validation Action */}
                              <div className="pt-2 flex flex-wrap items-center justify-between gap-2 border-t border-mist">
                                {!isVal && !isRej ? (
                                  <button
                                    type="button"
                                    onClick={() => handleValidateRelationship(rel)}
                                    disabled={isValidating}
                                    className="btn-secondary text-[11px] px-2.5 py-1 flex items-center gap-1.5"
                                  >
                                    {isValidating && <Spinner className="h-3 w-3 text-teal" />}
                                    {isValidating ? 'Validating Join…' : 'Validate Join for Metrics'}
                                  </button>
                                ) : isVal ? (
                                  <span className="text-[11px] text-emerald-600 font-medium flex items-center gap-1">
                                    ✓ Validated join key active for analytical metrics
                                  </span>
                                ) : (
                                  <span className="text-[11px] text-coral font-medium">
                                    Join key excluded from analytical use
                                  </span>
                                )}
                                <span className="text-[10px] text-[var(--muted)] italic">
                                  Manual rejection and mapping overrides are planned for future releases.
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </section>
      )}

      {/* Data Quality Alerts */}
      {qualityWarnings.length > 0 && (
        <div className="space-y-2">
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
