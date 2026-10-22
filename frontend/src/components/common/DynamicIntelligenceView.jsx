import { useEffect, useState } from 'react';
import Spinner from './Spinner';

export default function DynamicIntelligenceView({ token, initialDatasetId = null }) {
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState(initialDatasetId);
  const [intelligence, setIntelligence] = useState(null);
  const [opportunities, setOpportunities] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [refreshing, setRefreshing] = useState(false);

  // Filter for opportunities
  const [selectedTypeFilter, setSelectedTypeFilter] = useState('all');

  // Detail modal / drawer for an insight
  const [activeEvidenceInsight, setActiveEvidenceInsight] = useState(null);
  const [evidenceDetail, setEvidenceDetail] = useState(null);
  const [loadingEvidence, setLoadingEvidence] = useState(false);

  // Active executed analysis view
  const [activeAnalysisResult, setActiveAnalysisResult] = useState(null);
  const [executingOppId, setExecutingOppId] = useState(null);

  // 1. Fetch available datasets if no initialDatasetId or for selector
  useEffect(() => {
    if (!token) return;
    const fetchDatasets = async () => {
      try {
        const res = await fetch('/api/datasets', {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          const list = await res.json();
          setDatasets(list);
          if (!selectedDatasetId && list.length > 0) {
            setSelectedDatasetId(list[0].id);
          }
        }
      } catch (err) {
        console.error('Failed to load datasets:', err);
      }
    };
    fetchDatasets();
  }, [token]);

  // 2. Fetch Intelligence and Opportunities for selected dataset
  const fetchDatasetIntelligence = async (datasetId, forceRefresh = false) => {
    if (!datasetId || !token) return;
    if (forceRefresh) setRefreshing(true);
    else setLoading(true);
    setError('');

    try {
      // Fetch proactive insights & executed analyses
      const insightsRes = await fetch(
        `/api/datasets/${datasetId}/insights${forceRefresh ? '?refresh=true' : ''}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (!insightsRes.ok) {
        const errData = await insightsRes.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to load business intelligence');
      }
      const intelData = await insightsRes.json();
      setIntelligence(intelData);

      // Fetch all candidate opportunities
      const oppsRes = await fetch(
        `/api/datasets/${datasetId}/analysis-opportunities${forceRefresh ? '?refresh=true' : ''}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (oppsRes.ok) {
        const oppsData = await oppsRes.json();
        setOpportunities(oppsData.opportunities || []);
      }
    } catch (err) {
      console.error(err);
      setError(err.message || 'Error generating dynamic intelligence');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    if (selectedDatasetId) {
      fetchDatasetIntelligence(selectedDatasetId);
    }
  }, [selectedDatasetId, token]);

  // Run a specific opportunity on demand
  const handleRunOpportunity = async (oppId) => {
    if (!selectedDatasetId || !token) return;
    setExecutingOppId(oppId);
    try {
      const res = await fetch(`/api/datasets/${selectedDatasetId}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ opportunity_ids: [oppId], refresh: true }),
      });
      if (res.ok) {
        const data = await res.json();
        setIntelligence(data);
        const match = data.executed_analyses?.find((a) => a.opportunity_id === oppId);
        if (match) setActiveAnalysisResult(match);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setExecutingOppId(null);
    }
  };

  // View full explainability trace
  const handleOpenEvidence = async (ins) => {
    setActiveEvidenceInsight(ins);
    setLoadingEvidence(true);
    try {
      const res = await fetch(`/api/datasets/${selectedDatasetId}/insights/${ins.id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const detail = await res.json();
        setEvidenceDetail(detail);
      } else {
        setEvidenceDetail({
          insight: ins,
          finding: ins.title,
          metric_calculation: 'Calculated directly from non-null verified data observations',
          data_used: `Dataset '${intelligence?.dataset_name || 'Current'}'`,
          relevant_dimensions: ins.affected_segment ? [ins.affected_segment] : [],
          statistical_evidence: ins.evidence || [],
          confidence: ins.confidence,
          limitations: ins.limitations || [],
        });
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingEvidence(false);
    }
  };

  const getSeverityBadge = (sev) => {
    switch (sev) {
      case 'critical':
        return (
          <span className="badge bg-coral-soft text-coral border border-coral/30 font-semibold tracking-wide text-[10px]">
            CRITICAL
          </span>
        );
      case 'warning':
        return (
          <span className="badge bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30 font-semibold tracking-wide text-[10px]">
            WARNING
          </span>
        );
      case 'positive':
        return (
          <span className="badge bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 font-semibold tracking-wide text-[10px]">
            POSITIVE
          </span>
        );
      default:
        return (
          <span className="badge bg-mist text-ink/80 border border-mist font-medium tracking-wide text-[10px]">
            INFO
          </span>
        );
    }
  };

  const filteredOpportunities = opportunities.filter((opp) => {
    if (selectedTypeFilter === 'all') return true;
    return opp.analysis_type === selectedTypeFilter;
  });

  if (datasets.length === 0 && !loading) {
    return (
      <div className="p-8 text-center border border-mist rounded-control bg-surface">
        <h3 className="font-display text-lg font-semibold text-ink">No Datasets Available</h3>
        <p className="text-sm text-[var(--muted)] mt-1">
          Upload a dataset to generate dynamic business analysis opportunities and proactive insights.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-page-in">
      {/* Header & Dataset Switcher */}
      <div className="p-5 border border-mist rounded-control bg-surface flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="font-display text-lg font-semibold text-ink">
              Executive Findings & Automated Analyses
            </h2>
            <span className="status-pill-safe text-[11px]">
              Continuous Discovery Active
            </span>
          </div>
          <p className="text-xs text-[var(--muted)] mt-1 max-w-2xl">
            Proactive discovery of meaningful changes, operational patterns, and statistical anomalies synthesized across your verified data.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {datasets.length > 1 && (
            <select
              value={selectedDatasetId || ''}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              className="text-xs border border-mist rounded-control bg-surface px-3 py-1.5 font-medium text-ink focus:outline-none focus:border-teal"
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.row_count?.toLocaleString()} rows)
                </option>
              ))}
            </select>
          )}

          <button
            type="button"
            onClick={() => fetchDatasetIntelligence(selectedDatasetId, true)}
            disabled={refreshing || loading}
            className="btn-secondary text-xs px-3 py-1.5 flex items-center gap-1.5"
          >
            {refreshing && <Spinner className="h-3.5 w-3.5 text-teal" />}
            {refreshing ? 'Re-evaluating…' : 'Refresh Analysis'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 border border-coral/30 bg-coral-soft rounded-control text-xs text-ink flex items-start gap-2">
          <span className="text-coral font-bold">⚠️ Error:</span>
          <span>{error}</span>
        </div>
      )}

      {loading && !intelligence ? (
        <div className="p-12 text-center border border-mist rounded-control bg-surface">
          <Spinner className="h-7 w-7 mx-auto mb-3 text-teal" />
          <p className="text-sm font-semibold text-ink">Determining Relevant Business Analyses…</p>
          <p className="text-xs text-[var(--muted)] mt-1 max-w-md mx-auto">
            Reasoning over detected business roles, grains, and relationship graphs to synthesize high-priority insights with statistical evidence.
          </p>
        </div>
      ) : (
        <>
          {/* SECTION 1: PROACTIVE TOP BUSINESS INSIGHTS */}
          <section className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="font-display text-base font-semibold text-ink">
                  Top Business Insights
                </h3>
                <p className="text-xs text-[var(--muted)]">
                  Proactively surfaced, deduplicated, and ranked by business severity, magnitude, and confidence.
                </p>
              </div>
              <span className="text-xs text-[var(--muted)] font-mono">
                {intelligence?.top_insights?.length || 0} top findings
              </span>
            </div>

            {(!intelligence?.top_insights || intelligence.top_insights.length === 0) ? (
              <div className="p-6 border border-mist rounded-control bg-surface text-center text-xs text-[var(--muted)]">
                No high-confidence anomalies or major variances detected in this dataset.
              </div>
            ) : (
              <div className="grid md:grid-cols-2 gap-4">
                {intelligence.top_insights.map((ins) => (
                  <div
                    key={ins.id}
                    className="p-4 border border-mist rounded-control bg-surface hover:border-teal/50 transition-colors flex flex-col justify-between"
                  >
                    <div>
                      {/* Badge row */}
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <div className="flex items-center gap-2">
                          {getSeverityBadge(ins.severity)}
                          <span className="text-[10px] uppercase font-semibold text-[var(--muted)] px-1.5 py-0.5 rounded bg-mist/60">
                            {ins.category}
                          </span>
                        </div>
                        <span className="text-[10px] text-teal font-medium">
                          Confidence: {(ins.confidence * 100).toFixed(0)}%
                        </span>
                      </div>

                      {/* Headline */}
                      <h4 className="font-display text-sm font-bold text-ink mb-1">
                        {ins.title}
                      </h4>

                      {/* Affected Segment */}
                      {ins.affected_segment && (
                        <p className="text-[11px] text-teal font-medium mb-2">
                          Cohort: {ins.affected_segment}
                        </p>
                      )}

                      {/* Evidence summary */}
                      <div className="space-y-1 mb-3">
                        {ins.evidence?.slice(0, 2).map((ev, ei) => (
                          <p key={ei} className="text-xs text-ink/80 flex items-start gap-1.5 leading-snug">
                            <span className="text-teal font-bold shrink-0">✓</span>
                            <span>{ev}</span>
                          </p>
                        ))}
                      </div>

                      {/* Why it matters */}
                      <div className="p-2.5 rounded bg-paper border border-mist text-xs text-[var(--muted)] mb-3">
                        <span className="font-semibold text-ink text-[11px] block mb-0.5">Why this matters:</span>
                        {ins.why_it_matters}
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="pt-3 border-t border-mist flex flex-wrap items-center justify-between gap-2">
                      <button
                        type="button"
                        onClick={() => handleOpenEvidence(ins)}
                        className="text-xs text-teal hover:underline font-semibold flex items-center gap-1"
                      >
                        <span>Inspect Evidence & Calculations</span> →
                      </button>

                      <div className="flex items-center gap-2">
                        <a
                          href="/whatif"
                          className="text-xs text-muted hover:text-ink font-medium"
                          title="Simulate intervention levers for this insight"
                        >
                          Simulate in What-If →
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* SECTION 2: RECOMMENDED ANALYSES */}
          <section className="space-y-4 pt-4 border-t border-mist">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="font-display text-base font-semibold text-ink">
                  Recommended Business Analyses
                </h3>
                <p className="text-xs text-[var(--muted)]">
                  Data-driven analytical workflows prioritized by potential business impact, data completeness, and grain safety.
                </p>
              </div>

              {/* Type Filter */}
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                {['all', 'descriptive', 'trend', 'comparative', 'correlation', 'anomaly', 'segmentation', 'cross_dataset'].map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setSelectedTypeFilter(t)}
                    className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                      selectedTypeFilter === t
                        ? 'bg-teal text-white'
                        : 'bg-mist text-[var(--muted)] hover:text-ink'
                    }`}
                  >
                    {t.replace('_', ' ').toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredOpportunities.map((opp) => {
                const isExecuting = executingOppId === opp.id;
                let actionLabel = 'Explore Finding';
                if (opp.analysis_type === 'trend') actionLabel = 'Analyze Trend';
                else if (opp.analysis_type === 'comparative') actionLabel = 'Compare Segments';
                else if (opp.analysis_type === 'correlation') actionLabel = 'Test Association';
                else if (opp.analysis_type === 'anomaly') actionLabel = 'Audit Outliers';
                else if (opp.analysis_type === 'segmentation') actionLabel = 'Discover Segments';
                else if (opp.analysis_type === 'cross_dataset') actionLabel = 'Verify Connection';

                return (
                  <div
                    key={opp.id}
                    className="p-4 border border-mist rounded-control bg-surface flex flex-col justify-between hover:border-teal/40 transition-colors"
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-mist text-ink/80">
                          {opp.analysis_type.replace('_', ' ')}
                        </span>
                        <span className="badge text-[11px] bg-teal-soft text-teal border border-teal/20 font-semibold" title="Analytical relevance score">
                          {(opp.relevance_score * 100).toFixed(0)}% Relevance
                        </span>
                      </div>

                      <h4 className="font-display text-sm font-semibold text-ink mb-1">
                        {opp.title}
                      </h4>
                      <p className="text-xs text-[var(--muted)] mb-3 leading-relaxed">
                        {opp.description}
                      </p>

                      {/* Evidence Checklist: Recommended because */}
                      <div className="space-y-1 mb-4 p-2.5 rounded bg-paper border border-mist">
                        <p className="text-[10px] uppercase font-bold text-[var(--muted)] tracking-wider mb-1">
                          Recommended because:
                        </p>
                        {opp.evidence_factors?.map((ef, idx) => (
                          <p key={idx} className="text-[11px] text-ink/90 flex items-start gap-1.5">
                            <span className="text-teal font-bold shrink-0">✓</span>
                            <span>{ef}</span>
                          </p>
                        ))}

                        {opp.reason_unavailable && (
                          <p className="text-[11px] text-coral font-medium mt-1">
                            ⚠️ {opp.reason_unavailable}
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Run / Explore button with specific label */}
                    <div className="pt-2 border-t border-mist flex items-center justify-between">
                      <span className="text-[10px] text-[var(--muted)]">
                        {opp.grain_safe ? '🛡️ Safe Grain' : '⚠️ Grain Restricted'}
                      </span>
                      <button
                        type="button"
                        onClick={() => handleRunOpportunity(opp.id)}
                        disabled={isExecuting || opp.status === 'unavailable'}
                        className="btn-secondary text-xs px-2.5 py-1 flex items-center gap-1"
                      >
                        {isExecuting && <Spinner className="h-3 w-3 text-teal" />}
                        {isExecuting ? 'Evaluating…' : actionLabel}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* ACTIVE EXECUTED ANALYSIS MODAL */}
          {activeAnalysisResult && (
            <div className="fixed inset-0 z-50 bg-ink/50 flex items-center justify-center p-4">
              <div className="bg-surface border border-mist rounded-control w-full max-w-2xl p-6 max-h-[85vh] overflow-y-auto space-y-4 animate-page-in shadow-xl">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-mist text-teal">
                      {activeAnalysisResult.analysis_type}
                    </span>
                    <h3 className="font-display text-lg font-bold text-ink mt-1">
                      {activeAnalysisResult.title}
                    </h3>
                  </div>
                  <button
                    type="button"
                    onClick={() => setActiveAnalysisResult(null)}
                    className="text-xs text-[var(--muted)] hover:text-ink font-bold px-2 py-1"
                  >
                    ✕ Close
                  </button>
                </div>

                {/* Summary stats */}
                <div className="p-4 border border-mist rounded bg-paper space-y-2">
                  <h4 className="font-semibold text-xs text-ink uppercase tracking-wider">
                    Summary Statistics
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                    {Object.entries(activeAnalysisResult.summary_stats || {}).map(([k, v]) => {
                      if (typeof v === 'object' && v !== null) return null;
                      return (
                        <div key={k} className="p-2 border border-mist/60 rounded bg-surface">
                          <p className="text-[10px] text-[var(--muted)] uppercase">{k.replace('_', ' ')}</p>
                          <p className="text-sm font-semibold text-ink mt-0.5">{String(v)}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Evidence */}
                {activeAnalysisResult.evidence?.length > 0 && (
                  <div className="space-y-1">
                    <h4 className="font-semibold text-xs text-ink uppercase tracking-wider">
                      Empirical Evidence
                    </h4>
                    {activeAnalysisResult.evidence.map((ev, i) => (
                      <p key={i} className="text-xs text-ink/80 flex items-start gap-1.5">
                        <span className="text-teal font-bold">•</span>
                        <span>{ev}</span>
                      </p>
                    ))}
                  </div>
                )}

                {/* Limitations */}
                {activeAnalysisResult.limitations?.length > 0 && (
                  <div className="p-3 bg-paper border border-mist rounded text-xs space-y-1 text-[var(--muted)]">
                    <p className="font-semibold text-[11px] text-ink">Analytical Limitations:</p>
                    {activeAnalysisResult.limitations.map((lim, i) => (
                      <p key={i}>⚠️ {lim}</p>
                    ))}
                  </div>
                )}

                <div className="pt-3 border-t border-mist flex justify-end">
                  <button
                    type="button"
                    onClick={() => setActiveAnalysisResult(null)}
                    className="btn-secondary text-xs"
                  >
                    Done
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* EVIDENCE & EXPLAINABILITY MODAL */}
          {activeEvidenceInsight && (
            <div className="fixed inset-0 z-50 bg-ink/50 flex items-center justify-center p-4">
              <div className="bg-surface border border-mist rounded-control w-full max-w-xl p-6 space-y-4 animate-page-in shadow-xl max-h-[85vh] overflow-y-auto">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="text-[10px] uppercase font-bold text-teal">
                      Insight Explainability Trace
                    </span>
                    <h3 className="font-display text-base font-bold text-ink mt-0.5">
                      {activeEvidenceInsight.title}
                    </h3>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setActiveEvidenceInsight(null);
                      setEvidenceDetail(null);
                    }}
                    className="text-xs text-[var(--muted)] hover:text-ink font-bold px-2 py-1"
                  >
                    ✕ Close
                  </button>
                </div>

                {loadingEvidence ? (
                  <div className="py-8 text-center">
                    <Spinner className="h-5 w-5 mx-auto text-teal" />
                  </div>
                ) : (
                  <div className="space-y-3 text-xs">
                    <div className="p-3 rounded bg-paper border border-mist">
                      <p className="text-[10px] uppercase font-bold text-[var(--muted)]">Finding</p>
                      <p className="text-ink font-medium mt-0.5">{evidenceDetail?.finding || activeEvidenceInsight.title}</p>
                    </div>

                    <div className="p-3 rounded bg-paper border border-mist">
                      <p className="text-[10px] uppercase font-bold text-[var(--muted)]">Metric Calculation</p>
                      <p className="text-ink font-mono text-[11px] mt-0.5">{evidenceDetail?.metric_calculation}</p>
                    </div>

                    <div className="p-3 rounded bg-paper border border-mist">
                      <p className="text-[10px] uppercase font-bold text-[var(--muted)]">Data Source & Sample</p>
                      <p className="text-ink mt-0.5">{evidenceDetail?.data_used}</p>
                    </div>

                    <div className="space-y-1">
                      <p className="text-[10px] uppercase font-bold text-[var(--muted)]">Statistical Evidence</p>
                      {(evidenceDetail?.statistical_evidence || activeEvidenceInsight.evidence || []).map((ev, i) => (
                        <p key={i} className="text-ink/80 flex items-start gap-1.5">
                          <span className="text-teal font-bold">•</span>
                          <span>{ev}</span>
                        </p>
                      ))}
                    </div>

                    <div className="p-3 rounded bg-paper border border-mist text-[var(--muted)] space-y-1">
                      <p className="text-[10px] uppercase font-bold text-ink">Limitations & Non-Causal Boundary</p>
                      {(evidenceDetail?.limitations || activeEvidenceInsight.limitations || []).map((lim, i) => (
                        <p key={i}>⚠️ {lim}</p>
                      ))}
                    </div>
                  </div>
                )}

                <div className="pt-3 border-t border-mist flex justify-end">
                  <button
                    type="button"
                    onClick={() => {
                      setActiveEvidenceInsight(null);
                      setEvidenceDetail(null);
                    }}
                    className="btn-secondary text-xs"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
