import { useEffect, useState, useMemo } from 'react';
import Spinner from './Spinner';
import { useViewMode } from '../../context/ViewModeContext';
import {
  Sparkles,
  TrendingUp,
  AlertTriangle,
  Users,
  Link2,
  BarChart2,
  Search,
  ChevronDown,
  CheckCircle2,
  ShieldCheck,
  Filter,
  ArrowRight,
  Eye,
  Info
} from 'lucide-react';

export default function DynamicIntelligenceView({ token, initialDatasetId = null }) {
  const { plainEnglish } = useViewMode();

  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState(initialDatasetId);
  const [intelligence, setIntelligence] = useState(null);
  const [opportunities, setOpportunities] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [refreshing, setRefreshing] = useState(false);

  // Filters and curation
  const [selectedTypeFilter, setSelectedTypeFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showAllOpps, setShowAllOpps] = useState(false);
  const [expandedEvidenceId, setExpandedEvidenceId] = useState(null);

  // Detail modal / drawer for an insight
  const [activeEvidenceInsight, setActiveEvidenceInsight] = useState(null);
  const [evidenceDetail, setEvidenceDetail] = useState(null);
  const [loadingEvidence, setLoadingEvidence] = useState(false);

  // Active executed analysis view
  const [activeAnalysisResult, setActiveAnalysisResult] = useState(null);
  const [executingOppId, setExecutingOppId] = useState(null);

  // 1. Fetch available datasets
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
        const data = await res.json();
        setEvidenceDetail(data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingEvidence(false);
    }
  };

  // Helper for severity badges
  const getSeverityBadge = (severity) => {
    switch (severity) {
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

  // 1. Deduplicate opportunities to prevent repetitive card spam
  const uniqueOpportunities = useMemo(() => {
    const seen = new Set();
    const result = [];
    for (const opp of opportunities) {
      // Normalize key by type and title
      const cleanTitle = (opp.title || '')
        .replace(/⋈/g, '&')
        .replace(/\s+/g, ' ')
        .trim()
        .toLowerCase();
      const key = `${opp.analysis_type}__${cleanTitle}`;
      if (!seen.has(key)) {
        seen.add(key);
        result.push(opp);
      }
    }
    return result;
  }, [opportunities]);

  // 2. Count opportunities per category
  const categoryCounts = useMemo(() => {
    const counts = { all: uniqueOpportunities.length };
    for (const opp of uniqueOpportunities) {
      counts[opp.analysis_type] = (counts[opp.analysis_type] || 0) + 1;
    }
    return counts;
  }, [uniqueOpportunities]);

  const categories = [
    { id: 'all', label: plainEnglish ? 'All Recommended' : 'All Types', icon: '✨' },
    { id: 'trend', label: plainEnglish ? 'Trends & Movement' : 'Trends', icon: '📈' },
    { id: 'anomaly', label: plainEnglish ? 'Spikes & Outliers' : 'Anomalies', icon: '🚨' },
    { id: 'segmentation', label: plainEnglish ? 'Customer Cohorts' : 'Segmentation', icon: '👥' },
    { id: 'cross_dataset', label: plainEnglish ? 'Linked Tables' : 'Cross-Dataset', icon: '🔗' },
    { id: 'comparative', label: plainEnglish ? 'Segment Comparisons' : 'Comparative', icon: '⚖️' },
    { id: 'correlation', label: plainEnglish ? 'Metric Associations' : 'Correlation', icon: '🔄' },
    { id: 'descriptive', label: plainEnglish ? 'Summaries' : 'Descriptive', icon: '📊' },
  ];

  // 3. Filtered opportunities by category and search
  const filteredOpportunities = useMemo(() => {
    return uniqueOpportunities.filter((opp) => {
      if (selectedTypeFilter !== 'all' && opp.analysis_type !== selectedTypeFilter) {
        return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchTitle = (opp.title || '').toLowerCase().includes(q);
        const matchDesc = (opp.description || '').toLowerCase().includes(q);
        const matchCol =
          (opp.primary_column || '').toLowerCase().includes(q) ||
          (opp.secondary_column || '').toLowerCase().includes(q);
        return matchTitle || matchDesc || matchCol;
      }
      return true;
    });
  }, [uniqueOpportunities, selectedTypeFilter, searchQuery]);

  // 4. Curate to top 6 by default for a clean, non-overwhelming experience!
  const displayedOpportunities = showAllOpps
    ? filteredOpportunities
    : filteredOpportunities.slice(0, 6);

  const toggleEvidence = (id) => {
    setExpandedEvidenceId((prev) => (prev === id ? null : id));
  };

  const getCategoryMeta = (type) => {
    switch (type) {
      case 'trend':
        return { label: 'Trend Analysis', icon: '📈', cls: 'bg-blue-500/15 text-blue-300 border-blue-500/30' };
      case 'anomaly':
        return { label: 'Outlier Check', icon: '🚨', cls: 'bg-amber-500/15 text-amber-300 border-amber-500/30' };
      case 'segmentation':
        return { label: 'Behavioral Cohorts', icon: '👥', cls: 'bg-purple-500/15 text-purple-300 border-purple-500/30' };
      case 'cross_dataset':
        return { label: 'Linked Tables', icon: '🔗', cls: 'bg-teal-500/15 text-teal-300 border-teal-500/30' };
      case 'comparative':
        return { label: 'Comparison', icon: '⚖️', cls: 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30' };
      case 'correlation':
        return { label: 'Association', icon: '🔄', cls: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30' };
      default:
        return { label: 'Measure Summary', icon: '📊', cls: 'bg-mist text-ink/80 border-mist' };
    }
  };

  if (datasets.length === 0 && !loading) {
    return (
      <div className="p-8 text-center border border-mist rounded-control bg-surface">
        <h3 className="font-display text-lg font-semibold text-ink">No Tables Available</h3>
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
              {plainEnglish ? 'Key Findings & Proactive Business Analyses' : 'Executive Findings & Automated Analyses'}
            </h2>
            <span className="status-pill-safe text-[11px] flex items-center gap-1">
              <Sparkles size={11} className="text-teal" />
              <span>{plainEnglish ? 'Automatic Pattern Discovery' : 'Continuous Discovery Active'}</span>
            </span>
          </div>
          <p className="text-xs text-[var(--muted)] mt-1 max-w-2xl leading-relaxed">
            {plainEnglish
              ? 'Proactively discovers significant trends, unusual spikes, and customer segments across your business data without requiring SQL.'
              : 'Proactive discovery of meaningful changes, operational patterns, and statistical anomalies synthesized across your verified data.'}
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
          <p className="text-sm font-semibold text-ink">Synthesizing Business Insights…</p>
          <p className="text-xs text-[var(--muted)] mt-1 max-w-md mx-auto">
            Analyzing trends, outlier anomalies, and table connections to surface actionable opportunities.
          </p>
        </div>
      ) : (
        <>
          {/* SECTION 1: PROACTIVE TOP BUSINESS INSIGHTS */}
          <section className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="font-display text-base font-semibold text-ink flex items-center gap-2">
                  <span>{plainEnglish ? 'Top Discovered Business Insights' : 'Top Business Insights'}</span>
                  <span className="text-xs px-2 py-0.5 rounded bg-mist text-[var(--muted)] font-normal">
                    {intelligence?.top_insights?.length || 0} findings
                  </span>
                </h3>
                <p className="text-xs text-[var(--muted)]">
                  {plainEnglish
                    ? 'Important patterns surfaced automatically, ranked by business impact and certainty.'
                    : 'Proactively surfaced, deduplicated, and ranked by business severity, magnitude, and confidence.'}
                </p>
              </div>
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
                          Certainty: {(ins.confidence * 100).toFixed(0)}%
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
                        <span>Inspect Evidence & Math</span> →
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

          {/* SECTION 2: CURATED RECOMMENDED ANALYSES (DEDUPLICATED & FOCUSED) */}
          <section className="space-y-4 pt-6 border-t border-mist">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
              <div>
                <h3 className="font-display text-base font-semibold text-ink flex items-center gap-2">
                  <span>{plainEnglish ? 'Recommended Business Analyses' : 'Dynamic Analysis Opportunities'}</span>
                  <span className="text-xs px-2 py-0.5 rounded bg-teal/15 text-teal border border-teal/30 font-medium">
                    {uniqueOpportunities.length} unique opportunities
                  </span>
                </h3>
                <p className="text-xs text-[var(--muted)] mt-0.5">
                  {plainEnglish
                    ? 'Curated analytical workflows you can execute with 1 click to uncover trends, outliers, or segments.'
                    : 'Data-driven analytical workflows prioritized by potential business impact, data completeness, and grain safety.'}
                </p>
              </div>

              {/* Search Bar */}
              <div className="relative">
                <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                <input
                  type="text"
                  placeholder="Search analyses by metric or question…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="input text-xs py-1.5 pl-8 pr-3 w-64"
                />
              </div>
            </div>

            {/* Category Filter Pills with Live Counts */}
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
              {categories.map((c) => {
                const count = categoryCounts[c.id] || 0;
                if (c.id !== 'all' && count === 0) return null; // Don't show empty tabs

                return (
                  <button
                    key={c.id}
                    type="button"
                    onClick={() => setSelectedTypeFilter(c.id)}
                    className={`px-3 py-1.5 rounded-control text-[11px] font-medium transition-all shrink-0 flex items-center gap-1.5 ${
                      selectedTypeFilter === c.id
                        ? 'bg-teal text-paper font-semibold shadow-xs'
                        : 'bg-surface hover:bg-mist text-[var(--muted)] hover:text-ink border border-mist'
                    }`}
                  >
                    <span>{c.icon}</span>
                    <span>{c.label}</span>
                    <span className={`px-1.5 py-0.2 rounded text-[10px] ${selectedTypeFilter === c.id ? 'bg-black/20 text-paper' : 'bg-mist text-[var(--muted)]'}`}>
                      {count}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Curation Notification Bar */}
            <div className="flex items-center justify-between px-3.5 py-2 rounded bg-paper/60 border border-mist text-xs text-[var(--muted)]">
              <span>
                Showing <strong>{displayedOpportunities.length}</strong> {showAllOpps ? 'total' : 'curated top'} analyses
                {selectedTypeFilter !== 'all' && ` in ${selectedTypeFilter.replace('_', ' ')}`}
                {searchQuery && ` matching "${searchQuery}"`}
              </span>

              {filteredOpportunities.length > 6 && (
                <button
                  type="button"
                  onClick={() => setShowAllOpps(!showAllOpps)}
                  className="text-xs text-teal hover:underline font-semibold flex items-center gap-1"
                >
                  <span>{showAllOpps ? 'Show Curated Top 6 ▴' : `View All ${filteredOpportunities.length} Analyses ▾`}</span>
                </button>
              )}
            </div>

            {/* The Cards Grid */}
            {filteredOpportunities.length === 0 ? (
              <div className="p-8 text-center border border-mist rounded-control bg-surface text-xs text-[var(--muted)] space-y-2">
                <p className="text-ink font-medium">No matching analyses found</p>
                <p>Try clearing your search query or selecting a different category tab.</p>
                <button
                  type="button"
                  onClick={() => { setSelectedTypeFilter('all'); setSearchQuery(''); }}
                  className="btn-secondary text-xs px-3 py-1"
                >
                  Clear Filters
                </button>
              </div>
            ) : (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {displayedOpportunities.map((opp) => {
                  const isExecuting = executingOppId === opp.id;
                  const cat = getCategoryMeta(opp.analysis_type);
                  const isEvidenceOpen = expandedEvidenceId === opp.id;

                  // Clean title for display
                  let cleanTitle = opp.title || '';
                  if (plainEnglish) {
                    cleanTitle = cleanTitle.replace(/⋈/g, '&');
                  }

                  let actionLabel = 'Run Analysis';
                  if (opp.analysis_type === 'trend') actionLabel = 'Analyze Trend';
                  else if (opp.analysis_type === 'comparative') actionLabel = 'Compare Segments';
                  else if (opp.analysis_type === 'correlation') actionLabel = 'Test Association';
                  else if (opp.analysis_type === 'anomaly') actionLabel = 'Audit Outliers';
                  else if (opp.analysis_type === 'segmentation') actionLabel = 'Discover Segments';
                  else if (opp.analysis_type === 'cross_dataset') actionLabel = 'Explore Link';

                  return (
                    <div
                      key={opp.id}
                      className="p-4 border border-mist rounded-control bg-surface flex flex-col justify-between hover:border-teal/40 transition-all shadow-xs"
                    >
                      <div>
                        {/* Top Badges */}
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded border flex items-center gap-1 ${cat.cls}`}>
                            <span>{cat.icon}</span>
                            <span>{cat.label}</span>
                          </span>

                          <span className="text-[11px] font-semibold text-teal bg-teal-soft/40 px-2 py-0.5 rounded border border-teal/20" title="Analytical relevance score">
                            {(opp.relevance_score * 100).toFixed(0)}% Match
                          </span>
                        </div>

                        {/* Title */}
                        <h4 className="font-display text-sm font-semibold text-ink mb-1.5 leading-snug">
                          {cleanTitle}
                        </h4>

                        {/* Description */}
                        <p className="text-xs text-[var(--muted)] mb-3 leading-relaxed">
                          {opp.description}
                        </p>

                        {/* Collapsible Evidence & Safety Check */}
                        <div className="mb-3">
                          <button
                            type="button"
                            onClick={() => toggleEvidence(opp.id)}
                            className="text-[11px] text-[var(--muted)] hover:text-ink font-medium flex items-center gap-1 transition-colors"
                          >
                            <span>Why this is recommended ({opp.evidence_factors?.length || 0})</span>
                            <ChevronDown
                              size={12}
                              className={`transition-transform ${isEvidenceOpen ? 'rotate-180' : ''}`}
                            />
                          </button>

                          {isEvidenceOpen && (
                            <div className="mt-2 space-y-1.5 p-2.5 rounded bg-paper border border-mist text-xs animate-fade-in">
                              {opp.evidence_factors?.map((ef, idx) => (
                                <p key={idx} className="text-[11px] text-ink/90 flex items-start gap-1.5 leading-snug">
                                  <span className="text-teal font-bold shrink-0">✓</span>
                                  <span>{ef}</span>
                                </p>
                              ))}

                              {opp.reason_unavailable && (
                                <p className="text-[11px] text-amber-300 font-medium mt-1.5 p-1.5 rounded bg-amber-500/10 border border-amber-500/25">
                                  ⚠️ {opp.reason_unavailable}
                                </p>
                              )}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Card Footer: Safety Status + Run Button */}
                      <div className="pt-3 border-t border-mist flex items-center justify-between gap-2">
                        <div>
                          {opp.grain_safe ? (
                            <span className="text-[11px] text-emerald-400 font-medium flex items-center gap-1" title="Safe to calculate without multiplying totals">
                              <CheckCircle2 size={12} />
                              <span>{plainEnglish ? 'Safe to run' : 'Safe grain'}</span>
                            </span>
                          ) : opp.pre_aggregation_needed ? (
                            <span className="text-[11px] text-amber-400 font-medium flex items-center gap-1" title="Requires grouping to prevent duplicate counting">
                              <AlertTriangle size={12} />
                              <span>{plainEnglish ? 'Requires grouping' : 'Pre-aggregation'}</span>
                            </span>
                          ) : (
                            <span className="text-[11px] text-[var(--muted)]">
                              Pending check
                            </span>
                          )}
                        </div>

                        <button
                          type="button"
                          onClick={() => handleRunOpportunity(opp.id)}
                          disabled={isExecuting || opp.status === 'unavailable'}
                          className="btn-primary text-xs px-3 py-1.5 flex items-center gap-1 font-semibold"
                        >
                          {isExecuting && <Spinner className="h-3 w-3 text-paper" />}
                          <span>{isExecuting ? 'Running…' : actionLabel}</span>
                          {!isExecuting && <ArrowRight size={12} />}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Bottom Expand/Collapse Bar */}
            {!showAllOpps && filteredOpportunities.length > 6 && (
              <div className="text-center pt-2">
                <button
                  type="button"
                  onClick={() => setShowAllOpps(true)}
                  className="btn-secondary text-xs px-4 py-2 inline-flex items-center gap-1.5"
                >
                  <span>View All {filteredOpportunities.length} Analyses</span>
                  <ChevronDown size={13} />
                </button>
              </div>
            )}
          </section>

          {/* ACTIVE EXECUTED ANALYSIS MODAL */}
          {activeAnalysisResult && (
            <div className="fixed inset-0 z-50 bg-ink/50 backdrop-blur-xs flex items-center justify-center p-4">
              <div className="bg-surface border border-mist rounded-control w-full max-w-2xl p-6 max-h-[85vh] overflow-y-auto space-y-4 animate-page-in shadow-2xl">
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
                    className="btn-secondary text-xs px-4 py-1.5"
                  >
                    Done
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* EVIDENCE & EXPLAINABILITY MODAL */}
          {activeEvidenceInsight && (
            <div className="fixed inset-0 z-50 bg-ink/50 backdrop-blur-xs flex items-center justify-center p-4">
              <div className="bg-surface border border-mist rounded-control w-full max-w-xl p-6 space-y-4 animate-page-in shadow-2xl max-h-[85vh] overflow-y-auto">
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
                    className="btn-secondary text-xs px-4 py-1.5"
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
