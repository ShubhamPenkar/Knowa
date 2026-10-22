import { useState, useEffect } from 'react';
import Spinner from './Spinner';

export default function DataRelationshipsModel({ token, datasets = [] }) {
  const [relationships, setRelationships] = useState([]);
  const [grains, setGrains] = useState({});
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [viewMode, setViewMode] = useState('all'); // 'all', 'relationships', 'grains'
  const [safetyFilter, setSafetyFilter] = useState('all'); // 'all', 'safe', 'risks'
  const [searchQuery, setSearchQuery] = useState('');

  // Explainability Modal State
  const [explainingRel, setExplainingRel] = useState(null);

  // Join Simulation Modal State
  const [showSimulateModal, setShowSimulateModal] = useState(false);
  const [simSourceDataset, setSimSourceDataset] = useState('');
  const [simSourceColumn, setSimSourceColumn] = useState('');
  const [simTargetDataset, setSimTargetDataset] = useState('');
  const [simTargetColumn, setSimTargetColumn] = useState('');
  const [simulating, setSimulating] = useState(false);
  const [simulationResult, setSimulationResult] = useState(null);
  const [simulationError, setSimulationError] = useState('');

  // Validation State
  const [validatingId, setValidatingId] = useState(null);

  useEffect(() => {
    if (token) {
      loadData();
    }
  }, [token]);

  const loadData = async (forceRefresh = false) => {
    if (forceRefresh) setRefreshing(true);
    else setLoading(true);
    setError('');

    try {
      // 1. Fetch relationships
      const relRes = await fetch(
        forceRefresh ? '/api/datasets/relationships/discover' : '/api/datasets/relationships',
        {
          method: forceRefresh ? 'POST' : 'GET',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          ...(forceRefresh ? { body: JSON.stringify({ refresh: true }) } : {}),
        }
      );

      if (relRes.ok) {
        const relData = await relRes.json();
        setRelationships(relData.relationships || []);
      }

      // 2. Fetch grains for all datasets in parallel
      if (datasets.length > 0) {
        const grainResults = {};
        await Promise.all(
          datasets.map(async (d) => {
            try {
              const res = await fetch(`/api/datasets/${d.id}/grain`, {
                headers: { Authorization: `Bearer ${token}` },
              });
              if (res.ok) {
                grainResults[d.id] = await res.json();
              }
            } catch (err) {
              console.error(`Failed to load grain for ${d.name}`, err);
            }
          })
        );
        setGrains(grainResults);
      }
    } catch (err) {
      console.error(err);
      setError('Failed to load multi-dataset intelligence');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleValidateJoin = async (rel) => {
    if (!token || !rel) return;
    setValidatingId(rel.id);
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

      if (res.ok) {
        const updated = await res.json();
        setRelationships((prev) =>
          prev.map((r) => (r.id === updated.id ? { ...r, ...updated } : r))
        );
      }
    } catch (err) {
      console.error('Validation error', err);
    } finally {
      setValidatingId(null);
    }
  };

  const handleOpenSimulation = (rel = null) => {
    if (rel) {
      setSimSourceDataset(rel.source_dataset_id);
      setSimSourceColumn(rel.source_column);
      setSimTargetDataset(rel.target_dataset_id);
      setSimTargetColumn(rel.target_column);
    } else if (datasets.length >= 2) {
      setSimSourceDataset(datasets[0].id);
      setSimSourceColumn(datasets[0].columns?.[0]?.name || '');
      setSimTargetDataset(datasets[1].id);
      setSimTargetColumn(datasets[1].columns?.[0]?.name || '');
    }
    setSimulationResult(null);
    setSimulationError('');
    setShowSimulateModal(true);
  };

  const handleRunSimulation = async (e) => {
    e.preventDefault();
    if (!simSourceDataset || !simSourceColumn || !simTargetDataset || !simTargetColumn) return;
    setSimulating(true);
    setSimulationError('');
    setSimulationResult(null);

    try {
      const res = await fetch('/api/datasets/relationships/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          source_dataset_id: simSourceDataset,
          source_column: simSourceColumn,
          target_dataset_id: simTargetDataset,
          target_column: simTargetColumn,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Simulation execution failed');
      }

      const result = await res.json();
      setSimulationResult(result);
    } catch (err) {
      console.error(err);
      setSimulationError(err.message || 'Simulation error');
    } finally {
      setSimulating(false);
    }
  };

  const getGrainBadge = (grainType) => {
    switch (grainType) {
      case 'entity':
        return { label: '1 Row = 1 Entity (e.g. Account/Customer)', cls: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' };
      case 'transaction':
        return { label: '1 Row = 1 Event or Transaction', cls: 'bg-blue-500/20 text-blue-300 border-blue-500/30' };
      case 'bridge':
        return { label: '1 Row = 1 Bridge Association', cls: 'bg-purple-500/20 text-purple-300 border-purple-500/30' };
      case 'event':
        return { label: '1 Row = 1 Activity Log Entry', cls: 'bg-amber-500/20 text-amber-300 border-amber-500/30' };
      case 'snapshot':
        return { label: '1 Row = 1 Periodic Status Snapshot', cls: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30' };
      case 'time_series':
        return { label: '1 Row = 1 Time Period Interval', cls: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30' };
      case 'periodic_aggregate':
        return { label: '1 Row = 1 Aggregate Summary', cls: 'bg-teal-500/20 text-teal-300 border-teal-500/30' };
      case 'composite_key':
        return { label: '1 Row = Multi-Key Combination', cls: 'bg-violet-500/20 text-violet-300 border-violet-500/30' };
      default:
        return { label: 'Row Representation Unconfirmed', cls: 'bg-gray-500/20 text-gray-300 border-gray-500/30' };
    }
  };

  const getSafetyBadge = (safety, containmentRatio = 1.0) => {
    switch (safety) {
      case 'safe':
        if (containmentRatio != null && containmentRatio < 0.6) {
          return {
            label: `Partial Key Overlap (${Math.round(containmentRatio * 100)}% match)`,
            cls: 'bg-amber-500/20 text-amber-300 border-amber-500/30'
          };
        }
        return { label: 'Analytically Safe Join', cls: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' };
      case 'safe_with_aggregation':
        return { label: 'Safe with Prior Aggregation', cls: 'bg-blue-500/20 text-blue-300 border-blue-500/30' };
      case 'fanout_risk':
        return { label: '⚠️ Risk: Counts Records More Than Once', cls: 'bg-amber-500/20 text-amber-300 border-amber-500/30' };
      case 'many_to_many_risk':
        return { label: '⚠️ Severe Duplication Risk (Many-to-Many)', cls: 'bg-rose-500/20 text-rose-300 border-rose-500/30' };
      default:
        return { label: 'Incomplete Overlap Evidence', cls: 'bg-gray-500/20 text-gray-300 border-gray-500/30' };
    }
  };

  const getCardinalityLabel = (cardinality) => {
    switch (cardinality) {
      case 'one_to_one':
        return '1 : 1 (Exact)';
      case 'one_to_many':
        return '1 : Many';
      case 'many_to_one':
        return 'Many : 1';
      case 'many_to_many':
        return 'Many : Many (Cartesian Hazard)';
      default:
        return 'Unknown Connection';
    }
  };

  // Filtered relationships
  const filteredRelationships = relationships.filter((rel) => {
    if (safetyFilter === 'safe' && rel.join_safety !== 'safe' && rel.join_safety !== 'safe_with_aggregation') {
      return false;
    }
    if (safetyFilter === 'risks' && rel.join_safety !== 'fanout_risk' && rel.join_safety !== 'many_to_many_risk') {
      return false;
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchSrc = rel.source_dataset_name.toLowerCase().includes(q) || rel.source_column.toLowerCase().includes(q);
      const matchTgt = rel.target_dataset_name.toLowerCase().includes(q) || rel.target_column.toLowerCase().includes(q);
      return matchSrc || matchTgt;
    }
    return true;
  });

  const totalRisksCount = relationships.filter(
    (r) => r.join_safety === 'fanout_risk' || r.join_safety === 'many_to_many_risk'
  ).length;

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center p-16 space-y-3">
        <Spinner />
        <p className="text-sm text-[var(--muted)]">Analyzing dataset grains and discovering relationships…</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-page-in">
      {/* Top Banner & Metric Strip */}
      <div className="surface p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase font-semibold tracking-wider text-teal">Data Connections</span>
              <span className="px-2 py-0.5 text-[11px] rounded bg-teal/15 text-teal border border-teal/30 font-medium">
                Business Join Reality
              </span>
            </div>
            <h2 className="text-2xl font-display font-semibold mt-1">How Your Data Connects</h2>
            <p className="text-sm text-[var(--muted)] mt-1 max-w-2xl">
              Understand what one row represents in each table, how records connect, and whether combining tables will exclude records or count some values more than once.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => handleOpenSimulation()}
              className="btn-secondary text-xs px-3 py-2 flex items-center gap-1.5"
            >
              <span>🔬</span> Simulate Join
            </button>
            <button
              type="button"
              onClick={() => loadData(true)}
              disabled={refreshing}
              className="btn-primary text-xs px-3.5 py-2 flex items-center gap-1.5"
            >
              {refreshing ? <Spinner /> : <span>↻</span>}
              <span>Re-discover Relationships</span>
            </button>
          </div>
        </div>

        {/* Stats Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6 pt-6 border-t border-mist">
          <div className="p-3 bg-paper/60 border border-mist rounded-control">
            <p className="text-xs text-[var(--muted)]">Registered Datasets</p>
            <p className="text-2xl font-semibold font-display mt-0.5 text-ink">{datasets.length}</p>
          </div>
          <div className="p-3 bg-paper/60 border border-mist rounded-control">
            <p className="text-xs text-[var(--muted)]">Discovered Joins</p>
            <p className="text-2xl font-semibold font-display mt-0.5 text-teal">{relationships.length}</p>
          </div>
          <div className="p-3 bg-paper/60 border border-mist rounded-control">
            <p className="text-xs text-[var(--muted)]">Validated Joins</p>
            <p className="text-2xl font-semibold font-display mt-0.5 text-emerald-400">
              {relationships.filter((r) => r.validation_status === 'validated').length}
            </p>
          </div>
          <div className="p-3 bg-paper/60 border border-mist rounded-control">
            <p className="text-xs text-[var(--muted)]">Analytical Join Risks</p>
            <p className={`text-2xl font-semibold font-display mt-0.5 ${totalRisksCount > 0 ? 'text-amber-400' : 'text-muted'}`}>
              {totalRisksCount}
            </p>
          </div>
        </div>
      </div>

      {/* Filter / View Mode Switcher */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-1.5 bg-surface border border-mist p-1 rounded-control text-xs">
          <button
            type="button"
            onClick={() => setViewMode('all')}
            className={`px-3 py-1.5 rounded transition-colors ${
              viewMode === 'all' ? 'bg-teal text-paper font-semibold' : 'text-[var(--muted)] hover:text-ink'
            }`}
          >
            All Intelligence
          </button>
          <button
            type="button"
            onClick={() => setViewMode('relationships')}
            className={`px-3 py-1.5 rounded transition-colors ${
              viewMode === 'relationships' ? 'bg-teal text-paper font-semibold' : 'text-[var(--muted)] hover:text-ink'
            }`}
          >
            Relationships ({relationships.length})
          </button>
          <button
            type="button"
            onClick={() => setViewMode('grains')}
            className={`px-3 py-1.5 rounded transition-colors ${
              viewMode === 'grains' ? 'bg-teal text-paper font-semibold' : 'text-[var(--muted)] hover:text-ink'
            }`}
          >
            Dataset Grains ({datasets.length})
          </button>
        </div>

        {viewMode !== 'grains' && (
          <div className="flex items-center gap-2 w-full sm:w-auto">
            <div className="flex items-center gap-1 text-xs">
              <button
                type="button"
                onClick={() => setSafetyFilter('all')}
                className={`px-2.5 py-1 rounded border ${
                  safetyFilter === 'all'
                    ? 'border-teal bg-teal/10 text-teal font-medium'
                    : 'border-mist text-[var(--muted)] hover:text-ink'
                }`}
              >
                All
              </button>
              <button
                type="button"
                onClick={() => setSafetyFilter('safe')}
                className={`px-2.5 py-1 rounded border ${
                  safetyFilter === 'safe'
                    ? 'border-emerald-500 bg-emerald-500/10 text-emerald-300 font-medium'
                    : 'border-mist text-[var(--muted)] hover:text-ink'
                }`}
              >
                Safe Joins
              </button>
              <button
                type="button"
                onClick={() => setSafetyFilter('risks')}
                className={`px-2.5 py-1 rounded border ${
                  safetyFilter === 'risks'
                    ? 'border-amber-500 bg-amber-500/10 text-amber-300 font-medium'
                    : 'border-mist text-[var(--muted)] hover:text-ink'
                }`}
              >
                Warnings ({totalRisksCount})
              </button>
            </div>
            <input
              type="text"
              placeholder="Filter columns or tables…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input text-xs py-1 px-3 w-44"
            />
          </div>
        )}
      </div>

      {error && (
        <div className="p-4 bg-coral-soft border border-coral/40 text-coral rounded-control text-sm">
          {error}
        </div>
      )}

      {/* SECTION 1: WHAT ONE ROW REPRESENTS (GRAIN) */}
      {(viewMode === 'all' || viewMode === 'grains') && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-display text-lg font-semibold">What One Row Represents (Dataset Grain)</h3>
              <p className="text-xs text-[var(--muted)]">
                Explicit definition of what each record represents in the real world. Prevents inadvertent record duplication when merging tables.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map((dataset) => {
              const grain = grains[dataset.id];
              const badge = grain ? getGrainBadge(grain.grain_type) : getGrainBadge('unknown');

              return (
                <div key={dataset.id} className="surface p-5 flex flex-col justify-between space-y-4">
                  <div>
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <h4 className="font-display font-semibold text-ink text-base">{dataset.name}</h4>
                        <p className="text-xs text-[var(--muted)] mt-0.5">
                          {dataset.row_count?.toLocaleString()} rows · {dataset.column_count} columns
                        </p>
                      </div>
                      <span className={`px-2 py-0.5 text-xs font-medium rounded border ${badge.cls}`}>
                        {badge.label}
                      </span>
                    </div>

                    <div className="mt-3 p-3 bg-paper/60 border border-mist rounded-control">
                      <p className="text-xs font-semibold text-ink">
                        {grain?.grain_description || `Level of detail: ${dataset.name}`}
                      </p>
                      <p className="text-[11px] text-[var(--muted)] mt-1">
                        Inference Confidence: {grain ? Math.round(grain.confidence * 100) : 0}% ({grain?.confidence_type === 'heuristic' ? 'inferred from distinct keys' : 'verified from schema'})
                        {grain && grain.confidence < 0.6 ? ' · Tentative (confirm key column)' : ' · Confirmed unique grain'}
                      </p>
                    </div>

                    <div className="mt-3 space-y-2 text-xs">
                      {grain?.primary_key && grain.primary_key.length > 0 && (
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span className="text-[var(--muted)]">Key:</span>
                          {grain.primary_key.map((pk) => (
                            <span key={pk} className="px-1.5 py-0.5 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30 font-mono text-[11px]">
                              {pk}
                            </span>
                          ))}
                        </div>
                      )}

                      {grain?.entity_identifiers && grain.entity_identifiers.length > 0 && (
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span className="text-[var(--muted)]">Entity FKs:</span>
                          {grain.entity_identifiers.map((fk) => (
                            <span key={fk} className="px-1.5 py-0.5 rounded bg-teal-500/15 text-teal-300 border border-teal-500/30 font-mono text-[11px]">
                              {fk}
                            </span>
                          ))}
                        </div>
                      )}

                      {grain?.potential_measures && grain.potential_measures.length > 0 && (
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span className="text-[var(--muted)]">Vulnerable Measures:</span>
                          {grain.potential_measures.slice(0, 3).map((m) => (
                            <span key={m} className="px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-300 border border-blue-500/30 font-mono text-[11px]">
                              {m}
                            </span>
                          ))}
                          {grain.potential_measures.length > 3 && (
                            <span className="text-[10px] text-[var(--muted)]">+{grain.potential_measures.length - 3}</span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {grain?.evidence && grain.evidence.length > 0 && (
                    <details className="pt-3 border-t border-mist text-xs">
                      <summary className="cursor-pointer text-[var(--muted)] hover:text-ink font-medium">
                        Evidence ({grain.evidence.length})
                      </summary>
                      <ul className="mt-2 space-y-1 list-disc list-inside text-[var(--muted)] text-[11px]">
                        {grain.evidence.map((ev, i) => (
                          <li key={i}>{ev}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* SECTION 2: RELATIONSHIPS & JOIN SAFETY */}
      {(viewMode === 'all' || viewMode === 'relationships') && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-display text-lg font-semibold">How Your Data Connects (Cross-Dataset Joins)</h3>
              <p className="text-xs text-[var(--muted)]">
                Discovered key relationships showing match rates, cardinality, and whether combining tables will exclude records or multiply totals.
              </p>
            </div>
          </div>

          {filteredRelationships.length === 0 ? (
            <div className="surface p-8 text-center space-y-3">
              <p className="text-base text-ink font-medium">No matching relationships found</p>
              <p className="text-xs text-[var(--muted)] max-w-md mx-auto">
                Upload related business datasets (e.g. customers, orders, products) to allow Knowa to infer relationships and evaluate join safety.
              </p>
              <button
                type="button"
                onClick={() => handleOpenSimulation()}
                className="btn-primary text-xs"
              >
                Simulate Join Between Two Columns
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredRelationships.map((rel) => {
                const safety = getSafetyBadge(rel.join_safety, rel.containment_ratio);
                const cardLabel = getCardinalityLabel(rel.cardinality);
                const isValidating = validatingId === rel.id;

                return (
                  <div key={rel.id} className="surface p-6 space-y-4 hover:border-mist/80 transition-colors">
                    {/* Visual Relationship Node Diagram */}
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-mist">
                      <div className="flex flex-wrap items-center gap-3">
                        {/* Source Box */}
                        <div className="bg-paper border border-mist px-3.5 py-2 rounded-control">
                          <p className="text-[11px] text-[var(--muted)] font-mono">{rel.source_dataset_name}</p>
                          <p className="text-sm font-semibold text-ink font-mono mt-0.5">{rel.source_column}</p>
                        </div>

                        {/* Cardinality Connector */}
                        <div className="flex flex-col items-center px-2">
                          <div className="flex items-center gap-1.5">
                            <span className="w-4 h-[1px] bg-teal/60"></span>
                            <span className="px-2 py-0.5 rounded bg-teal/15 text-teal font-mono text-xs font-bold border border-teal/30">
                              {cardLabel}
                            </span>
                            <span className="w-4 h-[1px] bg-teal/60"></span>
                          </div>
                          <span className="text-[10px] text-[var(--muted)] font-mono mt-0.5">
                            {Math.round(rel.containment_ratio * 100)}% match
                          </span>
                        </div>

                        {/* Target Box */}
                        <div className="bg-paper border border-mist px-3.5 py-2 rounded-control">
                          <p className="text-[11px] text-[var(--muted)] font-mono">{rel.target_dataset_name}</p>
                          <p className="text-sm font-semibold text-ink font-mono mt-0.5">{rel.target_column}</p>
                        </div>
                      </div>

                      {/* Right-side status and action buttons */}
                      <div className="flex flex-wrap items-center gap-2">
                        <span className={`px-2.5 py-1 text-xs font-medium rounded border ${safety.cls}`}>
                          {safety.label}
                        </span>
                        <button
                          type="button"
                          onClick={() => setExplainingRel(rel)}
                          className="btn-secondary text-xs px-2.5 py-1"
                        >
                          Why Related?
                        </button>
                        <button
                          type="button"
                          onClick={() => handleOpenSimulation(rel)}
                          className="btn-secondary text-xs px-2.5 py-1"
                        >
                          Simulate Join
                        </button>
                        {rel.validation_status !== 'validated' && (
                          <button
                            type="button"
                            onClick={() => handleValidateJoin(rel)}
                            disabled={isValidating}
                            className="btn-primary text-xs px-2.5 py-1"
                          >
                            {isValidating ? <Spinner /> : 'Validate Join'}
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Join Safety Risk & Warnings Banner */}
                    {(rel.join_safety === 'fanout_risk' || rel.double_counting_risk) && (
                      <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-control text-xs text-amber-200 flex items-start gap-2">
                        <span className="text-base leading-none">⚠️</span>
                        <div>
                          <p className="font-semibold text-amber-300">Potential Double-Counting Warning</p>
                          <p className="mt-0.5">
                            Joining <strong>{rel.source_dataset_name}</strong> → <strong>{rel.target_dataset_name}</strong> creates a {cardLabel} expansion (estimated {rel.fan_out_ratio}x).
                            {rel.double_counting_measures?.length > 0 && (
                              <span> Source measures ({rel.double_counting_measures.join(', ')}) should not be summed after this join without pre-aggregation or grain normalization.</span>
                            )}
                          </p>
                        </div>
                      </div>
                    )}

                    {rel.join_safety === 'many_to_many_risk' && (
                      <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-control text-xs text-rose-200 flex items-start gap-2">
                        <span className="text-base leading-none">⚠️</span>
                        <div>
                          <p className="font-semibold text-rose-300">Cartesian Explosion Risk (Many-to-Many)</p>
                          <p className="mt-0.5">
                            Both datasets contain duplicate keys. Direct join risks geometric row expansion. Use a bridge table or aggregate both sides before combining.
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Data Quality Hygiene Alerts (whitespace, casing, nulls) */}
                    {rel.quality_notes && rel.quality_notes.length > 0 && (
                      <div className="p-3 bg-paper/80 border border-mist rounded-control text-xs space-y-1">
                        <p className="font-semibold text-ink flex items-center gap-1.5">
                          <span>🧹</span> Data Cleanliness & Hygiene Observations:
                        </p>
                        <ul className="list-disc list-inside text-[var(--muted)] space-y-0.5">
                          {rel.quality_notes.map((note, i) => (
                            <li key={i}>{note}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Key Metrics Footnote */}
                    <div className="flex flex-wrap items-center gap-4 text-xs text-[var(--muted)] pt-1">
                      <span>
                        Confidence: <strong className="text-ink">{Math.round(rel.confidence * 100)}%</strong> ({rel.confidence_type})
                      </span>
                      <span>
                        Jaccard Index: <strong className="text-ink">{rel.jaccard_similarity}</strong>
                      </span>
                      <span>
                        Overlapping Keys: <strong className="text-ink">{rel.overlapping_values_count?.toLocaleString()}</strong>
                      </span>
                      <span>
                        Unmatched Source: <strong className="text-ink">{Math.round((rel.source_unmatched_rate || 0) * 100)}%</strong>
                      </span>
                      <span>
                        Unmatched Target: <strong className="text-ink">{Math.round((rel.target_unmatched_rate || 0) * 100)}%</strong>
                      </span>
                      {rel.validation_status === 'validated' && (
                        <span className="text-emerald-400 font-medium">✓ Validated against actual data</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* EXPLAINABILITY MODAL */}
      {explainingRel && (
        <div className="fixed inset-0 z-50 bg-ink/40 flex items-center justify-center p-4">
          <div className="bg-surface border border-mist rounded-control w-full max-w-2xl p-6 max-h-[85vh] overflow-y-auto animate-page-in space-y-5">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-xs font-semibold uppercase text-teal tracking-wider">Explainability Engine</span>
                <h3 className="text-xl font-display font-semibold text-ink mt-0.5">
                  Why Knowa Inferred This Relationship
                </h3>
                <p className="text-xs font-mono text-[var(--muted)] mt-1">
                  {explainingRel.source_dataset_name}.{explainingRel.source_column} ⟷ {explainingRel.target_dataset_name}.{explainingRel.target_column}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setExplainingRel(null)}
                className="text-[var(--muted)] hover:text-ink text-lg"
              >
                ✕
              </button>
            </div>

            {/* Section 1: Validity & Discovery Evidence */}
            <div className="surface p-4 space-y-2">
              <h4 className="text-xs uppercase font-semibold text-teal tracking-wider">1. Cross-Dataset Evidence</h4>
              <ul className="space-y-1.5 text-xs text-ink list-disc list-inside">
                {explainingRel.evidence?.map((ev, i) => (
                  <li key={i}>{ev}</li>
                ))}
              </ul>
            </div>

            {/* Section 2: Cardinality Explanation */}
            <div className="surface p-4 space-y-2">
              <h4 className="text-xs uppercase font-semibold text-teal tracking-wider">
                2. Why is this relationship {getCardinalityLabel(explainingRel.cardinality)}?
              </h4>
              <p className="text-xs text-[var(--muted)]">
                Cardinality is determined mathematically from uniqueness distributions across non-null keys:
              </p>
              <div className="grid grid-cols-2 gap-3 text-xs pt-2">
                <div className="p-2.5 bg-paper rounded border border-mist">
                  <p className="text-[var(--muted)]">{explainingRel.source_dataset_name}.{explainingRel.source_column}</p>
                  <p className="text-sm font-semibold text-ink mt-0.5">
                    {Math.round(explainingRel.source_uniqueness_ratio * 100)}% Unique
                  </p>
                  <p className="text-[10px] text-[var(--muted)]">
                    {explainingRel.source_uniqueness_ratio >= 0.95 ? 'Unique row identifier' : 'Values repeat across rows'}
                  </p>
                </div>
                <div className="p-2.5 bg-paper rounded border border-mist">
                  <p className="text-[var(--muted)]">{explainingRel.target_dataset_name}.{explainingRel.target_column}</p>
                  <p className="text-sm font-semibold text-ink mt-0.5">
                    {Math.round(explainingRel.target_uniqueness_ratio * 100)}% Unique
                  </p>
                  <p className="text-[10px] text-[var(--muted)]">
                    {explainingRel.target_uniqueness_ratio >= 0.95 ? 'Unique row identifier' : 'Values repeat across rows'}
                  </p>
                </div>
              </div>
            </div>

            {/* Section 3: Analytical Safety vs Relationship Validity */}
            <div className="surface p-4 space-y-2">
              <h4 className="text-xs uppercase font-semibold text-teal tracking-wider">
                3. Analytical Safety Distinction
              </h4>
              <p className="text-xs text-ink leading-relaxed">
                Knowa explicitly distinguishes <strong>Relationship Validity</strong> from <strong>Analytical Safety</strong>.
                While this join is structurally valid (keys align to compatible entities), performing downstream operations
                like summing source metrics will result in duplication.
              </p>
              <div className="p-3 bg-paper rounded border border-mist text-xs space-y-1">
                <p className="font-semibold text-ink">Join Safety Status: {explainingRel.join_safety}</p>
                <p className="text-[var(--muted)]">
                  Estimated Fan-Out Multiplier: <strong>{explainingRel.fan_out_ratio}x</strong>
                </p>
                {explainingRel.double_counting_measures?.length > 0 && (
                  <p className="text-amber-300">
                    Vulnerable metrics: {explainingRel.double_counting_measures.join(', ')}
                  </p>
                )}
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                type="button"
                onClick={() => setExplainingRel(null)}
                className="btn-secondary text-xs px-4 py-2"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* JOIN SIMULATION MODAL */}
      {showSimulateModal && (
        <div className="fixed inset-0 z-50 bg-ink/40 flex items-center justify-center p-4">
          <div className="bg-surface border border-mist rounded-control w-full max-w-2xl p-6 max-h-[85vh] overflow-y-auto animate-page-in space-y-5">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-xs font-semibold uppercase text-teal tracking-wider">Join Simulation Lab</span>
                <h3 className="text-xl font-display font-semibold text-ink mt-0.5">
                  Non-Destructive Join Simulation
                </h3>
                <p className="text-xs text-[var(--muted)] mt-1">
                  Test and inspect join row expansion, unmatched keys, and Cartesian multiplication without altering original datasets.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowSimulateModal(false)}
                className="text-[var(--muted)] hover:text-ink text-lg"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleRunSimulation} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Source Selection */}
                <div className="p-3 bg-paper rounded border border-mist space-y-3">
                  <p className="text-xs font-semibold text-ink uppercase tracking-wider">Source Dataset</p>
                  <div>
                    <label className="block text-[11px] text-[var(--muted)] mb-1">Dataset</label>
                    <select
                      className="input text-xs w-full py-1.5"
                      value={simSourceDataset}
                      onChange={(e) => {
                        setSimSourceDataset(e.target.value);
                        const ds = datasets.find((d) => d.id === e.target.value);
                        setSimSourceColumn(ds?.columns?.[0]?.name || '');
                      }}
                    >
                      {datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name} ({d.row_count} rows)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-[11px] text-[var(--muted)] mb-1">Join Column</label>
                    <select
                      className="input text-xs w-full py-1.5"
                      value={simSourceColumn}
                      onChange={(e) => setSimSourceColumn(e.target.value)}
                    >
                      {datasets
                        .find((d) => d.id === simSourceDataset)
                        ?.columns?.map((c) => (
                          <option key={c.name} value={c.name}>
                            {c.name} ({c.semantic_type || c.role || 'col'})
                          </option>
                        ))}
                    </select>
                  </div>
                </div>

                {/* Target Selection */}
                <div className="p-3 bg-paper rounded border border-mist space-y-3">
                  <p className="text-xs font-semibold text-ink uppercase tracking-wider">Target Dataset</p>
                  <div>
                    <label className="block text-[11px] text-[var(--muted)] mb-1">Dataset</label>
                    <select
                      className="input text-xs w-full py-1.5"
                      value={simTargetDataset}
                      onChange={(e) => {
                        setSimTargetDataset(e.target.value);
                        const ds = datasets.find((d) => d.id === e.target.value);
                        setSimTargetColumn(ds?.columns?.[0]?.name || '');
                      }}
                    >
                      {datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name} ({d.row_count} rows)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-[11px] text-[var(--muted)] mb-1">Join Column</label>
                    <select
                      className="input text-xs w-full py-1.5"
                      value={simTargetColumn}
                      onChange={(e) => setSimTargetColumn(e.target.value)}
                    >
                      {datasets
                        .find((d) => d.id === simTargetDataset)
                        ?.columns?.map((c) => (
                          <option key={c.name} value={c.name}>
                            {c.name} ({c.semantic_type || c.role || 'col'})
                          </option>
                        ))}
                    </select>
                  </div>
                </div>
              </div>

              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowSimulateModal(false)}
                  className="btn-secondary text-xs px-3 py-2"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={simulating}
                  className="btn-primary text-xs px-4 py-2 flex items-center gap-1.5"
                >
                  {simulating ? <Spinner /> : <span>Run Simulation</span>}
                </button>
              </div>
            </form>

            {simulationError && (
              <div className="p-3 bg-coral-soft border border-coral/40 text-coral rounded-control text-xs">
                {simulationError}
              </div>
            )}

            {/* Simulation Results Card */}
            {simulationResult && (
              <div className="surface p-4 space-y-4 border-teal/40 animate-page-in">
                <div className="flex items-center justify-between">
                  <h4 className="font-display font-semibold text-ink text-sm">Simulation Results</h4>
                  <span className={`px-2.5 py-0.5 text-xs font-medium rounded border ${getSafetyBadge(simulationResult.join_safety).cls}`}>
                    {getSafetyBadge(simulationResult.join_safety).label}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-2.5 bg-paper rounded border border-mist">
                    <p className="text-[var(--muted)]">Source Rows</p>
                    <p className="text-base font-semibold text-ink mt-0.5">
                      {simulationResult.rows_before?.toLocaleString()}
                    </p>
                  </div>
                  <div className="p-2.5 bg-paper rounded border border-mist">
                    <p className="text-[var(--muted)]">Joined Rows</p>
                    <p className="text-base font-semibold text-teal mt-0.5">
                      {simulationResult.rows_after?.toLocaleString()}
                    </p>
                  </div>
                  <div className="p-2.5 bg-paper rounded border border-mist">
                    <p className="text-[var(--muted)]">Expansion Factor</p>
                    <p className={`text-base font-semibold mt-0.5 ${simulationResult.row_expansion_factor > 1.05 ? 'text-amber-400' : 'text-emerald-400'}`}>
                      {simulationResult.row_expansion_factor}x
                    </p>
                  </div>
                  <div className="p-2.5 bg-paper rounded border border-mist">
                    <p className="text-[var(--muted)]">Unmatched Source</p>
                    <p className="text-base font-semibold text-ink mt-0.5">
                      {simulationResult.unmatched_source_rows?.toLocaleString()}
                    </p>
                  </div>
                </div>

                {simulationResult.warnings?.length > 0 && (
                  <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded text-xs text-amber-200 space-y-1">
                    <p className="font-semibold text-amber-300">Analytical Warnings:</p>
                    <ul className="list-disc list-inside space-y-0.5">
                      {simulationResult.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {simulationResult.evidence?.length > 0 && (
                  <div className="p-3 bg-paper rounded border border-mist text-xs space-y-1">
                    <p className="font-semibold text-ink">Simulation Evidence:</p>
                    <ul className="list-disc list-inside text-[var(--muted)] space-y-0.5">
                      {simulationResult.evidence.map((ev, i) => (
                        <li key={i}>{ev}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
