import { useState, useEffect } from 'react';
import Spinner from './Spinner';
import { useViewMode } from '../../context/ViewModeContext';
import { 
  CheckCircle2, 
  AlertTriangle, 
  ArrowRight, 
  HelpCircle, 
  Sparkles, 
  ShieldCheck, 
  Layers, 
  RefreshCw, 
  FlaskConical, 
  Search,
  ChevronDown,
  Info,
  Check,
  X
} from 'lucide-react';

export default function DataRelationshipsModel({ token, datasets = [] }) {
  const [relationships, setRelationships] = useState([]);
  const [grains, setGrains] = useState({});
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  
  // Global View Mode
  const { plainEnglish, setPlainEnglish } = useViewMode();
  const [showHelpBanner, setShowHelpBanner] = useState(true);
  const [viewMode, setViewMode] = useState('all'); // 'all', 'relationships', 'grains'
  const [safetyFilter, setSafetyFilter] = useState('all'); // 'all', 'safe', 'risks'
  const [searchQuery, setSearchQuery] = useState('');
  
  // Custom grain confirmations in local state
  const [userConfirmedGrains, setUserConfirmedGrains] = useState({});

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
      setError('Failed to load table intelligence');
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

  const handleConfirmMeaning = (datasetId, meaningType, customLabel) => {
    setUserConfirmedGrains((prev) => ({
      ...prev,
      [datasetId]: {
        type: meaningType,
        label: customLabel,
        confirmedAt: new Date().toLocaleTimeString(),
      },
    }));
  };

  // Helper to choose a friendly emoji icon for each table
  const getDatasetIcon = (name = '') => {
    const n = name.toLowerCase();
    if (n.includes('card') || n.includes('pay') || n.includes('fraud') || n.includes('trans')) return '💳';
    if (n.includes('user') || n.includes('cust') || n.includes('churn') || n.includes('account') || n.includes('client')) return '👤';
    if (n.includes('order') || n.includes('invoice') || n.includes('bill') || n.includes('receipt')) return '🧾';
    if (n.includes('prod') || n.includes('item') || n.includes('sku') || n.includes('catalog')) return '📦';
    if (n.includes('event') || n.includes('log') || n.includes('session') || n.includes('click') || n.includes('activity')) return '⚡';
    if (n.includes('metric') || n.includes('stat') || n.includes('summary') || n.includes('daily') || n.includes('monthly')) return '📊';
    return '📁';
  };

  // Plain-English translations of grain types
  const getGrainInfo = (grainType, datasetName = '') => {
    const icon = getDatasetIcon(datasetName);
    
    switch (grainType) {
      case 'entity':
        return {
          plainBadge: '1 Row = 1 Customer / Account',
          plainDesc: 'Each record represents an individual customer, company, or user profile.',
          techBadge: 'Entity Grain (1:1 PK)',
          techDesc: 'Uniquely keyed entity dimension table',
          cls: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
          icon: icon || '👤',
          isReady: true,
        };
      case 'transaction':
        return {
          plainBadge: '1 Row = 1 Transaction / Purchase',
          plainDesc: 'Each record is a single financial transaction, order, or purchase event.',
          techBadge: 'Transaction Grain',
          techDesc: 'Discrete event record with timestamp and measures',
          cls: 'bg-blue-500/15 text-blue-300 border-blue-500/30',
          icon: icon || '💳',
          isReady: true,
        };
      case 'event':
        return {
          plainBadge: '1 Row = 1 Activity Event',
          plainDesc: 'Each record tracks a discrete action, click, or log entry.',
          techBadge: 'Event Grain',
          techDesc: 'High-frequency activity log',
          cls: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30',
          icon: icon || '⚡',
          isReady: true,
        };
      case 'periodic_aggregate':
      case 'snapshot':
        return {
          plainBadge: '1 Row = 1 Periodic Summary',
          desc: 'Each record summarizes key numbers across a specific date or time window.',
          techBadge: 'Aggregate Snapshot',
          techDesc: 'Pre-aggregated rollup table',
          cls: 'bg-purple-500/15 text-purple-300 border-purple-500/30',
          icon: icon || '📊',
          isReady: true,
        };
      case 'time_series':
        return {
          plainBadge: '1 Row = 1 Time Period',
          plainDesc: 'Sequential measurements tracked over recurring intervals.',
          techBadge: 'Time-Series Grain',
          techDesc: 'Ordered temporal series',
          cls: 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30',
          icon: icon || '📈',
          isReady: true,
        };
      case 'composite_key':
        return {
          plainBadge: '1 Row = Combined Multiple Columns',
          plainDesc: 'Records are identified by combining 2 or more columns together.',
          techBadge: 'Composite Key Grain',
          techDesc: 'Compound primary key',
          cls: 'bg-violet-500/15 text-violet-300 border-violet-500/30',
          icon: icon || '🗂️',
          isReady: true,
        };
      default:
        return {
          plainBadge: 'Needs Quick Check',
          plainDesc: 'Knowa scanned your rows. Confirm below whether each record represents a customer, a transaction, or an event.',
          techBadge: 'Unconfirmed Grain',
          techDesc: 'No unique identifier verified heuristically',
          cls: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
          icon: icon || '📁',
          isReady: false,
        };
    }
  };

  // Plain-English translations of join safety
  const getSafetyBadge = (safety, containmentRatio = 1.0) => {
    if (plainEnglish) {
      switch (safety) {
        case 'safe':
          if (containmentRatio != null && containmentRatio < 0.6) {
            return {
              label: `Partial Link (${Math.round(containmentRatio * 100)}% match)`,
              desc: 'Some records in Table A do not exist in Table B, but linking will not duplicate values.',
              cls: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
              icon: AlertTriangle,
              isWarning: true,
            };
          }
          return {
            label: 'Safe to Link (No Duplication)',
            desc: 'Safe to connect. Values and metric totals will not be multiplied.',
            cls: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
            icon: ShieldCheck,
            isWarning: false,
          };
        case 'safe_with_aggregation':
          return {
            label: 'Safe (Group Before Summing)',
            desc: 'Safe to connect, but summarize child records before calculating grand totals.',
            cls: 'bg-blue-500/15 text-blue-300 border-blue-500/30',
            icon: Info,
            isWarning: false,
          };
        case 'fanout_risk':
          return {
            label: '⚠️ Warning: Could Double-Count Numbers',
            desc: 'One record in the first table matches multiple records in the second. Summing totals directly may multiply numbers.',
            cls: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
            icon: AlertTriangle,
            isWarning: true,
          };
        case 'many_to_many_risk':
          return {
            label: '🚨 High Duplication Risk (Many-to-Many)',
            desc: 'Both tables contain repeating keys. Linking them directly produces duplicate rows on both sides.',
            cls: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
            icon: AlertTriangle,
            isWarning: true,
          };
        default:
          return {
            label: 'Pending Key Verification',
            desc: 'Awaiting further key match verification.',
            cls: 'bg-gray-500/15 text-gray-300 border-gray-500/30',
            icon: HelpCircle,
            isWarning: false,
          };
      }
    }

    // Technical Mode
    switch (safety) {
      case 'safe':
        return { label: 'Analytically Safe Join', cls: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' };
      case 'safe_with_aggregation':
        return { label: 'Safe with Prior Aggregation', cls: 'bg-blue-500/20 text-blue-300 border-blue-500/30' };
      case 'fanout_risk':
        return { label: '⚠️ Risk: Fan-Out Multiplier', cls: 'bg-amber-500/20 text-amber-300 border-amber-500/30' };
      case 'many_to_many_risk':
        return { label: '⚠️ Cartesian Hazard (N:M)', cls: 'bg-rose-500/20 text-rose-300 border-rose-500/30' };
      default:
        return { label: 'Incomplete Overlap Evidence', cls: 'bg-gray-500/20 text-gray-300 border-gray-500/30' };
    }
  };

  const getCardinalityLabel = (cardinality) => {
    if (plainEnglish) {
      switch (cardinality) {
        case 'one_to_one':
          return '1-to-1 Match';
        case 'one_to_many':
          return '1 Record matches Many';
        case 'many_to_one':
          return 'Many Records match 1';
        case 'many_to_many':
          return 'Many match Many (Repeating)';
        default:
          return 'Connected Records';
      }
    }

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

  const validatedCount = relationships.filter((r) => r.validation_status === 'validated').length;

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center p-16 space-y-3">
        <Spinner />
        <p className="text-sm text-[var(--muted)]">Analyzing tables and discovering safe connections…</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-page-in">
      {/* Top Banner & Metric Strip */}
      <div className="surface p-6 rounded-control border border-mist relative overflow-hidden">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="max-w-2xl">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs uppercase font-semibold tracking-wider text-teal">Table Relationships</span>
              <span className="px-2 py-0.5 text-[11px] rounded bg-teal/15 text-teal border border-teal/30 font-medium">
                Data Safety Shield
              </span>
            </div>
            
            <h2 className="text-2xl font-display font-semibold mt-1 text-ink">
              {plainEnglish ? 'How Your Tables Connect & Link' : 'Cross-Dataset Relational Discovery'}
            </h2>
            
            <p className="text-sm text-[var(--muted)] mt-1.5 leading-relaxed">
              {plainEnglish
                ? 'Knowa automatically analyzes your uploaded files to find matching records and protects your business metrics from being double-counted.'
                : 'Deterministic relational graph discovery, grain verification, and fan-out risk evaluation across organization datasets.'}
            </p>
          </div>

          {/* Right Action & Mode Controls */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            {/* Plain English Mode Toggle */}
            <div className="flex items-center bg-paper/80 border border-mist p-1 rounded-control text-xs">
              <button
                type="button"
                onClick={() => setPlainEnglish(true)}
                className={`px-3 py-1.5 rounded transition-all flex items-center gap-1.5 ${
                  plainEnglish
                    ? 'bg-teal text-paper font-semibold shadow-sm'
                    : 'text-[var(--muted)] hover:text-ink'
                }`}
                title="Plain English mode uses everyday business language"
              >
                <Sparkles size={13} />
                <span>Plain English</span>
              </button>
              <button
                type="button"
                onClick={() => setPlainEnglish(false)}
                className={`px-3 py-1.5 rounded transition-all ${
                  !plainEnglish
                    ? 'bg-teal text-paper font-semibold shadow-sm'
                    : 'text-[var(--muted)] hover:text-ink'
                }`}
                title="Technical mode displays raw database keys, containment ratios, and SQL mechanics"
              >
                <span>Technical</span>
              </button>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => handleOpenSimulation()}
                className="btn-secondary text-xs px-3.5 py-2 flex items-center gap-1.5"
                title="Test linking any two columns together before querying"
              >
                <FlaskConical size={14} className="text-teal" />
                <span>Test a Link</span>
              </button>

              <button
                type="button"
                onClick={() => loadData(true)}
                disabled={refreshing}
                className="btn-primary text-xs px-3.5 py-2 flex items-center gap-1.5"
              >
                {refreshing ? <Spinner /> : <RefreshCw size={13} />}
                <span>Re-check Links</span>
              </button>
            </div>
          </div>
        </div>

        {/* Educational Helper Banner */}
        {showHelpBanner && plainEnglish && (
          <div className="mt-5 p-3.5 bg-teal/10 border border-teal/25 rounded-control text-xs text-ink/90 flex items-start justify-between gap-3 animate-fade-in">
            <div className="flex items-start gap-2.5">
              <span className="text-base leading-none">💡</span>
              <p className="leading-relaxed">
                <strong className="text-teal font-medium">Why table linking matters:</strong> When you connect two tables (e.g. Customers and Purchases), Knowa verifies that each customer matches their own transactions without multiplying amounts or creating duplicate records. This keeps your dashboards and AI truthful.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowHelpBanner(false)}
              className="text-[var(--muted)] hover:text-ink text-sm p-0.5"
              title="Dismiss tip"
            >
              ✕
            </button>
          </div>
        )}

        {/* Stats Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6 pt-6 border-t border-mist">
          {/* Card 1: Datasets */}
          <div className="p-3.5 bg-paper/60 border border-mist rounded-control">
            <div className="flex items-center justify-between text-[var(--muted)] text-xs">
              <span>{plainEnglish ? 'Uploaded Tables' : 'Registered Datasets'}</span>
              <Layers size={14} className="text-teal" />
            </div>
            <p className="text-2xl font-semibold font-display mt-1 text-ink">{datasets.length}</p>
            <p className="text-[11px] text-[var(--muted)] mt-0.5">Profiled & ready</p>
          </div>

          {/* Card 2: Discovered Links */}
          <div className="p-3.5 bg-paper/60 border border-mist rounded-control">
            <div className="flex items-center justify-between text-[var(--muted)] text-xs">
              <span>{plainEnglish ? 'Smart Links Found' : 'Discovered Joins'}</span>
              <Sparkles size={14} className="text-teal" />
            </div>
            <p className="text-2xl font-semibold font-display mt-1 text-teal">{relationships.length}</p>
            <p className="text-[11px] text-[var(--muted)] mt-0.5">Auto-detected relationships</p>
          </div>

          {/* Card 3: Validated Links */}
          <div className="p-3.5 bg-paper/60 border border-mist rounded-control">
            <div className="flex items-center justify-between text-[var(--muted)] text-xs">
              <span>{plainEnglish ? 'Verified Against Data' : 'Validated Joins'}</span>
              <ShieldCheck size={14} className="text-emerald-400" />
            </div>
            <div className="flex items-baseline gap-2 mt-1">
              <p className="text-2xl font-semibold font-display text-emerald-400">{validatedCount}</p>
              <span className="text-xs text-[var(--muted)] font-medium">of {relationships.length}</span>
            </div>
            <p className="text-[11px] text-[var(--muted)] mt-0.5">
              {validatedCount === 0 ? 'Click "Verify" on links below' : 'Fully tested on disk'}
            </p>
          </div>

          {/* Card 4: Risks & Warnings */}
          <div className="p-3.5 bg-paper/60 border border-mist rounded-control">
            <div className="flex items-center justify-between text-[var(--muted)] text-xs">
              <span>{plainEnglish ? 'Double-Counting Risks' : 'Analytical Join Risks'}</span>
              <AlertTriangle size={14} className={totalRisksCount > 0 ? 'text-amber-400' : 'text-emerald-400'} />
            </div>
            <p className={`text-2xl font-semibold font-display mt-1 ${totalRisksCount > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
              {totalRisksCount}
            </p>
            <p className="text-[11px] mt-0.5 font-medium">
              {totalRisksCount === 0 ? (
                <span className="text-emerald-400">✓ Safe to combine directly</span>
              ) : (
                <span className="text-amber-400">⚠️ Needs grouping before summing</span>
              )}
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
            {plainEnglish ? 'Overview & All Links' : 'All Intelligence'}
          </button>
          <button
            type="button"
            onClick={() => setViewMode('relationships')}
            className={`px-3 py-1.5 rounded transition-colors ${
              viewMode === 'relationships' ? 'bg-teal text-paper font-semibold' : 'text-[var(--muted)] hover:text-ink'
            }`}
          >
            {plainEnglish ? 'Table Connections' : 'Relationships'} ({relationships.length})
          </button>
          <button
            type="button"
            onClick={() => setViewMode('grains')}
            className={`px-3 py-1.5 rounded transition-colors ${
              viewMode === 'grains' ? 'bg-teal text-paper font-semibold' : 'text-[var(--muted)] hover:text-ink'
            }`}
          >
            {plainEnglish ? 'What Each Row Means' : 'Dataset Grains'} ({datasets.length})
          </button>
        </div>

        {/* Safety Filter and Search */}
        {(viewMode === 'all' || viewMode === 'relationships') && (
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1 bg-surface border border-mist p-1 rounded-control text-xs">
              <button
                type="button"
                onClick={() => setSafetyFilter('all')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  safetyFilter === 'all' ? 'bg-mist text-ink font-medium' : 'text-[var(--muted)] hover:text-ink'
                }`}
              >
                All Links
              </button>
              <button
                type="button"
                onClick={() => setSafetyFilter('safe')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  safetyFilter === 'safe'
                    ? 'bg-emerald-500/20 text-emerald-300 font-medium'
                    : 'text-[var(--muted)] hover:text-ink'
                }`}
              >
                ✓ Safe
              </button>
              <button
                type="button"
                onClick={() => setSafetyFilter('risks')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  safetyFilter === 'risks'
                    ? 'bg-amber-500/20 text-amber-300 font-medium'
                    : 'text-[var(--muted)] hover:text-ink'
                }`}
              >
                ⚠️ Warnings ({totalRisksCount})
              </button>
            </div>

            <div className="relative">
              <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
              <input
                type="text"
                placeholder={plainEnglish ? 'Search tables or columns…' : 'Filter keys or datasets…'}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input text-xs py-1.5 pl-8 pr-3 w-48"
              />
            </div>
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
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="font-display text-lg font-semibold text-ink flex items-center gap-2">
                <span>{plainEnglish ? 'What Each Row Means (Table Structure)' : 'What One Row Represents (Dataset Grain)'}</span>
                <span className="text-xs px-2 py-0.5 rounded bg-mist text-[var(--muted)] font-normal">
                  {datasets.length} tables
                </span>
              </h3>
              <p className="text-xs text-[var(--muted)] mt-0.5">
                {plainEnglish
                  ? 'Confirm what one record represents in each table so your metrics (like revenue, churn, or transactions) aren\'t accidentally multiplied.'
                  : 'Explicit definition of what each record represents in the real world. Prevents inadvertent record duplication when merging tables.'}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map((dataset) => {
              const grain = grains[dataset.id];
              const confirmed = userConfirmedGrains[dataset.id];
              const grainInfo = getGrainInfo(confirmed ? confirmed.type : grain?.grain_type, dataset.name);
              const icon = grainInfo.icon;

              return (
                <div 
                  key={dataset.id} 
                  className={`surface p-5 flex flex-col justify-between space-y-4 border transition-all ${
                    confirmed ? 'border-emerald-500/40 bg-surface/90' : 'border-mist hover:border-mist/80'
                  }`}
                >
                  <div>
                    {/* Card Header: Icon + Name + Badge */}
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-start gap-2.5">
                        <span className="text-2xl p-2 rounded-control bg-paper/80 border border-mist leading-none">
                          {icon}
                        </span>
                        <div>
                          <h4 className="font-display font-semibold text-ink text-base flex items-center gap-1.5">
                            <span>{dataset.name}</span>
                          </h4>
                          <p className="text-xs text-[var(--muted)] mt-0.5 font-mono">
                            {dataset.row_count?.toLocaleString()} rows · {dataset.column_count} columns
                          </p>
                        </div>
                      </div>

                      <span className={`px-2 py-0.5 text-xs font-medium rounded border shrink-0 ${grainInfo.cls}`}>
                        {confirmed ? '✓ Confirmed by You' : (plainEnglish ? grainInfo.plainBadge : grainInfo.techBadge)}
                      </span>
                    </div>

                    {/* Human Explanation Box */}
                    <div className="mt-3.5 p-3 bg-paper/70 border border-mist rounded-control">
                      <p className="text-xs font-semibold text-ink flex items-center gap-1.5">
                        {confirmed ? (
                          <>
                            <CheckCircle2 size={13} className="text-emerald-400 shrink-0" />
                            <span>Confirmed: 1 row = 1 {confirmed.label}</span>
                          </>
                        ) : (
                          <>
                            <span>{plainEnglish ? (grainInfo.isReady ? '✓ ' : 'ℹ️ ') + grainInfo.plainDesc : (grain?.grain_description || `Level of detail: ${dataset.name}`)}</span>
                          </>
                        )}
                      </p>

                      <p className="text-[11px] text-[var(--muted)] mt-1.5 flex items-center justify-between">
                        <span>
                          {plainEnglish ? 'Automatic Confidence: ' : 'Inference Confidence: '}
                          <strong className="text-ink">{grain ? Math.round(grain.confidence * 100) : 0}%</strong>
                        </span>
                        {grain && grain.confidence >= 0.7 ? (
                          <span className="text-emerald-400 font-medium">✓ High Match</span>
                        ) : (
                          <span className="text-amber-400 font-medium">Quick review helpful</span>
                        )}
                      </p>
                    </div>

                    {/* Quick Confirmation Action (if unconfirmed) */}
                    {!confirmed && (
                      <div className="mt-3 p-2.5 bg-surface-elevated/70 border border-mist/80 rounded-control text-xs space-y-2">
                        <p className="text-[11px] text-[var(--muted)] font-medium">
                          Quick confirm what each row is:
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          <button
                            type="button"
                            onClick={() => handleConfirmMeaning(dataset.id, 'transaction', 'Transaction / Purchase')}
                            className="px-2 py-1 rounded bg-paper hover:bg-mist border border-mist text-[11px] text-ink transition-colors flex items-center gap-1"
                          >
                            <span>💳</span> Transaction
                          </button>
                          <button
                            type="button"
                            onClick={() => handleConfirmMeaning(dataset.id, 'entity', 'Customer / Account')}
                            className="px-2 py-1 rounded bg-paper hover:bg-mist border border-mist text-[11px] text-ink transition-colors flex items-center gap-1"
                          >
                            <span>👤</span> Customer
                          </button>
                          <button
                            type="button"
                            onClick={() => handleConfirmMeaning(dataset.id, 'event', 'Activity Log')}
                            className="px-2 py-1 rounded bg-paper hover:bg-mist border border-mist text-[11px] text-ink transition-colors flex items-center gap-1"
                          >
                            <span>⚡</span> Activity
                          </button>
                          <button
                            type="button"
                            onClick={() => handleConfirmMeaning(dataset.id, 'periodic_aggregate', 'Daily Summary')}
                            className="px-2 py-1 rounded bg-paper hover:bg-mist border border-mist text-[11px] text-ink transition-colors flex items-center gap-1"
                          >
                            <span>📊</span> Summary
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Key Columns Info */}
                    <div className="mt-3 space-y-2 text-xs">
                      {/* Primary Key / Unique ID */}
                      {grain?.primary_key && grain.primary_key.length > 0 && (
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span className="text-[var(--muted)]" title="Unique ID column identifying each row">
                            {plainEnglish ? 'Unique ID:' : 'Primary Key:'}
                          </span>
                          {grain.primary_key.map((pk) => (
                            <span key={pk} className="px-1.5 py-0.5 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30 font-mono text-[11px]">
                              {pk}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Foreign Keys / Link Columns */}
                      {grain?.entity_identifiers && grain.entity_identifiers.length > 0 && (
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span className="text-[var(--muted)]" title="Columns used to link this table to other tables">
                            {plainEnglish ? 'Linkable IDs:' : 'Entity FKs:'}
                          </span>
                          {grain.entity_identifiers.map((fk) => (
                            <span key={fk} className="px-1.5 py-0.5 rounded bg-teal-500/15 text-teal-300 border border-teal-500/30 font-mono text-[11px]">
                              {fk}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Measures / Numeric columns */}
                      {grain?.potential_measures && grain.potential_measures.length > 0 && (
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span className="text-[var(--muted)]" title="Amounts or totals that must not be counted twice">
                            {plainEnglish ? 'Key Totals/Metrics:' : 'Vulnerable Measures:'}
                          </span>
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

                  {/* Supporting Evidence Drawer */}
                  {grain?.evidence && grain.evidence.length > 0 && (
                    <details className="pt-3 border-t border-mist text-xs group">
                      <summary className="cursor-pointer text-[var(--muted)] hover:text-ink font-medium flex items-center justify-between">
                        <span>{plainEnglish ? 'How Knowa detected this' : `Evidence (${grain.evidence.length})`}</span>
                        <ChevronDown size={13} className="text-[var(--muted)] transition-transform group-open:rotate-180" />
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
              <h3 className="font-display text-lg font-semibold text-ink">
                {plainEnglish ? 'Table Connections & Safety Warnings' : 'How Your Data Connects (Cross-Dataset Joins)'}
              </h3>
              <p className="text-xs text-[var(--muted)] mt-0.5">
                {plainEnglish
                  ? 'Discovered connections between your tables with match rates, connection types, and double-counting safeguards.'
                  : 'Discovered key relationships showing match rates, cardinality, and whether combining tables will exclude records or multiply totals.'}
              </p>
            </div>
          </div>

          {filteredRelationships.length === 0 ? (
            <div className="surface p-8 text-center space-y-3 rounded-control border border-mist">
              <p className="text-base text-ink font-medium">No matching connections found</p>
              <p className="text-xs text-[var(--muted)] max-w-md mx-auto">
                {searchQuery
                  ? 'Try clearing your search filter to view all discovered connections.'
                  : 'Upload related tables (e.g. customers, orders, transactions) to allow Knowa to infer relationships and evaluate safety.'}
              </p>
              <button
                type="button"
                onClick={() => handleOpenSimulation()}
                className="btn-primary text-xs"
              >
                Test a Custom Connection
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredRelationships.map((rel) => {
                const safety = getSafetyBadge(rel.join_safety, rel.containment_ratio);
                const cardLabel = getCardinalityLabel(rel.cardinality);
                const isValidating = validatingId === rel.id;
                const srcIcon = getDatasetIcon(rel.source_dataset_name);
                const tgtIcon = getDatasetIcon(rel.target_dataset_name);

                return (
                  <div key={rel.id} className="surface p-6 space-y-4 rounded-control border border-mist hover:border-mist/80 transition-all">
                    {/* Visual Relationship Diagram */}
                    <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-mist">
                      <div className="flex flex-wrap items-center gap-3">
                        {/* Source Box */}
                        <div className="bg-paper border border-mist px-3.5 py-2.5 rounded-control shadow-sm">
                          <p className="text-[11px] text-[var(--muted)] flex items-center gap-1 font-medium">
                            <span>{srcIcon}</span> {rel.source_dataset_name}
                          </p>
                          <p className="text-sm font-semibold text-ink font-mono mt-0.5">{rel.source_column}</p>
                        </div>

                        {/* Connection Visual Pill */}
                        <div className="flex flex-col items-center px-2">
                          <div className="flex items-center gap-1.5">
                            <span className="w-4 h-[1px] bg-teal/60"></span>
                            <span className="px-2.5 py-1 rounded bg-teal/15 text-teal text-xs font-semibold border border-teal/30 shadow-sm">
                              {cardLabel}
                            </span>
                            <span className="w-4 h-[1px] bg-teal/60"></span>
                          </div>
                          <span className="text-[10px] text-[var(--muted)] mt-1 font-medium">
                            {Math.round(rel.containment_ratio * 100)}% match rate
                          </span>
                        </div>

                        {/* Target Box */}
                        <div className="bg-paper border border-mist px-3.5 py-2.5 rounded-control shadow-sm">
                          <p className="text-[11px] text-[var(--muted)] flex items-center gap-1 font-medium">
                            <span>{tgtIcon}</span> {rel.target_dataset_name}
                          </p>
                          <p className="text-sm font-semibold text-ink font-mono mt-0.5">{rel.target_column}</p>
                        </div>
                      </div>

                      {/* Right-side status and action buttons */}
                      <div className="flex flex-wrap items-center gap-2">
                        <span className={`px-2.5 py-1 text-xs font-medium rounded border flex items-center gap-1.5 ${safety.cls}`}>
                          {safety.label}
                        </span>

                        <button
                          type="button"
                          onClick={() => setExplainingRel(rel)}
                          className="btn-secondary text-xs px-2.5 py-1.5 flex items-center gap-1"
                          title="Learn why these two tables link together"
                        >
                          <HelpCircle size={13} />
                          <span>Why Linked?</span>
                        </button>

                        <button
                          type="button"
                          onClick={() => handleOpenSimulation(rel)}
                          className="btn-secondary text-xs px-2.5 py-1.5 flex items-center gap-1"
                          title="Test what happens when these tables combine"
                        >
                          <FlaskConical size={13} className="text-teal" />
                          <span>Test Link</span>
                        </button>

                        {rel.validation_status !== 'validated' ? (
                          <button
                            type="button"
                            onClick={() => handleValidateJoin(rel)}
                            disabled={isValidating}
                            className="btn-primary text-xs px-2.5 py-1.5 flex items-center gap-1"
                            title="Verify keys against actual disk records"
                          >
                            {isValidating ? <Spinner /> : <ShieldCheck size={13} />}
                            <span>Validate Link</span>
                          </button>
                        ) : (
                          <span className="text-xs px-2.5 py-1 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 flex items-center gap-1 font-medium">
                            <Check size={12} />
                            <span>Verified</span>
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Join Safety Risk & Warnings Banner */}
                    {(rel.join_safety === 'fanout_risk' || rel.double_counting_risk) && (
                      <div className="p-3.5 bg-amber-500/10 border border-amber-500/30 rounded-control text-xs text-amber-200 flex items-start gap-2.5">
                        <AlertTriangle size={16} className="text-amber-400 shrink-0 mt-0.5" />
                        <div className="space-y-1">
                          <p className="font-semibold text-amber-300">
                            {plainEnglish ? '⚠️ Double-Counting Warning on Grand Totals' : 'Potential Double-Counting Warning'}
                          </p>
                          <p className="leading-relaxed">
                            {plainEnglish ? (
                              <>
                                Connecting <strong>{rel.source_dataset_name}</strong> to <strong>{rel.target_dataset_name}</strong> matches 1 record to approximately <strong>{rel.fan_out_ratio}x records</strong>. 
                                {rel.double_counting_measures?.length > 0 && (
                                  <span> Summing metric columns like <code className="px-1 py-0.5 rounded bg-black/40 text-amber-200">{rel.double_counting_measures.join(', ')}</code> directly across this link will multiply totals by ~{rel.fan_out_ratio}x. Always group or summarize first!</span>
                                )}
                              </>
                            ) : (
                              <>
                                Joining <strong>{rel.source_dataset_name}</strong> → <strong>{rel.target_dataset_name}</strong> creates a {cardLabel} expansion (estimated {rel.fan_out_ratio}x).
                                {rel.double_counting_measures?.length > 0 && (
                                  <span> Source measures ({rel.double_counting_measures.join(', ')}) should not be summed after this join without pre-aggregation or grain normalization.</span>
                                )}
                              </>
                            )}
                          </p>
                        </div>
                      </div>
                    )}

                    {rel.join_safety === 'many_to_many_risk' && (
                      <div className="p-3.5 bg-rose-500/10 border border-rose-500/30 rounded-control text-xs text-rose-200 flex items-start gap-2.5">
                        <AlertTriangle size={16} className="text-rose-400 shrink-0 mt-0.5" />
                        <div>
                          <p className="font-semibold text-rose-300">
                            {plainEnglish ? '🚨 Caution: Severe Duplication Risk' : 'Cartesian Explosion Risk (Many-to-Many)'}
                          </p>
                          <p className="mt-0.5 leading-relaxed">
                            {plainEnglish
                              ? 'Both tables have repeating keys. Connecting them directly creates cross-multiplied duplicate rows on both sides.'
                              : 'Both datasets contain duplicate keys. Direct join risks geometric row expansion. Use a bridge table or aggregate both sides before combining.'}
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Data Cleanliness Notes */}
                    {rel.quality_notes && rel.quality_notes.length > 0 && (
                      <div className="p-3 bg-paper/80 border border-mist rounded-control text-xs space-y-1">
                        <p className="font-semibold text-ink flex items-center gap-1.5">
                          <span>🧹</span> Data Quality Observations:
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
                        Match Confidence: <strong className="text-ink">{Math.round(rel.confidence * 100)}%</strong>
                      </span>
                      <span>
                        Matching Records: <strong className="text-ink">{rel.overlapping_values_count?.toLocaleString()}</strong>
                      </span>
                      {!plainEnglish && (
                        <span>
                          Jaccard Similarity: <strong className="text-ink">{rel.jaccard_similarity}</strong>
                        </span>
                      )}
                      <span>
                        Unmatched in {rel.source_dataset_name}: <strong className="text-ink">{Math.round((rel.source_unmatched_rate || 0) * 100)}%</strong>
                      </span>
                      <span>
                        Unmatched in {rel.target_dataset_name}: <strong className="text-ink">{Math.round((rel.target_unmatched_rate || 0) * 100)}%</strong>
                      </span>
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
        <div className="fixed inset-0 z-50 bg-ink/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-surface border border-mist rounded-control w-full max-w-2xl p-6 max-h-[85vh] overflow-y-auto animate-page-in space-y-5 shadow-2xl">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-xs font-semibold uppercase text-teal tracking-wider flex items-center gap-1.5">
                  <Sparkles size={13} />
                  <span>{plainEnglish ? 'Link Explanation' : 'Explainability Engine'}</span>
                </span>
                <h3 className="text-xl font-display font-semibold text-ink mt-0.5">
                  {plainEnglish ? 'Why These Tables Connect' : 'Why Knowa Inferred This Relationship'}
                </h3>
                <p className="text-xs font-mono text-[var(--muted)] mt-1">
                  {explainingRel.source_dataset_name}.{explainingRel.source_column} ⟷ {explainingRel.target_dataset_name}.{explainingRel.target_column}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setExplainingRel(null)}
                className="text-[var(--muted)] hover:text-ink text-lg p-1"
              >
                ✕
              </button>
            </div>

            {/* Section 1: How Knowa Found This Connection */}
            <div className="surface p-4 space-y-2 border border-mist rounded-control">
              <h4 className="text-xs uppercase font-semibold text-teal tracking-wider">
                1. How Knowa Found This Connection
              </h4>
              <ul className="space-y-1.5 text-xs text-ink list-disc list-inside leading-relaxed">
                {explainingRel.evidence?.map((ev, i) => (
                  <li key={i}>{ev}</li>
                ))}
              </ul>
            </div>

            {/* Section 2: How Records Match */}
            <div className="surface p-4 space-y-2 border border-mist rounded-control">
              <h4 className="text-xs uppercase font-semibold text-teal tracking-wider">
                2. Connection Type: {getCardinalityLabel(explainingRel.cardinality)}
              </h4>
              <p className="text-xs text-[var(--muted)] leading-relaxed">
                {plainEnglish
                  ? 'We analyzed how frequently values repeat in each table to know if 1 record matches exactly 1 record or many.'
                  : 'Cardinality is determined mathematically from uniqueness distributions across non-null keys:'}
              </p>
              <div className="grid grid-cols-2 gap-3 text-xs pt-2">
                <div className="p-3 bg-paper rounded border border-mist">
                  <p className="text-[var(--muted)] font-mono text-[11px]">{explainingRel.source_dataset_name}.{explainingRel.source_column}</p>
                  <p className="text-base font-semibold text-ink mt-0.5">
                    {Math.round(explainingRel.source_uniqueness_ratio * 100)}% Unique
                  </p>
                  <p className="text-[11px] text-[var(--muted)] mt-0.5">
                    {explainingRel.source_uniqueness_ratio >= 0.95 ? 'Each value is unique (1 row per ID)' : 'Values repeat across rows'}
                  </p>
                </div>
                <div className="p-3 bg-paper rounded border border-mist">
                  <p className="text-[var(--muted)] font-mono text-[11px]">{explainingRel.target_dataset_name}.{explainingRel.target_column}</p>
                  <p className="text-base font-semibold text-ink mt-0.5">
                    {Math.round(explainingRel.target_uniqueness_ratio * 100)}% Unique
                  </p>
                  <p className="text-[11px] text-[var(--muted)] mt-0.5">
                    {explainingRel.target_uniqueness_ratio >= 0.95 ? 'Each value is unique (1 row per ID)' : 'Values repeat across rows'}
                  </p>
                </div>
              </div>
            </div>

            {/* Section 3: Safety & Calculations */}
            <div className="surface p-4 space-y-2 border border-mist rounded-control">
              <h4 className="text-xs uppercase font-semibold text-teal tracking-wider">
                3. Safety & Calculation Safeguards
              </h4>
              <p className="text-xs text-ink leading-relaxed">
                {plainEnglish
                  ? 'Even when two tables link cleanly, calculating totals across them can be hazardous if multiple records match. Knowa protects your charts by alerting you whenever grouping is required.'
                  : 'Knowa explicitly distinguishes Relationship Validity from Analytical Safety. Performing downstream operations like summing source metrics will result in duplication.'}
              </p>
              <div className="p-3 bg-paper rounded border border-mist text-xs space-y-1 mt-2">
                <p className="font-semibold text-ink">
                  Safety Status: <span className="text-teal">{getSafetyBadge(explainingRel.join_safety).label}</span>
                </p>
                <p className="text-[var(--muted)]">
                  Row Multiplier Factor: <strong>{explainingRel.fan_out_ratio}x</strong>
                </p>
                {explainingRel.double_counting_measures?.length > 0 && (
                  <p className="text-amber-300">
                    Metrics that require grouping: {explainingRel.double_counting_measures.join(', ')}
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
        <div className="fixed inset-0 z-50 bg-ink/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-surface border border-mist rounded-control w-full max-w-2xl p-6 max-h-[85vh] overflow-y-auto animate-page-in space-y-5 shadow-2xl">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-xs font-semibold uppercase text-teal tracking-wider flex items-center gap-1.5">
                  <FlaskConical size={13} />
                  <span>{plainEnglish ? 'Link Testing Playground' : 'Join Simulation Lab'}</span>
                </span>
                <h3 className="text-xl font-display font-semibold text-ink mt-0.5">
                  {plainEnglish ? 'Test Combining Two Tables' : 'Non-Destructive Join Simulation'}
                </h3>
                <p className="text-xs text-[var(--muted)] mt-1">
                  {plainEnglish
                    ? 'Safely test linking two tables to see how many records match and whether totals will be multiplied, without changing your real data.'
                    : 'Test and inspect join row expansion, unmatched keys, and Cartesian multiplication without altering original datasets.'}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowSimulateModal(false)}
                className="text-[var(--muted)] hover:text-ink text-lg p-1"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleRunSimulation} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Source Selection */}
                <div className="p-3.5 bg-paper rounded border border-mist space-y-3">
                  <p className="text-xs font-semibold text-ink uppercase tracking-wider">First Table (Source)</p>
                  <div>
                    <label className="block text-[11px] text-[var(--muted)] mb-1">Select Table</label>
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
                          {getDatasetIcon(d.name)} {d.name} ({d.row_count} rows)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-[11px] text-[var(--muted)] mb-1">Matching Column</label>
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
                <div className="p-3.5 bg-paper rounded border border-mist space-y-3">
                  <p className="text-xs font-semibold text-ink uppercase tracking-wider">Second Table (Target)</p>
                  <div>
                    <label className="block text-[11px] text-[var(--muted)] mb-1">Select Table</label>
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
                          {getDatasetIcon(d.name)} {d.name} ({d.row_count} rows)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-[11px] text-[var(--muted)] mb-1">Matching Column</label>
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

              <div className="flex justify-end gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setShowSimulateModal(false)}
                  className="btn-secondary text-xs px-3.5 py-2"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={simulating}
                  className="btn-primary text-xs px-4 py-2 flex items-center gap-1.5 font-semibold"
                >
                  {simulating ? <Spinner /> : <FlaskConical size={14} />}
                  <span>Run Link Simulation</span>
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
              <div className="surface p-4 space-y-4 border border-teal/40 rounded-control animate-page-in">
                <div className="flex items-center justify-between">
                  <h4 className="font-display font-semibold text-ink text-sm flex items-center gap-1.5">
                    <Sparkles size={14} className="text-teal" />
                    <span>Simulation Results</span>
                  </h4>
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
                    <p className="text-[var(--muted)]">Combined Rows</p>
                    <p className="text-base font-semibold text-teal mt-0.5">
                      {simulationResult.rows_after?.toLocaleString()}
                    </p>
                  </div>
                  <div className="p-2.5 bg-paper rounded border border-mist">
                    <p className="text-[var(--muted)]">Row Expansion</p>
                    <p className={`text-base font-semibold mt-0.5 ${simulationResult.row_expansion_factor > 1.05 ? 'text-amber-400' : 'text-emerald-400'}`}>
                      {simulationResult.row_expansion_factor}x
                    </p>
                  </div>
                  <div className="p-2.5 bg-paper rounded border border-mist">
                    <p className="text-[var(--muted)]">Unmatched Rows</p>
                    <p className="text-base font-semibold text-ink mt-0.5">
                      {simulationResult.unmatched_source_rows?.toLocaleString()}
                    </p>
                  </div>
                </div>

                {simulationResult.warnings?.length > 0 && (
                  <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded text-xs text-amber-200 space-y-1">
                    <p className="font-semibold text-amber-300">Safety Alerts:</p>
                    <ul className="list-disc list-inside space-y-0.5">
                      {simulationResult.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
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
