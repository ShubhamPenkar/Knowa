import { useState, useEffect, useMemo } from 'react';
import { useParams, useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function humanize(name) {
  return String(name || '')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function pct(p) {
  if (p == null || Number.isNaN(Number(p))) return '—';
  return `${(Number(p) * 100).toFixed(0)}%`;
}

export default function WhatIf() {
  const { projectId } = useParams();
  const [searchParams] = useSearchParams();
  const rowIdx = searchParams.get('row');

  const { token } = useAuth();
  const navigate = useNavigate();

  const [project, setProject] = useState(null);
  const [testData, setTestData] = useState(null);
  const [baseRow, setBaseRow] = useState(null);
  const [modifiedValues, setModifiedValues] = useState({});
  const [baselinePrediction, setBaselinePrediction] = useState(null);
  const [simulationResult, setSimulationResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);
  const [filterLever, setFilterLever] = useState(false);

  useEffect(() => {
    if (token && projectId) fetchProject();
  }, [projectId, token]);

  const fetchProject = async () => {
    try {
      const res = await fetch(`/api/projects/${projectId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setProject(data);
        if (data.status === 'trained' || data.status === 'ready') {
          fetchTestData(data);
        }
      }
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const fetchTestData = async (proj) => {
    try {
      const res = await fetch(`/api/projects/${projectId}/test-data?limit=50`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setTestData(data.rows);
        if (rowIdx != null && data.rows[parseInt(rowIdx, 10)]) {
          selectBaseRow(data.rows[parseInt(rowIdx, 10)], proj);
        }
      }
    } catch (err) {
      console.error(err);
    }
  };

  const selectBaseRow = async (row, proj = project) => {
    setBaseRow(row);
    setModifiedValues({});
    setSimulationResult(null);
    setError('');

    const features = {};
    (proj.feature_columns || []).forEach((col) => {
      features[col] = row[col];
    });

    const preferred = [
      'customerID',
      'CustomerID',
      'customer_id',
      'customerId',
      'entity_id',
      'account_id',
      'user_id',
      'id',
      'ID',
    ];
    let entity_id = null;
    for (const k of preferred) {
      if (row[k] != null && String(row[k]).trim() !== '') {
        entity_id = String(row[k]);
        break;
      }
    }
    if (!entity_id && rowIdx != null) entity_id = `row-${rowIdx}`;

    try {
      const res = await fetch(`/api/projects/${projectId}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          features,
          entity_id,
          include_explanations: true,
          include_recommendations: false,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setBaselinePrediction(data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleFeatureChange = (feature, value) => {
    const parsed =
      value === ''
        ? undefined
        : !Number.isNaN(Number(value)) && value.trim() !== ''
          ? Number(value)
          : value;
    setModifiedValues((prev) => ({ ...prev, [feature]: parsed }));
  };

  const activeChanges = useMemo(() => {
    const out = {};
    Object.entries(modifiedValues).forEach(([k, v]) => {
      if (v === undefined) return;
      if (baseRow && String(v) === String(baseRow[k])) return;
      out[k] = v;
    });
    return out;
  }, [modifiedValues, baseRow]);

  const runSimulation = async () => {
    if (!Object.keys(activeChanges).length) {
      setError('Change at least one value that differs from the original.');
      return;
    }
    setSimulating(true);
    setError('');

    const baseFeatures = {};
    project.feature_columns.forEach((col) => {
      baseFeatures[col] = baseRow[col];
    });

    try {
      const res = await fetch(`/api/projects/${projectId}/simulate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          base_features: baseFeatures,
          modified_features: activeChanges,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setSimulationResult(data);
        setHistory((prev) =>
          [
            {
              timestamp: new Date().toISOString(),
              changes: { ...activeChanges },
              result: data,
            },
            ...prev,
          ].slice(0, 10)
        );
      } else {
        const errData = await res.json().catch(() => ({}));
        setError(errData.detail || 'Simulation failed');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    }
    setSimulating(false);
  };

  const resetModifications = () => {
    setModifiedValues({});
    setSimulationResult(null);
    setError('');
  };

  const applySuggestion = (feature, value) => {
    if (value === undefined) return;
    setModifiedValues((prev) => ({ ...prev, [feature]: value }));
  };

  const applyHistoryItem = (item) => {
    setModifiedValues(item.changes);
    setSimulationResult(item.result);
  };

  const isReg = project?.problem_type === 'regression';
  const outcomeLabel = humanize(
    project?.target_description || project?.target_column || 'outcome'
  );

  const leverFeatures = useMemo(() => {
    const drivers =
      baselinePrediction?.explanations?.drivers ||
      baselinePrediction?.explanations?.shap?.top_features ||
      [];
    return new Set(drivers.map((d) => d.feature));
  }, [baselinePrediction]);

  const columnsToShow = useMemo(() => {
    const cols = project?.feature_columns || [];
    if (!filterLever || leverFeatures.size === 0) return cols;
    const preferred = cols.filter((c) => leverFeatures.has(c));
    return preferred.length ? preferred : cols;
  }, [project, filterLever, leverFeatures]);

  if (loading) {
    return (
      <div className="min-h-[40vh] flex items-center justify-center text-[var(--muted)] text-sm">
        Loading scenario tools…
      </div>
    );
  }

  if (!project) {
    return (
      <div className="p-6">
        <p className="text-[var(--muted)]">Project not found</p>
      </div>
    );
  }

  if (project.status !== 'trained' && project.status !== 'ready') {
    return (
      <div className="p-6 max-w-xl">
        <Link to={`/projects/${projectId}`} className="text-sm text-teal hover:underline">
          ← Back to project
        </Link>
        <h1 className="font-display text-2xl font-semibold text-ink mt-4">What-if analysis</h1>
        <p className="mt-2 text-sm text-[var(--muted)]">
          Prepare the project first so we can score before/after scenarios.
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-8">
      <header className="border-b border-mist pb-6">
        <Link to={`/projects/${projectId}`} className="text-sm text-teal hover:underline">
          ← Back to project
        </Link>
        <p className="page-kicker mt-4 mb-1">Phase 5 · Scenario</p>
        <h1 className="font-display text-3xl font-semibold text-ink tracking-tight">
          What if we changed something?
        </h1>
        <p className="mt-2 text-sm text-[var(--muted)] max-w-2xl leading-relaxed">
          Pick a person or account, tweak the levers that drive {outcomeLabel.toLowerCase()}, and
          compare the estimated chance before vs after — with a re-ranked next-step plan.
        </p>
      </header>

      {!baseRow && (
        <section className="surface overflow-hidden">
          <div className="px-5 py-4 border-b border-mist">
            <h2 className="font-display text-lg font-semibold text-ink">Choose a starting case</h2>
            <p className="text-sm text-[var(--muted)] mt-1">
              Held-out sample from prepare. Click a row to load the baseline.
            </p>
          </div>
          <div className="overflow-x-auto max-h-[28rem]">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Known</th>
                  {(project.feature_columns || []).slice(0, 5).map((col) => (
                    <th key={col}>{humanize(col)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(testData || []).map((row, idx) => (
                  <tr
                    key={idx}
                    className="cursor-pointer hover:bg-paper/80"
                    onClick={() => selectBaseRow(row)}
                  >
                    <td className="text-[var(--muted)]">{idx + 1}</td>
                    <td>
                      {isReg
                        ? Number(row[project.target_column]).toFixed(2)
                        : String(row[project.target_column]) === project.target_positive_label
                          ? 'Yes'
                          : 'No'}
                    </td>
                    {(project.feature_columns || []).slice(0, 5).map((col) => (
                      <td key={col} className="max-w-[8rem] truncate">
                        {String(row[col] ?? '')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {baseRow && (
        <div className="grid lg:grid-cols-12 gap-8">
          <div className="lg:col-span-7 space-y-6">
            {/* Baseline */}
            {baselinePrediction && (
              <div className="border border-mist px-5 py-5">
                <div className="flex justify-between items-start gap-4">
                  <div>
                    <p className="page-kicker mb-1">Baseline</p>
                    <p className="font-display text-3xl font-semibold tabular-nums text-ink">
                      {isReg
                        ? Number(baselinePrediction.predicted_value).toFixed(2)
                        : pct(baselinePrediction.probability)}
                    </p>
                    <p className="text-sm text-[var(--muted)] mt-1">
                      Current best guess for {outcomeLabel.toLowerCase()}
                      {baselinePrediction.risk_level
                        ? ` · ${baselinePrediction.risk_level}`
                        : ''}
                    </p>
                  </div>
                  <button
                    type="button"
                    className="btn-secondary text-sm"
                    onClick={() => {
                      setBaseRow(null);
                      setBaselinePrediction(null);
                      setSimulationResult(null);
                      setModifiedValues({});
                    }}
                  >
                    Change case
                  </button>
                </div>
              </div>
            )}

            {/* Suggested tweaks */}
            {baselinePrediction?.explanations?.drivers && (
              <div className="border border-mist px-5 py-4">
                <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] mb-3">
                  Suggested levers (from “why”)
                </h3>
                <ul className="space-y-2">
                  {(baselinePrediction.explanations.drivers || [])
                    .filter((d) => Number(d.impact) > 0)
                    .slice(0, 4)
                    .map((d) => (
                      <li
                        key={d.feature}
                        className="flex flex-wrap items-center justify-between gap-2 text-sm border-b border-mist/70 pb-2"
                      >
                        <span className="text-ink">
                          <span className="font-medium">{humanize(d.feature)}</span>
                          <span className="text-[var(--muted)]"> · currently raises risk</span>
                        </span>
                        <button
                          type="button"
                          className="text-xs text-teal hover:underline"
                          onClick={() => {
                            setFilterLever(true);
                            // scroll-ish focus: mark for edit
                            const el = document.getElementById(`feat-${d.feature}`);
                            el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            el?.querySelector('input,select')?.focus();
                          }}
                        >
                          Edit this
                        </button>
                      </li>
                    ))}
                </ul>
              </div>
            )}

            {/* Feature editor */}
            <div className="border border-mist">
              <div className="px-5 py-4 border-b border-mist flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="font-display text-lg font-semibold text-ink">Edit scenario</h3>
                  <p className="text-sm text-[var(--muted)] mt-0.5">
                    {Object.keys(activeChanges).length} change
                    {Object.keys(activeChanges).length === 1 ? '' : 's'} ready
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    className="btn-secondary text-sm"
                    onClick={() => setFilterLever((v) => !v)}
                  >
                    {filterLever ? 'Show all fields' : 'Focus key drivers'}
                  </button>
                  <button type="button" className="btn-secondary text-sm" onClick={resetModifications}>
                    Reset
                  </button>
                  <button
                    type="button"
                    className="btn-primary text-sm"
                    disabled={simulating || !Object.keys(activeChanges).length}
                    onClick={runSimulation}
                  >
                    {simulating ? 'Running…' : 'Run scenario'}
                  </button>
                </div>
              </div>

              {error && (
                <div className="mx-5 mt-4 px-3 py-2 border border-coral/40 bg-coral-soft text-sm text-ink">
                  {error}
                </div>
              )}

              <div className="p-5 grid sm:grid-cols-2 gap-3 max-h-[28rem] overflow-y-auto">
                {columnsToShow.map((col) => {
                  const originalValue = baseRow[col];
                  const isModified = activeChanges[col] !== undefined;
                  const display =
                    modifiedValues[col] !== undefined ? modifiedValues[col] : originalValue;
                  return (
                    <div
                      id={`feat-${col}`}
                      key={col}
                      className={`px-3 py-3 border text-sm ${
                        isModified ? 'border-teal/50 bg-teal-soft/10' : 'border-mist'
                      }`}
                    >
                      <label className="block text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
                        {humanize(col)}
                        {isModified && (
                          <span className="ml-2 normal-case tracking-normal text-teal">changed</span>
                        )}
                      </label>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={display ?? ''}
                          onChange={(e) => handleFeatureChange(col, e.target.value)}
                          className="flex-1 bg-transparent border border-mist px-2 py-1.5 text-ink focus:border-teal outline-none"
                        />
                        {isModified && (
                          <button
                            type="button"
                            className="text-xs text-[var(--muted)] hover:text-ink px-1"
                            onClick={() => {
                              setModifiedValues((prev) => {
                                const next = { ...prev };
                                delete next[col];
                                return next;
                              });
                            }}
                          >
                            ↩
                          </button>
                        )}
                      </div>
                      <p className="text-xs text-[var(--muted)] mt-1">Original: {String(originalValue)}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Result */}
            {simulationResult && (
              <div className="border border-mist">
                <div className="px-5 py-5 border-b border-mist">
                  <p className="page-kicker mb-1">Scenario outcome</p>
                  <p className="text-sm text-ink leading-relaxed max-w-2xl">
                    {simulationResult.plain_summary}
                  </p>
                  {Array.isArray(simulationResult.key_insights) &&
                    simulationResult.key_insights.length > 0 && (
                      <ul className="mt-3 space-y-1.5 text-sm text-[var(--muted)] list-disc pl-5">
                        {simulationResult.key_insights.map((line, i) => (
                          <li key={i} className="leading-relaxed">
                            {line}
                          </li>
                        ))}
                      </ul>
                    )}
                </div>

                <div className="grid grid-cols-3 gap-px bg-mist">
                  <div className="bg-paper px-4 py-5 text-center">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Before</div>
                    <div className="font-display text-2xl font-semibold tabular-nums text-ink mt-1">
                      {isReg
                        ? Number(simulationResult.original?.predicted_value).toFixed(2)
                        : pct(simulationResult.original?.probability)}
                    </div>
                    {!isReg && simulationResult.original?.risk_level && (
                      <div className="text-xs text-[var(--muted)] mt-1">
                        {simulationResult.original.risk_level}
                      </div>
                    )}
                  </div>
                  <div className="bg-paper px-4 py-5 text-center">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">After</div>
                    <div className="font-display text-2xl font-semibold tabular-nums text-ink mt-1">
                      {isReg
                        ? Number(simulationResult.modified?.predicted_value).toFixed(2)
                        : pct(simulationResult.modified?.probability)}
                    </div>
                    {!isReg && simulationResult.modified?.risk_level && (
                      <div className="text-xs text-[var(--muted)] mt-1">
                        {simulationResult.modified.risk_level}
                      </div>
                    )}
                  </div>
                  <div className="bg-paper px-4 py-5 text-center">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Impact</div>
                    <div
                      className={`font-display text-2xl font-semibold tabular-nums mt-1 ${
                        Number(simulationResult.impact) < 0
                          ? 'text-teal'
                          : Number(simulationResult.impact) > 0
                            ? 'text-coral'
                            : 'text-ink'
                      }`}
                    >
                      {Number(simulationResult.impact) > 0 ? '+' : ''}
                      {isReg
                        ? Number(simulationResult.impact).toFixed(2)
                        : `${(Number(simulationResult.impact) * 100).toFixed(1)} pp`}
                    </div>
                    <div className="text-xs text-[var(--muted)] mt-1">
                      {simulationResult.risk_level_change || simulationResult.direction}
                    </div>
                  </div>
                </div>

                {simulationResult.change_log?.length > 0 && (
                  <div className="px-5 py-4 border-t border-mist">
                    <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] mb-2">
                      Changes applied
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {simulationResult.change_log.map((c) => (
                        <span
                          key={c.feature}
                          className="text-xs border border-mist px-2.5 py-1 text-ink"
                        >
                          {humanize(c.label || c.feature)}: {String(c.before)} → {String(c.after)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {simulationResult.driver_shift?.length > 0 && (
                  <div className="px-5 py-4 border-t border-mist">
                    <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] mb-2">
                      How the story shifts
                    </h4>
                    <ul className="space-y-2 text-sm">
                      {simulationResult.driver_shift.slice(0, 4).map((d) => (
                        <li key={d.feature} className="flex justify-between gap-3 border-b border-mist/70 pb-2">
                          <span className="text-ink">{humanize(d.label || d.feature)}</span>
                          <span className="text-[var(--muted)] tabular-nums shrink-0">
                            {d.before_impact?.toFixed?.(3) ?? d.before_impact} →{' '}
                            {d.after_impact?.toFixed?.(3) ?? d.after_impact}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {simulationResult.decision_summary && (
                  <div className="px-5 py-4 border-t border-mist text-sm">
                    <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] mb-2">
                      After this scenario
                    </h4>
                    <p className="font-medium text-ink">{simulationResult.decision_summary.strategy}</p>
                    <p className="mt-1 text-[var(--muted)] leading-relaxed">
                      {simulationResult.decision_summary.description}
                    </p>
                  </div>
                )}

                {simulationResult.recommendations?.length > 0 && (
                  <div className="px-5 py-4 border-t border-mist">
                    <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] mb-2">
                      Re-ranked next steps
                    </h4>
                    <p className="text-xs text-[var(--muted)] mb-3 leading-relaxed">
                      Impact language in these ranks is illustrative (catalog heuristics), not a
                      re-simulated outcome.
                    </p>
                    <ol className="space-y-3 text-sm">
                      {simulationResult.recommendations.slice(0, 3).map((r, i) => (
                        <li key={r.action_code || i}>
                          <div className="font-medium text-ink">
                            {i + 1}. {r.action_name || r.name}
                          </div>
                          {r.reasoning && (
                            <p className="text-[var(--muted)] mt-0.5 leading-relaxed">{r.reasoning}</p>
                          )}
                          {r.expected_probability_reduction > 0.01 && (
                            <p className="text-xs text-[var(--muted)] mt-1 tabular-nums">
                              Illustrative est. ~−
                              {(Number(r.expected_probability_reduction) * 100).toFixed(0)} pp
                            </p>
                          )}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Sidebar */}
          <aside className="lg:col-span-5 space-y-6">
            <div className="border border-mist px-4 py-4 text-sm">
              <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] mb-2">
                How to use this
              </h3>
              <ol className="list-decimal pl-4 space-y-2 text-[var(--muted)] leading-relaxed">
                <li>Change levers that raise risk in the baseline “why”.</li>
                <li>Run the scenario to see before vs after chance.</li>
                <li>Read re-ranked actions as a guide — not a guarantee.</li>
              </ol>
            </div>

            {history.length > 0 && (
              <div className="border border-mist">
                <div className="px-4 py-3 border-b border-mist">
                  <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                    Recent scenarios ({history.length})
                  </h3>
                </div>
                <ul className="divide-y divide-mist max-h-80 overflow-y-auto">
                  {history.map((item, idx) => {
                    const imp = Number(item.result?.impact ?? 0);
                    return (
                      <li key={idx}>
                        <button
                          type="button"
                          className="w-full text-left px-4 py-3 hover:bg-paper text-sm"
                          onClick={() => applyHistoryItem(item)}
                        >
                          <div className="flex justify-between gap-2">
                            <span
                              className={`font-medium tabular-nums ${
                                imp < 0 ? 'text-teal' : imp > 0 ? 'text-coral' : 'text-ink'
                              }`}
                            >
                              {imp > 0 ? '+' : ''}
                              {(imp * 100).toFixed(1)} pp
                            </span>
                            <span className="text-xs text-[var(--muted)]">
                              {new Date(item.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <div className="text-xs text-[var(--muted)] mt-1">
                            {Object.keys(item.changes || {}).length} field change
                            {Object.keys(item.changes || {}).length === 1 ? '' : 's'}
                          </div>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {simulationResult?.suggested_tweaks?.length > 0 && (
              <div className="border border-mist">
                <div className="px-4 py-3 border-b border-mist">
                  <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                    Try dialing
                  </h3>
                </div>
                <ul className="divide-y divide-mist">
                  {simulationResult.suggested_tweaks.map((t, i) => (
                    <li key={`${t.feature}-${i}`} className="px-4 py-3 text-sm">
                      <div className="font-medium text-ink">{humanize(t.label || t.feature)}</div>
                      {t.hint && (
                        <p className="text-xs text-[var(--muted)] mt-1 leading-relaxed">{t.hint}</p>
                      )}
                      {t.suggested_value !== undefined && (
                        <button
                          type="button"
                          className="mt-2 text-xs text-teal hover:underline"
                          onClick={() => applySuggestion(t.feature, t.suggested_value)}
                        >
                          Set to {String(t.suggested_value)}
                        </button>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </aside>
        </div>
      )}
    </div>
  );
}
