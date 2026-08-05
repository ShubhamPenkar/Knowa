import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { PredictionPanel } from '../components/PredictionPanel';

export default function ProjectDetail() {
  const { id } = useParams();
  const { token, user } = useAuth();
  const navigate = useNavigate();
  const [project, setProject] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState('');
  const [predicting, setPredicting] = useState(false);
  const [dataPreview, setDataPreview] = useState(null);
  const [selectedRowIdx, setSelectedRowIdx] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [testData, setTestData] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  const [predictError, setPredictError] = useState('');
  const [knownOutcome, setKnownOutcome] = useState(null);
  const [spotCheck, setSpotCheck] = useState(null);
  const [spotLoading, setSpotLoading] = useState(false);
  const [feedbackSummary, setFeedbackSummary] = useState(null);
  const [ledger, setLedger] = useState(null);

  useEffect(() => {
    if (token && id) fetchProject();
  }, [id, token]);

  useEffect(() => {
    if (project?.status === 'trained' || project?.status === 'ready') {
      fetchTestData();
      fetchSpotCheck();
      fetchFeedbackSummary();
      fetchLedger();
    }
  }, [project?.status, id, token]);

  const fetchProject = async () => {
    try {
      const res = await fetch(`/api/projects/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setProject(data);
        if (data.dataset_id) fetchDataPreview(data.dataset_id);
      }
    } catch (err) {
      console.error('Error fetching project:', err);
    }
    setLoading(false);
  };

  const fetchDataPreview = async (datasetId) => {
    try {
      const res = await fetch(`/api/datasets/${datasetId}/preview?rows=20`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setDataPreview(await res.json());
    } catch (err) {
      console.error(err);
    }
  };

  const fetchTestData = async () => {
    try {
      const res = await fetch(`/api/projects/${id}/test-data?limit=30`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setTestData(await res.json());
    } catch (err) {
      console.error(err);
    }
  };

  const fetchSpotCheck = async () => {
    setSpotLoading(true);
    try {
      const res = await fetch(`/api/projects/${id}/spot-check?limit=50`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setSpotCheck(await res.json());
      else setSpotCheck(null);
    } catch {
      setSpotCheck(null);
    }
    setSpotLoading(false);
  };

  const fetchFeedbackSummary = async () => {
    try {
      const res = await fetch(`/api/projects/${id}/feedback-summary`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setFeedbackSummary(await res.json());
      else setFeedbackSummary(null);
    } catch {
      setFeedbackSummary(null);
    }
  };

  const fetchLedger = async () => {
    try {
      const res = await fetch(`/api/projects/${id}/decisions?limit=30`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setLedger(await res.json());
      else setLedger(null);
    } catch {
      setLedger(null);
    }
  };

  const checkInDecision = async (decisionId) => {
    try {
      const res = await fetch(`/api/projects/${id}/decisions/${decisionId}/check-in`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          notes: 'Manual check-in from project ledger.',
          schedule_next: true,
        }),
      });
      if (res.ok) fetchLedger();
    } catch (err) {
      console.error(err);
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setTrainError('');
    try {
      const res = await fetch(`/api/projects/${id}/train`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        await fetchProject();
        await fetchSpotCheck();
      } else {
        const data = await res.json();
        setTrainError(data.detail || 'Training failed');
      }
    } catch (err) {
      setTrainError('Training failed: ' + err.message);
    }
    setTraining(false);
  };

  const rows = testData?.rows || dataPreview?.rows || dataPreview?.data || [];
  const columnSource =
    (testData?.rows?.[0] && Object.keys(testData.rows[0])) ||
    dataPreview?.columns ||
    [];
  const featurePreviewCols = columnSource
    .filter((col) => col !== project?.target_column)
    .slice(0, 5);

  /** Stable customer/row id for feedback join later. */
  const resolveEntityId = (row, idx) => {
    if (!row) return `row-${idx}`;
    const preferred = [
      'customerID',
      'CustomerID',
      'customer_id',
      'customerId',
      'entity_id',
      'EntityId',
      'account_id',
      'AccountID',
      'user_id',
      'UserID',
      'id',
      'ID',
    ];
    for (const k of preferred) {
      if (row[k] != null && String(row[k]).trim() !== '') return String(row[k]);
    }
    const lowerMap = Object.fromEntries(
      Object.keys(row).map((k) => [k.toLowerCase(), k])
    );
    for (const k of preferred) {
      const real = lowerMap[k.toLowerCase()];
      if (real != null && row[real] != null && String(row[real]).trim() !== '') {
        return String(row[real]);
      }
    }
    return `row-${idx}`;
  };

  const formatApiError = (detail) => {
    if (detail == null) return 'Request failed';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
    }
    if (typeof detail === 'object') {
      if (detail.message) {
        const bits = [detail.message];
        if (detail.missing_features?.length) {
          bits.push(`Missing: ${detail.missing_features.slice(0, 8).join(', ')}`);
        }
        if (detail.empty_features?.length) {
          bits.push(`Empty: ${detail.empty_features.slice(0, 8).join(', ')}`);
        }
        return bits.join(' ');
      }
      return JSON.stringify(detail);
    }
    return String(detail);
  };

  const handlePredictRow = async (row, idx) => {
    setSelectedRowIdx(idx);
    setPredicting(true);
    setPredictError('');
    setPrediction(null);
    const actualKnown =
      project?.target_column != null
        ? String(row[project.target_column]) === String(project.target_positive_label)
        : null;
    setKnownOutcome(actualKnown);

    const features = {};
    project.feature_columns.forEach((col) => {
      features[col] = row[col];
    });
    const entity_id = resolveEntityId(row, idx);

    try {
      const res = await fetch(`/api/projects/${id}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          features,
          entity_id,
          include_explanations: true,
          include_recommendations: true,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setPrediction(data);
      } else {
        setPredictError(formatApiError(data.detail) || 'Prediction failed');
      }
    } catch (err) {
      setPredictError('Network error: ' + err.message);
    }
    setPredicting(false);
  };

  const handleDeleteProject = async () => {
    if (!window.confirm(`Delete project "${project.name}"?`)) return;
    setDeleting(true);
    setDeleteError('');
    try {
      const res = await fetch(`/api/projects/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        navigate('/projects');
        return;
      }
      const data = await res.json();
      setDeleteError(data.detail || 'Failed to delete project');
    } catch {
      setDeleteError('Network error while deleting project');
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="page flex justify-center items-center min-h-[40vh]">
        <div className="h-6 w-6 border-2 border-teal border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!project) {
    return (
      <div className="page">
        <p className="text-[var(--muted)]">Project not found</p>
      </div>
    );
  }

  const isReady = project.status === 'trained' || project.status === 'ready';
  const outcomeLabel = (project.target_description || project.target_column || 'outcome')
    .replace(/[_-]+/g, ' ')
    .trim();

  const readiness =
    isReady && project.active_model
      ? (() => {
          const m = project.active_model;
          if (project.problem_type === 'regression') {
            if (m.r2_score == null) return { label: 'Ready', detail: 'You can review cases and plan actions.' };
            const r2 = Number(m.r2_score);
            if (r2 >= 0.5)
              return { label: 'Strong guide', detail: 'Patterns look reliable enough to prioritize work.' };
            if (r2 >= 0.2)
              return { label: 'Useful guide', detail: 'Directionally helpful — double-check edge cases.' };
            return { label: 'Rough guide', detail: 'Use as a hint, not the only signal.' };
          }
          const auc = m.auc_roc != null ? Number(m.auc_roc) : m.accuracy != null ? Number(m.accuracy) : null;
          if (auc == null) return { label: 'Ready', detail: 'You can review cases and plan actions.' };
          if (auc >= 0.8)
            return { label: 'Strong guide', detail: 'Good at ranking who needs attention first.' };
          if (auc >= 0.7)
            return { label: 'Solid guide', detail: 'Solid enough to drive a daily priority list.' };
          if (auc >= 0.6)
            return { label: 'Useful guide', detail: 'Helpful direction — confirm before big spends.' };
          return { label: 'Rough guide', detail: 'Treat as a first filter; add human judgment.' };
        })()
      : null;

  return (
    <div className="page">
      <button type="button" onClick={() => navigate('/projects')} className="btn-ghost -ml-2 mb-4 text-sm">
        ← Projects
      </button>

      <div className="page-header">
        <div>
          <p className="page-kicker">Project · Decisions</p>
          <h1 className="page-title">{project.name}</h1>
          {project.description && <p className="page-sub">{project.description}</p>}
          {!project.description && (
            <p className="page-sub">
              Spot who&apos;s at risk of {outcomeLabel.toLowerCase()}, understand why, and choose a next step.
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className={`badge ${
              isReady
                ? 'bg-teal-soft/50 text-ink border border-teal/20'
                : project.status === 'error'
                  ? 'bg-coral-soft text-ink border border-coral/30'
                  : 'bg-mist text-ink'
            }`}
          >
            {isReady ? 'Ready to use' : project.status === 'error' ? 'Needs attention' : 'Not prepared yet'}
          </span>
          {(user?.role === 'owner' || user?.role === 'admin') && (
            <button type="button" onClick={handleDeleteProject} disabled={deleting} className="btn-danger text-xs py-1.5">
              {deleting ? 'Deleting…' : 'Delete'}
            </button>
          )}
        </div>
      </div>

      {deleteError && (
        <div className="mb-4 text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control">{deleteError}</div>
      )}

      <dl className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-3 mb-8 text-sm border-y border-mist py-4">
        <div>
          <dt className="text-[11px] uppercase tracking-wide text-[var(--muted)]">We&apos;re watching</dt>
          <dd className="font-medium text-ink mt-0.5 capitalize">{outcomeLabel}</dd>
        </div>
        <div>
          <dt className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Signals used</dt>
          <dd className="font-medium text-ink mt-0.5">{project.feature_columns?.length || 0} factors</dd>
        </div>
        <div>
          <dt className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Kind of question</dt>
          <dd className="font-medium text-ink mt-0.5">
            {project.problem_type === 'regression' ? 'How much / how many' : 'Will it happen?'}
          </dd>
        </div>
        <div>
          <dt className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Last prepared</dt>
          <dd className="font-medium text-ink mt-0.5">
            {project.active_model?.version
              ? project.active_model.trained_at
                ? new Date(project.active_model.trained_at).toLocaleDateString()
                : project.active_model.version
              : '—'}
          </dd>
        </div>
      </dl>

      {(project.status === 'created' || project.status === 'draft') && (
        <section className="surface p-6 mb-8">
          <h2 className="font-display text-lg font-semibold text-ink">Prepare this project</h2>
          <p className="text-sm text-[var(--muted)] mt-1 mb-4 max-w-lg">
            Learn patterns from your data so you can rank who needs care for {outcomeLabel.toLowerCase()} and why.
          </p>
          {trainError && (
            <div className="mb-4 text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control">
              {trainError}
            </div>
          )}
          <button type="button" onClick={handleTrain} disabled={training} className="btn-primary">
            {training ? 'Preparing…' : 'Prepare project'}
          </button>
        </section>
      )}

      {isReady && readiness && (
        <section className="mb-8">
          <div className="flex flex-wrap items-end justify-between gap-3 mb-4">
            <div>
              <h2 className="font-display text-lg font-semibold text-ink">How to use this</h2>
              <p className="text-sm text-[var(--muted)]">Guidance quality for day-to-day decisions</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Link to={`/projects/${id}/whatif`} className="btn-secondary text-sm">
                Try a scenario
              </Link>
              <button type="button" onClick={handleTrain} disabled={training} className="btn-ghost text-sm">
                {training ? 'Refreshing…' : 'Refresh with latest data'}
              </button>
            </div>
          </div>
          <div className="border border-mist px-5 py-5 max-w-2xl">
            <p className="font-display text-xl font-semibold text-ink">{readiness.label}</p>
            <p className="mt-2 text-sm text-[var(--muted)] leading-relaxed">{readiness.detail}</p>
            <p className="mt-3 text-sm text-ink">
              Next: pick a person or account in the list, read why they stand out, and choose an action — or simulate one first.
            </p>
          </div>
        </section>
      )}

      {/* Spot-check: known outcomes vs predictions on held-out rows */}
      {isReady && project.problem_type !== 'regression' && (
        <section className="mb-8 border border-mist">
          <div className="px-5 py-4 border-b border-mist flex flex-wrap items-center justify-between gap-2">
            <div>
              <p className="page-kicker">Verification</p>
              <h2 className="font-display text-lg font-semibold text-ink mt-0.5">
                Does this match known outcomes?
              </h2>
            </div>
            <button
              type="button"
              onClick={fetchSpotCheck}
              disabled={spotLoading}
              className="btn-ghost text-sm"
            >
              {spotLoading ? 'Checking…' : 'Re-run check'}
            </button>
          </div>
          <div className="px-5 py-5">
            {spotLoading && !spotCheck && (
              <p className="text-sm text-[var(--muted)]">Comparing held-out labels to predictions…</p>
            )}
            {spotCheck && spotCheck.supported === false && (
              <p className="text-sm text-[var(--muted)]">{spotCheck.message}</p>
            )}
            {spotCheck && spotCheck.n === 0 && (
              <p className="text-sm text-[var(--muted)]">
                {spotCheck.message || 'No held-out rows to score yet.'}
              </p>
            )}
            {spotCheck && spotCheck.n > 0 && (
              <>
                <p className="font-display text-xl font-semibold text-ink">{spotCheck.grade_label}</p>
                <p className="mt-2 text-sm text-[var(--muted)] leading-relaxed max-w-2xl">
                  {spotCheck.grade_detail}
                </p>
                <p className="mt-3 text-sm text-ink max-w-2xl">{spotCheck.plain_summary}</p>
                <div className="mt-5 grid grid-cols-2 md:grid-cols-4 gap-px bg-mist border border-mist">
                  <div className="bg-paper px-3 py-3">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Yes/No match</div>
                    <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                      {Math.round((spotCheck.agree_rate || 0) * 100)}%
                    </div>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      {spotCheck.agree_count}/{spotCheck.n} held-out rows
                    </p>
                  </div>
                  <div className="bg-paper px-3 py-3">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                      High-risk → true Yes
                    </div>
                    <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                      {spotCheck.high_risk_precision != null
                        ? `${Math.round(spotCheck.high_risk_precision * 100)}%`
                        : '—'}
                    </div>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      {spotCheck.flagged_high || 0} flagged high
                    </p>
                  </div>
                  <div className="bg-paper px-3 py-3">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                      Low-risk → true No
                    </div>
                    <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                      {spotCheck.low_risk_true_negative_rate != null
                        ? `${Math.round(spotCheck.low_risk_true_negative_rate * 100)}%`
                        : '—'}
                    </div>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      {spotCheck.calm_low || 0} calm calls
                    </p>
                  </div>
                  <div className="bg-paper px-3 py-3">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Soft ranges</div>
                    <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                      {Math.round((spotCheck.soft_signal_share || 0) * 100)}%
                    </div>
                    <p className="text-xs text-[var(--muted)] mt-1">of scored cases</p>
                  </div>
                </div>
                <p className="mt-4 text-xs text-[var(--muted)] max-w-2xl">
                  Soft ranges flag mid-priority scores or nearly open residual bands — not every fat
                  residual bar (those often look wide even when ranking is clear). Held-out check
                  answers “is the compass pointing the right way?” — not whether every % is destiny.
                </p>
              </>
            )}
          </div>
        </section>
      )}

      {isReady && project.problem_type !== 'regression' && feedbackSummary && (
        <section className="mb-8 border border-mist">
          <div className="px-5 py-4 border-b border-mist">
            <p className="page-kicker">Learning log (A7)</p>
            <h2 className="font-display text-lg font-semibold text-ink mt-0.5">
              Outcomes you recorded
            </h2>
          </div>
          <div className="px-5 py-5">
            <p className="text-sm text-ink leading-relaxed max-w-2xl">
              {feedbackSummary.plain_summary}
            </p>
            <div className="mt-4 grid sm:grid-cols-3 gap-px bg-mist">
              <div className="bg-paper px-3 py-3">
                <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Logged</div>
                <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                  {feedbackSummary.with_feedback || 0}
                </div>
                <p className="text-xs text-[var(--muted)] mt-1">
                  of {feedbackSummary.total_predictions || 0} predictions
                </p>
              </div>
              <div className="bg-paper px-3 py-3">
                <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                  Model match
                </div>
                <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                  {feedbackSummary.model_match_rate != null
                    ? `${Math.round(feedbackSummary.model_match_rate * 100)}%`
                    : '—'}
                </div>
                <p className="text-xs text-[var(--muted)] mt-1">on known Yes/No logs</p>
              </div>
              <div className="bg-paper px-3 py-3">
                <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                  Actions tracked
                </div>
                <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                  {Object.keys(feedbackSummary.action_effectiveness || {}).length}
                </div>
                <p className="text-xs text-[var(--muted)] mt-1">types with outcome + action</p>
              </div>
            </div>
            {Array.isArray(feedbackSummary.action_effectiveness_ranked) &&
              feedbackSummary.action_effectiveness_ranked.length > 0 && (
                <div className="mt-4 border-t border-mist pt-4">
                  <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] mb-2">
                    Action hit rate
                  </div>
                  <ul className="space-y-2">
                    {feedbackSummary.action_effectiveness_ranked.slice(0, 5).map((a) => (
                      <li
                        key={a.action_code}
                        className="flex flex-wrap items-baseline justify-between gap-2 text-sm"
                      >
                        <span className="text-ink font-medium">
                          {a.action_name || a.action_code}
                        </span>
                        <span className="tabular-nums text-[var(--muted)]">
                          {a.success_n}/{a.n} avoided
                          {!a.reliable ? ' · small n' : ''}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
          </div>
        </section>
      )}

      {isReady && project.problem_type !== 'regression' && (
        <section className="mb-8 border border-mist">
          <div className="px-5 py-4 border-b border-mist">
            <p className="page-kicker">Decision ledger (B3)</p>
            <h2 className="font-display text-lg font-semibold text-ink mt-0.5">
              Committed actions
            </h2>
          </div>
          <div className="px-5 py-5">
            <p className="text-sm text-ink leading-relaxed max-w-2xl">
              {ledger?.plain_summary ||
                'No decisions committed yet. Open a case, pick an action, and commit it with a 30/60/90 recheck.'}
            </p>
            {Array.isArray(ledger?.decisions) && ledger.decisions.length > 0 ? (
              <ul className="mt-4 divide-y divide-mist border border-mist">
                {ledger.decisions.slice(0, 12).map((d) => (
                  <li
                    key={d.id}
                    className="px-4 py-3 flex flex-wrap items-start justify-between gap-3 text-sm"
                  >
                    <div className="min-w-0">
                      <div className="font-medium text-ink">{d.action_name}</div>
                      <p className="text-xs text-[var(--muted)] mt-1 leading-relaxed">
                        {d.status}
                        {d.probability_at_commit != null
                          ? ` · at ${(d.probability_at_commit * 100).toFixed(0)}%`
                          : ''}
                        {d.recheck_at
                          ? ` · recheck ${String(d.recheck_at).slice(0, 10)}`
                          : ''}
                        {d.due_for_recheck ? ' · due now' : ''}
                      </p>
                      {d.autopsy_narrative && (
                        <p className="text-xs text-[var(--muted)] mt-2 max-w-xl leading-relaxed">
                          {d.autopsy_narrative}
                        </p>
                      )}
                    </div>
                    {d.status !== 'closed' && d.status !== 'cancelled' && (
                      <button
                        type="button"
                        onClick={() => checkInDecision(d.id)}
                        className="shrink-0 text-xs px-3 py-1.5 border border-mist hover:border-teal text-ink"
                      >
                        Check in
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        </section>
      )}

      {isReady && rows.length > 0 && (
        <section className="grid lg:grid-cols-5 gap-8">
          <div className="lg:col-span-3 min-w-0">
            <h2 className="font-display text-lg font-semibold text-ink mb-1">Who to review</h2>
            <p className="text-sm text-[var(--muted)] mb-4">
              Known outcome is from your data. Click a row to compare it with the estimated chance.
            </p>
            <div className="surface overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Known outcome</th>
                    {featurePreviewCols.map((col) => (
                      <th key={col} className="truncate max-w-[6rem]">
                        {String(col).replace(/[_-]+/g, ' ')}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, 15).map((row, idx) => {
                    const actual =
                      String(row[project.target_column]) === String(project.target_positive_label);
                    const selected = selectedRowIdx === idx;
                    return (
                      <tr
                        key={idx}
                        onClick={() => handlePredictRow(row, idx)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            handlePredictRow(row, idx);
                          }
                        }}
                        tabIndex={0}
                        role="button"
                        aria-pressed={selected}
                        className={`cursor-pointer ${selected ? 'bg-teal-soft/30' : ''}`}
                      >
                        <td className="text-[var(--muted)]">{idx + 1}</td>
                        <td>
                          <span
                            className={`badge ${
                              actual ? 'bg-coral-soft text-ink' : 'bg-mist text-ink'
                            }`}
                          >
                            {actual ? 'Yes' : 'No'}
                          </span>
                        </td>
                        {featurePreviewCols.map((col) => (
                          <td key={col} className="truncate max-w-[6rem] text-[var(--muted)]">
                            {String(row[col] ?? '')}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="lg:col-span-2">
            <div className="lg:sticky lg:top-6 space-y-3">
              {predictError && (
                <div className="text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control">
                  {predictError}
                </div>
              )}
              {predicting && (
                <div className="surface p-6 text-sm text-[var(--muted)]">Building the case view…</div>
              )}
              {!predicting && !prediction && (
                <div className="surface p-6">
                  <p className="page-kicker">Case brief</p>
                  <h3 className="font-display text-lg font-semibold text-ink mt-1">Select someone</h3>
                  <p className="text-sm text-muted mt-2 leading-relaxed">
                    You&apos;ll see estimated chance of {outcomeLabel.toLowerCase()}, known outcome (if any), why,
                    and next steps.
                  </p>
                  <div className="mt-6 h-2 rounded-[2px] bg-mist" aria-hidden="true" />
                  <p className="mt-3 text-xs text-muted">Waiting for a selection</p>
                </div>
              )}
              {!predicting && prediction && (
                <PredictionPanel
                  result={prediction}
                  knownOutcome={knownOutcome}
                  outcomeYesLabel="Yes"
                  outcomeNoLabel="No"
                  projectId={id}
                  authToken={token}
                  onFeedbackSaved={() => fetchFeedbackSummary()}
                  onDecisionCommitted={() => fetchLedger()}
                  simulateHref={`/projects/${id}/whatif${
                    selectedRowIdx != null ? `?row=${selectedRowIdx}` : ''
                  }`}
                  simulateLabel="What if we changed something?"
                />
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
