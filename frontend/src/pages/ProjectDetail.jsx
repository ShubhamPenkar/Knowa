import { useState, useEffect, useMemo, useRef } from 'react';
import { useParams, useNavigate, Link, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { PredictionPanel } from '../components/PredictionPanel';
import Spinner from '../components/common/Spinner';

export default function ProjectDetail() {
  const { id } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
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
  const [extraDecision, setExtraDecision] = useState(null);
  const [focusDecisionId, setFocusDecisionId] = useState(null);
  const [checkInTarget, setCheckInTarget] = useState(null);
  const [checkInOutcome, setCheckInOutcome] = useState('');
  const [checkInNotes, setCheckInNotes] = useState('');
  const [checkInInterval, setCheckInInterval] = useState(30);
  const [checkInMode, setCheckInMode] = useState('reschedule'); // reschedule | close | keep
  const [checkInSaving, setCheckInSaving] = useState(false);
  const [checkInError, setCheckInError] = useState('');
  const [lastAutopsy, setLastAutopsy] = useState(null);
  const [showAllDecisions, setShowAllDecisions] = useState(false);
  const deepLinkRef = useRef({ decision: null, prediction: null });
  const fromFollowUps =
    searchParams.get('from') === 'follow-ups' || searchParams.get('from') === 'priorities';

  useEffect(() => {
    deepLinkRef.current = { decision: null, prediction: null };
    setExtraDecision(null);
    setFocusDecisionId(null);
    setPrediction(null);
    setSelectedRowIdx(null);
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
      const res = await fetch(`/api/projects/${id}/decisions?limit=80`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setLedger(await res.json());
      else setLedger(null);
    } catch {
      setLedger(null);
    }
  };

  const openCheckIn = (decision) => {
    setCheckInTarget(decision);
    setCheckInOutcome(decision.actual_outcome || '');
    setCheckInNotes('');
    setCheckInInterval(decision.recheck_interval_days || 30);
    setCheckInMode('reschedule');
    setCheckInError('');
    setLastAutopsy(null);
  };

  // Priorities → case deep-link (?prediction=)
  useEffect(() => {
    const predictionId = searchParams.get('prediction');
    if (!token || !id || !predictionId) return;
    if (deepLinkRef.current.prediction === predictionId) return;
    if (!(project?.status === 'trained' || project?.status === 'ready')) return;

    deepLinkRef.current.prediction = predictionId;
    let cancelled = false;
    (async () => {
      setPredicting(true);
      setPredictError('');
      try {
        const res = await fetch(`/api/projects/${id}/predictions/${predictionId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json().catch(() => ({}));
        if (cancelled) return;
        if (!res.ok) {
          setPredictError(
            typeof data.detail === 'string' ? data.detail : 'Could not open that case'
          );
          setPredicting(false);
          return;
        }
        setPrediction(data);
        setSelectedRowIdx(null);
        setKnownOutcome(null);
        requestAnimationFrame(() => {
          document
            .getElementById('case-brief')
            ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      } catch (err) {
        if (!cancelled) setPredictError(err?.message || 'Could not open that case');
      }
      if (!cancelled) setPredicting(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [searchParams, id, token, project?.status]);

  // Priorities → follow-up deep-link (?decision=&checkin=1)
  useEffect(() => {
    const decisionId = searchParams.get('decision');
    if (!token || !id || !decisionId || !ledger) return;
    if (deepLinkRef.current.decision === decisionId) return;

    deepLinkRef.current.decision = decisionId;
    setFocusDecisionId(decisionId);
    const wantCheckin = searchParams.get('checkin') === '1';

    let cancelled = false;
    (async () => {
      let target = (ledger.decisions || []).find((d) => d.id === decisionId) || null;
      if (!target) {
        try {
          const res = await fetch(`/api/projects/${id}/decisions/${decisionId}`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (res.ok) {
            target = await res.json();
            if (!cancelled) setExtraDecision(target);
          }
        } catch {
          /* ignore — highlight still attempts */
        }
      }
      if (cancelled) return;

      requestAnimationFrame(() => {
        document
          .getElementById(`decision-${decisionId}`)
          ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      });

      if (
        wantCheckin &&
        target &&
        target.status !== 'closed' &&
        target.status !== 'cancelled'
      ) {
        openCheckIn(target);
        const next = new URLSearchParams(searchParams);
        next.delete('checkin');
        setSearchParams(next, { replace: true });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [ledger, searchParams, id, token]);

  const closeCheckIn = () => {
    if (checkInSaving) return;
    setCheckInTarget(null);
    setCheckInError('');
  };

  const submitCheckIn = async () => {
    if (!checkInTarget) return;
    setCheckInSaving(true);
    setCheckInError('');
    try {
      const body = {
        notes: checkInNotes.trim() || null,
        actual_outcome: checkInOutcome || null,
        close: checkInMode === 'close',
        schedule_next: checkInMode === 'reschedule',
      };
      if (checkInMode === 'reschedule') {
        body.recheck_interval_days = Number(checkInInterval) || 30;
      }
      const res = await fetch(`/api/projects/${id}/decisions/${checkInTarget.id}/check-in`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail;
        let message = 'Could not save update';
        if (typeof detail === 'string') message = detail;
        else if (Array.isArray(detail) && detail.length) {
          message = detail
            .map((e) => (typeof e === 'string' ? e : e?.msg || JSON.stringify(e)))
            .join('; ');
        }
        setCheckInError(message);
        return;
      }
      setLastAutopsy(data.autopsy_narrative || null);
      await fetchLedger();
      await fetchFeedbackSummary();
      setCheckInTarget(null);
    } catch (err) {
      console.error(err);
      setCheckInError('Could not save update');
    } finally {
      setCheckInSaving(false);
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
  /** Mobile: keep table scannable — outcome + first signal only. */
  const mobileFeatureCols = featurePreviewCols.slice(0, 1);

  const allDecisions = useMemo(() => {
    const all = [...(ledger?.decisions || [])];
    if (extraDecision && !all.some((d) => d.id === extraDecision.id)) {
      all.unshift(extraDecision);
    }
    return all;
  }, [ledger, extraDecision]);

  const visibleDecisions = useMemo(() => {
    const cap = showAllDecisions ? 80 : 12;
    if (!focusDecisionId) return allDecisions.slice(0, cap);
    const focused = allDecisions.find((d) => d.id === focusDecisionId);
    const rest = allDecisions.filter((d) => d.id !== focusDecisionId).slice(0, cap - 1);
    return focused ? [focused, ...rest] : allDecisions.slice(0, cap);
  }, [allDecisions, focusDecisionId, showAllDecisions]);

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
        requestAnimationFrame(() => {
          document
            .getElementById('case-brief-panel')
            ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        });
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
        <Spinner />
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

  const checkInDeepLink =
    Boolean(focusDecisionId) && searchParams.get('checkin') === '1' && !checkInTarget;

  return (
    <div className="page">
      {checkInDeepLink && (
        <div className="sticky top-14 lg:top-0 z-30 -mx-6 md:-mx-8 lg:-mx-10 mb-4 px-6 md:px-8 lg:px-10 py-2.5 border-b border-teal/30 bg-teal-soft/50 backdrop-blur-sm flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm text-ink">
            Updating this follow-up — scroll to <span className="font-medium">Your follow-ups</span> or open the case brief.
          </p>
          <a href="#follow-ups" className="text-xs font-medium text-teal hover:underline shrink-0">
            Jump to follow-ups
          </a>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 mb-4 -ml-2">
        {fromFollowUps ? (
          <Link to="/follow-ups" className="btn-ghost text-sm">
            ← Follow-ups
          </Link>
        ) : (
          <button type="button" onClick={() => navigate('/projects')} className="btn-ghost text-sm">
            ← Projects
          </button>
        )}
        <Link to={`/cases?project=${id}`} className="btn-ghost text-sm text-[var(--muted)]">
          Cases
        </Link>
      </div>

      <div className="page-header">
        <div>
          <p className="page-kicker">Project</p>
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
          {isReady && (
            <>
              <Link to={`/cases?project=${id}`} className="btn-secondary text-sm">
                Open cases
              </Link>
              <Link to={`/whatif/${id}`} className="btn-ghost text-sm">
                What-if
              </Link>
            </>
          )}
        </div>
      </div>

      {deleteError && (
        <div className="mb-4 text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control">{deleteError}</div>
      )}

      {(project.status === 'created' || project.status === 'draft') && (
        <section className="surface p-6 mb-8">
          <h2 className="font-display text-lg font-semibold text-ink">Turn on guidance</h2>
          <p className="text-sm text-[var(--muted)] mt-1 mb-4 max-w-lg">
            We&apos;ll learn from your data who needs attention for {outcomeLabel.toLowerCase()} — and
            why — so your team can act.
          </p>
          {trainError && (
            <div className="mb-4 text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control">
              {trainError}
            </div>
          )}
          <button type="button" onClick={handleTrain} disabled={training} className="btn-primary">
            {training ? 'Getting ready…' : 'Get started'}
          </button>
        </section>
      )}

      {isReady && (
        <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
          <p className="text-sm text-[var(--muted)]">
            {readiness ? (
              <>
                <span className="text-ink font-medium">{readiness.label}</span>
                <span> — {readiness.detail}</span>
              </>
            ) : (
              'Open a person or account to see risk and next steps.'
            )}
          </p>
          <button type="button" onClick={handleTrain} disabled={training} className="btn-ghost text-sm">
            {training ? 'Updating…' : 'Update with latest data'}
          </button>
        </div>
      )}

      {/* Follow-ups always visible when ready (non-regression) */}
      {isReady && project.problem_type !== 'regression' && (
        <section id="follow-ups" className="mb-8 border border-mist">
          <div className="px-5 py-4 border-b border-mist flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="font-display text-lg font-semibold text-ink">Your follow-ups</h2>
              <p className="text-sm text-[var(--muted)] mt-1">
                {ledger?.plain_summary ||
                  'Actions you saved from case briefs — check in when they come due.'}
              </p>
            </div>
            {allDecisions.length > 12 && (
              <button
                type="button"
                onClick={() => setShowAllDecisions((v) => !v)}
                className="btn-ghost text-sm shrink-0"
              >
                {showAllDecisions ? 'Show fewer' : `Show all (${allDecisions.length})`}
              </button>
            )}
          </div>
          {lastAutopsy && (
            <div className="px-5 py-3 border-b border-mist bg-teal-soft/40 text-sm text-ink">
              <div className="text-[11px] uppercase tracking-wide text-teal mb-1">Latest check-in</div>
              <p>{lastAutopsy}</p>
            </div>
          )}
          {visibleDecisions.length === 0 ? (
            <div className="px-5 py-8 text-center">
              <p className="text-sm text-ink font-medium">No follow-ups saved yet</p>
              <p className="text-sm text-[var(--muted)] mt-2 max-w-md mx-auto">
                Open a case below, pick an action, and save a follow-up. It will show up here and under Follow-ups.
              </p>
              <a href="#case-brief" className="inline-flex mt-4 text-sm font-medium text-teal hover:underline">
                Open a case and save a follow-up →
              </a>
            </div>
          ) : (
            <ul className="divide-y divide-mist">
              {visibleDecisions.map((d) => {
                const statusLabel =
                  d.status === 'committed'
                    ? 'Scheduled'
                    : d.status === 'checking'
                      ? 'In review'
                      : d.status === 'closed'
                        ? 'Done'
                        : d.status === 'cancelled'
                          ? 'Cancelled'
                          : d.status === 'proposed'
                            ? 'Proposed'
                            : String(d.status || '').replace(/_/g, ' ');
                const focused = focusDecisionId === d.id;
                return (
                  <li
                    id={`decision-${d.id}`}
                    key={d.id}
                    className={`px-5 py-3 flex flex-wrap items-start justify-between gap-3 text-sm transition-colors ${
                      focused ? 'bg-teal-soft/35 ring-1 ring-inset ring-teal/30' : ''
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="font-medium text-ink">{d.action_name}</div>
                      <p className="text-xs text-[var(--muted)] mt-1">
                        {statusLabel}
                        {d.recheck_at ? ` · check back ${String(d.recheck_at).slice(0, 10)}` : ''}
                        {d.due_for_recheck ? ' · due now' : ''}
                        {d.actual_outcome ? ` · outcome: ${d.actual_outcome}` : ''}
                        {d.checkin_count > 0 ? ` · ${d.checkin_count} update(s)` : ''}
                      </p>
                      {d.autopsy_narrative && (
                        <p className="text-xs text-[var(--muted)] mt-2 max-w-2xl leading-relaxed">
                          {d.autopsy_narrative}
                        </p>
                      )}
                    </div>
                    <div className="shrink-0 flex flex-wrap gap-2">
                      {d.prediction_id && (
                        <Link
                          to={`/cases?project=${id}&prediction=${d.prediction_id}`}
                          className="text-xs px-3 py-1.5 border border-mist hover:border-teal text-ink rounded-control"
                        >
                          Open case
                        </Link>
                      )}
                      {d.status !== 'closed' && d.status !== 'cancelled' && (
                        <button
                          type="button"
                          onClick={() => openCheckIn(d)}
                          className="text-xs px-3 py-1.5 border border-mist hover:border-teal text-ink rounded-control"
                        >
                          Update
                        </button>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      )}

      {isReady && (rows.length > 0 || prediction || predicting || predictError) && (
        <section id="case-brief" className="grid lg:grid-cols-5 gap-8 mb-10">
          {/* On mobile: brief first when a case is open */}
          <div
            id="case-brief-panel"
            className={`lg:col-span-2 ${prediction || predicting ? 'order-1' : 'order-2'} lg:order-2`}
          >
            <div className="lg:sticky lg:top-6 space-y-3">
              {predictError && (
                <div className="text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control">
                  {predictError}
                </div>
              )}
              {predicting && (
                <div className="surface p-6 text-sm text-[var(--muted)] flex items-center gap-3">
                  <Spinner className="h-4 w-4" /> Opening brief…
                </div>
              )}
              {!predicting && !prediction && (
                <div className="surface p-6">
                  <p className="page-kicker">Brief</p>
                  <h3 className="font-display text-lg font-semibold text-ink mt-1">
                    Select someone to begin
                  </h3>
                  <p className="text-sm text-[var(--muted)] mt-2 leading-relaxed">
                    You&apos;ll see how likely {outcomeLabel.toLowerCase()} is, what&apos;s driving
                    it, and what to do next.
                  </p>
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
                  simulateHref={`/whatif/${id}${
                    selectedRowIdx != null ? `?row=${selectedRowIdx}` : ''
                  }`}
                  simulateLabel="Explore a what-if"
                />
              )}
            </div>
          </div>

          <div className={`lg:col-span-3 min-w-0 ${prediction || predicting ? 'order-2' : 'order-1'} lg:order-1`}>
            <h2 className="font-display text-lg font-semibold text-ink mb-1">Who needs attention?</h2>
            <p className="text-sm text-[var(--muted)] mb-4">
              {rows.length > 0
                ? 'Click a row to open their brief. “In data” is the labeled outcome in your dataset (when available).'
                : 'Opened from Priorities — full case brief is beside this list on larger screens.'}
            </p>
            {rows.length > 0 ? (
              <div className="surface overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>In data</th>
                      {mobileFeatureCols.map((col) => (
                        <th key={col} className="truncate max-w-[7rem] md:hidden">
                          {String(col).replace(/[_-]+/g, ' ')}
                        </th>
                      ))}
                      {featurePreviewCols.map((col) => (
                        <th key={`d-${col}`} className="truncate max-w-[6rem] hidden md:table-cell">
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
                          aria-label={`Open case ${idx + 1}`}
                          className={`cursor-pointer transition-colors hover:bg-mist/40 ${
                            selected ? 'bg-teal-soft/30 ring-1 ring-inset ring-teal/30' : ''
                          }`}
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
                          {mobileFeatureCols.map((col) => (
                            <td key={col} className="truncate max-w-[7rem] text-[var(--muted)] md:hidden">
                              {String(row[col] ?? '')}
                            </td>
                          ))}
                          {featurePreviewCols.map((col) => (
                            <td
                              key={`d-${col}`}
                              className="truncate max-w-[6rem] text-[var(--muted)] hidden md:table-cell"
                            >
                              {String(row[col] ?? '')}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="border border-mist px-5 py-6 text-sm text-[var(--muted)]">
                No sample rows loaded here — the stored case brief still opens beside this list.
              </div>
            )}
          </div>
        </section>
      )}

      <details className="mb-8 border border-mist group">
        <summary className="px-5 py-4 cursor-pointer list-none hover:bg-mist/20">
          <span className="font-display text-base font-semibold text-ink">Project details</span>
          <span className="block text-sm text-[var(--muted)] mt-1">
            Watching for {outcomeLabel.toLowerCase()} · {project.feature_columns?.length || 0} signals
          </span>
        </summary>
        <dl className="grid grid-cols-2 md:grid-cols-3 gap-x-6 gap-y-3 px-5 py-4 text-sm border-t border-mist">
          <div>
            <dt className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Watching for</dt>
            <dd className="font-medium text-ink mt-0.5 capitalize">{outcomeLabel}</dd>
          </div>
          <div>
            <dt className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Using</dt>
            <dd className="font-medium text-ink mt-0.5">
              {project.feature_columns?.length || 0} signals from your data
            </dd>
          </div>
          <div>
            <dt className="text-[11px] uppercase tracking-wide text-[var(--muted)]">Updated</dt>
            <dd className="font-medium text-ink mt-0.5">
              {project.active_model?.trained_at
                ? new Date(project.active_model.trained_at).toLocaleDateString()
                : isReady
                  ? 'Ready'
                  : '—'}
            </dd>
          </div>
        </dl>
        {(user?.role === 'owner' || user?.role === 'admin') && (
          <div className="px-5 py-4 border-t border-mist">
            <button
              type="button"
              onClick={handleDeleteProject}
              disabled={deleting}
              className="btn-ghost text-xs text-coral hover:text-coral"
            >
              {deleting ? 'Deleting…' : 'Delete this project'}
            </button>
          </div>
        )}
      </details>

      {checkInTarget && (
        <div
          className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/55 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="checkin-title"
          onClick={closeCheckIn}
        >
          <div
            className="w-full max-w-md border border-mist bg-surface shadow-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-5 py-4 border-b border-mist">
              <h2 id="checkin-title" className="font-display text-lg font-semibold text-ink">
                Update follow-up
              </h2>
              <p className="text-sm text-[var(--muted)] mt-1">{checkInTarget.action_name}</p>
            </div>
            <div className="px-5 py-4 space-y-4">
              <div>
                <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] mb-2">
                  What happened?
                </div>
                <div className="flex flex-wrap gap-2">
                  {[
                    { value: 'yes', label: 'Yes — outcome occurred' },
                    { value: 'no', label: 'No — did not occur' },
                    { value: 'unknown', label: 'Still unknown' },
                  ].map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setCheckInOutcome(opt.value)}
                      className={`text-xs px-3 py-1.5 border rounded-control ${
                        checkInOutcome === opt.value
                          ? 'border-teal text-teal'
                          : 'border-mist text-ink hover:border-teal'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label
                  htmlFor="checkin-notes"
                  className="block text-[11px] uppercase tracking-wide text-[var(--muted)] mb-2"
                >
                  Notes
                </label>
                <textarea
                  id="checkin-notes"
                  rows={3}
                  value={checkInNotes}
                  onChange={(e) => setCheckInNotes(e.target.value)}
                  placeholder="What did you try? Anything unexpected?"
                  className="w-full bg-paper border border-mist px-3 py-2 text-sm text-ink rounded-control focus:outline-none focus:border-teal"
                />
              </div>

              <div>
                <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] mb-2">
                  Next step
                </div>
                <div className="space-y-2">
                  <label className="flex items-start gap-2 text-sm text-ink cursor-pointer">
                    <input
                      type="radio"
                      name="checkin-mode"
                      checked={checkInMode === 'reschedule'}
                      onChange={() => setCheckInMode('reschedule')}
                      className="mt-1"
                    />
                    <span>
                      Check back again
                      {checkInMode === 'reschedule' && (
                        <span className="ml-2 inline-flex gap-1">
                          {[30, 60, 90].map((n) => (
                            <button
                              key={n}
                              type="button"
                              onClick={() => setCheckInInterval(n)}
                              className={`text-xs px-2 py-0.5 border rounded-control ${
                                checkInInterval === n
                                  ? 'border-teal text-teal'
                                  : 'border-mist text-[var(--muted)]'
                              }`}
                            >
                              {n}d
                            </button>
                          ))}
                        </span>
                      )}
                    </span>
                  </label>
                  <label className="flex items-start gap-2 text-sm text-ink cursor-pointer">
                    <input
                      type="radio"
                      name="checkin-mode"
                      checked={checkInMode === 'close'}
                      onChange={() => setCheckInMode('close')}
                      className="mt-1"
                    />
                    <span>Close as done</span>
                  </label>
                  <label className="flex items-start gap-2 text-sm text-ink cursor-pointer">
                    <input
                      type="radio"
                      name="checkin-mode"
                      checked={checkInMode === 'keep'}
                      onChange={() => setCheckInMode('keep')}
                      className="mt-1"
                    />
                    <span>Keep open (same recheck date)</span>
                  </label>
                </div>
              </div>

              {checkInError && (
                <p className="text-sm text-coral">{checkInError}</p>
              )}
            </div>
            <div className="px-5 py-4 border-t border-mist flex justify-end gap-2">
              <button
                type="button"
                onClick={closeCheckIn}
                disabled={checkInSaving}
                className="btn-ghost text-sm"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={submitCheckIn}
                disabled={checkInSaving}
                className="btn-primary text-sm"
              >
                {checkInSaving ? 'Saving…' : 'Save update'}
              </button>
            </div>
          </div>
        </div>
      )}

      {isReady && project.problem_type !== 'regression' && (
        <details className="mb-4 border border-mist group">
          <summary className="px-5 py-4 cursor-pointer list-none hover:bg-mist/20">
            <span className="font-display text-base font-semibold text-ink">Quality & learning</span>
            <span className="block text-sm text-[var(--muted)] mt-1">
              Optional — how well guidance matches real outcomes
              {spotCheck?.n > 0
                ? ` · ${Math.round((spotCheck.agree_rate || 0) * 100)}% matched so far`
                : ''}
            </span>
          </summary>
          <div className="px-5 py-5 border-t border-mist space-y-8">
            <div>
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <h3 className="font-medium text-ink">Does the ranking help?</h3>
                <button
                  type="button"
                  onClick={fetchSpotCheck}
                  disabled={spotLoading}
                  className="btn-ghost text-sm"
                >
                  {spotLoading ? 'Checking…' : 'Refresh'}
                </button>
              </div>
              {spotLoading && !spotCheck && (
                <p className="text-sm text-[var(--muted)]">Checking against known outcomes…</p>
              )}
              {spotCheck && spotCheck.supported === false && (
                <p className="text-sm text-[var(--muted)]">{spotCheck.message}</p>
              )}
              {spotCheck && spotCheck.n === 0 && (
                <p className="text-sm text-[var(--muted)]">
                  {spotCheck.message || 'Not enough known outcomes to check yet.'}
                </p>
              )}
              {spotCheck && spotCheck.n > 0 && (
                <>
                  <p className="text-sm text-ink max-w-2xl leading-relaxed">
                    {spotCheck.plain_summary}
                  </p>
                  <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-px bg-mist border border-mist">
                    <div className="bg-paper px-3 py-3">
                      <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                        Matched known outcomes
                      </div>
                      <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                        {Math.round((spotCheck.agree_rate || 0) * 100)}%
                      </div>
                    </div>
                    <div className="bg-paper px-3 py-3">
                      <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                        Right when flagged high
                      </div>
                      <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                        {spotCheck.high_risk_precision != null
                          ? `${Math.round(spotCheck.high_risk_precision * 100)}%`
                          : '—'}
                      </div>
                    </div>
                    <div className="bg-paper px-3 py-3">
                      <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                        Right when flagged low
                      </div>
                      <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                        {spotCheck.low_risk_true_negative_rate != null
                          ? `${Math.round(spotCheck.low_risk_true_negative_rate * 100)}%`
                          : '—'}
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>

            {feedbackSummary && (
              <div>
                <h3 className="font-medium text-ink mb-2">Outcomes your team recorded</h3>
                <p className="text-sm text-[var(--muted)] leading-relaxed max-w-2xl mb-4">
                  {feedbackSummary.plain_summary}
                </p>
                {feedbackSummary.learning?.plain && (
                  <p className="text-sm text-teal mb-4 leading-relaxed max-w-2xl">
                    {feedbackSummary.learning.plain}
                  </p>
                )}
                <div className="grid sm:grid-cols-3 gap-px bg-mist border border-mist">
                  <div className="bg-paper px-3 py-3">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                      Recorded
                    </div>
                    <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                      {feedbackSummary.with_feedback || 0}
                    </div>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      of {feedbackSummary.total_predictions || 0} cases reviewed
                    </p>
                  </div>
                  <div className="bg-paper px-3 py-3">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                      Estimate matched
                    </div>
                    <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                      {feedbackSummary.model_match_rate != null
                        ? `${Math.round(feedbackSummary.model_match_rate * 100)}%`
                        : '—'}
                    </div>
                  </div>
                  <div className="bg-paper px-3 py-3">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted)]">
                      Reshaping rankings
                    </div>
                    <div className="mt-1 font-display text-2xl font-semibold tabular-nums text-ink">
                      {feedbackSummary.learning?.actions_reshaping_rankings ?? 0}
                    </div>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      of {Object.keys(feedbackSummary.action_effectiveness || {}).length} tracked
                      actions (need 3+ outcomes)
                    </p>
                  </div>
                </div>
                {Array.isArray(feedbackSummary.action_effectiveness_ranked) &&
                  feedbackSummary.action_effectiveness_ranked.length > 0 && (
                    <ul className="mt-4 space-y-2">
                      {feedbackSummary.action_effectiveness_ranked.slice(0, 5).map((a) => (
                        <li
                          key={a.action_code}
                          className="flex flex-wrap items-baseline justify-between gap-2 text-sm"
                        >
                          <span className="text-ink font-medium">
                            {a.action_name || a.action_code}
                          </span>
                          <span className="tabular-nums text-[var(--muted)] text-right">
                            {a.success_n}/{a.n} favorable
                            {a.reliable ? ' · reshapes ranking' : ' · small sample'}
                            {a.learning_note ? (
                              <span className="block text-xs text-teal mt-0.5">{a.learning_note}</span>
                            ) : null}
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
              </div>
            )}
          </div>
        </details>
      )}
    </div>
  );
}
