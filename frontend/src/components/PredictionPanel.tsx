import type { ReactNode } from 'react';
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { TrustSpine } from './TrustSpine';
import { HalftoneGlow } from './HalftoneGlow';
import { assessTrust, matchLabel } from '../lib/trustAssessment';

export type ExplanationDriver = {
  feature: string;
  label?: string;
  impact?: number;
  direction?: string;
  strength?: string;
  text?: string;
  value?: unknown;
};

export type InsightItem = {
  text?: string;
  reason?: string;
  suggestion?: string;
  severity?: string;
  message?: string;
  display_name?: string;
  category?: string;
  strength?: string;
  contribution?: string;
};

export type PredictionPayload = {
  probability?: number;
  predicted_value?: number;
  confidence?: number;
  risk_level?: string;
  target?: string;
  problem_type?: string;
  confidence_interval?: {
    lower: number;
    upper: number;
    level?: number;
    width?: number;
  } | null;
  low_confidence?: boolean;
  abstention_reason?: string | null;
  model_disagreement?: number | null;
  insights?: InsightItem[];
  insight_brief?: {
    headline?: string;
    summary?: string;
    theme_rollup?: Array<{ category: string; label: string; features: string[]; count: number }>;
    risk_factors?: string[];
    protective_factors?: string[];
    overall_severity?: string;
    trust_note?: string | null;
    action_context?: {
      primary_lever?: { display_name?: string; feature?: string; suggestion?: string } | null;
      addressable_factors?: Array<{ display_name?: string; suggestion?: string }>;
    };
  };
  recommendations?: Array<{
    action_code?: string;
    action_name?: string;
    name?: string;
    description?: string;
    reasoning?: string;
    final_score?: number;
    impact_score?: number;
    cost_score?: number;
    relevance_score?: number;
    cost_label?: string;
    expected_probability_reduction?: number;
    new_probability_estimate?: number;
    probability_reduction_percent?: number;
    impact_is_illustrative?: boolean;
    impact_disclaimer?: string;
    implementation_time?: string;
    rank?: number;
  }>;
  prediction_id?: string;
  entity_id?: string;
  source?: string;
  persisted?: boolean;
  feedback?: {
    actual_outcome?: string;
    action_taken?: string;
    plain_summary?: string;
    model_matched_outcome?: boolean | null;
  } | null;
  decision_summary?: {
    strategy?: string;
    description?: string;
    current_probability?: number;
    expected_new_probability?: number;
    expected_reduction?: number;
  };
  recommendation_scoring?: {
    impact_weight?: number;
    cost_weight?: number;
    relevance_weight?: number;
    soft_case?: boolean;
  };
  explanation_consistency?: {
    score?: number | null;
    trust_level?: string;
    flag?: boolean;
    plain?: string;
  };
  explanations?: {
    shap?: { top_features?: ExplanationDriver[] };
    lime?: { top_features?: ExplanationDriver[] };
    drivers?: ExplanationDriver[];
    consistency?: {
      score?: number | null;
      trust_level?: string;
      flag?: boolean;
      plain?: string;
    };
    all_factors?: Array<{ feature: string; shap_value?: number; impact?: number; lime_weight?: number }>;
    degraded?: boolean;
    error?: string;
    methods_available?: string[];
  };
};

type Props = {
  result: PredictionPayload;
  children?: ReactNode;
  simulateHref?: string;
  simulateLabel?: string;
  knownOutcome?: boolean | null;
  outcomeYesLabel?: string;
  outcomeNoLabel?: string;
  /** A7 basic feedback log (optional) */
  projectId?: string;
  authToken?: string | null;
  onFeedbackSaved?: (fb: NonNullable<PredictionPayload['feedback']>) => void;
  /** B3 decision ledger callback */
  onDecisionCommitted?: (decision: Record<string, unknown>) => void;
};

function humanize(name: string): string {
  return name
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function attentionCopy(risk?: string, isSoft?: boolean): string {
  const r = String(risk || '').toLowerCase();
  if (r === 'critical' || r === 'high') {
    return isSoft ? 'Needs attention (soft)' : 'Needs attention';
  }
  if (r === 'medium') return isSoft ? 'Watch closely (soft)' : 'Watch closely';
  return isSoft ? 'Likely stable (soft range)' : 'Looking stable';
}

function consistencyTone(level?: string): string {
  const l = String(level || '').toLowerCase();
  if (l === 'high') return 'border-teal/30 bg-teal-soft/20';
  if (l === 'medium' || l === 'single_method') return 'border-mist bg-paper';
  if (l === 'low' || l === 'unavailable') return 'border-coral/40 bg-coral-soft';
  return 'border-mist';
}

/**
 * Decision surface for business users: likelihood, why, what to do.
 */
export function PredictionPanel({
  result,
  children,
  simulateHref,
  simulateLabel = 'Test a different plan',
  knownOutcome = null,
  outcomeYesLabel = 'Yes',
  outcomeNoLabel = 'No',
  projectId,
  authToken,
  onFeedbackSaved,
  onDecisionCommitted,
}: Props) {
  const isReg = result.problem_type === 'regression';
  const point = isReg
    ? Number(result.predicted_value ?? 0)
    : Number(result.probability ?? 0);

  const ci = result.confidence_interval;
  const lower = ci?.lower ?? (isReg ? point * 0.9 : Math.max(0, point - 0.15));
  const upper = ci?.upper ?? (isReg ? point * 1.1 : Math.min(1, point + 0.15));
  const level = ci?.level ?? 0.9;

  const trust = assessTrust({
    point,
    lower,
    upper,
    lowConfidence: result.low_confidence,
    isRegression: isReg,
  });

  const outcome = result.target ? humanize(String(result.target)) : 'this outcome';
  const title = isReg
    ? `Expected ${outcome.toLowerCase()}`
    : `Chance of ${outcome.toLowerCase()}`;

  const agreement = !isReg ? matchLabel(knownOutcome, point) : 'unknown';

  const consistency =
    result.explanation_consistency || result.explanations?.consistency || null;

  const brief = result.insight_brief;
  const primaryLever = brief?.action_context?.primary_lever;

  const drivers: ExplanationDriver[] =
    result.explanations?.drivers ??
    result.explanations?.shap?.top_features ??
    result.explanations?.all_factors?.slice(0, 5).map((f) => ({
      feature: f.feature,
      impact: Number(f.impact ?? f.shap_value ?? f.lime_weight ?? 0),
      direction: Number(f.impact ?? f.shap_value ?? f.lime_weight ?? 0) > 0 ? 'increases' : 'decreases',
    })) ??
    [];

  const methods = result.explanations?.methods_available ?? [];
  const explainError = result.explanations?.error;

  const [fbOutcome, setFbOutcome] = useState(() => {
    const o = result.feedback?.actual_outcome;
    if (o === 'positive' || o === 'yes') return 'positive';
    if (o === 'negative' || o === 'no') return 'negative';
    if (o === 'unknown') return 'unknown';
    return knownOutcome === true ? 'positive' : knownOutcome === false ? 'negative' : '';
  });
  const [fbAction, setFbAction] = useState(result.feedback?.action_taken || '');
  const [fbStatus, setFbStatus] = useState<string | null>(result.feedback?.plain_summary || null);
  const [fbError, setFbError] = useState('');
  const [fbSaving, setFbSaving] = useState(false);
  const [fbMatch, setFbMatch] = useState<boolean | null | undefined>(
    result.feedback?.model_matched_outcome
  );

  const [decAction, setDecAction] = useState(() => {
    const top = result.recommendations?.[0];
    return top?.action_code || top?.action_name || '';
  });
  const [decInterval, setDecInterval] = useState(30);
  const [decStatus, setDecStatus] = useState<string | null>(null);
  const [decError, setDecError] = useState('');
  const [decSaving, setDecSaving] = useState(false);

  const canLog =
    Boolean(projectId && authToken && result.prediction_id) && !isReg;
  const canCommit =
    Boolean(projectId && authToken) &&
    !isReg &&
    Boolean(result.recommendations && result.recommendations.length > 0);

  const submitDecision = async () => {
    if (!canCommit || !decAction) {
      setDecError('Choose an action to commit.');
      return;
    }
    setDecSaving(true);
    setDecError('');
    try {
      const chosen =
        result.recommendations?.find(
          (r) => (r.action_code || r.action_name || r.name) === decAction
        ) || result.recommendations?.[0];
      const code = chosen?.action_code || decAction;
      const res = await fetch(`/api/projects/${projectId}/decisions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          action_code: code,
          action_name: chosen?.action_name || chosen?.name || code,
          action_description: chosen?.description || null,
          prediction_id: result.prediction_id || null,
          entity_id: result.entity_id || null,
          probability: result.probability ?? null,
          risk_level: result.risk_level || null,
          expected_probability_after: chosen?.new_probability_estimate ?? null,
          expected_lift:
            chosen?.expected_probability_reduction != null
              ? -Number(chosen.expected_probability_reduction)
              : null,
          decision_summary:
            result.decision_summary?.description ||
            `Committed “${chosen?.action_name || code}” with a ${decInterval}-day recheck.`,
          recheck_interval_days: decInterval,
          case_snapshot: {
            drivers: result.explanations?.drivers || null,
            recommendation_scoring: result.recommendation_scoring || null,
          },
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setDecError(data.detail || 'Could not commit decision');
      } else {
        setDecStatus(data.plain_summary || 'Decision committed to ledger.');
        onDecisionCommitted?.(data);
      }
    } catch (e) {
      setDecError(e instanceof Error ? e.message : 'Network error');
    }
    setDecSaving(false);
  };

  const submitFeedback = async () => {
    if (!canLog || !fbOutcome) {
      setFbError('Choose an outcome first.');
      return;
    }
    setFbSaving(true);
    setFbError('');
    try {
      const res = await fetch(`/api/projects/${projectId}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          prediction_id: result.prediction_id,
          actual_outcome: fbOutcome,
          action_taken: fbAction || null,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setFbError(data.detail || 'Could not save feedback');
      } else {
        setFbStatus(data.plain_summary || 'Outcome logged.');
        setFbMatch(data.model_matched_outcome);
        onFeedbackSaved?.(data);
      }
    } catch (e) {
      setFbError(e instanceof Error ? e.message : 'Network error');
    }
    setFbSaving(false);
  };

  return (
    <div className="space-y-7">
      {!isReg && (
        <div className="border-b border-mist pb-5">
          <p className="page-kicker mb-1">This case</p>
          <p className="font-display text-2xl font-semibold text-ink tracking-tight">
            {brief?.headline || attentionCopy(result.risk_level, trust.isSoft)}
          </p>
          <p className="mt-2 text-sm text-[var(--muted)] leading-relaxed">
            {brief?.summary || trust.summary}
          </p>
          <p className="mt-2 text-sm text-ink">
            Best guess:{' '}
            <span className="font-semibold tabular-nums">{(point * 100).toFixed(0)}%</span>{' '}
            chance of {outcome.toLowerCase()}.
          </p>

          {knownOutcome != null && (
            <div
              className={`mt-4 px-3 py-3 border text-sm ${
                agreement === 'agrees'
                  ? 'border-teal/30 bg-teal-soft/20'
                  : agreement === 'conflicts'
                    ? 'border-coral/40 bg-coral-soft'
                    : 'border-mist'
              }`}
            >
              <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
                Check against known data
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-ink">
                <span>
                  Known outcome:{' '}
                  <strong>{knownOutcome ? outcomeYesLabel : outcomeNoLabel}</strong>
                </span>
                <span>
                  Call at 50%:{' '}
                  <strong>{point >= 0.5 ? outcomeYesLabel : outcomeNoLabel}</strong>
                </span>
              </div>
              <p className="mt-1.5 text-[var(--muted)]">
                {agreement === 'agrees' &&
                  'Matches the held-out label for this row — good sign for this case.'}
                {agreement === 'conflicts' &&
                  'Disagrees with the held-out label — open drivers and treat as a miss to learn from.'}
              </p>
            </div>
          )}
        </div>
      )}

      <TrustSpine
        point={point}
        lower={lower}
        upper={upper}
        level={level}
        lowConfidence={trust.isSoft}
        abstentionReason={result.abstention_reason}
        disagreement={null}
        domain={isReg ? undefined : [0, 1]}
        outcomeLabel={title}
        businessCopy
        badgeLabel={trust.badge}
        rangeNote={trust.rangeNote}
      />

      {simulateHref && (
        <HalftoneGlow
          className="min-h-[9.5rem] w-full rounded-control border border-mist"
          background="#0A0908"
          focalPoints={[
            { x: 0.36, y: 0.5, color: '#FF5A1F' },
            { x: 0.64, y: 0.5, color: '#00C8B4' },
          ]}
          dotSpacing={12}
          maxDotRadius={4.5}
          glowRadius={180}
          animated={false}
        >
          <Link
            to={simulateHref}
            className="inline-flex items-center justify-center px-6 py-3 text-sm font-semibold tracking-wide text-ink bg-paper/40 border border-ink/20 hover:border-teal/60 hover:text-teal focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal rounded-control transition-colors backdrop-blur-[1px]"
          >
            {simulateLabel}
          </Link>
        </HalftoneGlow>
      )}

      {canLog && (
        <div className="border border-mist px-3 py-4 text-sm space-y-3">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
              Log real outcome (A7)
            </div>
            <p className="text-[var(--muted)] leading-relaxed">
              When you know what happened, record it so we can score model hits and action
              effectiveness. This is a basic log — not a 30/60/90 decision autopsy.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {(
              [
                ['positive', outcomeYesLabel],
                ['negative', outcomeNoLabel],
                ['unknown', 'Unknown'],
              ] as const
            ).map(([val, label]) => (
              <button
                key={val}
                type="button"
                onClick={() => setFbOutcome(val)}
                className={`px-3 py-1.5 text-xs border ${
                  fbOutcome === val
                    ? 'border-teal bg-teal-soft/30 text-ink'
                    : 'border-mist text-[var(--muted)] hover:text-ink'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {result.recommendations && result.recommendations.length > 0 && (
            <div>
              <label className="block text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
                Action taken (optional)
              </label>
              <select
                value={fbAction}
                onChange={(e) => setFbAction(e.target.value)}
                className="w-full bg-paper border border-mist px-2 py-2 text-ink text-sm"
              >
                <option value="">None / not recorded</option>
                {result.recommendations.map((r, i) => {
                  const code = r.action_code || r.action_name || r.name || `action_${i}`;
                  return (
                    <option key={code} value={code}>
                      {r.action_name || r.name || code}
                    </option>
                  );
                })}
              </select>
            </div>
          )}
          <button
            type="button"
            onClick={submitFeedback}
            disabled={fbSaving || !fbOutcome}
            className="px-4 py-2 text-sm font-medium border border-ink/20 hover:border-teal text-ink disabled:opacity-50"
          >
            {fbSaving ? 'Saving…' : 'Save outcome'}
          </button>
          {fbError && <p className="text-coral text-xs">{fbError}</p>}
          {fbStatus && (
            <p
              className={`text-xs leading-relaxed ${
                fbMatch === false
                  ? 'text-coral'
                  : fbMatch === true
                    ? 'text-teal'
                    : 'text-[var(--muted)]'
              }`}
            >
              {fbStatus}
            </p>
          )}
        </div>
      )}

      {consistency && consistency.plain && (
        <div className={`px-3 py-3 border text-sm ${consistencyTone(consistency.trust_level)}`}>
          <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
            Why trust check
            {consistency.score != null && (
              <span className="ml-2 normal-case tracking-normal text-ink font-medium tabular-nums">
                {Math.round(Number(consistency.score) * 100)}% agreement · {consistency.trust_level}
              </span>
            )}
          </div>
          <p className="text-ink leading-relaxed">{consistency.plain}</p>
          {methods.length > 0 && (
            <p className="mt-1.5 text-xs text-[var(--muted)]">
              Methods: {methods.map((m) => m.toUpperCase()).join(' + ')}
            </p>
          )}
        </div>
      )}

      {explainError && (
        <p className="text-sm text-coral">Could not explain this case: {explainError}</p>
      )}

      {(brief?.theme_rollup?.length || primaryLever) && (
        <div className="space-y-3">
          {brief?.theme_rollup && brief.theme_rollup.length > 0 && (
            <div>
              <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] mb-2">
                Themes in this case
              </h4>
              <div className="flex flex-wrap gap-2">
                {brief.theme_rollup.map((t) => (
                  <span
                    key={t.category}
                    className="text-xs border border-mist px-2.5 py-1 text-ink"
                  >
                    {t.label}
                    <span className="text-[var(--muted)]"> · {t.count}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
          {primaryLever?.suggestion && (
            <div className="border border-mist px-3 py-3 text-sm">
              <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
                Highest-leverage next focus
              </div>
              <p className="font-medium text-ink">
                {primaryLever.display_name || humanize(String(primaryLever.feature || 'Driver'))}
              </p>
              <p className="mt-1 text-[var(--muted)] leading-relaxed">{primaryLever.suggestion}</p>
            </div>
          )}
        </div>
      )}

      {result.insights && result.insights.length > 0 && (
        <div>
          <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] mb-3">
            In plain language
          </h4>
          <ul className="space-y-3">
            {result.insights.slice(0, 5).map((ins, i) => (
              <li key={i} className="text-sm text-ink leading-relaxed border-l-2 border-mist pl-3">
                {ins.text || ins.message || ins.reason}
                {ins.suggestion && (
                  <span className="block mt-1 text-[var(--muted)]">{ins.suggestion}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {drivers.length > 0 && (
        <div>
          <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] mb-3">
            Why we think this
          </h4>
          <ul className="space-y-2.5">
            {drivers.slice(0, 5).map((f) => {
              const impact = Number(f.impact ?? 0);
              const up = f.direction === 'increases' || impact > 0;
              const strength =
                f.strength ||
                (Math.abs(impact) >= 0.12
                  ? 'strong'
                  : Math.abs(impact) >= 0.04
                    ? 'moderate'
                    : 'mild');
              return (
                <li key={f.feature} className="text-sm border-b border-mist/80 pb-2">
                  <div className="flex justify-between gap-4">
                    <span className="text-ink font-medium">{f.label || humanize(f.feature)}</span>
                    <span className={`shrink-0 font-medium ${up ? 'text-coral' : 'text-teal'}`}>
                      {strength === 'strong'
                        ? 'Strong · '
                        : strength === 'moderate'
                          ? 'Moderate · '
                          : 'Mild · '}
                      {up ? 'raises risk' : 'lowers risk'}
                    </span>
                  </div>
                  {f.text && <p className="mt-1 text-[var(--muted)] leading-relaxed">{f.text}</p>}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {result.recommendations && result.recommendations.length > 0 && (
        <div>
          <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] mb-3">
            What to do next
          </h4>
          <p className="text-xs text-[var(--muted)] mb-4 leading-relaxed">
            Ranked action catalog. Impact numbers are illustrative heuristics, not a
            re-simulated outcome for this case.
          </p>
          {result.decision_summary && (
            <div className="mb-4 border border-mist px-3 py-3 text-sm">
              <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
                Decision strategy
              </div>
              <p className="font-medium text-ink">{result.decision_summary.strategy}</p>
              {result.decision_summary.description && (
                <p className="mt-1 text-[var(--muted)] leading-relaxed">
                  {result.decision_summary.description}
                </p>
              )}
              {result.decision_summary.expected_new_probability != null &&
                result.decision_summary.current_probability != null && (
                  <p className="mt-2 text-xs text-ink tabular-nums">
                    Illustrative stack estimate:{" "}
                    {(result.decision_summary.current_probability * 100).toFixed(0)}% →{" "}
                    {(result.decision_summary.expected_new_probability * 100).toFixed(0)}%
                    <span className="text-[var(--muted)]">
                      {" "}
                      (catalog heuristic, not re-simulated)
                    </span>
                  </p>
                )}
            </div>
          )}
          <ol className="space-y-5">
            {result.recommendations.slice(0, 4).map((r, i) => {
              const impact = Number(r.impact_score ?? 0);
              const cost = Number(r.cost_score ?? 0);
              const rel = Number(r.relevance_score ?? 0);
              const final = Number(r.final_score ?? 0);
              return (
                <li key={r.action_code || r.action_name || r.name || i} className="text-sm border-b border-mist/80 pb-4 last:border-0">
                  <div className="flex justify-between gap-3 items-start">
                    <div className="font-medium text-ink">
                      {i + 1}. {r.action_name || r.name}
                    </div>
                    <span className="shrink-0 text-xs tabular-nums text-[var(--muted)]">
                      Rank score {(final * 100).toFixed(0)}
                    </span>
                  </div>
                  {r.description && (
                    <p className="mt-1 text-[var(--muted)] leading-relaxed">{r.description}</p>
                  )}
                  {r.reasoning && (
                    <p className="mt-1.5 text-ink/90 leading-relaxed">{r.reasoning}</p>
                  )}
                  <div className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
                    <ScoreBar label="Illustrative impact" value={impact} tone="coral" />
                    <ScoreBar label="Low cost" value={1 - cost} tone="teal" />
                    <ScoreBar label="Relevance" value={rel} tone="teal" />
                  </div>
                  {r.expected_probability_reduction != null &&
                    r.expected_probability_reduction > 0.01 && (
                      <p className="mt-2 text-xs text-[var(--muted)] tabular-nums">
                        Illustrative est. ~−
                        {(r.expected_probability_reduction * 100).toFixed(0)} pp
                        {r.implementation_time ? ` · ${r.implementation_time}` : ""}
                        <span className="block mt-0.5">
                          {r.impact_disclaimer ||
                            "Catalog heuristic — not a re-simulated outcome"}
                        </span>
                      </p>
                    )}
                </li>
              );
            })}
          </ol>
        </div>
      )}

      {canCommit && (
        <div className="border border-mist px-3 py-4 text-sm space-y-3">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
              Commit to ledger (B3)
            </div>
            <p className="text-[var(--muted)] leading-relaxed">
              Turn a recommended action into an accountable decision with a scheduled
              recheck. This is the decision ledger — not just an A7 outcome stamp.
            </p>
          </div>
          <div>
            <label className="block text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
              Action
            </label>
            <select
              value={decAction}
              onChange={(e) => setDecAction(e.target.value)}
              className="w-full bg-paper border border-mist px-2 py-2 text-ink text-sm"
            >
              {(result.recommendations || []).map((r, i) => {
                const code = r.action_code || r.action_name || r.name || `action_${i}`;
                return (
                  <option key={code} value={code}>
                    {r.action_name || r.name || code}
                  </option>
                );
              })}
            </select>
          </div>
          <div className="flex flex-wrap gap-2">
            {([30, 60, 90] as const).map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setDecInterval(d)}
                className={`px-3 py-1.5 text-xs border ${
                  decInterval === d
                    ? 'border-teal bg-teal-soft/30 text-ink'
                    : 'border-mist text-[var(--muted)] hover:text-ink'
                }`}
              >
                {d}-day recheck
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={submitDecision}
            disabled={decSaving || !decAction}
            className="px-4 py-2 text-sm font-medium border border-ink/20 hover:border-teal text-ink disabled:opacity-50"
          >
            {decSaving ? 'Committing…' : 'Commit decision'}
          </button>
          {decError && <p className="text-coral text-xs">{decError}</p>}
          {decStatus && <p className="text-teal text-xs leading-relaxed">{decStatus}</p>}
        </div>
      )}

      {children}
    </div>
  );
}

function ScoreBar({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: 'coral' | 'teal';
}) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  const bar = tone === 'coral' ? 'bg-coral' : 'bg-teal';
  return (
    <div>
      <div className="flex justify-between text-[var(--muted)] mb-1">
        <span>{label}</span>
        <span className="tabular-nums">{pct}</span>
      </div>
      <div className="h-1 bg-mist">
        <div className={`h-full ${bar}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default PredictionPanel;
