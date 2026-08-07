import type { ReactNode } from 'react';
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { TrustSpine } from './TrustSpine';
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
  projectId?: string;
  authToken?: string | null;
  onFeedbackSaved?: (fb: NonNullable<PredictionPayload['feedback']>) => void;
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
    return isSoft ? 'Needs a closer look' : 'Needs attention';
  }
  if (r === 'medium') return isSoft ? 'Keep an eye on this' : 'Watch closely';
  return isSoft ? 'Looking stable — still verify' : 'Looking stable';
}

/**
 * Decision surface for business users: likelihood, why, what to do.
 */
export function PredictionPanel({
  result,
  children,
  simulateHref,
  simulateLabel = 'Explore a what-if',
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
      setDecError('Choose an action first.');
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
            `Saved “${chosen?.action_name || code}” — check back in ${decInterval} days.`,
          recheck_interval_days: decInterval,
          case_snapshot: {
            drivers: result.explanations?.drivers || null,
            recommendation_scoring: result.recommendation_scoring || null,
          },
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setDecError(data.detail || 'Could not save follow-up');
      } else {
        setDecStatus(data.plain_summary || 'Follow-up saved.');
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
        setFbError(data.detail || 'Could not save');
      } else {
        setFbStatus(data.plain_summary || 'Saved — thanks.');
        setFbMatch(data.model_matched_outcome);
        onFeedbackSaved?.(data);
      }
    } catch (e) {
      setFbError(e instanceof Error ? e.message : 'Network error');
    }
    setFbSaving(false);
  };

  return (
    <div className="space-y-6">
      {!isReg && (
        <div>
          <p className="page-kicker mb-1">Brief</p>
          <p className="font-display text-2xl font-semibold text-ink tracking-tight">
            {brief?.headline || attentionCopy(result.risk_level, trust.isSoft)}
          </p>
          <p className="mt-2 text-sm text-[var(--muted)] leading-relaxed">
            {brief?.summary || trust.summary}
          </p>
          <p className="mt-3 text-sm text-ink">
            About{' '}
            <span className="font-semibold tabular-nums">{(point * 100).toFixed(0)}%</span>{' '}
            chance of {outcome.toLowerCase()}
            {trust.isSoft ? ' — treat as a guide, not a sure thing.' : '.'}
          </p>
          {knownOutcome != null && (
            <p
              className={`mt-3 text-sm ${
                agreement === 'agrees'
                  ? 'text-teal'
                  : agreement === 'conflicts'
                    ? 'text-coral'
                    : 'text-[var(--muted)]'
              }`}
            >
              {agreement === 'agrees' &&
                `Matches the labeled outcome in your dataset (${knownOutcome ? outcomeYesLabel : outcomeNoLabel}).`}
              {agreement === 'conflicts' &&
                `Dataset label is ${knownOutcome ? outcomeYesLabel : outcomeNoLabel} — dig into why below.`}
            </p>
          )}
        </div>
      )}

      {result.recommendations && result.recommendations.length > 0 && (
        <div>
          <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-teal mb-3">
            Suggested next step
          </h4>
          {trust.isSoft && (
            <p className="text-sm text-[var(--muted)] mb-3 leading-relaxed">
              Certainty is soft on this case — prefer lighter check-ins first, and confirm with a
              what-if before big spends.
            </p>
          )}
          {result.decision_summary?.strategy && (
            <p className="text-sm text-ink mb-3 leading-relaxed">
              {result.decision_summary.strategy}
              {result.decision_summary.description
                ? ` — ${result.decision_summary.description}`
                : ''}
            </p>
          )}
          <p className="text-xs text-[var(--muted)] mb-3 leading-relaxed">
            Playbook suggestions ranked for this case. Impact labels are guides — run a what-if to
            see a real before/after for this person.
          </p>
          <ol className="space-y-4">
            {result.recommendations.slice(0, 3).map((r, i) => {
              const impact = Number(r.impact_score ?? 0);
              const cost = Number(r.cost_score ?? 0);
              const effort =
                cost <= 0.35 ? 'Low effort' : cost <= 0.65 ? 'Medium effort' : 'Higher effort';
              let punch =
                impact >= 0.65 ? 'High potential' : impact >= 0.4 ? 'Solid potential' : 'Light touch';
              if (trust.isSoft && impact >= 0.4) {
                punch = 'Worth testing carefully';
              }
              return (
                <li
                  key={r.action_code || r.action_name || r.name || i}
                  className="text-sm border border-mist rounded-control px-3 py-3"
                >
                  <div className="font-medium text-ink">
                    {i + 1}. {r.action_name || r.name}
                  </div>
                  <p className="mt-1 text-xs text-[var(--muted)]">
                    {punch} · {effort}
                    {r.implementation_time ? ` · ${r.implementation_time}` : ''}
                  </p>
                  {(r.description || r.reasoning) && (
                    <p className="mt-2 text-[var(--muted)] leading-relaxed">
                      {r.reasoning || r.description}
                    </p>
                  )}
                </li>
              );
            })}
          </ol>
        </div>
      )}

      {canCommit && (
        <div className="border border-teal/25 bg-teal-soft/10 px-3 py-4 text-sm space-y-3 rounded-control">
          <div className="text-[11px] uppercase tracking-wide text-teal">Save a follow-up</div>
          <p className="text-[var(--muted)] leading-relaxed">
            Lock in what you&apos;ll do and when you&apos;ll check back. Any expected lift stored
            here is a playbook estimate until you confirm with a what-if or a real outcome.
          </p>
          <div>
            <label className="block text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
              Action
            </label>
            <select
              value={decAction}
              onChange={(e) => setDecAction(e.target.value)}
              className="w-full bg-paper border border-mist px-2 py-2 text-ink text-sm rounded-control"
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
                className={`px-3 py-1.5 text-xs border rounded-control ${
                  decInterval === d
                    ? 'border-teal bg-teal-soft/30 text-ink'
                    : 'border-mist text-[var(--muted)] hover:text-ink'
                }`}
              >
                {d} days
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={submitDecision}
            disabled={decSaving || !decAction}
            className="btn-primary text-sm"
          >
            {decSaving ? 'Saving…' : 'Save follow-up'}
          </button>
          {decError && <p className="text-coral text-xs">{decError}</p>}
          {decStatus && <p className="text-teal text-xs leading-relaxed">{decStatus}</p>}
        </div>
      )}

      {simulateHref && (
        <Link to={simulateHref} className="btn-secondary text-sm w-full">
          {simulateLabel || 'Explore a what-if'}
        </Link>
      )}

      {(brief?.theme_rollup?.length ||
        primaryLever ||
        (result.insights && result.insights.length > 0) ||
        drivers.length > 0) && (
        <div className="space-y-4 border-t border-mist pt-5">
          <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
            Why it looks this way
          </h4>
          {brief?.theme_rollup && brief.theme_rollup.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {brief.theme_rollup.map((t) => (
                <span
                  key={t.category}
                  className="text-xs border border-mist px-2.5 py-1 text-ink rounded-control"
                >
                  {t.label}
                </span>
              ))}
            </div>
          )}
          {primaryLever?.suggestion && (
            <p className="text-sm text-ink leading-relaxed">
              <span className="font-medium">
                {primaryLever.display_name || humanize(String(primaryLever.feature || 'Driver'))}
              </span>
              <span className="text-[var(--muted)]"> — {primaryLever.suggestion}</span>
            </p>
          )}
          {result.insights && result.insights.length > 0 && (
            <ul className="space-y-2.5">
              {result.insights.slice(0, 4).map((ins, i) => (
                <li key={i} className="text-sm text-ink leading-relaxed border-l-2 border-mist pl-3">
                  {ins.text || ins.message || ins.reason}
                  {ins.suggestion && (
                    <span className="block mt-1 text-[var(--muted)]">{ins.suggestion}</span>
                  )}
                </li>
              ))}
            </ul>
          )}
          {drivers.length > 0 && !(result.insights && result.insights.length > 0) && (
            <ul className="space-y-2">
              {drivers.slice(0, 4).map((f) => {
                const impact = Number(f.impact ?? 0);
                const up = f.direction === 'increases' || impact > 0;
                return (
                  <li key={f.feature} className="text-sm flex justify-between gap-3">
                    <span className="text-ink">{f.label || humanize(f.feature)}</span>
                    <span className={`shrink-0 ${up ? 'text-coral' : 'text-teal'}`}>
                      {up ? 'raises risk' : 'lowers risk'}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}

      {explainError && (
        <p className="text-sm text-coral">Could not explain this case: {explainError}</p>
      )}

      <details className="border border-mist rounded-control group">
        <summary className="px-3 py-3 cursor-pointer text-sm list-none hover:bg-mist/20">
          <span className="font-medium text-ink">How sure should you be?</span>
          <span className="block text-xs text-[var(--muted)] mt-0.5">{trust.badge}</span>
        </summary>
        <div className="px-3 pb-4 space-y-4 border-t border-mist pt-3">
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
          {consistency?.plain && (
            <p className="text-sm text-[var(--muted)] leading-relaxed">{consistency.plain}</p>
          )}
        </div>
      </details>

      {canLog && (
        <details className="border border-mist rounded-control">
          <summary className="px-3 py-3 cursor-pointer text-sm list-none hover:bg-mist/20">
            <span className="font-medium text-ink">Record what happened</span>
            <span className="block text-xs text-[var(--muted)] mt-0.5">
              After you know the real result
            </span>
          </summary>
          <div className="px-3 pb-4 space-y-3 border-t border-mist pt-3 text-sm">
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
                  className={`px-3 py-1.5 text-xs border rounded-control ${
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
              <select
                value={fbAction}
                onChange={(e) => setFbAction(e.target.value)}
                className="w-full bg-paper border border-mist px-2 py-2 text-ink text-sm rounded-control"
              >
                <option value="">Action taken (optional)</option>
                {result.recommendations.map((r, i) => {
                  const code = r.action_code || r.action_name || r.name || `action_${i}`;
                  return (
                    <option key={code} value={code}>
                      {r.action_name || r.name || code}
                    </option>
                  );
                })}
              </select>
            )}
            <button
              type="button"
              onClick={submitFeedback}
              disabled={fbSaving || !fbOutcome}
              className="btn-secondary text-sm"
            >
              {fbSaving ? 'Saving…' : 'Save what happened'}
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
        </details>
      )}

      {children}
    </div>
  );
}

export default PredictionPanel;
