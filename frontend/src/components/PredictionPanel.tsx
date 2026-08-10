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
  intervenability?: string;
  blindspot?: boolean;
  blindspot_codes?: string[];
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
      blindspot_reranked?: boolean;
      previous_primary_feature?: string;
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
    effectiveness_rate?: number | null;
    n_outcomes?: number;
    success_n?: number;
    learning_applied?: boolean;
    learning_note?: string | null;
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
    uses_feedback_effectiveness?: boolean;
    effectiveness_n_actions?: number;
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
  blindspot_warnings?: Array<{
    code?: string;
    feature?: string;
    severity?: string;
    plain?: string;
  }>;
  blindspots?: {
    warnings?: Array<{
      code?: string;
      feature?: string;
      severity?: string;
      plain?: string;
    }>;
    plain_summary?: string;
    layer?: string;
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
  const blindspotWarnings =
    result.blindspot_warnings || result.blindspots?.warnings || [];

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
  const [showMoreActions, setShowMoreActions] = useState(false);

  const formatApiError = (detail: unknown, fallback: string) => {
    if (detail == null) return fallback;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) =>
          typeof d === 'object' && d && 'msg' in d
            ? String((d as { msg?: string }).msg || JSON.stringify(d))
            : JSON.stringify(d)
        )
        .join('; ');
    }
    if (typeof detail === 'object') {
      const obj = detail as { message?: string };
      if (obj.message) return obj.message;
      try {
        return JSON.stringify(detail);
      } catch {
        return fallback;
      }
    }
    return String(detail);
  };

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
        setDecError(formatApiError(data.detail, 'Could not save follow-up'));
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
        setFbError(formatApiError(data.detail, 'Could not save'));
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

  const recs = result.recommendations || [];
  const topRec = recs[0];
  const extraRecs = recs.slice(1, 3);

  const renderRecCard = (
    r: (typeof recs)[number],
    i: number,
    opts?: { emphasize?: boolean }
  ) => {
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
        className={`text-sm border rounded-control px-3 py-3 ${
          opts?.emphasize ? 'border-teal/35 bg-teal-soft/10' : 'border-mist'
        }`}
      >
        <div className="font-medium text-ink">
          {opts?.emphasize ? 'Top action: ' : `${i + 1}. `}
          {r.action_name || r.name}
        </div>
        <p className="mt-1 text-xs text-[var(--muted)]">
          {punch} · {effort}
          {r.implementation_time ? ` · ${r.implementation_time}` : ''}
          {r.learning_applied ? ' · learned from outcomes' : ''}
        </p>
        {(r.description || r.reasoning) && (
          <p className="mt-2 text-[var(--muted)] leading-relaxed">
            {r.reasoning || r.description}
          </p>
        )}
        {r.learning_note && (
          <p className="mt-2 text-xs text-teal leading-relaxed">{r.learning_note}</p>
        )}
      </li>
    );
  };

  return (
    <div className="space-y-5">
      {!isReg && (
        <div className="case-brief-stage">
          <div className="px-5 pt-5 pb-4 md:px-6 md:pt-6">
            <p className="page-kicker mb-1">Brief</p>
            <p className="font-display text-2xl md:text-[1.75rem] font-semibold text-ink tracking-tight leading-tight">
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
          <TrustSpine
            point={point}
            lower={lower}
            upper={upper}
            level={level}
            lowConfidence={trust.isSoft}
            abstentionReason={result.abstention_reason}
            disagreement={null}
            domain={[0, 1]}
            outcomeLabel={title}
            businessCopy
            badgeLabel={trust.badge}
            rangeNote={trust.rangeNote}
          />
          {consistency?.plain && (
            <p className="px-5 pb-5 md:px-6 md:pb-6 text-sm text-[var(--muted)] leading-relaxed border-t border-mist pt-3">
              {consistency.plain}
            </p>
          )}
        </div>
      )}

      {/* Act first: top recommendation + save follow-up */}
      {topRec && (
        <div className="space-y-3">
          <h4 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-teal">
            What to do next
          </h4>
          {trust.isSoft && (
            <p className="text-sm text-[var(--muted)] leading-relaxed">
              Certainty is soft — prefer lighter check-ins first, and confirm with a what-if before
              big spends.
            </p>
          )}
          <ol className="space-y-3">{renderRecCard(topRec, 0, { emphasize: true })}</ol>
          {extraRecs.length > 0 && (
            <>
              <button
                type="button"
                onClick={() => setShowMoreActions((v) => !v)}
                className="text-xs font-medium text-teal hover:underline"
              >
                {showMoreActions ? 'Hide other actions' : `More actions (${extraRecs.length})`}
              </button>
              {showMoreActions && (
                <ol className="space-y-3">
                  {extraRecs.map((r, i) => renderRecCard(r, i + 1))}
                </ol>
              )}
            </>
          )}
        </div>
      )}

      {canCommit && (
        <div className="border border-teal/30 bg-teal-soft/15 px-3 py-4 text-sm space-y-3 rounded-control">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-teal">Save what you&apos;ll do</div>
            <p className="text-[var(--muted)] leading-relaxed mt-1">
              Commit a follow-up and a check-back date. Expected lift is a playbook estimate until
              you confirm with a what-if or a real outcome.
            </p>
          </div>
          <div>
            <label className="block text-[11px] uppercase tracking-wide text-[var(--muted)] mb-1">
              Action
            </label>
            <select
              value={decAction}
              onChange={(e) => setDecAction(e.target.value)}
              className="w-full bg-paper border border-mist px-2 py-2 text-ink text-sm rounded-control"
            >
              {recs.map((r, i) => {
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
          <HalftoneGlow
            className="min-h-[2.75rem] rounded-control overflow-hidden"
            background="transparent"
            focalPoints={[
              { x: 0.35, y: 0.5, color: '#00C8B4' },
              { x: 0.7, y: 0.5, color: '#FF5A1F' },
            ]}
            dotSpacing={14}
            maxDotRadius={3}
            glowRadius={140}
          >
            <button
              type="button"
              onClick={submitDecision}
              disabled={decSaving || !decAction}
              className="btn-primary text-sm w-full relative z-10"
            >
              {decSaving ? 'Saving…' : 'Save follow-up'}
            </button>
          </HalftoneGlow>
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
        drivers.length > 0 ||
        blindspotWarnings.length > 0) && (
        <details className="border border-mist rounded-control">
          <summary className="px-3 py-3 cursor-pointer text-sm list-none hover:bg-mist/20">
            <span className="font-medium text-ink">Why it looks this way</span>
            <span className="block text-xs text-[var(--muted)] mt-0.5">
              Drivers, context flags, and suggestions
            </span>
          </summary>
          <div className="px-3 pb-4 space-y-4 border-t border-mist pt-3">
            {blindspotWarnings.length > 0 && (
              <div className="border border-mist px-3 py-3 space-y-2 bg-mist/20">
                <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-ink">
                  Treat as context, not a dial
                </div>
                <ul className="space-y-2">
                  {blindspotWarnings.slice(0, 3).map((w, i) => (
                    <li
                      key={`${w.code || 'bs'}-${w.feature || i}`}
                      className={`text-sm leading-relaxed border-l-2 pl-3 ${
                        w.severity === 'critical'
                          ? 'border-coral text-ink'
                          : 'border-mist text-[var(--muted)]'
                      }`}
                    >
                      {w.plain}
                    </li>
                  ))}
                </ul>
              </div>
            )}
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
        </details>
      )}

      {explainError && (
        <p className="text-sm text-coral">Could not explain this case: {explainError}</p>
      )}

      {isReg && (
        <TrustSpine
          point={point}
          lower={lower}
          upper={upper}
          level={level}
          lowConfidence={trust.isSoft}
          abstentionReason={result.abstention_reason}
          disagreement={null}
          outcomeLabel={title}
          businessCopy
          badgeLabel={trust.badge}
          rangeNote={trust.rangeNote}
        />
      )}

      {canLog && (
        <details className="border border-mist rounded-control">
          <summary className="px-3 py-3 cursor-pointer text-sm list-none hover:bg-mist/20">
            <span className="font-medium text-ink">Log what already happened</span>
            <span className="block text-xs text-[var(--muted)] mt-0.5">
              Different from saving a follow-up — use this after you know the real result
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
            {recs.length > 0 && (
              <select
                value={fbAction}
                onChange={(e) => setFbAction(e.target.value)}
                className="w-full bg-paper border border-mist px-2 py-2 text-ink text-sm rounded-control"
              >
                <option value="">Action taken (optional)</option>
                {recs.map((r, i) => {
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
              {fbSaving ? 'Saving…' : 'Save outcome log'}
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
