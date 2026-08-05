/**
 * Soft-range assessment so residual conformal bands don't cry wolf on every case.
 *
 * Residual |y−p| intervals often use a large global quantile. Full lo/hi almost
 * always spans calm→act after clipping — that is not a useful per-case soft flag.
 *
 * Soft = prioritization is murky for *this* score: mid (watch) point, nearly open
 * residual band on [0,1], or backend abstention — not “far ends of residual bar
 * would map to different action tiers.”
 */

export type SoftReason =
  | 'none'
  | 'near_full_band'
  | 'mid_priority'
  | 'backend_abstention'
  | 'regression_wide';

export type TrustAssessment = {
  isSoft: boolean;
  softReason: SoftReason;
  badge: string;
  summary: string;
  rangeNote: string;
  width: number;
};

/** Action tiers used for prioritization (matches risk buckets, simplified). */
export function actionTier(p: number): 'calm' | 'watch' | 'act' {
  if (p >= 0.6) return 'act';
  if (p >= 0.4) return 'watch';
  return 'calm';
}

/** Nearly covers whole probability domain after residual clipping. */
export const NEAR_FULL_WIDTH = 0.9;

/** Kept for callers/tests; no longer a soft width floor by itself. */
export const TIER_CONFLICT_MIN_WIDTH = 0.35;

export function intervalIsSoft(opts: {
  point: number;
  lower: number;
  upper: number;
  lowConfidence?: boolean;
  isRegression?: boolean;
}): { isSoft: boolean; reason: SoftReason; width: number; lo: number; hi: number } {
  const lo = Math.min(opts.lower, opts.upper);
  const hi = Math.max(opts.lower, opts.upper);
  const width = hi - lo;
  const p = opts.point;

  if (opts.isRegression) {
    const scale = Math.max(Math.abs(p), 1.0);
    const soft = width > 0.75 * scale;
    return {
      isSoft: soft || Boolean(opts.lowConfidence),
      reason: soft
        ? 'regression_wide'
        : opts.lowConfidence
          ? 'backend_abstention'
          : 'none',
      width,
      lo,
      hi,
    };
  }

  // Float eps: residual clips often yield width ≈ 0.9 exactly
  if (width + 1e-9 >= NEAR_FULL_WIDTH) {
    return { isSoft: true, reason: 'near_full_band', width, lo, hi };
  }

  if (actionTier(p) === 'watch') {
    return { isSoft: true, reason: 'mid_priority', width, lo, hi };
  }

  if (opts.lowConfidence) {
    return { isSoft: true, reason: 'backend_abstention', width, lo, hi };
  }

  return { isSoft: false, reason: 'none', width, lo, hi };
}

export function assessTrust(opts: {
  point: number;
  lower: number;
  upper: number;
  lowConfidence?: boolean;
  isRegression?: boolean;
}): TrustAssessment {
  const soft = intervalIsSoft(opts);
  const p = opts.point;
  const isSoft = soft.isSoft;

  if (opts.isRegression) {
    return {
      isSoft,
      softReason: soft.reason,
      badge: isSoft ? 'Wide range — use carefully' : 'Range is usable',
      summary: isSoft
        ? 'The estimate comes with a wide band — check domain sense before big moves.'
        : 'The range is tight enough to plan around the estimate.',
      rangeNote: isSoft
        ? 'Wide band: treat the number as a guide, not a precise target.'
        : 'Best estimate (tick) with a realistic range around it.',
      width: soft.width,
    };
  }

  let badge: string;
  let summary: string;
  let rangeNote: string;

  if (isSoft) {
    if (soft.reason === 'near_full_band') {
      badge = 'Range nearly open';
      summary =
        'The plausible range covers almost everything — the point estimate is a weak plan by itself. Confirm with context.';
      rangeNote =
        'When the residual band fills most of 0–100%, treat the tick as directional only.';
    } else if (soft.reason === 'backend_abstention') {
      badge = 'Low certainty flag';
      summary =
        'Uncertainty policy flagged this estimate. Prefer a lighter next step until you can confirm.';
      rangeNote =
        'Soft because the model’s uncertainty gate fired — not only because the residual bar looks wide.';
    } else {
      // mid_priority
      badge = 'Mid-range priority';
      summary =
        'The chance sits between calm and act. Ranking still works; treat the exact % loosely.';
      rangeNote =
        'Soft for mid-band scores. Clear low / clear high stays firm even when residual bars look fat.';
    }
  } else {
    badge = 'Clear enough to plan';
    if (p >= 0.6) {
      summary =
        'Higher chance of the outcome and a clear priority band. Reasonable to plan intervention.';
    } else if (p >= 0.4) {
      summary =
        'Moderate chance — still a usable watch list item.';
    } else {
      summary =
        'Lower chance of the outcome. Fine to deprioritize vs elevated cases (residual bars can still look wide).';
    }
    rangeNote =
      'Soft is reserved for mid-range scores or nearly open residual bands — not every wide residual bar.';
  }

  return {
    isSoft,
    softReason: soft.reason,
    badge,
    summary,
    rangeNote,
    width: soft.width,
  };
}

export function matchLabel(
  knownYes: boolean | null | undefined,
  probability: number
): 'agrees' | 'conflicts' | 'unknown' {
  if (knownYes == null) return 'unknown';
  const predYes = probability >= 0.5;
  if (knownYes === predYes) return 'agrees';
  return 'conflicts';
}
