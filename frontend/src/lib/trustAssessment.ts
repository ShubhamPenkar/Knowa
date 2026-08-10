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
      badge = 'Wide uncertainty';
      summary =
        'The likely range is almost everything — use the % as a starting point and confirm with context.';
      rangeNote =
        'When the likely range covers nearly 0–100%, treat the estimate as directional only.';
    } else if (soft.reason === 'backend_abstention') {
      badge = 'Less sure on this one';
      summary =
        'We flagged this estimate as less reliable. Prefer a lighter next step until you can confirm.';
      rangeNote = 'Start with a lighter action while you gather more context.';
    } else {
      // mid_priority
      badge = 'Borderline priority';
      summary =
        'This sits between “calm” and “act.” Still useful for ranking — don’t over-weight the exact %.';
      rangeNote =
        'Mid-range cases are soft by design. Clear high or low risk stays firmer.';
    }
  } else {
    badge = 'Clear enough to plan';
    if (p >= 0.6) {
      summary =
        'Higher chance of the outcome and a clear priority. Reasonable to plan an intervention.';
    } else if (p >= 0.4) {
      summary = 'Moderate chance — a sensible watch-list item.';
    } else {
      summary =
        'Lower chance of the outcome. Fine to deprioritize versus elevated cases.';
    }
    rangeNote = 'Best estimate (tick) with a realistic range around it.';
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

/** Plain-language Don't-act reason for queue rows and banners. */
export function softReasonLabel(reason: SoftReason | string | null | undefined): string {
  switch (reason) {
    case 'near_full_band':
      return 'Likely range is too wide'
    case 'mid_priority':
      return 'Borderline priority'
    case 'backend_abstention':
      return 'Model is less sure'
    case 'regression_wide':
      return 'Wide estimate range'
    case 'none':
    case null:
    case undefined:
    case '':
      return ''
    default:
      return 'Elevated uncertainty'
  }
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
