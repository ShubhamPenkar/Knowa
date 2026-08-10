import { useId, useMemo } from 'react';

export type TrustSpineProps = {
  /** Point estimate (probability or continuous value). */
  point: number;
  /** Conformal interval lower bound. */
  lower: number;
  /** Conformal interval upper bound. */
  upper: number;
  /** Target coverage level, e.g. 0.9 */
  level?: number;
  lowConfidence?: boolean;
  abstentionReason?: string | null;
  disagreement?: number | null;
  /** Domain for the track; defaults to [0, 1] for classification. */
  domain?: [number, number];
  /** Optional human label for the outcome. */
  outcomeLabel?: string;
  /** Prefer animation unless reduced motion or animate=false. */
  animate?: boolean;
  className?: string;
  /** Plain language for business users (hides technical meta). */
  businessCopy?: boolean;
  /** Override badge text (trust assessment). */
  badgeLabel?: string | null;
  /** Override footer explanation. */
  rangeNote?: string | null;
};

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

function formatVal(n: number, isProbDomain: boolean): string {
  if (isProbDomain) return `${Math.round(n * 100)}%`;
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(1);
  return n.toFixed(2);
}

/**
 * Signature product moment: range of plausible outcomes as a trust spine.
 */
export function TrustSpine({
  point,
  lower,
  upper,
  level = 0.9,
  lowConfidence = false,
  abstentionReason = null,
  disagreement = null,
  domain,
  outcomeLabel,
  animate = true,
  className = '',
  businessCopy = false,
  badgeLabel = null,
  rangeNote = null,
}: TrustSpineProps) {
  const labelId = useId();
  const descId = useId();

  const [dMin, dMax] = useMemo(() => {
    if (domain) return domain;
    const vals = [point, lower, upper];
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    if (minV >= 0 && maxV <= 1) return [0, 1] as [number, number];
    const pad = Math.max((maxV - minV) * 0.15, 0.05);
    return [minV - pad, maxV + pad] as [number, number];
  }, [domain, point, lower, upper]);

  const span = dMax - dMin || 1;
  const toPct = (v: number) => clamp01((v - dMin) / span) * 100;

  const lo = Math.min(lower, upper);
  const hi = Math.max(lower, upper);
  const left = toPct(lo);
  const right = toPct(hi);
  const width = Math.max(right - left, 0.8);
  const tick = toPct(point);
  const isProbDomain = dMin === 0 && dMax === 1;
  const coverage = Math.round(level * 100);

  const metaParts: string[] = businessCopy
    ? []
    : [`${coverage}% conformal coverage`];
  if (!businessCopy && disagreement != null && !Number.isNaN(disagreement)) {
    metaParts.push(`disagreement ${disagreement.toFixed(2)}`);
  }

  return (
    <figure
      className={`trust-band ${className}`}
      aria-labelledby={labelId}
      aria-describedby={descId}
    >
      <div className="flex flex-wrap items-start justify-between gap-3 mb-5">
        <div>
          <p className="page-kicker mb-1">
            {businessCopy ? 'How sure we are' : 'Calibrated confidence'}
          </p>
          <h3 id={labelId} className="font-display text-xl md:text-2xl font-semibold text-ink tracking-tight">
            {outcomeLabel
              ? outcomeLabel
              : businessCopy
                ? 'Likely range'
                : 'Prediction interval'}
          </h3>
        </div>
        {lowConfidence ? (
          <span className="badge bg-coral-soft text-ink border border-coral/30" role="status">
            {badgeLabel || (businessCopy ? 'Less sure — verify first' : 'Low confidence')}
          </span>
        ) : (
          <span className="badge bg-teal-soft/60 text-ink border border-teal/20">
            {badgeLabel || (businessCopy ? 'Clear enough to act' : 'Interval ready')}
          </span>
        )}
      </div>

      <div className="relative pt-2 pb-6 select-none">
        <div
          className="relative h-3 rounded-[2px] bg-mist/80"
          role="img"
          aria-label={`Estimate ${formatVal(point, isProbDomain)}, range ${formatVal(lo, isProbDomain)} to ${formatVal(hi, isProbDomain)}`}
        >
          <div
            className={`absolute top-0 h-full rounded-[2px] origin-left ${
              lowConfidence ? 'bg-coral/75' : 'bg-teal'
            } ${animate ? 'motion-safe:animate-spine-in' : ''}`}
            style={{
              left: `${left}%`,
              width: `${width}%`,
              ...(animate ? { animationDelay: '120ms' } : {}),
            }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-0.5 h-6 bg-ink rounded-sm z-10"
            style={{ left: `${tick}%` }}
            aria-hidden="true"
          />
        </div>

        <div className="relative mt-4 h-5 text-xs font-medium tabular-nums text-[var(--muted)]">
          <span className="absolute left-0">{formatVal(dMin, isProbDomain)}</span>
          <span
            className="absolute -translate-x-1/2 text-ink font-display text-sm font-semibold"
            style={{ left: `${Math.min(92, Math.max(8, tick))}%` }}
          >
            {formatVal(point, isProbDomain)}
          </span>
          <span className="absolute right-0">{formatVal(dMax, isProbDomain)}</span>
        </div>
        <div className="relative mt-1 h-4 text-[11px] tabular-nums text-[var(--muted)]">
          <span
            className="absolute -translate-x-1/2"
            style={{ left: `${Math.min(90, Math.max(6, left))}%` }}
          >
            {formatVal(lo, isProbDomain)}
          </span>
          <span
            className="absolute -translate-x-1/2"
            style={{ left: `${Math.min(94, Math.max(10, right))}%` }}
          >
            {formatVal(hi, isProbDomain)}
          </span>
        </div>
      </div>

      <figcaption id={descId} className="text-sm text-[var(--muted)] space-y-1.5">
        {businessCopy ? (
          <p>{rangeNote || 'Best estimate (tick) with a realistic range around it.'}</p>
        ) : (
          <>
            <p>
              Point estimate with a {coverage}% conformal interval
              {isProbDomain ? ' on probability' : ''}.
            </p>
            {metaParts.length > 0 && (
              <p className="text-xs tracking-wide uppercase text-ink/50 font-medium">
                {metaParts.join(' · ')}
              </p>
            )}
          </>
        )}
        {lowConfidence && abstentionReason && !businessCopy && (
          <p className="text-sm text-ink border-l-2 border-coral pl-3 mt-2">
            {abstentionReason}
          </p>
        )}
        {lowConfidence && businessCopy && (
          <p className="text-sm text-ink border-l-2 border-coral pl-3 mt-2">
            Prefer lighter actions first when certainty is soft.
          </p>
        )}
      </figcaption>
    </figure>
  );
}

export default TrustSpine;
