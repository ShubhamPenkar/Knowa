/**
 * HalftoneGlow — KNOWA signature/accent element.
 *
 * Renders a canvas-based dot-matrix "halftone print" glow behind its children.
 * Use ONLY for primary CTA moments (1–2 per page maximum), e.g. a simulate/act
 * control on the Trust Spine panel. Not a general-purpose page background —
 * overuse defeats the purpose.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';

export type HalftoneFocalPoint = {
  /** Fractional x position in the container (0–1). */
  x: number;
  /** Fractional y position in the container (0–1). */
  y: number;
  color: string;
};

export type HalftoneGlowProps = {
  children?: ReactNode;
  focalPoints?: HalftoneFocalPoint[];
  background?: string;
  /** Distance between grid dots in CSS pixels. */
  dotSpacing?: number;
  /** Maximum filled-dot radius in CSS pixels (near a focal point). */
  maxDotRadius?: number;
  /** Glow falloff radius in CSS pixels from each focal point. */
  glowRadius?: number;
  /**
   * When true, slowly drifts focal points. Forced off when
   * prefers-reduced-motion: reduce.
   */
  animated?: boolean;
  className?: string;
  style?: CSSProperties;
};

type CachedDot = {
  x: number;
  y: number;
  /** Per-focal strengths (0–1), length === focalPoints.length */
  strengths: number[];
};

type DotCache = {
  width: number;
  height: number;
  spacing: number;
  maxDotRadius: number;
  glowRadius: number;
  focalKey: string;
  dots: CachedDot[];
};

function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function parseColor(hex: string): { r: number; g: number; b: number } {
  const raw = hex.replace('#', '').trim();
  const full =
    raw.length === 3
      ? raw
          .split('')
          .map((c) => c + c)
          .join('')
      : raw.padEnd(6, '0').slice(0, 6);
  const n = parseInt(full, 16);
  return {
    r: (n >> 16) & 255,
    g: (n >> 8) & 255,
    b: n & 255,
  };
}

function buildDotCache(
  width: number,
  height: number,
  spacing: number,
  maxDotRadius: number,
  glowRadius: number,
  focals: HalftoneFocalPoint[],
  focalKey: string
): DotCache {
  const dots: CachedDot[] = [];
  if (width <= 0 || height <= 0 || spacing <= 0) {
    return { width, height, spacing, maxDotRadius, glowRadius, focalKey, dots };
  }

  const glow = Math.max(glowRadius, 1);
  // Slight inset so edge dots don’t clip harsh
  for (let y = spacing * 0.5; y < height; y += spacing) {
    for (let x = spacing * 0.5; x < width; x += spacing) {
      const strengths: number[] = [];
      let any = false;
      for (const f of focals) {
        const fx = f.x * width;
        const fy = f.y * height;
        const dx = x - fx;
        const dy = y - fy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        // Smooth falloff: full at center → 0 at glowRadius
        const t = 1 - Math.min(dist / glow, 1);
        const strength = t * t * t; // ease-out cubic density
        strengths.push(strength);
        if (strength > 0.01) any = true;
      }
      if (any || focals.length === 0) {
        dots.push({ x, y, strengths });
      }
    }
  }

  return { width, height, spacing, maxDotRadius, glowRadius, focalKey, dots };
}

function paintDots(
  ctx: CanvasRenderingContext2D,
  cache: DotCache,
  focals: HalftoneFocalPoint[],
  background: string,
  /** Pixel offsets per focal for animation drift */
  drifts: Array<{ dx: number; dy: number }>,
  maxDotRadius: number,
  glowRadius: number
) {
  const { width, height, dots } = cache;
  // Assumes caller already applied ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  // so coordinates are CSS pixels.

  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  if (focals.length === 0) return;

  const colors = focals.map((f) => parseColor(f.color));
  const glow = Math.max(glowRadius, 1);

  for (const dot of dots) {
    let bestStrength = 0;
    let mixR = 0;
    let mixG = 0;
    let mixB = 0;
    let mixW = 0;

    for (let i = 0; i < focals.length; i++) {
      const f = focals[i];
      const drift = drifts[i] || { dx: 0, dy: 0 };
      const fx = f.x * width + drift.dx;
      const fy = f.y * height + drift.dy;
      const dx = dot.x - fx;
      const dy = dot.y - fy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const t = 1 - Math.min(dist / glow, 1);
      const strength = t * t * t;
      if (strength <= 0.01) continue;
      const c = colors[i];
      mixR += c.r * strength;
      mixG += c.g * strength;
      mixB += c.b * strength;
      mixW += strength;
      if (strength > bestStrength) bestStrength = strength;
    }

    if (mixW <= 0.01 || bestStrength <= 0.01) continue;

    const r = (maxDotRadius * bestStrength) / 1.15;
    if (r < 0.25) continue;

    const opacity = Math.min(0.95, bestStrength * 1.05);
    const rr = Math.round(mixR / mixW);
    const gg = Math.round(mixG / mixW);
    const bb = Math.round(mixB / mixW);

    ctx.beginPath();
    ctx.fillStyle = `rgba(${rr},${gg},${bb},${opacity})`;
    ctx.arc(dot.x, dot.y, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

const DEFAULT_FOCALS: HalftoneFocalPoint[] = [
  { x: 0.36, y: 0.5, color: '#FF5A1F' },
  { x: 0.64, y: 0.5, color: '#00C8B4' },
];

export function HalftoneGlow({
  children,
  focalPoints = DEFAULT_FOCALS,
  background = '#0A0908',
  dotSpacing = 13,
  maxDotRadius = 4,
  glowRadius = 220,
  animated = false,
  className = '',
  style,
}: HalftoneGlowProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cacheRef = useRef<DotCache | null>(null);
  const rafRef = useRef<number | null>(null);
  const sizeRef = useRef({ w: 0, h: 0, dpr: 1 });
  const [reduceMotion, setReduceMotion] = useState(prefersReducedMotion);

  const focalKey = useMemo(
    () =>
      focalPoints.map((f) => `${f.x},${f.y},${f.color}`).join('|') +
      `|${dotSpacing}|${maxDotRadius}|${glowRadius}|${background}`,
    [focalPoints, dotSpacing, maxDotRadius, glowRadius, background]
  );

  const shouldAnimate = animated && !reduceMotion;

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => setReduceMotion(mq.matches);
    onChange();
    mq.addEventListener?.('change', onChange);
    return () => mq.removeEventListener?.('change', onChange);
  }, []);

  const ensureCache = useCallback(
    (w: number, h: number) => {
      const existing = cacheRef.current;
      if (
        existing &&
        existing.width === w &&
        existing.height === h &&
        existing.focalKey === focalKey
      ) {
        return existing;
      }
      const next = buildDotCache(
        w,
        h,
        dotSpacing,
        maxDotRadius,
        glowRadius,
        focalPoints,
        focalKey
      );
      cacheRef.current = next;
      return next;
    },
    [dotSpacing, maxDotRadius, glowRadius, focalPoints, focalKey]
  );

  const draw = useCallback(
    (drifts: Array<{ dx: number; dy: number }>) => {
      const canvas = canvasRef.current;
      const { w, h, dpr } = sizeRef.current;
      if (!canvas || w <= 0 || h <= 0) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Reset + apply DPR scale so drawing uses CSS pixels
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const cache = ensureCache(w, h);
      paintDots(ctx, cache, focalPoints, background, drifts, maxDotRadius, glowRadius);
    },
    [ensureCache, focalPoints, background, maxDotRadius, glowRadius]
  );

  const resizeAndDraw = useCallback(() => {
    const wrapper = wrapperRef.current;
    const canvas = canvasRef.current;
    if (!wrapper || !canvas) return;

    const rect = wrapper.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);

    sizeRef.current = { w, h, dpr };
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    // Invalidate cache dimensions
    if (
      cacheRef.current &&
      (cacheRef.current.width !== w || cacheRef.current.height !== h)
    ) {
      cacheRef.current = null;
    }

    draw(focalPoints.map(() => ({ dx: 0, dy: 0 })));
  }, [draw, focalPoints]);

  // ResizeObserver
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;

    resizeAndDraw();

    const ro = new ResizeObserver(() => {
      resizeAndDraw();
    });
    ro.observe(wrapper);

    return () => {
      ro.disconnect();
    };
  }, [resizeAndDraw]);

  // Recompute when visual props change (not every RAF)
  useEffect(() => {
    cacheRef.current = null;
    resizeAndDraw();
  }, [focalKey, resizeAndDraw]);

  // Optional subtle focal drift
  useEffect(() => {
    if (!shouldAnimate) {
      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      draw(focalPoints.map(() => ({ dx: 0, dy: 0 })));
      return;
    }

    const start = performance.now();
    const periodMs = 8000;
    const amplitude = 15;

    const tick = (now: number) => {
      const t = ((now - start) % periodMs) / periodMs;
      // Smooth loop
      const angle = t * Math.PI * 2;
      const drifts = focalPoints.map((_, i) => {
        const phase = i * 1.7;
        return {
          dx: Math.cos(angle + phase) * amplitude,
          dy: Math.sin(angle * 0.85 + phase) * amplitude * 0.65,
        };
      });
      draw(drifts);
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [shouldAnimate, draw, focalPoints]);

  return (
    <div
      ref={wrapperRef}
      className={`relative overflow-hidden ${className}`}
      style={style}
    >
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 block h-full w-full"
        aria-hidden="true"
      />
      <div className="relative z-10 flex h-full min-h-[inherit] w-full items-center justify-center p-3">
        {children}
      </div>
    </div>
  );
}

export default HalftoneGlow;
