"""Soft-range rules for case trust UI and spot-check (aligned with frontend).

Residual conformal bands on probabilities often use a large global quantile
(|y−p| ~ 0.5–0.7). Lo/hi then almost always span calm→act after clipping — that
is a *global* property of residual intervals, not a useful per-case soft flag.

Soft means prioritization is *case-murky*:
  - point sits in the watch band,
  - residual band is nearly open on [0, 1] after clipping, or
  - backend abstention (model disagreement / uninformative width policy).
"""

from __future__ import annotations

from typing import Literal, Optional

SoftReason = Literal[
    "none",
    "near_full_band",
    "mid_priority",
    "backend_abstention",
    "regression_wide",
]

# After residual clip, ~0.9+ means the band fills almost the whole [0,1] domain.
NEAR_FULL_WIDTH = 0.9


def action_tier(p: float) -> Literal["calm", "watch", "act"]:
    if p >= 0.6:
        return "act"
    if p >= 0.4:
        return "watch"
    return "calm"


def interval_is_soft(
    *,
    point: float,
    lower: float,
    upper: float,
    low_confidence: bool = False,
    is_regression: bool = False,
) -> dict:
    lo = float(min(lower, upper))
    hi = float(max(lower, upper))
    width = hi - lo
    p = float(point)

    if is_regression:
        scale = max(abs(p), 1.0)
        soft_reg = width > 0.75 * scale
        if soft_reg:
            return {
                "is_soft": True,
                "reason": "regression_wide",
                "width": width,
                "lower": lo,
                "upper": hi,
            }
        if low_confidence:
            return {
                "is_soft": True,
                "reason": "backend_abstention",
                "width": width,
                "lower": lo,
                "upper": hi,
            }
        return {
            "is_soft": False,
            "reason": "none",
            "width": width,
            "lower": lo,
            "upper": hi,
        }

    # Nearly uninformative residual band (often mid scores after [0,1] clip)
    # Float eps: residual clips often yield width ≈ 0.9 exactly
    if width + 1e-9 >= NEAR_FULL_WIDTH:
        return {
            "is_soft": True,
            "reason": "near_full_band",
            "width": width,
            "lower": lo,
            "upper": hi,
        }

    # Watch-band point: priority is inherently less firm than clear calm/act
    if action_tier(p) == "watch":
        return {
            "is_soft": True,
            "reason": "mid_priority",
            "width": width,
            "lower": lo,
            "upper": hi,
        }

    # Backend policy (disagreement or width threshold) still surfaces as soft
    if low_confidence:
        return {
            "is_soft": True,
            "reason": "backend_abstention",
            "width": width,
            "lower": lo,
            "upper": hi,
        }

    return {
        "is_soft": False,
        "reason": "none",
        "width": width,
        "lower": lo,
        "upper": hi,
    }
