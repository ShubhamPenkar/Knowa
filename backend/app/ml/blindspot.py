"""B4 Causal blindspot heuristics.

Flags when SHAP/LIME drivers look confounded or not intervention-ready.
This is a "needs scrutiny" trust layer — not a claim of causal fact.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from app.insights.feature_mapping import get_feature_info, resolve_feature_key


# Identity / demographic / immutable-ish — never treat as a short-term dial
_HARD_LOW_CANONICAL = {
    "age",
    "gender",
    "senior_citizen",
    "geography",
    "partner",
    "dependents",
    "marital_status",
    "race",
    "ethnicity",
    "nationality",
    "employee_number",
    "employee_count",
    "over18",
    "standard_hours",
    "customer_id",
}

# Strong association but slow / non-dial for near-term action
_CONTEXT_CANONICAL = {
    "tenure",
    "total_charges",
    "years_at_company",
    "num_companies_worked",
    "totalworkingyears",
    "years_in_current_role",
    "years_since_last_promotion",
    "distance_from_home",
    "credit_score",
}

_HARD_LOW_NAME_BITS = (
    "gender",
    "seniorcitizen",
    "senior_citizen",
    "ethnicity",
    "race",
    "nationality",
    "employee_number",
    "employeenumber",
    "customerid",
    "customer_id",
)


def _norm(name: str) -> str:
    return (
        str(name or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _is_missing_indicator(feature: str) -> bool:
    n = _norm(feature)
    return n.endswith("__is_missing") or n.endswith("_is_missing") or n.endswith("_missing")


def _intervenability(
    feature: str,
    feature_config: Optional[dict[str, Any]],
) -> tuple[str, Optional[str]]:
    """Return (level, reason_code) where level is high|medium|low."""
    n = _norm(feature)
    cfg = (feature_config or {}).get(feature) if feature_config else None
    if isinstance(cfg, dict):
        if cfg.get("derived") or _is_missing_indicator(feature):
            return "low", "derived_or_missingness"
        ftype = str(cfg.get("type") or "").lower()
        if ftype in ("id", "identifier"):
            return "low", "non_intervenable"

    if _is_missing_indicator(feature):
        return "low", "derived_or_missingness"

    compact = n.replace("_", "")
    if any(bit.replace("_", "") in compact for bit in _HARD_LOW_NAME_BITS):
        return "low", "non_intervenable"

    canon = resolve_feature_key(feature)
    if canon in _HARD_LOW_CANONICAL:
        return "low", "non_intervenable"
    if canon in _CONTEXT_CANONICAL:
        return "low", "context_not_dial"

    info = get_feature_info(feature)
    has_hint = bool(info.get("action_hint_risk") or info.get("action_hint_protect"))
    category = str(info.get("category") or "other")
    if category == "other" and not has_hint and not info.get("canonical"):
        return "medium", "unclear_lever"
    if has_hint:
        return "high", None
    if category in ("engagement", "services", "contract", "support", "financial"):
        return "medium", None
    return "medium", "unclear_lever"


def _display(feature: str) -> str:
    info = get_feature_info(feature)
    return str(info.get("display_name") or feature.replace("_", " ").title())


def _encode_target(
    series: pd.Series,
    positive_label: Optional[str],
) -> Optional[pd.Series]:
    if series is None or series.empty:
        return None
    pos = str(positive_label or "1").strip().lower()
    low = series.astype(str).str.strip().str.lower()
    if pos and (low == pos).any():
        return (low == pos).astype(float)
    # numeric / boolean fallback
    try:
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().sum() >= max(20, int(0.5 * len(series))):
            return num.astype(float)
    except Exception:
        pass
    return None


def _feature_numeric(series: pd.Series) -> Optional[pd.Series]:
    if series is None:
        return None
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    # binary-ish categoricals
    uniq = {str(v).strip().lower() for v in series.dropna().unique()[:12]}
    if uniq <= {"yes", "no", "true", "false", "1", "0", "y", "n"}:
        mapping = {
            "yes": 1.0,
            "true": 1.0,
            "1": 1.0,
            "y": 1.0,
            "no": 0.0,
            "false": 0.0,
            "0": 0.0,
            "n": 0.0,
        }
        return series.astype(str).str.strip().str.lower().map(mapping)
    try:
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().mean() >= 0.8:
            return num.astype(float)
    except Exception:
        pass
    return None


def _pairwise_proxy(
    training_data: pd.DataFrame,
    features: list[str],
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    nums: dict[str, pd.Series] = {}
    for f in features:
        if f not in training_data.columns:
            continue
        enc = _feature_numeric(training_data[f])
        if enc is not None and enc.notna().sum() >= 40:
            nums[f] = enc
    keys = list(nums.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            joined = pd.concat([nums[a], nums[b]], axis=1).dropna()
            if len(joined) < 40:
                continue
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if np.isfinite(corr) and abs(corr) >= threshold:
                pairs.append((a, b, corr))
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs


def _simpson_flip(
    training_data: pd.DataFrame,
    feature: str,
    target_column: str,
    positive_label: Optional[str],
    min_segment_n: int = 30,
) -> Optional[dict[str, Any]]:
    if feature not in training_data.columns or target_column not in training_data.columns:
        return None
    y = _encode_target(training_data[target_column], positive_label)
    x = _feature_numeric(training_data[feature])
    if y is None or x is None:
        return None
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 80:
        return None
    overall = float(frame["x"].corr(frame["y"]))
    if not np.isfinite(overall) or abs(overall) < 0.05:
        return None

    # Pick a low-cardinality segmenter among other columns
    candidates = []
    for col in training_data.columns:
        if col in (feature, target_column):
            continue
        nunique = training_data[col].nunique(dropna=True)
        if 2 <= nunique <= 6:
            candidates.append((nunique, col))
    candidates.sort()
    for _, seg_col in candidates[:4]:
        flips = []
        for seg_val, grp in training_data.groupby(seg_col):
            idx = grp.index.intersection(frame.index)
            sub = frame.loc[idx]
            if len(sub) < min_segment_n:
                continue
            c = float(sub["x"].corr(sub["y"]))
            if np.isfinite(c) and abs(c) >= 0.05 and (c * overall) < 0:
                flips.append(
                    {
                        "segment": f"{seg_col}={seg_val}",
                        "corr": round(c, 3),
                        "n": int(len(sub)),
                    }
                )
        if flips:
            return {
                "overall_corr": round(overall, 3),
                "flips": flips[:3],
                "segment_column": seg_col,
            }
    return None


def detect_blindspots(
    *,
    top_factors: list[dict[str, Any]],
    features: Optional[dict[str, Any]] = None,
    feature_config: Optional[dict[str, Any]] = None,
    consistency: Optional[dict[str, Any]] = None,
    training_data: Optional[pd.DataFrame] = None,
    target_column: Optional[str] = None,
    target_positive_label: Optional[str] = None,
    outcome_label: str = "the outcome",
    max_warnings: int = 3,
) -> dict[str, Any]:
    """
    Inspect top explanation drivers and emit scrutiny warnings.

    Returns warnings, per-driver flags, and a short plain summary.
    """
    del features  # reserved for future case-specific probes
    drivers = list(top_factors or [])[:5]
    warnings: list[dict[str, Any]] = []
    driver_flags: dict[str, dict[str, Any]] = {}
    outcome = (outcome_label or "the outcome").replace("_", " ").strip()

    ranked_names = []
    for i, d in enumerate(drivers):
        feat = str(d.get("feature") or "")
        if not feat:
            continue
        ranked_names.append(feat)
        level, reason = _intervenability(feat, feature_config)
        driver_flags[feat] = {
            "intervenability": level,
            "blindspot": level == "low",
            "codes": [reason] if reason else [],
            "shap_rank": i + 1,
        }

    # 1) Non-intervenable / context-not-dial on top drivers
    for feat, flags in driver_flags.items():
        codes = flags.get("codes") or []
        if "non_intervenable" in codes:
            warnings.append(
                {
                    "code": "non_intervenable",
                    "feature": feat,
                    "severity": "warning",
                    "plain": (
                        f"{_display(feat)} helps explain {outcome} here, but it is not a "
                        f"short-term lever — use it for targeting, not as the action."
                    ),
                    "evidence": {
                        "shap_rank": flags.get("shap_rank"),
                        "intervenability": "low",
                    },
                }
            )
        elif "context_not_dial" in codes:
            warnings.append(
                {
                    "code": "context_not_dial",
                    "feature": feat,
                    "severity": "warning",
                    "plain": (
                        f"{_display(feat)} often correlates with {outcome}, but changing it "
                        f"quickly is rarely realistic — treat it as context, not a dial."
                    ),
                    "evidence": {
                        "shap_rank": flags.get("shap_rank"),
                        "intervenability": "low",
                    },
                }
            )
        elif "derived_or_missingness" in codes:
            warnings.append(
                {
                    "code": "missingness_artifact",
                    "feature": feat,
                    "severity": "info",
                    "plain": (
                        f"{_display(feat)} looks like a missingness/derived pattern — "
                        f"not a business dial you can turn."
                    ),
                    "evidence": {
                        "shap_rank": flags.get("shap_rank"),
                        "intervenability": "low",
                    },
                }
            )

    # 2) Proxy / collider heuristic among top drivers
    if training_data is not None and len(ranked_names) >= 2:
        for a, b, corr in _pairwise_proxy(training_data, ranked_names)[:2]:
            warnings.append(
                {
                    "code": "proxy_pair",
                    "feature": a,
                    "severity": "warning",
                    "plain": (
                        f"{_display(a)} and {_display(b)} move together "
                        f"(corr {corr:+.2f}) — one may be a proxy for the other. "
                        f"Don’t treat them as independent levers."
                    ),
                    "evidence": {
                        "paired_feature": b,
                        "correlation": round(float(corr), 3),
                    },
                }
            )
            for f in (a, b):
                if f in driver_flags:
                    driver_flags[f]["blindspot"] = True
                    codes = driver_flags[f].setdefault("codes", [])
                    if "proxy_pair" not in codes:
                        codes.append("proxy_pair")

    # 3) Simpson / segment reversal on #1 driver
    if (
        training_data is not None
        and target_column
        and ranked_names
    ):
        top = ranked_names[0]
        flip = _simpson_flip(
            training_data,
            top,
            target_column,
            target_positive_label,
        )
        if flip:
            seg = flip["flips"][0]["segment"]
            warnings.append(
                {
                    "code": "segment_reversal",
                    "feature": top,
                    "severity": "critical",
                    "plain": (
                        f"The link between {_display(top)} and {outcome} flips in some "
                        f"segments (e.g. {seg}). Needs scrutiny before you act on it."
                    ),
                    "evidence": flip,
                }
            )
            if top in driver_flags:
                driver_flags[top]["blindspot"] = True
                codes = driver_flags[top].setdefault("codes", [])
                if "segment_reversal" not in codes:
                    codes.append("segment_reversal")

    # 4) Consistency trap — reasons agree but top driver isn't a lever
    trust = (consistency or {}).get("trust_level")
    top_flags = driver_flags.get(ranked_names[0]) if ranked_names else None
    if (
        trust in ("high", "medium")
        and top_flags
        and top_flags.get("intervenability") == "low"
    ):
        warnings.append(
            {
                "code": "consistency_trap",
                "feature": ranked_names[0],
                "severity": "info",
                "plain": (
                    "Explanation methods largely agree — but agreement is not causation. "
                    f"{_display(ranked_names[0])} still isn’t a safe short-term intervention."
                ),
                "evidence": {
                    "trust_level": trust,
                    "intervenability": "low",
                },
            }
        )

    # Deduplicate by code+feature, keep severity order, cap
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    seen: set[tuple[str, str]] = set()
    uniq: list[dict[str, Any]] = []
    for w in sorted(warnings, key=lambda x: severity_rank.get(x.get("severity"), 9)):
        key = (str(w.get("code")), str(w.get("feature")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(w)
    uniq = uniq[: max(1, int(max_warnings))]

    if not uniq:
        plain = (
            "No major blindspots flagged on the top drivers — still treat explanations "
            "as associative, not proven cause."
        )
    elif len(uniq) == 1:
        plain = uniq[0]["plain"]
    else:
        plain = (
            f"{len(uniq)} scrutiny notes on top drivers — treat some signals as context, "
            f"not dials."
        )

    # Prefer an actionable primary lever among risk-raising drivers
    preferred = None
    for d in drivers:
        feat = str(d.get("feature") or "")
        if not feat:
            continue
        impact = float(d.get("impact") or 0)
        direction = str(d.get("direction") or "")
        raises = direction == "increases" or impact > 0
        if not raises:
            continue
        flags = driver_flags.get(feat) or {}
        if flags.get("intervenability") == "low" or flags.get("blindspot"):
            continue
        preferred = feat
        break

    return {
        "warnings": uniq,
        "driver_flags": driver_flags,
        "preferred_primary_feature": preferred,
        "plain_summary": plain,
        "n_warnings": len(uniq),
        "layer": "B4_causal_blindspots",
    }


def annotate_drivers(
    drivers: list[dict[str, Any]],
    blindspots: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach intervenability / blindspot flags onto driver dicts (copy)."""
    flags = (blindspots or {}).get("driver_flags") or {}
    out = []
    for d in drivers or []:
        item = dict(d)
        feat = str(item.get("feature") or "")
        meta = flags.get(feat) or {}
        if meta:
            item["intervenability"] = meta.get("intervenability")
            item["blindspot"] = bool(meta.get("blindspot"))
            if meta.get("codes"):
                item["blindspot_codes"] = list(meta["codes"])
        out.append(item)
    return out
