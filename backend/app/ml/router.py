"""Dataset-aware model routing (Phase 1b).

Chooses between a foundation-model path (small/medium tabular) and the
stacked ensemble path (larger / wider data) from lightweight dataset stats.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import pandas as pd

Strategy = Literal["foundation_model", "ensemble"]


@dataclass(frozen=True)
class RoutingDecision:
    """Result of inspecting a training frame."""

    strategy: Strategy
    reason: str
    row_count: int
    feature_count: int
    categorical_fraction: float
    missing_density: float
    forced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_categorical_series(s: pd.Series) -> bool:
    return (
        s.dtype == object
        or str(s.dtype) in ("category", "string", "str")
        or pd.api.types.is_bool_dtype(s)
    )


def inspect_features(
    X: pd.DataFrame,
    *,
    max_sample_rows: int = 50_000,
) -> dict[str, float | int]:
    """Compute cheap data-profile stats used for routing."""
    if X is None or len(X.columns) == 0:
        return {
            "row_count": 0,
            "feature_count": 0,
            "categorical_fraction": 0.0,
            "missing_density": 0.0,
        }

    n_rows = int(len(X))
    n_features = int(X.shape[1])
    sample = X if n_rows <= max_sample_rows else X.sample(max_sample_rows, random_state=42)

    cat_count = sum(1 for col in sample.columns if _is_categorical_series(sample[col]))
    missing = float(sample.isna().to_numpy().mean()) if sample.size else 0.0

    return {
        "row_count": n_rows,
        "feature_count": n_features,
        "categorical_fraction": float(cat_count / max(n_features, 1)),
        "missing_density": missing,
    }


def route_training(
    X: pd.DataFrame,
    *,
    max_foundation_rows: int = 10_000,
    max_foundation_features: int = 500,
    force_strategy: Optional[Strategy] = None,
) -> RoutingDecision:
    """
    Route small/medium tabular data to foundation models; larger sets to ensemble.

    Rules (in order):
      1. Explicit force_strategy wins.
      2. Zero / empty data → ensemble (safe default for training pipeline errors later).
      3. Too many features (> max_foundation_features) → ensemble.
      4. row_count <= max_foundation_rows → foundation_model.
      5. Otherwise → ensemble.
    """
    stats = inspect_features(X)
    row_count = int(stats["row_count"])
    feature_count = int(stats["feature_count"])
    cat_frac = float(stats["categorical_fraction"])
    miss = float(stats["missing_density"])

    if force_strategy in ("foundation_model", "ensemble"):
        return RoutingDecision(
            strategy=force_strategy,
            reason=f"Forced strategy={force_strategy}",
            row_count=row_count,
            feature_count=feature_count,
            categorical_fraction=cat_frac,
            missing_density=miss,
            forced=True,
        )

    if row_count == 0 or feature_count == 0:
        return RoutingDecision(
            strategy="ensemble",
            reason="Empty feature matrix; defaulting to ensemble",
            row_count=row_count,
            feature_count=feature_count,
            categorical_fraction=cat_frac,
            missing_density=miss,
        )

    if feature_count > max_foundation_features:
        return RoutingDecision(
            strategy="ensemble",
            reason=(
                f"Feature count {feature_count} exceeds foundation limit "
                f"({max_foundation_features}); using stacked ensemble"
            ),
            row_count=row_count,
            feature_count=feature_count,
            categorical_fraction=cat_frac,
            missing_density=miss,
        )

    if row_count <= max_foundation_rows:
        return RoutingDecision(
            strategy="foundation_model",
            reason=(
                f"Rows={row_count} ≤ {max_foundation_rows} and features={feature_count} "
                f"≤ {max_foundation_features} (cat={cat_frac:.0%}, missing={miss:.1%}); "
                "foundation model preferred for small/medium tabular data"
            ),
            row_count=row_count,
            feature_count=feature_count,
            categorical_fraction=cat_frac,
            missing_density=miss,
        )

    return RoutingDecision(
        strategy="ensemble",
        reason=(
            f"Rows={row_count} > {max_foundation_rows}; "
            "using stacked ensemble for larger data"
        ),
        row_count=row_count,
        feature_count=feature_count,
        categorical_fraction=cat_frac,
        missing_density=miss,
    )


def route_from_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    max_foundation_rows: int = 10_000,
    max_foundation_features: int = 500,
    force_strategy: Optional[Strategy] = None,
) -> RoutingDecision:
    """Convenience: route from full frame + feature column names."""
    cols = [c for c in feature_columns if c in df.columns]
    return route_training(
        df[cols] if cols else df.iloc[:, 0:0],
        max_foundation_rows=max_foundation_rows,
        max_foundation_features=max_foundation_features,
        force_strategy=force_strategy,
    )
