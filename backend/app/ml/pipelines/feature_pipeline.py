"""Train-safe feature preparation (Phase 1.5).

Fit statistics only on training rows to avoid leakage, add missingness
indicators, drop obvious identifier/leak columns, and export a config for
prediction-time encoding.

ID / numeric typing rules are owned by ``app.ml.dataset_profiler`` so upload,
project-create, and train-time transforms stay consistent.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from app.ml.dataset_profiler import (
    NUMERIC_PARSE_THRESHOLD,
    coerce_numeric_series,
    is_likely_id_column,
    is_likely_leakage_column,
    is_stringy_dtype,
    normalize_col_name,
    numeric_parse_stats,
)

# Back-compat aliases used by older call sites / tests
_normalize_col_name = normalize_col_name
_NUMERIC_OBJECT_FRAC = NUMERIC_PARSE_THRESHOLD


def mostly_numeric(s: pd.Series, threshold: float = NUMERIC_PARSE_THRESHOLD) -> bool:
    """Whether a series should be treated as numeric (incl. stringified numbers)."""
    if pd.api.types.is_bool_dtype(s):
        return False
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return True
    stats = numeric_parse_stats(s)
    if stats["n_present"] == 0:
        return False
    return float(stats["parse_frac"]) >= threshold


def _is_stringy_dtype(s: pd.Series) -> bool:
    return is_stringy_dtype(s)


def _is_categorical(s: pd.Series) -> bool:
    """Categorical if not mostly-numeric and stringy / low-card integer."""
    if mostly_numeric(s):
        return False
    if is_stringy_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 25:
        return True
    if pd.api.types.is_bool_dtype(s):
        return True
    return False


class FeatureTransformer:
    """
    Fit-on-train / transform-everywhere feature pipeline.

    - Drops ID-like (name + uniqueness) and leaky columns
    - Treats mostly-numeric object columns as numeric (e.g. Store_Sales '116320')
    - String categoricals stay categorical (never zero-filled as continuous)
    - Adds missingness indicators when train missing rate > 0
    - Numerics: train median impute
    - Categoricals: train vocabulary ordinal codes; unknowns → -1
    - High-cardinality cats: keep top_k + __OTHER__
    """

    def __init__(
        self,
        top_k_categories: int = 50,
        drop_leakage: bool = True,
        add_missing_indicators: bool = True,
        max_missing_indicator_frac: float = 0.0,
        numeric_object_frac: float = NUMERIC_PARSE_THRESHOLD,
    ):
        self.top_k_categories = top_k_categories
        self.drop_leakage = drop_leakage
        self.add_missing_indicators = add_missing_indicators
        self.max_missing_indicator_frac = max_missing_indicator_frac
        self.numeric_object_frac = numeric_object_frac

        self.input_columns: list[str] = []
        self.dropped_columns: list[str] = []
        self.dropped_reasons: dict[str, str] = {}
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.feature_names_out_: list[str] = []
        self.medians_: dict[str, float] = {}
        self.cat_maps_: dict[str, dict[str, int]] = {}
        self.indicator_columns: list[str] = []
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        *,
        protected_columns: Optional[list[str]] = None,
    ) -> "FeatureTransformer":
        protected = set(protected_columns or [])
        X = X.copy()
        self.input_columns = list(X.columns)
        self.dropped_columns = []
        self.dropped_reasons = {}

        keep: list[str] = []
        for col in X.columns:
            if self.drop_leakage and (
                is_likely_id_column(col, X[col])
                or is_likely_leakage_column(col, protected=protected)
            ):
                self.dropped_columns.append(col)
                reason = (
                    "id_column_auto_excluded"
                    if is_likely_id_column(col, X[col])
                    else "leakage_or_label_name_heuristic"
                )
                self.dropped_reasons[col] = reason
                continue
            if X[col].nunique(dropna=True) <= 1:
                self.dropped_columns.append(col)
                self.dropped_reasons[col] = "constant"
                continue
            keep.append(col)

        X = X[keep]
        self.numeric_columns = []
        self.categorical_columns = []
        self.indicator_columns = []
        self.medians_ = {}
        self.cat_maps_ = {}

        for col in keep:
            series = X[col]
            if _is_categorical(series):
                self.categorical_columns.append(col)
                s = series.astype(str).fillna("__MISSING__")
                counts = s.value_counts()
                kept = set(counts.head(self.top_k_categories).index.tolist())
                codes: dict[str, int] = {}
                for i, val in enumerate(sorted(kept)):
                    codes[val] = i
                codes["__OTHER__"] = len(codes)
                if "__MISSING__" not in codes:
                    codes["__MISSING__"] = len(codes)
                self.cat_maps_[col] = codes
            else:
                self.numeric_columns.append(col)
                num = coerce_numeric_series(series)
                med = float(num.median()) if num.notna().any() else 0.0
                if np.isnan(med):
                    med = 0.0
                self.medians_[col] = med

            miss_rate = float(series.isna().mean())
            if self.add_missing_indicators and miss_rate > self.max_missing_indicator_frac:
                if miss_rate > 0:
                    self.indicator_columns.append(col)

        out: list[str] = []
        out.extend(self.numeric_columns)
        out.extend(self.categorical_columns)
        for col in self.indicator_columns:
            out.append(f"{col}__is_missing")
        self.feature_names_out_ = out
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("FeatureTransformer must be fitted before transform")

        frames: dict[str, np.ndarray] = {}
        n = len(X)

        for col in self.numeric_columns:
            if col in X.columns:
                num = coerce_numeric_series(X[col])
            else:
                num = pd.Series(np.full(n, np.nan))
            med = self.medians_.get(col, 0.0)
            frames[col] = num.fillna(med).to_numpy(dtype=float)

        for col in self.categorical_columns:
            codes_map = self.cat_maps_[col]
            other = codes_map.get("__OTHER__", -1)
            if col in X.columns:
                s = X[col]
                raw = s.where(s.notna(), other="__MISSING__").astype(str)
            else:
                raw = pd.Series(["__MISSING__"] * n)
            mapped = raw.map(lambda v: codes_map.get(v, other)).astype(float)
            frames[col] = mapped.to_numpy(dtype=float)

        for col in self.indicator_columns:
            name = f"{col}__is_missing"
            if col in X.columns:
                frames[name] = X[col].isna().astype(float).to_numpy()
            else:
                frames[name] = np.ones(n, dtype=float)

        out = pd.DataFrame(frames, index=X.index)
        return out[self.feature_names_out_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y, **kwargs).transform(X)

    def to_feature_config(self) -> dict[str, Any]:
        """Back-compat config for project UI / older predict paths."""
        cfg: dict[str, Any] = {}
        for col in self.numeric_columns:
            cfg[col] = {
                "type": "numeric",
                "median": self.medians_.get(col, 0.0),
            }
        for col in self.categorical_columns:
            codes = self.cat_maps_.get(col, {})
            cats = [
                k
                for k, _ in sorted(
                    (
                        (k, v)
                        for k, v in codes.items()
                        if k not in ("__OTHER__", "__MISSING__")
                    ),
                    key=lambda kv: kv[1],
                )
            ]
            cfg[col] = {
                "type": "categorical",
                "categories": cats,
                "code_map": codes,
            }
        for col in self.indicator_columns:
            cfg[f"{col}__is_missing"] = {"type": "numeric", "derived": True}
        cfg["_pipeline"] = {
            "version": "1.7",
            "dropped_columns": self.dropped_columns,
            "dropped_reasons": self.dropped_reasons,
            "feature_names_out": self.feature_names_out_,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
        }
        return cfg

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        joblib.dump(
            {
                "top_k_categories": self.top_k_categories,
                "drop_leakage": self.drop_leakage,
                "add_missing_indicators": self.add_missing_indicators,
                "max_missing_indicator_frac": self.max_missing_indicator_frac,
                "numeric_object_frac": self.numeric_object_frac,
                "input_columns": self.input_columns,
                "dropped_columns": self.dropped_columns,
                "dropped_reasons": self.dropped_reasons,
                "numeric_columns": self.numeric_columns,
                "categorical_columns": self.categorical_columns,
                "feature_names_out_": self.feature_names_out_,
                "medians_": self.medians_,
                "cat_maps_": self.cat_maps_,
                "indicator_columns": self.indicator_columns,
                "is_fitted": self.is_fitted,
            },
            os.path.join(path, "feature_transformer.joblib"),
        )
        with open(os.path.join(path, "feature_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_feature_config(), f, indent=2)

    def load(self, path: str) -> "FeatureTransformer":
        joblib_path = os.path.join(path, "feature_transformer.joblib")
        if os.path.exists(joblib_path):
            data = joblib.load(joblib_path)
            if isinstance(data, FeatureTransformer):
                self.__dict__.update(data.__dict__)
                return self
            for k, v in data.items():
                setattr(self, k, v)
            self.is_fitted = True
            return self
        cfg_path = os.path.join(path, "feature_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"No feature transformer at {path}")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        self.numeric_columns = []
        self.categorical_columns = []
        self.medians_ = {}
        self.cat_maps_ = {}
        self.indicator_columns = []
        for col, meta in cfg.items():
            if col.startswith("_") or not isinstance(meta, dict):
                continue
            if meta.get("derived"):
                base = col.replace("__is_missing", "")
                if base not in self.indicator_columns:
                    self.indicator_columns.append(base)
            elif meta.get("type") == "categorical":
                self.categorical_columns.append(col)
                self.cat_maps_[col] = meta.get("code_map") or {
                    c: i for i, c in enumerate(meta.get("categories") or [])
                }
            else:
                self.numeric_columns.append(col)
                self.medians_[col] = float(meta.get("median", 0.0))
        pipe = cfg.get("_pipeline") or {}
        self.feature_names_out_ = pipe.get("feature_names_out") or (
            self.numeric_columns
            + self.categorical_columns
            + [f"{c}__is_missing" for c in self.indicator_columns]
        )
        self.dropped_columns = pipe.get("dropped_columns") or []
        self.dropped_reasons = pipe.get("dropped_reasons") or {}
        self.is_fitted = True
        return self
