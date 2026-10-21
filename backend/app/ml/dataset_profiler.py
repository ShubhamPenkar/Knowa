"""Deterministic dataset hygiene profiler.

Runs on upload and again at project create / train. Catches the bug classes
we hit on Telco / Superstore / HR without LLM heuristics:

1. Numeric ↔ categorical mistyping (coercion success rate)
2. ID columns (uniqueness + sequential + name substrings)
3. Target positive-label presence
4. Constant / zero-variance columns
5. High missing density (warning)
6. High-cardinality categoricals (warning)

Actions (exclude ID / drop constant) are applied by default at project create
and training; blocking issues raise ProfilingError with a structured detail
payload (same shape family as FeatureValidationError).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.ml.semantic_inferencer import (
    BusinessRole,
    SemanticAnnotation,
    SemanticRole,
    SemanticType,
    infer_column_semantics,
)

# ---------------------------------------------------------------------------
# Thresholds (deterministic, documented)
# ---------------------------------------------------------------------------

NUMERIC_PARSE_THRESHOLD = 0.95  # ≥95% of non-null values parse → numeric
ID_UNIQUENESS_RATIO = 0.98  # ≥98% unique among non-null → ID-like
MISSING_WARN_FRAC = 0.30  # ≥30% null → warning
HIGH_CARD_RATIO = 0.50  # categorical unique/n ≥ 50% → high-cardinality warn
SEQUENTIAL_UNIQUE_MIN = 0.90  # for monotonic/sequential integer check
MAX_FAILED_SAMPLES = 8

# Name substrings that suggest identifiers (normalized: store_id, employee_number, …)
# Note: bare "num" is too aggressive (matches almost_num, amount_num); use number/no/id.
_ID_NAME_TOKEN_RE = re.compile(
    r"(^|_)(id|uuid|guid|pk|key|ref|no|number|code)(_|$)|"
    r"(^|_)(row_?id|row_?number|rownum|index)(_|$)|"
    r"customer_?id|user_?id|account_?id|store_?id|entity_?id|"
    r"member_?id|employee_?id|employee_?number|employee_?no|emp_?id|emp_?no|"
    r"staff_?id|personnel_?number|ref_?no|pk_"
)


class ProfilingError(ValueError):
    """Blocking dataset/project hygiene failure with structured detail."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "dataset_profiling_blocked",
        blocking_issues: Optional[list[dict[str, Any]]] = None,
        warnings: Optional[list[dict[str, Any]]] = None,
        present_target_values: Optional[list[str]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.blocking_issues = list(blocking_issues or [])
        self.warnings = list(warnings or [])
        self.present_target_values = (
            list(present_target_values) if present_target_values is not None else None
        )

    def as_detail(self) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "code": self.code,
            "message": str(self),
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
        }
        if self.present_target_values is not None:
            detail["present_target_values"] = self.present_target_values
        return detail


def normalize_col_name(name: str) -> str:
    """Lowercase + collapse spaces/punct so 'Store ID ' → 'store_id'."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return s.strip("_")


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Parse numbers from object/string columns (strip $ , % spaces); failures → NaN."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = (
        s.astype(str)
        .str.strip()
        .str.replace(r"[\$,%]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    cleaned = cleaned.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "none": np.nan,
            "null": np.nan,
            "na": np.nan,
            "<na>": np.nan,
        }
    )
    return pd.to_numeric(cleaned, errors="coerce")


def is_stringy_dtype(s: pd.Series) -> bool:
    if s.dtype == object:
        return True
    dt = str(s.dtype)
    if dt in ("string", "str", "category"):
        return True
    try:
        return bool(pd.api.types.is_string_dtype(s))
    except (TypeError, ValueError):
        return False


def numeric_parse_stats(s: pd.Series) -> dict[str, Any]:
    """Coercion success among non-null present values."""
    present_mask = s.notna()
    if is_stringy_dtype(s):
        stripped = s.astype(str).str.strip()
        present_mask = present_mask & (stripped != "") & (
            ~stripped.str.lower().isin({"nan", "none", "null", "na"})
        )
    n_present = int(present_mask.sum())
    if n_present == 0:
        return {
            "n_present": 0,
            "n_parsed": 0,
            "parse_frac": 0.0,
            "failed_samples": [],
        }
    parsed = coerce_numeric_series(s)
    ok = present_mask & parsed.notna()
    n_parsed = int(ok.sum())
    failed_idx = present_mask & parsed.isna()
    failed_samples = (
        s.loc[failed_idx].astype(str).head(MAX_FAILED_SAMPLES).tolist()
        if failed_idx.any()
        else []
    )
    return {
        "n_present": n_present,
        "n_parsed": n_parsed,
        "parse_frac": float(n_parsed / n_present),
        "failed_samples": failed_samples,
    }


def name_suggests_id(name: str) -> bool:
    n = normalize_col_name(name)
    if not n:
        return False
    if n in {"id", "uuid", "guid", "pk", "key", "index", "rowid", "row_id"}:
        return True
    return bool(_ID_NAME_TOKEN_RE.search(n))


def looks_sequential_integer(s: pd.Series, *, name: Optional[str] = None) -> bool:
    """
    Surrogate-key shaped integers: near-unique, mostly unit gaps, and either
    an ID-ish name or a 0/1-based contiguous index span.

    Avoids flagging measurements like incomes that happen to be unique.
    """
    num = coerce_numeric_series(s).dropna()
    if len(num) < 5:
        return False
    if not np.all(np.isfinite(num.to_numpy())):
        return False
    if not np.allclose(num.to_numpy(), np.round(num.to_numpy())):
        return False
    uniq_ratio = float(num.nunique() / len(num))
    if uniq_ratio < SEQUENTIAL_UNIQUE_MIN:
        return False
    ordered = np.sort(np.unique(num.to_numpy(dtype=float)))
    if len(ordered) < 5:
        return False
    diffs = np.diff(ordered)
    unitish = float(np.mean(diffs <= 1.0 + 1e-9))
    if unitish < 0.7:
        return False
    if name is not None and name_suggests_id(name):
        return True
    start = float(ordered[0])
    if start not in (0.0, 1.0):
        return False
    span = float(ordered[-1] - ordered[0])
    return span <= len(ordered) * 1.05


def is_likely_id_column(
    name: str,
    series: Optional[pd.Series] = None,
    *,
    uniqueness_ratio: float = ID_UNIQUENESS_RATIO,
) -> bool:
    """
    ID detection (deterministic):

    1. Name tokens (id, number, ref, key, …) after normalize.
    2. Surrogate-key shaped sequential integers (0/1-based or ID-named).

    Near-unique continuous numerics (income/sales) and free-text high-cardinality
    fields are NOT auto-excluded — those get type/warning handling instead.
    """
    if name_suggests_id(name):
        return True
    if series is None:
        return False
    return looks_sequential_integer(series, name=name)


def is_likely_leakage_column(name: str, protected: Optional[set[str]] = None) -> bool:
    """IDs always leak; other outcome/date-ish names droppable unless protected."""
    if is_likely_id_column(name):
        return True
    if protected and name in protected:
        return False
    n = normalize_col_name(name)
    leak_pats = (
        r"^target$",
        r"^label$",
        r"^y$",
        r"^churn$",
        r"^churned$",
        r"outcome$",
        r"^prediction",
        r"^score$",
        r"^probability$",
        r"created_?at$",
        r"updated_?at$",
        r"^timestamp$",
        r"^date$",
        r"^datetime$",
    )
    return any(re.search(p, n) for p in leak_pats)


@dataclass
class ColumnProfile:
    column_name: str
    inferred_type: str  # numeric | categorical | id | constant | text
    n_rows: int
    n_unique: int
    null_frac: float
    numeric_parse_frac: float
    issues: list[str] = field(default_factory=list)
    action_taken: str = "keep"
    # keep_as_numeric | keep_as_categorical | exclude_as_id | drop_constant | warn_only
    failed_parse_samples: list[str] = field(default_factory=list)
    sample_values: list[str] = field(default_factory=list)
    reason: str = ""
    # Semantic data understanding (Phase 1)
    semantic_type: str = SemanticType.UNKNOWN.value
    role: str = SemanticRole.UNKNOWN.value
    business_role: str = BusinessRole.UNKNOWN.value
    confidence: float = 0.0  # Rule-based heuristic score in [0.0, 1.0], not a calibrated probability
    confidence_type: str = "heuristic"  # 'heuristic' for inferred rules, 'user_assigned' for manual overrides
    evidence: list[str] = field(default_factory=list)
    is_user_corrected: bool = False
    physical_type: str = ""
    nullable: bool = False
    missing_ratio: float = 0.0
    unique_ratio: float = 0.0
    cardinality: int = 0
    is_identifier: bool = False
    is_target_candidate: bool = False
    target_recommendation: Optional[dict[str, Any]] = None
    distribution_stats: dict[str, Any] = field(default_factory=dict)
    quality_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetProfileReport:
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    blocking: bool = False
    blocking_issues: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    excluded_as_id: list[str] = field(default_factory=list)
    dropped_as_constant: list[str] = field(default_factory=list)
    recommended_features: list[str] = field(default_factory=list)
    target_column: Optional[str] = None
    target_positive_label: Optional[str] = None
    present_target_values: Optional[list[str]] = None
    suggested_positive_label: Optional[str] = None
    user_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Phase 1 Extended Fields
    duplicate_rows_count: int = 0
    duplicate_rows_ratio: float = 0.0
    quality_score: float = 100.0
    potential_targets: list[dict[str, Any]] = field(default_factory=list)
    column_relationships: list[dict[str, Any]] = field(default_factory=list)
    quality_summary: dict[str, Any] = field(default_factory=dict)

    def column_map(self) -> dict[str, ColumnProfile]:
        return {c.column_name: c for c in self.columns}

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": [c.to_dict() for c in self.columns],
            "blocking": self.blocking,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "excluded_as_id": self.excluded_as_id,
            "dropped_as_constant": self.dropped_as_constant,
            "recommended_features": self.recommended_features,
            "target_column": self.target_column,
            "target_positive_label": self.target_positive_label,
            "present_target_values": self.present_target_values,
            "suggested_positive_label": self.suggested_positive_label,
            "user_overrides": self.user_overrides,
            "duplicate_rows_count": self.duplicate_rows_count,
            "duplicate_rows_ratio": self.duplicate_rows_ratio,
            "quality_score": self.quality_score,
            "potential_targets": self.potential_targets,
            "column_relationships": self.column_relationships,
            "quality_summary": self.quality_summary,
        }

    def to_semantic_profile(self, dataset_name: str = "") -> dict[str, Any]:
        """Produce structured Phase 1 semantic data understanding profile."""
        cols = []
        for c in self.columns:
            cols.append({
                "name": c.column_name,
                "physical_type": c.physical_type,
                "inferred_type": c.inferred_type,
                "semantic_type": c.semantic_type,
                "role": c.role,
                "business_role": c.business_role,
                "confidence": c.confidence,
                "confidence_type": c.confidence_type,
                "evidence": c.evidence,
                "is_user_corrected": c.is_user_corrected,
                "nullable": c.nullable,
                "missing_ratio": c.missing_ratio,
                "unique_ratio": c.unique_ratio,
                "cardinality": c.cardinality,
                "is_identifier": c.is_identifier,
                "is_target_candidate": c.is_target_candidate,
                "target_recommendation": c.target_recommendation,
                "distribution_stats": c.distribution_stats,
                "quality_issues": c.quality_issues,
                "sample_values": c.sample_values,
                "action_taken": c.action_taken,
                "reason": c.reason,
            })
        return {
            "dataset": {
                "name": dataset_name,
                "row_count": self.n_rows,
                "column_count": self.n_columns,
                "duplicate_rows_count": self.duplicate_rows_count,
                "duplicate_rows_ratio": self.duplicate_rows_ratio,
                "quality_score": self.quality_score,
            },
            "columns": cols,
            "potential_targets": self.potential_targets,
            "column_relationships": self.column_relationships,
            "quality_summary": self.quality_summary,
            "warnings": self.warnings,
            "blocking": self.blocking,
            "blocking_issues": self.blocking_issues,
            "excluded_as_id": self.excluded_as_id,
            "dropped_as_constant": self.dropped_as_constant,
            "recommended_features": self.recommended_features,
            "target_column": self.target_column,
            "target_positive_label": self.target_positive_label,
            "present_target_values": self.present_target_values,
            "suggested_positive_label": self.suggested_positive_label,
            "user_overrides": self.user_overrides,
        }

    def raise_if_blocking(self) -> None:
        if not self.blocking:
            return
        msgs = [i.get("message") or i.get("code") for i in self.blocking_issues]
        raise ProfilingError(
            "; ".join(str(m) for m in msgs if m),
            blocking_issues=self.blocking_issues,
            warnings=self.warnings,
            present_target_values=self.present_target_values,
        )


_PREFERRED_POSITIVE = (
    "yes",
    "y",
    "true",
    "t",
    "1",
    "positive",
    "pos",
    "high",
    "churn",
    "churned",
    "attrition",
    "left",
    "exited",
)


def suggest_positive_label(present_values: list[str]) -> Optional[str]:
    if not present_values:
        return None
    by_lower = {str(v).lower(): str(v) for v in present_values}
    for key in _PREFERRED_POSITIVE:
        if key in by_lower:
            return by_lower[key]
    return None


def profile_column(
    name: str,
    s: pd.Series,
    n_rows: int,
    *,
    user_override: Optional[dict[str, Any]] = None,
) -> ColumnProfile:
    null_frac = float(s.isna().mean()) if n_rows else 0.0
    present = s.dropna()
    n_unique = int(present.nunique()) if len(present) else 0
    stats = numeric_parse_stats(s)
    parse_frac = float(stats["parse_frac"])
    samples = present.astype(str).head(5).tolist()

    # Semantic data understanding (Phase 1)
    sem = infer_column_semantics(name, s, user_override=user_override, n_rows=n_rows)

    issues: list[str] = []
    reason = ""
    inferred = "categorical"
    action = "keep_as_categorical"

    # Constant / zero-variance (after dropping nulls)
    if len(present) == 0 or n_unique <= 1:
        issues.append("constant_or_empty")
        inferred = "constant"
        action = "drop_constant"
        reason = "Only one distinct non-null value (or all null) — zero variance."
    elif is_likely_id_column(name, s):
        why = []
        if name_suggests_id(name):
            why.append("name_looks_like_id")
        if looks_sequential_integer(s, name=name):
            why.append("sequential_surrogate_key")
        uniq_ratio = float(n_unique / max(len(present), 1))
        if uniq_ratio >= ID_UNIQUENESS_RATIO:
            why.append(f"uniqueness={uniq_ratio:.3f}")
        issues.append("likely_identifier")
        inferred = "id"
        action = "exclude_as_id"
        reason = (
            "Excluded as likely identifier — override if this is a real feature. "
            f"({', '.join(why) or 'heuristic'})"
        )
    else:
        # Missing density warning
        if null_frac >= MISSING_WARN_FRAC:
            issues.append("high_missing_density")

        already_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if already_numeric or parse_frac >= NUMERIC_PARSE_THRESHOLD:
            inferred = "numeric"
            action = "keep_as_numeric"
            if not already_numeric and parse_frac >= NUMERIC_PARSE_THRESHOLD:
                issues.append("coerced_numeric_from_text")
                reason = (
                    f"Numeric-looking text coerced to numeric "
                    f"(parse_frac={parse_frac:.3f} >= {NUMERIC_PARSE_THRESHOLD})."
                )
                if stats["failed_samples"]:
                    issues.append("partial_numeric_parse_failures")
                    reason += f" Failed samples (kept as NaN): {stats['failed_samples'][:5]}"
            else:
                reason = "Native or coerced numeric."
        else:
            issues.append("treated_as_categorical")
            if parse_frac > 0 and parse_frac < NUMERIC_PARSE_THRESHOLD:
                issues.append("failed_numeric_coercion_below_threshold")
                reason = (
                    f"Parse success {parse_frac:.3f} < {NUMERIC_PARSE_THRESHOLD}; "
                    f"kept categorical. Failed samples: {stats['failed_samples'][:5]}"
                )
            else:
                reason = "Non-numeric values — categorical encoding."

            uniq_ratio = float(n_unique / max(n_rows, 1))
            if uniq_ratio >= HIGH_CARD_RATIO:
                issues.append("high_cardinality_categorical")

    all_issues = list(dict.fromkeys(issues + getattr(sem, "quality_issues", [])))
    return ColumnProfile(
        column_name=name,
        inferred_type=inferred,
        n_rows=n_rows,
        n_unique=n_unique,
        null_frac=null_frac,
        numeric_parse_frac=parse_frac,
        issues=all_issues,
        action_taken=action,
        failed_parse_samples=stats["failed_samples"],
        sample_values=samples,
        reason=reason,
        semantic_type=sem.semantic_type,
        role=sem.role,
        business_role=getattr(sem, "business_role", BusinessRole.UNKNOWN.value),
        confidence=sem.confidence,
        confidence_type=sem.confidence_type,
        evidence=sem.evidence,
        is_user_corrected=sem.is_user_corrected,
        physical_type=str(s.dtype),
        nullable=bool(null_frac > 0),
        missing_ratio=round(null_frac, 4),
        unique_ratio=round(float(n_unique / max(n_rows, 1)), 4),
        cardinality=n_unique,
        is_identifier=getattr(sem, "is_identifier", False) or (inferred == "id"),
        is_target_candidate=getattr(sem, "is_target_candidate", False),
        target_recommendation=(
            {"target_type": sem.target_type, "confidence": sem.target_confidence}
            if getattr(sem, "is_target_candidate", False)
            else None
        ),
        distribution_stats=getattr(sem, "distribution_summary", {}),
        quality_issues=all_issues,
    )


def profile_dataframe(
    df: pd.DataFrame,
    *,
    target_column: Optional[str] = None,
    target_positive_label: Optional[str] = None,
    problem_type: str = "binary_classification",
    feature_columns: Optional[list[str]] = None,
    user_overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> DatasetProfileReport:
    """Profile every column; optionally validate target label configuration."""
    if df is None or len(df.columns) == 0:
        report = DatasetProfileReport(n_rows=0, n_columns=0, columns=[])
        report.blocking = True
        report.blocking_issues.append(
            {
                "code": "empty_dataframe",
                "message": "Dataset is empty or has no columns.",
            }
        )
        return report

    n_rows = len(df)
    columns: list[ColumnProfile] = []
    for col in df.columns:
        override = (user_overrides or {}).get(str(col))
        columns.append(profile_column(str(col), df[col], n_rows, user_override=override))

    report = DatasetProfileReport(
        n_rows=n_rows,
        n_columns=len(df.columns),
        columns=columns,
        target_column=target_column,
        target_positive_label=target_positive_label,
        user_overrides=dict(user_overrides or {}),
    )

    for c in columns:
        if c.action_taken == "exclude_as_id":
            report.excluded_as_id.append(c.column_name)
            report.warnings.append(
                {
                    "code": "excluded_as_id",
                    "column": c.column_name,
                    "message": c.reason,
                }
            )
        elif c.action_taken == "drop_constant":
            report.dropped_as_constant.append(c.column_name)
            report.warnings.append(
                {
                    "code": "dropped_constant",
                    "column": c.column_name,
                    "message": c.reason,
                }
            )
        if "high_missing_density" in c.issues:
            report.warnings.append(
                {
                    "code": "high_missing_density",
                    "column": c.column_name,
                    "message": (
                        f"Column '{c.column_name}' is {c.null_frac:.0%} missing "
                        f"(>= {MISSING_WARN_FRAC:.0%}). Imputation will dominate."
                    ),
                    "null_frac": c.null_frac,
                }
            )
        if "high_cardinality_categorical" in c.issues:
            report.warnings.append(
                {
                    "code": "high_cardinality_categorical",
                    "column": c.column_name,
                    "message": (
                        f"Column '{c.column_name}' is a high-cardinality categorical "
                        f"({c.n_unique} uniques / {c.n_rows} rows)."
                    ),
                    "n_unique": c.n_unique,
                }
            )
        if "partial_numeric_parse_failures" in c.issues:
            report.warnings.append(
                {
                    "code": "partial_numeric_parse_failures",
                    "column": c.column_name,
                    "message": (
                        f"Some values in '{c.column_name}' failed numeric parse "
                        f"and become NaN: {c.failed_parse_samples[:5]}"
                    ),
                    "failed_samples": c.failed_parse_samples[:5],
                }
            )

    # Recommended training features = non-id, non-constant (and not target)
    skip = set(report.excluded_as_id) | set(report.dropped_as_constant)
    if target_column:
        skip.add(target_column)
    candidates = feature_columns if feature_columns is not None else [str(c) for c in df.columns]
    report.recommended_features = [c for c in candidates if c not in skip and c in df.columns]

    # Duplicate rows analysis
    dup_count = int(df.duplicated().sum()) if n_rows > 0 else 0
    dup_ratio = round(float(dup_count / max(n_rows, 1)), 4)
    report.duplicate_rows_count = dup_count
    report.duplicate_rows_ratio = dup_ratio
    if dup_count > 0:
        report.warnings.append(
            {
                "code": "duplicate_rows_detected",
                "message": f"Dataset contains {dup_count} duplicate rows ({dup_ratio:.1%}).",
                "duplicate_count": dup_count,
                "duplicate_ratio": dup_ratio,
            }
        )

    # Outlier detection on numeric columns
    for c in columns:
        if c.inferred_type == "numeric" and c.column_name in df.columns:
            nums = pd.to_numeric(df[c.column_name], errors="coerce").dropna()
            if len(nums) >= 20:
                q25 = float(np.percentile(nums, 25))
                q75 = float(np.percentile(nums, 75))
                iqr = q75 - q25
                if iqr > 1e-6:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers_count = int(((nums < lower_bound) | (nums > upper_bound)).sum())
                    if outliers_count > 0:
                        outlier_ratio = round(outliers_count / len(nums), 4)
                        if outlier_ratio >= 0.02:
                            if "contains_outliers" not in c.quality_issues:
                                c.quality_issues.append("contains_outliers")
                            if "contains_outliers" not in c.issues:
                                c.issues.append("contains_outliers")

    # Potential targets identification & ranking
    potential_targets: list[dict[str, Any]] = []
    seen_target_cols: set[str] = set()
    for c in columns:
        if c.is_target_candidate and c.column_name not in seen_target_cols:
            seen_target_cols.add(c.column_name)
            rec = c.target_recommendation or {}
            potential_targets.append({
                "column_name": c.column_name,
                "target_type": rec.get("target_type", "classification"),
                "confidence": rec.get("confidence", 0.7),
                "business_role": c.business_role,
                "semantic_type": c.semantic_type,
                "evidence": c.evidence,
            })
        elif c.column_name.lower() in {"target", "label", "outcome", "churn", "churned", "converted", "default"} and c.column_name not in seen_target_cols:
            seen_target_cols.add(c.column_name)
            ttype = "classification" if c.cardinality <= 10 or c.inferred_type == "categorical" else "regression"
            potential_targets.append({
                "column_name": c.column_name,
                "target_type": ttype,
                "confidence": 0.88,
                "business_role": c.business_role if c.business_role != BusinessRole.UNKNOWN.value else "target",
                "semantic_type": c.semantic_type,
                "evidence": ["Column name indicates machine learning target", f"Type: {c.inferred_type}"],
            })
    # If no target identified yet, check candidate binary/low-cardinality flags (non-constant, non-id, non-datetime)
    if not potential_targets and n_rows >= 10:
        for c in columns:
            if c.column_name not in seen_target_cols and not c.is_identifier and c.action_taken != "drop_constant":
                if c.cardinality == 2 and c.missing_ratio < 0.5:
                    seen_target_cols.add(c.column_name)
                    potential_targets.append({
                        "column_name": c.column_name,
                        "target_type": "binary_classification",
                        "confidence": 0.65,
                        "business_role": c.business_role,
                        "semantic_type": c.semantic_type,
                        "evidence": ["Binary feature with two unique outcomes; candidate for classification target"],
                    })

    potential_targets.sort(key=lambda x: x["confidence"], reverse=True)
    report.potential_targets = potential_targets

    # Column relationships & correlations
    column_relationships: list[dict[str, Any]] = []
    numeric_cols = [c.column_name for c in columns if c.inferred_type == "numeric" and c.column_name in df.columns]
    if len(numeric_cols) >= 2:
        sampled_df = df[numeric_cols[:25]]
        if len(sampled_df) > 5000:
            sampled_df = sampled_df.sample(5000, random_state=42)
        try:
            numeric_data = sampled_df.apply(pd.to_numeric, errors="coerce")
            corr_matrix = numeric_data.corr()
            cols = list(corr_matrix.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    col_a = cols[i]
                    col_b = cols[j]
                    val = corr_matrix.loc[col_a, col_b]
                    if pd.notna(val) and abs(val) >= 0.85:
                        column_relationships.append({
                            "type": "high_correlation",
                            "columns": [col_a, col_b],
                            "strength": round(float(val), 3),
                            "description": f"High linear correlation ({val:+.2f}) between '{col_a}' and '{col_b}'. Potential multicollinearity.",
                        })
        except Exception:
            pass

    # Entity link candidate detection
    for c in columns:
        cn = c.column_name.lower()
        if (cn.endswith("_id") or cn.endswith("id") or cn.endswith("_ref") or cn.endswith("_fk")) and not c.is_identifier:
            if 0.02 < c.unique_ratio < 0.98 and c.cardinality > 2:
                column_relationships.append({
                    "type": "entity_link_candidate",
                    "columns": [c.column_name],
                    "strength": 0.8,
                    "description": f"Column '{c.column_name}' appears to be a foreign key reference or dimension link.",
                })

    report.column_relationships = column_relationships

    # Overall dataset quality score computation (0 to 100)
    total_cells = max(n_rows * len(df.columns), 1)
    missing_cells = int(df.isna().sum().sum())
    missing_cells_ratio = round(missing_cells / total_cells, 4)

    score = 100.0
    score -= min(missing_cells_ratio * 40.0, 25.0)
    score -= min(dup_ratio * 30.0, 15.0)
    score -= min(len(report.dropped_as_constant) * 5.0, 15.0)
    high_missing_cols = sum(1 for c in columns if c.null_frac >= MISSING_WARN_FRAC)
    score -= min(high_missing_cols * 5.0, 15.0)
    format_issues = sum(
        1 for c in columns
        if any(iss in ("inconsistent_casing", "contains_whitespace_padding", "impossible_negative_numeric") for iss in c.quality_issues)
    )
    score -= min(format_issues * 2.0, 10.0)

    quality_score = max(0.0, min(100.0, round(score, 1)))

    if quality_score >= 90.0:
        grade = "Excellent"
    elif quality_score >= 75.0:
        grade = "Good"
    elif quality_score >= 60.0:
        grade = "Fair"
    else:
        grade = "Poor"

    report.quality_score = quality_score
    report.quality_summary = {
        "overall_score": quality_score,
        "grade": grade,
        "total_rows": n_rows,
        "total_columns": len(df.columns),
        "duplicate_rows": dup_count,
        "duplicate_ratio": dup_ratio,
        "missing_cells_count": missing_cells,
        "missing_cells_ratio": missing_cells_ratio,
        "constant_columns_count": len(report.dropped_as_constant),
        "identifier_columns_count": len(report.excluded_as_id),
        "target_candidates_count": len(potential_targets),
        "high_missing_columns_count": high_missing_cols,
    }

    # Target label validation
    if target_column and problem_type != "regression":
        if target_column not in df.columns:
            report.blocking = True
            report.blocking_issues.append(
                {
                    "code": "target_column_missing",
                    "message": f"Target column '{target_column}' not found in dataset.",
                }
            )
        else:
            present = (
                df[target_column].dropna().astype(str).unique().tolist()
            )
            report.present_target_values = sorted(present)[:100]
            report.suggested_positive_label = suggest_positive_label(present)

            if len(present) < 2:
                report.blocking = True
                report.blocking_issues.append(
                    {
                        "code": "target_single_class",
                        "message": (
                            f"Target column '{target_column}' has fewer than 2 classes. "
                            f"Found: {report.present_target_values}."
                        ),
                        "present_target_values": report.present_target_values,
                    }
                )

            if target_positive_label is not None and str(target_positive_label).strip() != "":
                pos = str(target_positive_label)
                if pos not in present:
                    report.blocking = True
                    msg = (
                        f"Positive label '{pos}' does not appear in target column "
                        f"'{target_column}'. Found values: {report.present_target_values}."
                    )
                    if report.suggested_positive_label:
                        msg += f" Suggested: '{report.suggested_positive_label}'."
                    report.blocking_issues.append(
                        {
                            "code": "positive_label_not_in_data",
                            "message": msg,
                            "positive_label": pos,
                            "present_target_values": report.present_target_values,
                            "suggested_positive_label": report.suggested_positive_label,
                        }
                    )

            if (
                problem_type == "binary_classification"
                and len(present) > 2
                and target_positive_label
                and str(target_positive_label) in present
            ):
                report.warnings.append(
                    {
                        "code": "multiclass_forced_binary",
                        "message": (
                            f"Target '{target_column}' has {len(present)} classes; "
                            f"training will binarize as '{target_positive_label}' vs rest."
                        ),
                        "present_target_values": report.present_target_values,
                    }
                )

    if feature_columns is not None and not report.recommended_features:
        # All selected features excluded
        report.blocking = True
        report.blocking_issues.append(
            {
                "code": "no_usable_features",
                "message": (
                    "All selected feature columns were excluded as identifiers or "
                    "constants. Choose predictive features instead."
                ),
                "excluded_as_id": report.excluded_as_id,
                "dropped_as_constant": report.dropped_as_constant,
            }
        )

    return report


def apply_feature_exclusions(
    feature_columns: list[str],
    report: DatasetProfileReport,
) -> list[str]:
    """Drop ID + constant columns from a feature list (default product behavior)."""
    ban = set(report.excluded_as_id) | set(report.dropped_as_constant)
    return [c for c in feature_columns if c not in ban]


def save_profile_report(report: DatasetProfileReport, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


def load_profile_report(path: str) -> Optional[dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def profile_path_for_parquet(parquet_path: str) -> str:
    if parquet_path.endswith(".parquet"):
        return parquet_path[: -len(".parquet")] + ".profile.json"
    return parquet_path + ".profile.json"


def enrich_column_metadata(
    base_columns: list[dict[str, Any]],
    report: DatasetProfileReport,
) -> list[dict[str, Any]]:
    """Merge profiler fields into Dataset.columns entries."""
    cmap = report.column_map()
    out: list[dict[str, Any]] = []
    for col in base_columns:
        name = col.get("name")
        entry = dict(col)
        prof = cmap.get(name)
        if prof:
            entry["inferred_type"] = prof.inferred_type
            entry["action_taken"] = prof.action_taken
            entry["issues"] = list(prof.issues)
            entry["null_frac"] = prof.null_frac
            entry["numeric_parse_frac"] = prof.numeric_parse_frac
            entry["profile_reason"] = prof.reason
            entry["semantic_type"] = prof.semantic_type
            entry["role"] = prof.role
            entry["business_role"] = getattr(prof, "business_role", BusinessRole.UNKNOWN.value)
            entry["confidence"] = prof.confidence
            entry["confidence_type"] = getattr(prof, "confidence_type", "heuristic")
            entry["evidence"] = list(prof.evidence)
            entry["is_user_corrected"] = prof.is_user_corrected
            entry["physical_type"] = getattr(prof, "physical_type", "")
            entry["nullable"] = getattr(prof, "nullable", False)
            entry["missing_ratio"] = getattr(prof, "missing_ratio", 0.0)
            entry["unique_ratio"] = getattr(prof, "unique_ratio", 0.0)
            entry["cardinality"] = getattr(prof, "cardinality", 0)
            entry["is_identifier"] = getattr(prof, "is_identifier", False)
            entry["is_target_candidate"] = getattr(prof, "is_target_candidate", False)
            entry["target_recommendation"] = getattr(prof, "target_recommendation", None)
            entry["distribution_stats"] = getattr(prof, "distribution_stats", {})
            entry["quality_issues"] = getattr(prof, "quality_issues", [])
        out.append(entry)
    return out
