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


def profile_column(name: str, s: pd.Series, n_rows: int) -> ColumnProfile:
    null_frac = float(s.isna().mean()) if n_rows else 0.0
    present = s.dropna()
    n_unique = int(present.nunique()) if len(present) else 0
    stats = numeric_parse_stats(s)
    parse_frac = float(stats["parse_frac"])
    samples = present.astype(str).head(5).tolist()

    issues: list[str] = []
    reason = ""

    # Constant / zero-variance (after dropping nulls)
    if len(present) == 0 or n_unique <= 1:
        issues.append("constant_or_empty")
        return ColumnProfile(
            column_name=name,
            inferred_type="constant",
            n_rows=n_rows,
            n_unique=n_unique,
            null_frac=null_frac,
            numeric_parse_frac=parse_frac,
            issues=issues,
            action_taken="drop_constant",
            failed_parse_samples=stats["failed_samples"],
            sample_values=samples,
            reason="Only one distinct non-null value (or all null) — zero variance.",
        )

    # ID detection (behavior + name)
    if is_likely_id_column(name, s):
        why = []
        if name_suggests_id(name):
            why.append("name_looks_like_id")
        if looks_sequential_integer(s, name=name):
            why.append("sequential_surrogate_key")
        uniq_ratio = float(n_unique / max(len(present), 1))
        if uniq_ratio >= ID_UNIQUENESS_RATIO:
            why.append(f"uniqueness={uniq_ratio:.3f}")
        issues.append("likely_identifier")
        return ColumnProfile(
            column_name=name,
            inferred_type="id",
            n_rows=n_rows,
            n_unique=n_unique,
            null_frac=null_frac,
            numeric_parse_frac=parse_frac,
            issues=issues,
            action_taken="exclude_as_id",
            failed_parse_samples=stats["failed_samples"],
            sample_values=samples,
            reason=(
                "Excluded as likely identifier — override if this is a real feature. "
                f"({', '.join(why) or 'heuristic'})"
            ),
        )

    # Missing density warning
    if null_frac >= MISSING_WARN_FRAC:
        issues.append("high_missing_density")

    # Type inference via coercion rate
    already_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if already_numeric or parse_frac >= NUMERIC_PARSE_THRESHOLD:
        inferred = "numeric"
        action = "keep_as_numeric"
        if not already_numeric and parse_frac >= NUMERIC_PARSE_THRESHOLD:
            issues.append("coerced_numeric_from_text")
            reason = (
                f"Numeric-looking text coerced to numeric "
                f"(parse_frac={parse_frac:.3f} ≥ {NUMERIC_PARSE_THRESHOLD})."
            )
            if stats["failed_samples"]:
                issues.append("partial_numeric_parse_failures")
                reason += f" Failed samples (kept as NaN): {stats['failed_samples'][:5]}"
        else:
            reason = "Native or coerced numeric."
        # Low-cardinality integers can still be fine as numeric for trees
        profile = ColumnProfile(
            column_name=name,
            inferred_type=inferred,
            n_rows=n_rows,
            n_unique=n_unique,
            null_frac=null_frac,
            numeric_parse_frac=parse_frac,
            issues=issues,
            action_taken=action,
            failed_parse_samples=stats["failed_samples"],
            sample_values=samples,
            reason=reason,
        )
        if null_frac >= MISSING_WARN_FRAC:
            profile.action_taken = "warn_high_missing" if action == "keep_as_numeric" else action
            # still keep as numeric; warning is separate
            profile.action_taken = "keep_as_numeric"
        return profile

    # Non-numeric → categorical (do NOT silently zero as continuous)
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

    return ColumnProfile(
        column_name=name,
        inferred_type="categorical",
        n_rows=n_rows,
        n_unique=n_unique,
        null_frac=null_frac,
        numeric_parse_frac=parse_frac,
        issues=issues,
        action_taken="keep_as_categorical",
        failed_parse_samples=stats["failed_samples"],
        sample_values=samples,
        reason=reason,
    )


def profile_dataframe(
    df: pd.DataFrame,
    *,
    target_column: Optional[str] = None,
    target_positive_label: Optional[str] = None,
    problem_type: str = "binary_classification",
    feature_columns: Optional[list[str]] = None,
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
        columns.append(profile_column(str(col), df[col], n_rows))

    report = DatasetProfileReport(
        n_rows=n_rows,
        n_columns=len(df.columns),
        columns=columns,
        target_column=target_column,
        target_positive_label=target_positive_label,
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
                        f"(≥ {MISSING_WARN_FRAC:.0%}). Imputation will dominate."
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
                    # Stale API default "1" + clear suggestion → still block at profiler
                    # level when explicitly validating; create_project may auto-adopt
                    # suggestion only for default "1" (handled in service).
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
        out.append(entry)
    return out
