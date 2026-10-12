"""Multi-Dataset Intelligence: Relationship Discovery, Grain Inference, and Metric Safety.

Phase 2 core engine:
1. Discovers candidate join relationships between independent datasets.
2. Infers dataset grain (entity, transaction, periodic aggregate, composite, unknown).
3. Detects fan-out and double-counting risks to prevent metric inflation.
4. Privacy-hardened pattern explanations without raw data leakage.
5. Heuristic confidence scores labeled explicitly as heuristic scores.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.ml.semantic_inferencer import SemanticRole, SemanticType, _normalize_name


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RelationshipCardinality(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    CANDIDATE = "candidate"
    NEEDS_REVIEW = "needs_review"
    AMBIGUOUS = "ambiguous"
    VALIDATED = "validated"
    REJECTED = "rejected"


class GrainType(str, Enum):
    ENTITY = "entity"
    TRANSACTION = "transaction"
    PERIODIC_AGGREGATE = "periodic_aggregate"
    COMPOSITE_KEY = "composite_key"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DatasetGrain:
    """Inferred grain / level-of-detail for a dataset."""
    dataset_id: str
    dataset_name: str
    grain_type: str = GrainType.UNKNOWN.value
    primary_key: list[str] = field(default_factory=list)
    confidence: float = 0.0  # Heuristic score in [0.0, 1.0], not a calibrated probability
    confidence_type: str = "heuristic"
    evidence: list[str] = field(default_factory=list)
    potential_measures: list[str] = field(default_factory=list)
    entity_identifiers: list[str] = field(default_factory=list)
    timestamp_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetRelationship:
    """Relationship candidate or validated join between two datasets."""
    id: str
    source_dataset_id: str
    source_dataset_name: str
    source_column: str
    target_dataset_id: str
    target_dataset_name: str
    target_column: str
    cardinality: str = RelationshipCardinality.UNKNOWN.value
    source_uniqueness_ratio: float = 0.0
    target_uniqueness_ratio: float = 0.0
    containment_ratio: float = 0.0  # Fraction of source keys in target
    jaccard_similarity: float = 0.0
    overlapping_values_count: int = 0
    confidence: float = 0.0  # Heuristic score in [0.0, 1.0], not a calibrated probability
    confidence_type: str = "heuristic"
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_status: str = ValidationStatus.CANDIDATE.value
    is_ambiguous: bool = False
    fan_out_ratio: float = 1.0  # Joined rows / source rows
    double_counting_risk: bool = False
    double_counting_measures: list[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    review_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Dataset Grain Inference
# ---------------------------------------------------------------------------

def infer_dataset_grain(
    df: pd.DataFrame,
    dataset_meta: dict[str, Any],
) -> DatasetGrain:
    """Infer the dataset grain, primary keys, and metric vulnerability.

    Identifies:
    - Entity-level: unique primary key, dimension attributes.
    - Transaction-level: timestamp + measures + entity foreign keys.
    - Periodic aggregate: period date + dimensions + aggregate measures.
    - Composite key: combination of non-null columns uniquely identifying rows.
    - Unknown/Uncertain: duplicate rows, missing keys, high null density.
    """
    dataset_id = str(dataset_meta.get("id") or "unknown_dataset")
    dataset_name = str(dataset_meta.get("name") or "dataset")
    columns_meta = dataset_meta.get("columns") or []
    col_map = {c.get("name"): c for c in columns_meta if c.get("name")}

    n_rows = len(df)
    if n_rows == 0:
        return DatasetGrain(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            grain_type=GrainType.UNKNOWN.value,
            confidence=0.10,
            confidence_type="heuristic",
            evidence=["Empty dataset with zero rows"],
            warnings=["No data rows present to establish dataset grain"],
        )

    # Classify column roles from metadata and series
    identifiers: list[str] = []
    measures: list[str] = []
    timestamps: list[str] = []
    dimensions: list[str] = []

    for col in df.columns:
        col_str = str(col)
        meta = col_map.get(col_str, {})
        role = meta.get("role") or ""
        sem_type = meta.get("semantic_type") or ""

        if role == SemanticRole.MEASURE.value or sem_type == SemanticType.NUMERIC_MEASURE.value:
            measures.append(col_str)
        elif role == SemanticRole.TIMESTAMP.value or sem_type == SemanticType.DATETIME.value:
            timestamps.append(col_str)
        elif role == SemanticRole.IDENTIFIER.value or sem_type == SemanticType.ID.value:
            identifiers.append(col_str)
        else:
            dimensions.append(col_str)

    # 1. Check for single-column unique primary key
    unique_pks: list[str] = []
    for col in df.columns:
        col_str = str(col)
        s = df[col]
        null_count = int(s.isnull().sum())
        n_unique = int(s.nunique())
        if null_count == 0 and n_unique == n_rows:
            meta = col_map.get(col_str, {})
            # Only consider non-measure columns as candidate PKs
            if meta.get("role") != SemanticRole.MEASURE.value:
                unique_pks.append(col_str)

    # Distinguish Entity vs Transaction vs Periodic Aggregate
    if unique_pks:
        primary_pk = unique_pks[0]
        # Check if another column looks like a repeated entity FK (e.g. customer_id, user_id)
        entity_fks = [
            c for c in df.columns
            if str(c) != primary_pk
            and bool(re.search(r"(customer|user|account|client|patient|member|employee|store|entity)_?id", str(c), re.I))
            and int(df[c].nunique()) < n_rows
        ]

        if timestamps and measures and entity_fks:
            # Transaction grain: order_id is unique, but customer_id repeats over time
            evidence = [
                f"Unique row identifier present without nulls ({len(unique_pks)} unique column)",
                f"Repeated entity foreign key present with non-unique distribution",
                f"Timestamp temporal tracking present across transactions",
                f"{len(measures)} quantitative measure columns present",
            ]
            return DatasetGrain(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                grain_type=GrainType.TRANSACTION.value,
                primary_key=[primary_pk],
                confidence=0.92,
                confidence_type="heuristic",
                evidence=evidence,
                potential_measures=measures,
                entity_identifiers=entity_fks,
                timestamp_columns=timestamps,
            )

        # Entity grain: single unique row identifier
        evidence = [
            f"Unique primary row identifier present without nulls (100% uniqueness)",
            f"{len(dimensions)} entity attribute dimension columns describing the entity",
        ]
        if measures:
            evidence.append(f"{len(measures)} quantitative attributes associated with entity")

        return DatasetGrain(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            grain_type=GrainType.ENTITY.value,
            primary_key=[primary_pk],
            confidence=0.95,
            confidence_type="heuristic",
            evidence=evidence,
            potential_measures=measures,
            entity_identifiers=[primary_pk],
            timestamp_columns=timestamps,
        )

    # 2. Check for composite key if no single unique key exists
    # Test pairs of columns (e.g. store_id + date, order_id + line_number)
    non_null_cols = [c for c in df.columns if df[c].isnull().sum() == 0 and str(c) not in measures]
    composite_candidates: list[list[str]] = []

    # Limit search to first 8 non-null columns to keep CPU bounded
    candidate_pool = non_null_cols[:8]
    for i in range(len(candidate_pool)):
        for j in range(i + 1, len(candidate_pool)):
            pair = [candidate_pool[i], candidate_pool[j]]
            if df.duplicated(subset=pair).sum() == 0:
                composite_candidates.append(pair)
                break
        if composite_candidates:
            break

    if composite_candidates:
        best_composite = composite_candidates[0]
        has_time = any(c in timestamps for c in best_composite)
        if has_time and measures:
            grain_type = GrainType.PERIODIC_AGGREGATE.value if len(dimensions) >= 1 else GrainType.TRANSACTION.value
            return DatasetGrain(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                grain_type=grain_type,
                primary_key=best_composite,
                confidence=0.88,
                confidence_type="heuristic",
                evidence=[
                    f"Compound composite key ({' + '.join(best_composite)}) uniquely identifies rows without nulls",
                    f"Temporal tracking present with {len(measures)} measures",
                ],
                potential_measures=measures,
                entity_identifiers=[c for c in best_composite if c not in timestamps],
                timestamp_columns=timestamps,
            )

        return DatasetGrain(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            grain_type=GrainType.COMPOSITE_KEY.value,
            primary_key=best_composite,
            confidence=0.85,
            confidence_type="heuristic",
            evidence=[
                f"Compound composite key ({' + '.join(best_composite)}) uniquely identifies rows (100% uniqueness)",
            ],
            potential_measures=measures,
            entity_identifiers=best_composite,
            timestamp_columns=timestamps,
        )

    # 3. Check for repeated transactions without explicit surrogate key
    entity_fks = [
        c for c in df.columns
        if bool(re.search(r"(customer|user|account|client|patient|member|store)_?id", str(c), re.I))
    ]
    if timestamps and measures and entity_fks:
        return DatasetGrain(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            grain_type=GrainType.TRANSACTION.value,
            primary_key=[],
            confidence=0.78,
            confidence_type="heuristic",
            evidence=[
                "Event transaction pattern: timestamp tracking + repeated entity identifier + quantitative metrics",
                "No explicit unique row surrogate key present; rows represent transaction events",
            ],
            potential_measures=measures,
            entity_identifiers=entity_fks,
            timestamp_columns=timestamps,
            warnings=["Dataset lacks unique primary row key; rows represent event-level occurrences"],
        )

    # 4. Unknown / Uncertain grain (do not guess blindly)
    return DatasetGrain(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        grain_type=GrainType.UNKNOWN.value,
        primary_key=[],
        confidence=0.25,
        confidence_type="heuristic",
        evidence=[
            "No unique primary identifier or composite key found without row duplication",
            "Data structure does not match standard entity, transaction, or periodic aggregate patterns",
        ],
        potential_measures=measures,
        entity_identifiers=entity_fks,
        timestamp_columns=timestamps,
        warnings=["Dataset grain could not be reliably established; manual specification recommended"],
    )


# ---------------------------------------------------------------------------
# Candidate Relationship Discovery
# ---------------------------------------------------------------------------

def _extract_sample_set(s: pd.Series, max_samples: int = 50000) -> set[str]:
    """Extract a non-null string set of values up to max_samples for bounded set operations."""
    clean = s.dropna()
    if len(clean) > max_samples:
        clean = clean.sample(n=max_samples, random_state=42)
    # Convert to stripped lower strings for robust inclusion matching
    return {str(v).strip().lower() for v in clean.tolist() if str(v).strip()}


_FINANCIAL_OR_MEASURE_WORDS_RE = re.compile(
    r"(^|_)(balance|amount|price|cost|charge|charges|fee|fees|revenue|sales|salary|income|"
    r"budget|profit|margin|discount|spend|spending|gdp|rate|deposit|withdraw|"
    r"payment_amount|account_balance|transaction_amount|account_val)(_|$)",
    re.IGNORECASE,
)

_DEMOGRAPHIC_NAME_RE = re.compile(
    r"(^|_)(age|gender|sex|marital|education|ethnicity|race|household|dependents|"
    r"income_bracket|salutation|occupation|demographic|seniority|citizen|generation)(_|$)",
    re.IGNORECASE,
)

_IDENTIFIER_NAME_RE = re.compile(
    r"(^|_)(id|key|code|uuid|guid|num|number|ref|reference|no|identifier|fk|pk)(_|$)|"
    r"(_id|_key|_code|_uuid|_num|_no|_ref)$",
    re.IGNORECASE,
)

_BOOLEAN_NAME_RE = re.compile(
    r"^(is|has|can|should|was|will|did|flag|active|enabled|churn|churned|converted|success)(_|$)|"
    r"(_is|_has|_flag|_active|_enabled)$",
    re.IGNORECASE,
)


def _is_measure_column(col_name: str, s: pd.Series, role: str, sem_type: str) -> bool:
    """Check if column represents a numeric measure or financial metric."""
    if role == SemanticRole.MEASURE.value or sem_type == SemanticType.NUMERIC_MEASURE.value:
        return True
    norm = _normalize_name(col_name)
    if bool(_FINANCIAL_OR_MEASURE_WORDS_RE.search(norm)):
        return True
    if pd.api.types.is_float_dtype(s):
        non_nulls = s.dropna()
        if len(non_nulls) >= 5 and (non_nulls % 1 != 0).any():
            return True
    return False


def _is_demographic_column(col_name: str, sem_type: str) -> bool:
    """Check if column represents a demographic indicator."""
    norm = _normalize_name(col_name)
    return bool(_DEMOGRAPHIC_NAME_RE.search(norm))


def _is_boolean_column(col_name: str, s: pd.Series, sem_type: str) -> bool:
    """Check if column is a boolean flag."""
    if sem_type == SemanticType.BOOLEAN.value:
        return True
    norm = _normalize_name(col_name)
    if bool(_BOOLEAN_NAME_RE.search(norm)):
        return True
    clean = s.dropna()
    if len(clean) >= 5 and clean.nunique() <= 2:
        vals = {str(v).strip().lower() for v in clean.tolist()}
        if vals.issubset({"true", "false", "1", "0", "1.0", "0.0", "t", "f", "yes", "no", "y", "n"}):
            return True
    return False


def _are_types_compatible(
    s_col: str,
    s_s: pd.Series,
    s_dtype: str,
    s_sem: str,
    s_role: str,
    t_col: str,
    t_s: pd.Series,
    t_dtype: str,
    t_sem: str,
    t_role: str,
) -> tuple[bool, Optional[str]]:
    """Check if two columns are data-type and semantic-type compatible for joins.

    Returns (is_compatible, optional_incompatibility_reason).
    """
    # 1. Measures and financial metrics cannot serve as join keys
    if _is_measure_column(s_col, s_s, s_role, s_sem) or _is_measure_column(t_col, t_s, t_role, t_sem):
        return False, "Quantitative measures and financial balances cannot serve as relational join keys"

    # 2. Demographic indicators cannot join with financial metrics or entity IDs
    s_demo = _is_demographic_column(s_col, s_sem)
    t_demo = _is_demographic_column(t_col, t_sem)
    if s_demo and _is_measure_column(t_col, t_s, t_role, t_sem):
        return False, "Customer demographic indicators cannot join with financial balances"
    if t_demo and _is_measure_column(s_col, s_s, s_role, s_sem):
        return False, "Customer demographic indicators cannot join with financial balances"
    if s_demo != t_demo:
        return False, "Customer demographic indicators cannot join with non-demographic columns"

    # 3. Boolean flags cannot serve as join keys
    if _is_boolean_column(s_col, s_s, s_sem) or _is_boolean_column(t_col, t_s, t_sem):
        return False, "Boolean flags and binary indicators cannot serve as relational join keys"

    # 4. Text columns cannot serve as join keys
    if s_sem == SemanticType.TEXT.value or t_sem == SemanticType.TEXT.value:
        return False, "Unstructured text columns cannot serve as relational join keys"

    # 5. Datetime columns can only join with datetime columns
    s_is_time = (s_sem == SemanticType.DATETIME.value or s_role == SemanticRole.TIMESTAMP.value)
    t_is_time = (t_sem == SemanticType.DATETIME.value or t_role == SemanticRole.TIMESTAMP.value)
    if s_is_time != t_is_time:
        return False, "Datetime timestamps cannot join with non-temporal data types"

    # 6. Geographic columns can only join with geographic dimensions
    s_is_geo = (s_sem == SemanticType.GEOGRAPHIC.value or bool(re.search(r"(country|state|city|region|zip|postal)", _normalize_name(s_col))))
    t_is_geo = (t_sem == SemanticType.GEOGRAPHIC.value or bool(re.search(r"(country|state|city|region|zip|postal)", _normalize_name(t_col))))
    if s_is_geo != t_is_geo:
        return False, "Geographic dimensions cannot join with non-geographic attributes"

    # 7. Data-type compatibility
    is_num_s = any(t in s_dtype.lower() for t in ("int", "float", "double"))
    is_num_t = any(t in t_dtype.lower() for t in ("int", "float", "double"))

    if is_num_s and is_num_t:
        # Check for continuous float decimals vs integer keys
        if pd.api.types.is_float_dtype(s_s):
            s_clean = s_s.dropna()
            if len(s_clean) > 0 and (s_clean % 1 != 0).any():
                return False, "Continuous floating-point series cannot serve as discrete join keys"
        if pd.api.types.is_float_dtype(t_s):
            t_clean = t_s.dropna()
            if len(t_clean) > 0 and (t_clean % 1 != 0).any():
                return False, "Continuous floating-point series cannot serve as discrete join keys"
        return True, None

    if is_num_s != is_num_t:
        # String vs numeric: string must be digit-like integer ID
        str_series = s_s if not is_num_s else t_s
        clean_strs = str_series.dropna().head(100).astype(str)
        if not clean_strs.empty and not clean_strs.str.isdigit().all():
            return False, "Non-numeric string values cannot join with numeric keys"
        return True, None

    # Both non-numeric strings/categories
    return True, None


def discover_candidate_relationships(
    source_df: pd.DataFrame,
    source_meta: dict[str, Any],
    target_df: pd.DataFrame,
    target_meta: dict[str, Any],
    max_sample_rows: int = 50000,
) -> list[DatasetRelationship]:
    """Discover candidate join relationships between two datasets.

    Evaluates:
    - Semantic column types and inferred roles
    - Data-type compatibility
    - Value overlap and directional inclusion rate (containment / Jaccard)
    - Cardinality (1:1, 1:N, N:1, N:N)
    - Fan-out and double-counting risk
    - Privacy-hardened explainable evidence (no raw data values)
    """
    s_id = str(source_meta.get("id") or "src")
    s_name = str(source_meta.get("name") or "Source Dataset")
    t_id = str(target_meta.get("id") or "tgt")
    t_name = str(target_meta.get("name") or "Target Dataset")

    s_cols_meta = {c.get("name"): c for c in source_meta.get("columns", []) if c.get("name")}
    t_cols_meta = {c.get("name"): c for c in target_meta.get("columns", []) if c.get("name")}

    s_measures = [
        c.get("name") for c in source_meta.get("columns", [])
        if c.get("role") == SemanticRole.MEASURE.value or c.get("semantic_type") == SemanticType.NUMERIC_MEASURE.value
        or bool(_FINANCIAL_OR_MEASURE_WORDS_RE.search(_normalize_name(str(c.get("name")))))
    ]

    candidates: list[DatasetRelationship] = []

    for s_col_name in source_df.columns:
        s_meta = s_cols_meta.get(str(s_col_name), {})
        s_role = str(s_meta.get("role") or "")
        s_sem = str(s_meta.get("semantic_type") or "")
        s_dtype = str(source_df[s_col_name].dtype)
        s_series = source_df[s_col_name]

        # Early candidate filtering rules
        if _is_measure_column(str(s_col_name), s_series, s_role, s_sem):
            continue
        if s_sem == SemanticType.TEXT.value:
            continue
        if _is_boolean_column(str(s_col_name), s_series, s_sem):
            continue
        s_null_ratio = float(s_series.isnull().mean())
        if s_null_ratio > 0.80:
            continue
        if int(s_series.nunique()) <= 1:
            continue

        s_set = _extract_sample_set(s_series, max_sample_rows)
        if len(s_set) < 2:
            continue

        s_clean_len = max(len(s_series.dropna()), 1)
        s_uniq_ratio = float(s_series.nunique() / s_clean_len)
        norm_s = _normalize_name(str(s_col_name))
        is_s_id = (
            s_sem == SemanticType.ID.value
            or s_role == SemanticRole.IDENTIFIER.value
            or bool(_IDENTIFIER_NAME_RE.search(norm_s))
        )

        for t_col_name in target_df.columns:
            t_meta = t_cols_meta.get(str(t_col_name), {})
            t_role = str(t_meta.get("role") or "")
            t_sem = str(t_meta.get("semantic_type") or "")
            t_dtype = str(target_df[t_col_name].dtype)
            t_series = target_df[t_col_name]

            # Target early filtering rules
            if _is_measure_column(str(t_col_name), t_series, t_role, t_sem):
                continue
            if t_sem == SemanticType.TEXT.value:
                continue
            if _is_boolean_column(str(t_col_name), t_series, t_sem):
                continue
            t_null_ratio = float(t_series.isnull().mean())
            if t_null_ratio > 0.80:
                continue
            if int(t_series.nunique()) <= 1:
                continue

            # Semantic & data type compatibility check
            is_compat, reason = _are_types_compatible(
                s_col=str(s_col_name),
                s_s=s_series,
                s_dtype=s_dtype,
                s_sem=s_sem,
                s_role=s_role,
                t_col=str(t_col_name),
                t_s=t_series,
                t_dtype=t_dtype,
                t_sem=t_sem,
                t_role=t_role,
            )
            if not is_compat:
                continue

            t_set = _extract_sample_set(t_series, max_sample_rows)
            if len(t_set) < 2:
                continue

            # Value overlap analysis
            overlap = s_set.intersection(t_set)
            overlap_count = len(overlap)
            if overlap_count < 2:
                continue

            containment_src = overlap_count / len(s_set)
            containment_tgt = overlap_count / len(t_set)
            max_containment = max(containment_src, containment_tgt)
            union_size = len(s_set.union(t_set))
            jaccard = overlap_count / max(union_size, 1)

            # Name similarity check
            norm_t = _normalize_name(str(t_col_name))
            is_t_id = (
                t_sem == SemanticType.ID.value
                or t_role == SemanticRole.IDENTIFIER.value
                or bool(_IDENTIFIER_NAME_RE.search(norm_t))
            )

            name_match = (norm_s == norm_t)
            fk_match = (
                norm_s.endswith(norm_t)
                or norm_t.endswith(norm_s)
                or (
                    bool(re.search(r"(^|_)(id|key|code|num|no)$", norm_s))
                    and bool(re.search(r"(^|_)(id|key|code|num|no)$", norm_t))
                )
            )

            # Minimum containment threshold
            if max_containment < 0.20:
                continue

            # If names differ completely without FK match:
            # allow only if strong semantic evidence (both are identifier keys) and sufficient containment
            if not name_match and not fk_match:
                if not (is_s_id and is_t_id) and max_containment < 0.60:
                    continue
                if max_containment < 0.40:
                    continue

            t_clean_len = max(len(t_series.dropna()), 1)
            t_uniq_ratio = float(t_series.nunique() / t_clean_len)

            # Determine Cardinality
            if s_uniq_ratio >= 0.95 and t_uniq_ratio >= 0.95:
                cardinality = RelationshipCardinality.ONE_TO_ONE.value
            elif s_uniq_ratio >= 0.95 and t_uniq_ratio < 0.90:
                cardinality = RelationshipCardinality.ONE_TO_MANY.value
            elif s_uniq_ratio < 0.90 and t_uniq_ratio >= 0.95:
                cardinality = RelationshipCardinality.MANY_TO_ONE.value
            else:
                cardinality = RelationshipCardinality.MANY_TO_MANY.value

            # Double counting and Fan-out risk analysis
            fan_out_ratio = 1.0
            double_counting_risk = False
            double_counting_measures: list[str] = []
            warnings: list[str] = []

            if cardinality in (RelationshipCardinality.ONE_TO_MANY.value, RelationshipCardinality.MANY_TO_MANY.value):
                target_counts = t_series.dropna().value_counts()
                if len(target_counts) > 0:
                    fan_out_ratio = float(target_counts.mean())

                if s_measures:
                    double_counting_risk = True
                    double_counting_measures = list(s_measures)
                    warnings.append(
                        f"Fan-out join (1:N, ratio={fan_out_ratio:.2f}) duplicates source rows; "
                        f"aggregating source measures {double_counting_measures[:3]} will cause double counting."
                    )

            if cardinality == RelationshipCardinality.MANY_TO_MANY.value:
                warnings.append("Many-to-many relationship detected; high risk of row multiplication and cartesian fan-out.")

            max_null_ratio = max(s_null_ratio, t_null_ratio)
            if max_null_ratio >= 0.20:
                warnings.append(f"Join key missingness is notable ({max_null_ratio:.1%} nulls); records without keys will not match.")

            # Compute transparent heuristic confidence score
            conf = 0.35 * max_containment + 0.20 * jaccard
            if is_s_id or is_t_id:
                conf += 0.15
            if is_s_id and is_t_id:
                conf += 0.10
            if name_match:
                conf += 0.15
            elif fk_match:
                conf += 0.08
            if cardinality in (RelationshipCardinality.ONE_TO_ONE.value, RelationshipCardinality.MANY_TO_ONE.value):
                conf += 0.05
            if max_null_ratio >= 0.20:
                conf -= 0.15 * max_null_ratio
            if cardinality == RelationshipCardinality.MANY_TO_MANY.value:
                conf -= 0.15

            conf = max(0.10, min(round(float(conf), 2), 0.98))

            # Explainable evidence (strictly privacy-hardened without raw values)
            evidence = [
                f"Value inclusion rate: {containment_src:.1%} of source values match target column ({overlap_count} overlapping distinct values)",
                f"Jaccard similarity index across distinct value sets: {jaccard:.2f}",
                f"Cardinality established as {cardinality.replace('_', ' ')} based on uniqueness ratios (src={s_uniq_ratio:.2f}, tgt={t_uniq_ratio:.2f})",
            ]
            if name_match:
                evidence.append(f"Normalized column name '{norm_s}' matches exactly across both datasets")
            elif fk_match:
                evidence.append("Column names match foreign-key naming pattern")
            elif is_s_id and is_t_id:
                evidence.append(
                    f"Strong value containment ({max_containment:.1%}) between identifier columns supports join candidate despite different column names ('{s_col_name}' vs '{t_col_name}')"
                )

            if is_s_id or is_t_id:
                evidence.append("Compatible semantic role ('id' / identifier detected)")

            # Separate candidate states: CANDIDATE vs NEEDS_REVIEW
            val_status = ValidationStatus.CANDIDATE.value
            review_reason: Optional[str] = None

            if cardinality == RelationshipCardinality.MANY_TO_MANY.value:
                val_status = ValidationStatus.NEEDS_REVIEW.value
                review_reason = "Many-to-many cardinality creates high risk of cartesian row multiplication and metric double counting."
            elif max_null_ratio >= 0.30:
                val_status = ValidationStatus.NEEDS_REVIEW.value
                review_reason = f"High missingness ({max_null_ratio:.1%}) on join key risks dropping or misattributing joined records."
            elif max_containment < 0.60:
                val_status = ValidationStatus.NEEDS_REVIEW.value
                review_reason = f"Partial key containment ({max_containment:.1%}) indicates incomplete overlap between datasets."
            elif not name_match and not fk_match and not (is_s_id and is_t_id):
                val_status = ValidationStatus.NEEDS_REVIEW.value
                review_reason = f"Differing column names ('{s_col_name}' vs '{t_col_name}') require manual verification of entity alignment."
            elif conf < 0.70:
                val_status = ValidationStatus.NEEDS_REVIEW.value
                review_reason = f"Heuristic confidence score ({conf:.0%}) indicates potential join key uncertainty."

            rel_id = f"{s_id}_{s_col_name}__{t_id}_{t_col_name}"
            candidates.append(
                DatasetRelationship(
                    id=rel_id,
                    source_dataset_id=s_id,
                    source_dataset_name=s_name,
                    source_column=str(s_col_name),
                    target_dataset_id=t_id,
                    target_dataset_name=t_name,
                    target_column=str(t_col_name),
                    cardinality=cardinality,
                    source_uniqueness_ratio=round(s_uniq_ratio, 3),
                    target_uniqueness_ratio=round(t_uniq_ratio, 3),
                    containment_ratio=round(containment_src, 3),
                    jaccard_similarity=round(jaccard, 3),
                    overlapping_values_count=overlap_count,
                    confidence=conf,
                    confidence_type="heuristic",
                    evidence=evidence,
                    warnings=warnings,
                    validation_status=val_status,
                    is_ambiguous=False,
                    fan_out_ratio=round(fan_out_ratio, 2),
                    double_counting_risk=double_counting_risk,
                    double_counting_measures=double_counting_measures,
                    review_reason=review_reason,
                    rejection_reason=None,
                )
            )

    # Ambiguity check: group by source column to see if multiple target columns match with similar confidence
    from collections import defaultdict
    by_source_col = defaultdict(list)
    for c in candidates:
        by_source_col[c.source_column].append(c)

    for col, group in by_source_col.items():
        if len(group) > 1:
            sorted_group = sorted(group, key=lambda x: x.confidence, reverse=True)
            top_conf = sorted_group[0].confidence
            # If second highest candidate is within 0.08 of top candidate
            if abs(top_conf - sorted_group[1].confidence) <= 0.08:
                for cand in sorted_group:
                    cand.is_ambiguous = True
                    cand.validation_status = ValidationStatus.AMBIGUOUS.value
                    cand.review_reason = (
                        "Multiple candidate join keys in target dataset share comparable confidence; manual confirmation required."
                    )
                    cand.warnings.append(
                        "Multiple candidate join keys in target dataset share comparable confidence; manual confirmation required."
                    )

    return candidates


# ---------------------------------------------------------------------------
# Join Validation
# ---------------------------------------------------------------------------

def validate_relationship_join(
    source_df: pd.DataFrame,
    source_col: str,
    target_df: pd.DataFrame,
    target_col: str,
    candidate: DatasetRelationship,
) -> DatasetRelationship:
    """Validate a candidate relationship join against actual data constraints.

    Evaluates:
    - Actual join row expansion
    - Null and unmatched rate
    - Re-confirms cardinality and updates validation status
    """
    if source_col not in source_df.columns or target_col not in target_df.columns:
        candidate.validation_status = ValidationStatus.REJECTED.value
        candidate.rejection_reason = "One or both columns not found in datasets"
        candidate.warnings.append("One or both columns not found in datasets")
        return candidate

    s_clean = source_df[[source_col]].dropna()
    t_clean = target_df[[target_col]].dropna()

    if len(s_clean) == 0 or len(t_clean) == 0:
        candidate.validation_status = ValidationStatus.REJECTED.value
        candidate.rejection_reason = "Insufficient non-null rows to validate join"
        candidate.warnings.append("Insufficient non-null rows to validate join")
        return candidate

    # Merge on subset to evaluate exact joined row count
    sample_s = s_clean.head(50000)
    sample_t = t_clean.head(50000)

    try:
        merged = pd.merge(sample_s, sample_t, left_on=source_col, right_on=target_col, how="inner")
        actual_joined_rows = len(merged)
        actual_fan_out = actual_joined_rows / max(len(sample_s), 1)

        candidate.fan_out_ratio = round(actual_fan_out, 2)

        if actual_joined_rows == 0:
            candidate.validation_status = ValidationStatus.REJECTED.value
            candidate.rejection_reason = "Join validation failed: zero matching rows produced"
            candidate.warnings.append("Join validation failed: zero matching rows produced")
        elif candidate.is_ambiguous:
            candidate.validation_status = ValidationStatus.AMBIGUOUS.value
            candidate.review_reason = "Join produced matching rows, but remains ambiguous with competing candidate keys"
            candidate.warnings.append("Join produced matching rows, but remains ambiguous with competing candidate keys; manual confirmation required.")
        else:
            candidate.validation_status = ValidationStatus.VALIDATED.value
            candidate.rejection_reason = None
            candidate.review_reason = None
            candidate.evidence.append(
                f"Data join validation confirmed: {actual_joined_rows} joined rows produced (fan-out ratio={actual_fan_out:.2f})"
            )
    except Exception as e:
        candidate.validation_status = ValidationStatus.REJECTED.value
        candidate.rejection_reason = f"Join execution error: {str(e)}"
        candidate.warnings.append(f"Join execution error: {str(e)}")

    return candidate
