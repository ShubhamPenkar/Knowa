"""Semantic data understanding and column inference engine.

Infers column roles and semantic types using:
1. Column names and naming conventions
2. Data types (native and parsed)
3. Cardinality, uniqueness, and missingness
4. Value distribution and sample patterns
5. Explainable evidence and calibrated confidence

Supports:
- Roles: identifier, measure, dimension, timestamp, unknown
- Semantic Types: id, numeric_measure, categorical, boolean, datetime, geographic, text, unknown
- User corrections with preservation across profiling runs
- Raw data preservation (semantic inference is non-destructive metadata only)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Semantic Roles & Types
# ---------------------------------------------------------------------------

class SemanticRole(str, Enum):
    IDENTIFIER = "identifier"
    MEASURE = "measure"
    DIMENSION = "dimension"
    TIMESTAMP = "timestamp"
    UNKNOWN = "unknown"


class SemanticType(str, Enum):
    ID = "id"
    NUMERIC_MEASURE = "numeric_measure"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    GEOGRAPHIC = "geographic"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class SemanticAnnotation:
    """Structured semantic inference for a column."""
    semantic_type: str = SemanticType.UNKNOWN.value
    role: str = SemanticRole.UNKNOWN.value
    confidence: float = 0.0  # Heuristic score in [0.0, 1.0], not a calibrated probability
    confidence_type: str = "heuristic"  # 'heuristic' for rule inferences, 'user_assigned' for manual overrides
    evidence: list[str] = field(default_factory=list)
    is_user_corrected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Regex Patterns & Domain Constants
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

_ID_NAME_RE = re.compile(
    r"(^|_)(id|idx|index|uuid|guid|pk|key|ref|code|number|no|ssn|ein|row_?id|row_?num)(_|$)|"
    r"customer_?id|user_?id|account_?id|store_?id|employee_?id|"
    r"member_?id|order_?id|transaction_?id|session_?id|patient_?id|"
    r"product_?id|item_?id|sku|device_?id|token",
    re.IGNORECASE,
)


def detect_identifier(
    name: str,
    s: pd.Series,
    n_rows: int,
    n_unique: int,
    non_null_count: int,
) -> Optional[SemanticAnnotation]:
    """Detect primary or surrogate row identifiers."""
    norm = _normalize_name(name)
    has_id_name = bool(_ID_NAME_RE.search(norm))
    uniq_ratio = float(n_unique / max(non_null_count, 1))

    # Check for UUIDs first (UUID format is unambiguous)
    samples = s.dropna().astype(str).head(10).tolist()
    if samples:
        uuid_matches = sum(1 for v in samples if bool(_UUID_RE.match(v.strip())))
        if uuid_matches / len(samples) >= 0.8:
            return SemanticAnnotation(
                semantic_type=SemanticType.ID.value,
                role=SemanticRole.IDENTIFIER.value,
                confidence=0.98,
                confidence_type="heuristic",
                evidence=[
                    "Values match standard RFC 4122 UUID pattern",
                    f"Uniqueness ratio = {uniq_ratio:.2f}",
                ],
            )

    # Low-cardinality exception: 'category_id' with 4 values is a dimension, not row identifier
    if has_id_name and (n_unique <= 10 or uniq_ratio < 0.20):
        return None

    # Monotonic sequential integers (surrogate key)
    if non_null_count >= 5 and pd.api.types.is_numeric_dtype(s):
        numeric_vals = pd.to_numeric(s.dropna(), errors="coerce").dropna()
        if len(numeric_vals) >= 5 and np.allclose(numeric_vals.to_numpy(), np.round(numeric_vals.to_numpy())):
            if uniq_ratio >= 0.95:
                ordered = np.sort(np.unique(numeric_vals.to_numpy()))
                diffs = np.diff(ordered)
                if len(diffs) > 0 and np.mean(diffs == 1.0) >= 0.75:
                    if has_id_name or float(ordered[0]) in (0.0, 1.0):
                        return SemanticAnnotation(
                            semantic_type=SemanticType.ID.value,
                            role=SemanticRole.IDENTIFIER.value,
                            confidence=0.95 if has_id_name else 0.86,
                            confidence_type="heuristic",
                            evidence=[
                                "Sequential integer surrogate key pattern (step=1)",
                                f"Uniqueness ratio = {uniq_ratio:.2f}",
                                "Monotonic integer progression with unit increment",
                            ],
                        )

    # High uniqueness + identifier name
    if has_id_name and uniq_ratio >= 0.90 and non_null_count >= 5:
        evidence = [
            f"Column name '{name}' matches identifier naming convention",
            f"High uniqueness: {n_unique} unique values among {non_null_count} rows ({uniq_ratio:.1%})",
        ]
        return SemanticAnnotation(
            semantic_type=SemanticType.ID.value,
            role=SemanticRole.IDENTIFIER.value,
            confidence=0.95,
            evidence=evidence,
        )

    return None

_DATETIME_NAME_RE = re.compile(
    r"(^|_)(date|time|timestamp|datetime|dt|dob|epoch)(_|$)|"
    r"created_?at|updated_?at|deleted_?at|modified_?at|start_?date|end_?date|"
    r"birth_?date|opened_?at|closed_?at|expiry|expiration|due_?date|"
    r"signup_?date|join_?date|event_?time",
    re.IGNORECASE,
)

_ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$"
)

_SLASH_DATE_RE = re.compile(
    r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:[ T]\d{1,2}:\d{2}(?:\:\d{2})?)?$"
)

_BOOLEAN_NAME_RE = re.compile(
    r"^(is|has|can|should|was|will|did|flag|active|enabled|churn|churned|converted|success)(_|$)|"
    r"(_is|_has|_flag|_active|_enabled)$",
    re.IGNORECASE,
)

_BOOLEAN_VAL_SETS = {
    frozenset({"true", "false"}),
    frozenset({"t", "f"}),
    frozenset({"yes", "no"}),
    frozenset({"y", "n"}),
    frozenset({"1", "0"}),
    frozenset({"1.0", "0.0"}),
}

_GEO_NAME_RE = re.compile(
    r"(^|_)(country|nation|state|province|region|city|town|county|zip|zipcode|"
    r"postal_?code|postcode|latitude|longitude|lat|lon|lng|coords|address|street|location)(_|$)",
    re.IGNORECASE,
)

_US_ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$")

_COUNTRY_CODES_ISO2 = {
    "US", "GB", "CA", "DE", "FR", "IN", "AU", "JP", "CN", "BR", "MX", "IT",
    "ES", "NL", "SE", "NO", "DK", "FI", "CH", "AT", "IE", "NZ", "SG", "ZA",
    "KR", "RU", "PL", "BE", "PT", "GR", "IL", "AE", "SA", "AR", "CL", "CO",
}

_COUNTRY_NAMES = {
    "united states", "usa", "united kingdom", "uk", "canada", "germany",
    "france", "india", "australia", "japan", "china", "brazil", "mexico",
    "italy", "spain", "netherlands", "sweden", "norway", "denmark", "finland",
    "switzerland", "austria", "ireland", "new zealand", "singapore", "south africa",
}

_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

_MEASURE_NAME_RE = re.compile(
    r"(^|_)(amount|price|cost|charge|charges|fee|fees|revenue|sales|salary|income|"
    r"budget|profit|margin|discount|balance|spend|spending|gdp|rate|pct|percent|"
    r"percentage|ratio|score|tenure|age|count|num|total|quantity|qty|distance|"
    r"weight|height|length|duration|seconds|minutes|hours|days|weeks|months|years|"
    r"speed|temperature|volume|capacity|size)(_|$)",
    re.IGNORECASE,
)

_TEXT_NAME_RE = re.compile(
    r"(^|_)(description|desc|comment|comments|note|notes|feedback|review|reviews|"
    r"summary|details|body|message|text|bio|transcript|explanation|narrative)(_|$)",
    re.IGNORECASE,
)

_CATEGORICAL_NAME_RE = re.compile(
    r"(^|_)(type|category|status|group|tier|segment|department|gender|sex|level|"
    r"mode|method|role|plan|channel|brand|contract|payment|industry|service|"
    r"priority|severity|state_code|class|code)(_|$)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Individual Detection Heuristics
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return s.strip("_")


def detect_datetime(
    name: str,
    s: pd.Series,
    non_null_count: int,
) -> Optional[SemanticAnnotation]:
    """Detect timestamps, dates, and temporal indicators."""
    norm = _normalize_name(name)
    has_dt_name = bool(_DATETIME_NAME_RE.search(norm))

    # Native pandas datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        return SemanticAnnotation(
            semantic_type=SemanticType.DATETIME.value,
            role=SemanticRole.TIMESTAMP.value,
            confidence=0.99,
            evidence=[
                "Native datetime dtype in dataframe",
                f"Column name: '{name}'",
            ],
        )

    # String-based dates
    if non_null_count > 0:
        samples = s.dropna().astype(str).str.strip().head(20).tolist()
        if samples:
            iso_matches = sum(1 for v in samples if bool(_ISO_DATE_RE.match(v)))
            slash_matches = sum(1 for v in samples if bool(_SLASH_DATE_RE.match(v)))
            total_dt_pattern = (iso_matches + slash_matches) / len(samples)

            if total_dt_pattern >= 0.75:
                conf = 0.96 if has_dt_name else 0.88
                return SemanticAnnotation(
                    semantic_type=SemanticType.DATETIME.value,
                    role=SemanticRole.TIMESTAMP.value,
                    confidence=conf,
                    confidence_type="heuristic",
                    evidence=[
                        f"{total_dt_pattern:.0%} of values match standard calendar date or timestamp patterns",
                        f"Name '{name}' matches temporal token" if has_dt_name else "Detected via date string pattern",
                    ],
                )

    # Numeric Unix epoch timestamps
    if non_null_count >= 5 and pd.api.types.is_numeric_dtype(s) and has_dt_name:
        clean = pd.to_numeric(s.dropna(), errors="coerce").dropna()
        if len(clean) >= 5:
            min_v, max_v = float(clean.min()), float(clean.max())
            # Seconds: ~2001 (1e9) to 2038+ (2.5e9)
            if 1_000_000_000 <= min_v and max_v <= 2_500_000_000:
                return SemanticAnnotation(
                    semantic_type=SemanticType.DATETIME.value,
                    role=SemanticRole.TIMESTAMP.value,
                    confidence=0.90,
                    confidence_type="heuristic",
                    evidence=[
                        "Numeric values fall within standard Unix epoch seconds range",
                        f"Column name '{name}' indicates timestamp",
                    ],
                )
            # Milliseconds: ~2001 (1e12) to 2038+ (2.5e12)
            if 1_000_000_000_000 <= min_v and max_v <= 2_500_000_000_000:
                return SemanticAnnotation(
                    semantic_type=SemanticType.DATETIME.value,
                    role=SemanticRole.TIMESTAMP.value,
                    confidence=0.90,
                    confidence_type="heuristic",
                    evidence=[
                        "Numeric values fall within standard Unix epoch millisecond range",
                        f"Column name '{name}' indicates timestamp",
                    ],
                )

    return None


def detect_boolean(
    name: str,
    s: pd.Series,
    n_unique: int,
    non_null_count: int,
) -> Optional[SemanticAnnotation]:
    """Detect boolean flags and binary indicators."""
    norm = _normalize_name(name)
    has_bool_name = bool(_BOOLEAN_NAME_RE.search(norm))

    # Native boolean
    if pd.api.types.is_bool_dtype(s):
        return SemanticAnnotation(
            semantic_type=SemanticType.BOOLEAN.value,
            role=SemanticRole.DIMENSION.value,
            confidence=0.99,
            evidence=[
                "Native boolean data type",
                f"Column name '{name}'",
            ],
        )

    # Binary distinct values check
    if non_null_count > 0 and n_unique in (1, 2):
        present_str_set = frozenset(
            s.dropna().astype(str).str.strip().str.lower().unique().tolist()
        )
        if any(present_str_set.issubset(val_set) for val_set in _BOOLEAN_VAL_SETS):
            # Check if 0/1 without boolean name
            is_numeric_binary = present_str_set.issubset({"0", "1", "0.0", "1.0"})
            if is_numeric_binary:
                conf = 0.92 if has_bool_name else 0.75
            else:
                conf = 0.95

            return SemanticAnnotation(
                semantic_type=SemanticType.BOOLEAN.value,
                role=SemanticRole.DIMENSION.value,
                confidence=conf,
                confidence_type="heuristic",
                evidence=[
                    "Values strictly restricted to standard binary boolean indicator set",
                    f"Column name '{name}' suggests boolean flag" if has_bool_name else "Binary categorical indicator",
                ],
            )

    return None


def detect_geographic(
    name: str,
    s: pd.Series,
    n_unique: int,
    non_null_count: int,
) -> Optional[SemanticAnnotation]:
    """Detect geographic dimensions (country, state, city, zip, coordinates)."""
    norm = _normalize_name(name)
    has_geo_name = bool(_GEO_NAME_RE.search(norm))

    # Coordinates
    if pd.api.types.is_numeric_dtype(s) and non_null_count >= 5:
        clean = pd.to_numeric(s.dropna(), errors="coerce").dropna()
        if len(clean) >= 5:
            min_v, max_v = float(clean.min()), float(clean.max())
            if re.search(r"(^|_)(latitude|lat)(_|$)", norm):
                if -90.0 <= min_v and max_v <= 90.0:
                    return SemanticAnnotation(
                        semantic_type=SemanticType.GEOGRAPHIC.value,
                        role=SemanticRole.DIMENSION.value,
                        confidence=0.96,
                        confidence_type="heuristic",
                        evidence=[
                            "Numeric values fall within valid geographic latitude range [-90, +90]",
                            f"Column name '{name}' specifies latitude",
                        ],
                    )
            if re.search(r"(^|_)(longitude|lon|lng)(_|$)", norm):
                if -180.0 <= min_v and max_v <= 180.0:
                    return SemanticAnnotation(
                        semantic_type=SemanticType.GEOGRAPHIC.value,
                        role=SemanticRole.DIMENSION.value,
                        confidence=0.96,
                        confidence_type="heuristic",
                        evidence=[
                            "Numeric values fall within valid geographic longitude range [-180, +180]",
                            f"Column name '{name}' specifies longitude",
                        ],
                    )

    # Postal / Zip Codes
    if has_geo_name and re.search(r"(zip|postal)", norm):
        samples = s.dropna().astype(str).str.strip().head(20).tolist()
        if samples:
            zip_matches = sum(1 for v in samples if bool(_US_ZIP_RE.match(v)))
            if zip_matches / len(samples) >= 0.70:
                return SemanticAnnotation(
                    semantic_type=SemanticType.GEOGRAPHIC.value,
                    role=SemanticRole.DIMENSION.value,
                    confidence=0.94,
                    confidence_type="heuristic",
                    evidence=[
                        "Values match standard postal/ZIP code format pattern",
                        f"Column name '{name}' indicates postal code",
                    ],
                )

    # Country / State
    if non_null_count > 0:
        samples_raw = s.dropna().astype(str).str.strip().head(30).tolist()
        if samples_raw:
            upper_samples = {v.upper() for v in samples_raw}
            lower_samples = {v.lower() for v in samples_raw}

            country_code_match = len(upper_samples.intersection(_COUNTRY_CODES_ISO2)) / len(upper_samples)
            country_name_match = len(lower_samples.intersection(_COUNTRY_NAMES)) / len(lower_samples)
            state_match = len(upper_samples.intersection(_US_STATES)) / len(upper_samples)

            if has_geo_name and (country_code_match >= 0.4 or country_name_match >= 0.4 or state_match >= 0.4):
                return SemanticAnnotation(
                    semantic_type=SemanticType.GEOGRAPHIC.value,
                    role=SemanticRole.DIMENSION.value,
                    confidence=0.93,
                    evidence=[
                        f"Column name '{name}' matches geographic entity",
                        "Sample values contain standard geographic codes or names",
                    ],
                )
            if not has_geo_name and (country_code_match >= 0.75 or country_name_match >= 0.75):
                return SemanticAnnotation(
                    semantic_type=SemanticType.GEOGRAPHIC.value,
                    role=SemanticRole.DIMENSION.value,
                    confidence=0.82,
                    evidence=[
                        "High proportion of values match standard country names or ISO country codes",
                    ],
                )

    return None


def detect_text(
    name: str,
    s: pd.Series,
    n_unique: int,
    non_null_count: int,
) -> Optional[SemanticAnnotation]:
    """Detect unstructured free-form text columns."""
    norm = _normalize_name(name)
    has_text_name = bool(_TEXT_NAME_RE.search(norm))

    if non_null_count > 0 and not pd.api.types.is_numeric_dtype(s):
        samples = s.dropna().astype(str).head(30).tolist()
        if samples:
            avg_len = float(np.mean([len(v) for v in samples]))
            avg_words = float(np.mean([len(v.split()) for v in samples]))

            # Long multi-word text
            if (avg_len > 40 and avg_words >= 4) or (has_text_name and avg_words >= 2):
                return SemanticAnnotation(
                    semantic_type=SemanticType.TEXT.value,
                    role=SemanticRole.DIMENSION.value,
                    confidence=0.92 if has_text_name else 0.85,
                    evidence=[
                        f"Unstructured text content (avg length: {avg_len:.0f} chars, avg word count: {avg_words:.1f})",
                        f"Column name '{name}' indicates text field" if has_text_name else "High natural language character density",
                    ],
                )

    return None


def detect_measure(
    name: str,
    s: pd.Series,
    n_unique: int,
    non_null_count: int,
) -> Optional[SemanticAnnotation]:
    """Detect quantitative continuous or discrete metrics (revenue, age, count, etc.)."""
    norm = _normalize_name(name)
    has_measure_name = bool(_MEASURE_NAME_RE.search(norm))

    is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if not is_num:
        # Check string coercion rate
        cleaned = (
            s.dropna()
            .astype(str)
            .str.strip()
            .str.replace(r"[\$,%]", "", regex=True)
            .str.replace(r"\s+", "", regex=True)
        )
        parsed = pd.to_numeric(cleaned, errors="coerce")
        if len(cleaned) > 0 and (parsed.notna().sum() / len(cleaned)) >= 0.95:
            is_num = True

    if is_num and non_null_count > 0:
        numeric_series = pd.to_numeric(
            s.dropna().astype(str).str.replace(r"[\$,%]", "", regex=True),
            errors="coerce",
        ).dropna()

        if len(numeric_series) > 0:
            min_v, max_v = float(numeric_series.min()), float(numeric_series.max())
            is_float = not np.allclose(numeric_series.to_numpy(), np.round(numeric_series.to_numpy()))

            # If it has a quantitative name
            if has_measure_name:
                return SemanticAnnotation(
                    semantic_type=SemanticType.NUMERIC_MEASURE.value,
                    role=SemanticRole.MEASURE.value,
                    confidence=0.95,
                    confidence_type="heuristic",
                    evidence=[
                        f"Column name '{name}' indicates quantitative metric",
                        f"Quantitative numeric distribution with {n_unique} distinct values",
                    ],
                )

            # Continuous floats with sufficient distinct values
            if is_float and n_unique >= 10:
                return SemanticAnnotation(
                    semantic_type=SemanticType.NUMERIC_MEASURE.value,
                    role=SemanticRole.MEASURE.value,
                    confidence=0.91,
                    confidence_type="heuristic",
                    evidence=[
                        f"Continuous numeric distribution with {n_unique} unique values",
                        f"Quantitative continuous scale without discrete clustering",
                    ],
                )

            # High-cardinality integers without ID pattern
            if n_unique >= 15:
                return SemanticAnnotation(
                    semantic_type=SemanticType.NUMERIC_MEASURE.value,
                    role=SemanticRole.MEASURE.value,
                    confidence=0.85,
                    confidence_type="heuristic",
                    evidence=[
                        f"Quantitative numeric metric with {n_unique} distinct values",
                        "High cardinality integer distribution consistent with measure",
                    ],
                )

    return None


def detect_categorical(
    name: str,
    s: pd.Series,
    n_unique: int,
    non_null_count: int,
) -> Optional[SemanticAnnotation]:
    """Detect discrete categorical dimensions."""
    norm = _normalize_name(name)
    has_cat_name = bool(_CATEGORICAL_NAME_RE.search(norm))
    uniq_ratio = float(n_unique / max(non_null_count, 1))

    # Low-cardinality discrete values (strings or coded numbers)
    if non_null_count > 0 and (n_unique <= 30 or uniq_ratio <= 0.20):
        return SemanticAnnotation(
            semantic_type=SemanticType.CATEGORICAL.value,
            role=SemanticRole.DIMENSION.value,
            confidence=0.92 if has_cat_name else 0.86,
            confidence_type="heuristic",
            evidence=[
                f"Low-cardinality discrete values ({n_unique} categories)",
                "Repeated discrete categories across rows without exposing raw sample values",
                f"Name '{name}' matches categorical dimension" if has_cat_name else "Repeated discrete values across rows",
            ],
        )

    return None


# ---------------------------------------------------------------------------
# Main Inference Function
# ---------------------------------------------------------------------------

def infer_column_semantics(
    name: str,
    series: pd.Series,
    *,
    user_override: Optional[dict[str, Any]] = None,
    n_rows: Optional[int] = None,
) -> SemanticAnnotation:
    """Infer semantic type, role, confidence, and explainable evidence for a column.

    If a user override is provided, user corrections take absolute precedence
    and are marked with `is_user_corrected = True`.
    """
    # 1. User corrections take absolute precedence
    if user_override and isinstance(user_override, dict):
        sem_type = user_override.get("semantic_type")
        role = user_override.get("role")
        if sem_type or role:
            resolved_type = str(sem_type or SemanticType.UNKNOWN.value)
            resolved_role = str(role or SemanticRole.DIMENSION.value)
            notes = user_override.get("notes") or "User confirmed/corrected column meaning"
            return SemanticAnnotation(
                semantic_type=resolved_type,
                role=resolved_role,
                confidence=1.0,
                confidence_type="user_assigned",
                evidence=[f"User override: {notes}"],
                is_user_corrected=True,
            )

    # 2. Extract series properties
    total_rows = n_rows if n_rows is not None else len(series)
    non_null = series.dropna()
    non_null_count = len(non_null)
    n_unique = int(non_null.nunique()) if non_null_count else 0

    # 3. Handle zero-variance / completely empty columns
    if non_null_count == 0 or n_unique <= 1:
        return SemanticAnnotation(
            semantic_type=SemanticType.UNKNOWN.value,
            role=SemanticRole.UNKNOWN.value,
            confidence=0.10,
            evidence=[
                "All values null or constant single value — zero variance",
                "Ambiguous/uninformative for semantic analysis",
            ],
            is_user_corrected=False,
        )

    # 4. Try high-precision detectors in priority order
    # (a) Datetime / Timestamps
    dt_res = detect_datetime(name, series, non_null_count)
    if dt_res and dt_res.confidence >= 0.85:
        return dt_res

    # (b) Booleans / Binary Flags
    bool_res = detect_boolean(name, series, n_unique, non_null_count)
    if bool_res and bool_res.confidence >= 0.80:
        return bool_res

    # (c) Geographic Dimensions
    geo_res = detect_geographic(name, series, n_unique, non_null_count)
    if geo_res and geo_res.confidence >= 0.80:
        return geo_res

    # (d) Primary & Surrogate Identifiers
    id_res = detect_identifier(name, series, total_rows, n_unique, non_null_count)
    if id_res and id_res.confidence >= 0.80:
        return id_res

    # (e) Free-form Unstructured Text
    text_res = detect_text(name, series, n_unique, non_null_count)
    if text_res and text_res.confidence >= 0.80:
        return text_res

    # (f) Quantitative Measures
    measure_res = detect_measure(name, series, n_unique, non_null_count)
    if measure_res and measure_res.confidence >= 0.80:
        return measure_res

    # (g) Categorical Dimensions
    cat_res = detect_categorical(name, series, n_unique, non_null_count)
    if cat_res:
        return cat_res

    # (h) Fallback: Check if lower-confidence date or measure exists
    if dt_res:
        return dt_res
    if measure_res:
        return measure_res

    # 5. Truly ambiguous / unknown columns (do not force-guess)
    return SemanticAnnotation(
        semantic_type=SemanticType.UNKNOWN.value,
        role=SemanticRole.UNKNOWN.value,
        confidence=0.30,
        evidence=[
            f"No dominant date, geographic, identifier, or numeric pattern recognized",
            f"{n_unique} unique values among {non_null_count} rows ({n_unique / max(non_null_count, 1):.1%} uniqueness)",
            "Ambiguous distribution kept unknown for user clarification",
        ],
        is_user_corrected=False,
    )
