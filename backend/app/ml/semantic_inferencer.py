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
    # Core types (matching existing definitions)
    ID = "id"
    NUMERIC_MEASURE = "numeric_measure"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    GEOGRAPHIC = "geographic"
    TEXT = "text"
    UNKNOWN = "unknown"

    # Extended Phase 1 types
    IDENTIFIER = "identifier"
    ENTITY_ID = "entity_id"
    MEASURE = "measure"
    DIMENSION = "dimension"
    ORDINAL = "ordinal"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    DATE = "date"
    DURATION = "duration"
    EMAIL = "email"
    PHONE = "phone"
    TARGET_CANDIDATE = "target_candidate"


class BusinessRole(str, Enum):
    # Monetary / Financial
    REVENUE = "revenue"
    SALES = "sales"
    PROFIT = "profit"
    COST = "cost"
    PRICE = "price"
    BUDGET = "budget"
    SALARY = "salary"
    BALANCE = "balance"

    # Quantitative / Operational
    QUANTITY = "quantity"
    COUNT = "count"
    PERCENTAGE = "percentage"
    SCORE = "score"
    RATING = "rating"
    DURATION = "duration"
    TENURE = "tenure"
    AGE = "age"

    # Entities / Actors
    CUSTOMER = "customer"
    PRODUCT = "product"
    ORDER = "order"
    TRANSACTION = "transaction"
    EMPLOYEE = "employee"
    USER = "user"

    # Temporal
    DATE = "date"
    TIMESTAMP = "timestamp"

    # Geographic
    COUNTRY = "country"
    REGION = "region"
    STATE = "state"
    CITY = "city"
    POSTAL_CODE = "postal_code"
    LOCATION = "location"

    # Categorical / Status
    STATUS = "status"
    CATEGORY = "category"
    SEGMENT = "segment"
    TIER = "tier"
    CONTRACT = "contract"
    PAYMENT_METHOD = "payment_method"

    # Targets / Outcomes
    CHURN = "churn"
    CONVERSION = "conversion"
    DEFAULT = "default"
    FRAUD = "fraud"
    TARGET = "target"

    # Contact / Structural
    EMAIL = "email"
    PHONE = "phone"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"


@dataclass
class SemanticAnnotation:
    """Structured semantic inference for a column."""
    semantic_type: str = SemanticType.UNKNOWN.value
    role: str = SemanticRole.UNKNOWN.value
    business_role: str = BusinessRole.UNKNOWN.value
    confidence: float = 0.0  # Heuristic score in [0.0, 1.0], not a calibrated probability
    confidence_type: str = "heuristic"  # 'heuristic' for rule inferences, 'user_assigned' for manual overrides
    evidence: list[str] = field(default_factory=list)
    is_user_corrected: bool = False
    is_identifier: bool = False
    is_target_candidate: bool = False
    target_type: Optional[str] = None  # classification | regression
    target_confidence: float = 0.0
    quality_issues: list[str] = field(default_factory=list)
    distribution_summary: dict[str, Any] = field(default_factory=dict)

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
# Extended Phase 1 Detectors
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

_PHONE_RE = re.compile(
    r"^(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{3,5}$|"
    r"^(\+?\d{1,3}[-.\s]?)?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,5}$|"
    r"^\+?\d{9,15}$|"
    r"^\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{3,5}$"
)

_CURRENCY_SYMBOL_RE = re.compile(
    r"^\s*[\$€£¥₹]\s*-?[\d,]+(\.\d+)?\s*$|"
    r"^\s*-?[\d,]+(\.\d+)?\s*[\$€£¥₹]\s*$|"
    r"^\s*(USD|EUR|GBP|CAD|AUD|INR)\s*-?[\d,]+(\.\d+)?\s*$",
    re.IGNORECASE,
)

_PERCENT_STR_RE = re.compile(r"^\s*-?[\d,]+(\.\d+)?\s*%\s*$")


def detect_currency(name: str, s: pd.Series, non_null_count: int) -> Optional[SemanticAnnotation]:
    """Detect currency / monetary values formatted with currency symbols or tokens."""
    if non_null_count == 0:
        return None
    samples = s.dropna().astype(str).str.strip().head(40).tolist()
    if not samples:
        return None
    matches = sum(1 for v in samples if bool(_CURRENCY_SYMBOL_RE.match(v)))
    ratio = matches / len(samples)
    norm = _normalize_name(name)
    has_name = bool(re.search(r"(^|_)(price|cost|amount|revenue|fee|salary|charge|budget|spend|balance|wage|invoice|money)(_|$)", norm))

    if ratio >= 0.60 or (has_name and ratio >= 0.30):
        conf = 0.96 if ratio >= 0.8 else 0.88
        return SemanticAnnotation(
            semantic_type=SemanticType.CURRENCY.value,
            role=SemanticRole.MEASURE.value,
            business_role=BusinessRole.PRICE.value,
            confidence=conf,
            confidence_type="heuristic",
            evidence=[
                f"Values formatted with currency/monetary symbols ({ratio:.0%} of sample)",
                "Monetary quantity scale consistent with financial measure",
            ],
        )
    return None


def detect_percentage(name: str, s: pd.Series, non_null_count: int) -> Optional[SemanticAnnotation]:
    """Detect percentages from '%' formatting or [0.0, 1.0] numeric distribution with percentage token."""
    if non_null_count == 0:
        return None
    samples = s.dropna().astype(str).str.strip().head(40).tolist()
    if not samples:
        return None
    matches = sum(1 for v in samples if bool(_PERCENT_STR_RE.match(v)))
    ratio = matches / len(samples)
    norm = _normalize_name(name)
    has_pct_name = bool(re.search(r"(^|_)(percent|percentage|pct|rate|ratio|share|portion|discount)(_|$)", norm))

    if ratio >= 0.60:
        return SemanticAnnotation(
            semantic_type=SemanticType.PERCENTAGE.value,
            role=SemanticRole.MEASURE.value,
            business_role=BusinessRole.PERCENTAGE.value,
            confidence=0.95,
            confidence_type="heuristic",
            evidence=[
                f"Values explicitly formatted with percentage sign '%' ({ratio:.0%} of sample)",
                "Proportional scale bounded between 0% and 100%",
            ],
        )
    if pd.api.types.is_numeric_dtype(s) and has_pct_name:
        nums = s.dropna()
        if len(nums) > 0 and 0.0 <= float(nums.min()) and float(nums.max()) <= 1.0:
            return SemanticAnnotation(
                semantic_type=SemanticType.PERCENTAGE.value,
                role=SemanticRole.MEASURE.value,
                business_role=BusinessRole.PERCENTAGE.value,
                confidence=0.90,
                confidence_type="heuristic",
                evidence=[
                    f"Numeric values bounded in [0.0, 1.0] (min={nums.min():.2f}, max={nums.max():.2f})",
                    f"Column name '{name}' indicates percentage/rate",
                ],
            )
    return None


_ORDINAL_SEQUENCES = [
    {"low", "medium", "high"},
    {"low", "med", "high"},
    {"small", "medium", "large"},
    {"bronze", "silver", "gold"},
    {"bronze", "silver", "gold", "platinum"},
    {"tier 1", "tier 2", "tier 3"},
    {"tier 1", "tier 2", "tier 3", "tier 4"},
    {"tier_1", "tier_2", "tier_3"},
    {"tier_1", "tier_2", "tier_3", "tier_4"},
    {"poor", "fair", "good", "very good", "excellent"},
    {"poor", "fair", "good", "excellent"},
    {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"},
    {"beginner", "intermediate", "advanced", "expert"},
    {"entry", "mid", "senior", "lead", "principal"},
]


def detect_email(name: str, s: pd.Series, non_null_count: int) -> Optional[SemanticAnnotation]:
    """Detect email address columns via RFC 5322 pattern matching."""
    if non_null_count == 0 or pd.api.types.is_numeric_dtype(s):
        return None
    samples = s.dropna().astype(str).str.strip().head(30).tolist()
    if not samples:
        return None
    matches = sum(1 for v in samples if bool(_EMAIL_RE.match(v)))
    ratio = matches / len(samples)
    norm = _normalize_name(name)
    has_name = bool(re.search(r"(^|_)(email|e_mail|mail)(_|$)", norm))

    if ratio >= 0.70 or (has_name and ratio >= 0.40):
        conf = 0.98 if (ratio >= 0.8 and has_name) else (0.92 if ratio >= 0.8 else 0.82)
        return SemanticAnnotation(
            semantic_type=SemanticType.EMAIL.value,
            role=SemanticRole.DIMENSION.value,
            business_role=BusinessRole.EMAIL.value,
            confidence=conf,
            confidence_type="heuristic",
            evidence=[
                f"Value pattern matches standard email syntax ({ratio:.0%} of sample)",
                f"Column name '{name}' indicates email address" if has_name else "Pattern-based email detection",
            ],
        )
    return None


def detect_phone(name: str, s: pd.Series, non_null_count: int) -> Optional[SemanticAnnotation]:
    """Detect telephone numbers via formatting pattern matching."""
    if non_null_count == 0:
        return None
    norm = _normalize_name(name)
    has_name = bool(re.search(r"(^|_)(phone|telephone|mobile|cell|fax)(_|$)", norm))
    # Unformatted pure numeric integers should not be guessed as phone numbers unless the column name says so
    if pd.api.types.is_numeric_dtype(s) and not has_name:
        return None
    samples = s.dropna().astype(str).str.strip().head(30).tolist()
    if not samples:
        return None
    matches = sum(1 for v in samples if bool(_PHONE_RE.match(v)))
    ratio = matches / len(samples)

    if ratio >= 0.70 or (has_name and ratio >= 0.40):
        conf = 0.96 if (ratio >= 0.8 and has_name) else (0.90 if ratio >= 0.8 else 0.80)
        return SemanticAnnotation(
            semantic_type=SemanticType.PHONE.value,
            role=SemanticRole.DIMENSION.value,
            business_role=BusinessRole.PHONE.value,
            confidence=conf,
            confidence_type="heuristic",
            evidence=[
                f"Value pattern matches telephone formatting pattern ({ratio:.0%} of sample)",
                f"Column name '{name}' indicates phone number" if has_name else "Pattern-based telephone detection",
            ],
        )
    return None


def detect_ordinal(name: str, s: pd.Series, n_unique: int, non_null_count: int) -> Optional[SemanticAnnotation]:
    """Detect ordinal dimension with recognized ranking scale."""
    if non_null_count == 0 or n_unique < 2 or n_unique > 7 or pd.api.types.is_numeric_dtype(s):
        return None
    sample_set = {str(v).strip().lower() for v in s.dropna().unique()}
    for scale in _ORDINAL_SEQUENCES:
        if sample_set == scale or (sample_set.issubset(scale) and len(sample_set) >= 2):
            return SemanticAnnotation(
                semantic_type=SemanticType.ORDINAL.value,
                role=SemanticRole.DIMENSION.value,
                business_role=BusinessRole.TIER.value,
                confidence=0.92,
                confidence_type="heuristic",
                evidence=[
                    f"Values match recognized ordered sequence scale ({n_unique} distinct progression levels)",
                    f"Column name '{name}' represents ordinal rank or tier",
                ],
            )
    return None


def detect_business_role(
    name: str,
    s: pd.Series,
    semantic_type: str,
    role: str,
    n_unique: int,
    non_null_count: int,
) -> tuple[str, list[str]]:
    """Infer granular domain business role using names, value distributions, and ranges.
    
    Returns (business_role, additional_evidence).
    Does NOT depend exclusively on column names: distribution statistics are evaluated.
    """
    norm = _normalize_name(name)
    ev: list[str] = []

    # Distribution statistics when numeric
    is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    clean = pd.to_numeric(s.dropna(), errors="coerce").dropna() if non_null_count > 0 else pd.Series([], dtype=float)
    has_vals = len(clean) > 0
    min_v = float(clean.min()) if has_vals else 0.0
    max_v = float(clean.max()) if has_vals else 0.0
    mean_v = float(clean.mean()) if has_vals else 0.0
    std_v = float(clean.std()) if len(clean) > 1 else 0.0
    is_int = bool(np.allclose(clean.to_numpy(), np.round(clean.to_numpy()))) if has_vals else False

    # 1. Target candidate roles
    if re.search(r"(^|_)(churn|churned|attrition|exited|left|cancelled|retention)(_|$)", norm):
        return BusinessRole.CHURN.value, [f"Column name '{name}' indicates customer churn or retention outcome"]
    if re.search(r"(^|_)(converted|conversion|subscribed|opt_?in|signed_?up|signup)(_|$)", norm):
        return BusinessRole.CONVERSION.value, [f"Column name '{name}' indicates business conversion outcome"]
    if re.search(r"(^|_)(default|delinquent|charged_?off|bad_?loan)(_|$)", norm):
        return BusinessRole.DEFAULT.value, [f"Column name '{name}' indicates credit default or delinquency"]
    if re.search(r"(^|_)(fraud|is_?fraud|anomaly)(_|$)", norm):
        return BusinessRole.FRAUD.value, [f"Column name '{name}' indicates fraud detection indicator"]
    if re.search(r"^(target|label|outcome|y)$", norm):
        return BusinessRole.TARGET.value, [f"Column name '{name}' indicates explicit supervised modeling target"]

    # 2. Financial / Monetary roles
    if re.search(r"(^|_)(revenue|total_?revenue|income|arr|mrr|turnover)(_|$)", norm):
        return BusinessRole.REVENUE.value, [f"Column name '{name}' indicates top-line revenue or recurring income metric"]
    if re.search(r"(^|_)(sales|gross_?sales|net_?sales)(_|$)", norm):
        return BusinessRole.SALES.value, [f"Column name '{name}' indicates sales volume or monetary sales"]
    if re.search(r"(^|_)(profit|net_?profit|margin|operating_?margin)(_|$)", norm):
        return BusinessRole.PROFIT.value, [f"Column name '{name}' indicates profit or financial margin"]
    if re.search(r"(^|_)(cost|cogs|expense|expenses|spend|spending|charges|fee|fees)(_|$)", norm):
        return BusinessRole.COST.value, [f"Column name '{name}' indicates operational cost or expenditure"]
    if re.search(r"(^|_)(price|unit_?price|fare|rate|ticket_?price)(_|$)", norm):
        return BusinessRole.PRICE.value, [f"Column name '{name}' indicates pricing or unit rate"]
    if re.search(r"(^|_)(salary|wage|compensation|pay)(_|$)", norm):
        return BusinessRole.SALARY.value, [f"Column name '{name}' indicates employee compensation or salary"]
    if re.search(r"(^|_)(balance|deposit|equity)(_|$)", norm):
        return BusinessRole.BALANCE.value, [f"Column name '{name}' indicates financial balance or capital balance"]
    if re.search(r"(^|_)(budget|allocation)(_|$)", norm):
        return BusinessRole.BUDGET.value, [f"Column name '{name}' indicates planned budget allocation"]

    # 3. Demographic & Behavioral (Supports generic names like X17 via distribution evidence)
    # Age
    if re.search(r"(^|_)(age|cust_?age|customer_?age|user_?age|client_?age)(_|$)", norm):
        return BusinessRole.AGE.value, [f"Column name '{name}' and numeric distribution indicate chronological age"]
    if is_num and is_int and len(clean) >= 10:
        if 15.0 <= min_v and max_v <= 115.0 and 22.0 <= mean_v <= 65.0 and 5.0 <= std_v <= 28.0 and n_unique >= 10:
            return BusinessRole.AGE.value, [
                f"Integer distribution bounded in human age range [{int(min_v)}-{int(max_v)}], median ~{int(mean_v)}, zero negative values"
            ]

    # Rating / Score
    if re.search(r"(^|_)(rating|score|satisfaction|nps|csat|stars)(_|$)", norm):
        return BusinessRole.RATING.value, [f"Column name '{name}' and bounded range indicate rating score"]
    if is_num and len(clean) >= 10:
        if 0.0 <= min_v and max_v <= 5.0 and 1.5 <= mean_v <= 4.8 and n_unique <= 11:
            return BusinessRole.RATING.value, [
                f"Values strictly bounded in 0-5 rating scale range with discrete interval clusters (min={min_v}, max={max_v})"
            ]

    # Percentage / Ratio
    if re.search(r"(^|_)(pct|percent|percentage|ratio|rate|discount_?rate|interest_?rate)(_|$)", norm):
        return BusinessRole.PERCENTAGE.value, [f"Column name '{name}' indicates percentage or ratio"]
    if is_num and not is_int and len(clean) >= 10:
        if 0.0 <= min_v and max_v <= 1.0 and n_unique >= 8:
            return BusinessRole.PERCENTAGE.value, [
                "Continuous numeric distribution strictly bounded between 0.0 and 1.0 consistent with ratio/probability"
            ]

    # Tenure & Duration
    if re.search(r"(^|_)(tenure|months_?active|years_?active|duration_?months)(_|$)", norm):
        return BusinessRole.TENURE.value, [f"Column name '{name}' indicates customer or employee tenure duration"]
    if re.search(r"(^|_)(duration|seconds|minutes|hours|days|weeks|months|elapsed)(_|$)", norm):
        return BusinessRole.DURATION.value, [f"Column name '{name}' indicates elapsed time duration"]

    # Quantity / Count
    if re.search(r"(^|_)(quantity|qty|units|items|count|volume|headcount)(_|$)", norm):
        return BusinessRole.QUANTITY.value, [f"Column name '{name}' indicates discrete unit quantity or count"]
    if is_num and is_int and min_v >= 0 and n_unique >= 5 and role == SemanticRole.MEASURE.value:
        return BusinessRole.QUANTITY.value, ["Non-negative discrete integer distribution consistent with quantity or count"]

    # 4. Entities / Actors
    if re.search(r"(^|_)(customer|cust|client|subscriber|shopper|buyer|member)(_|$)", norm):
        return BusinessRole.CUSTOMER.value, [f"Column name '{name}' associates with customer entity"]
    if re.search(r"(^|_)(product|prod|item|sku|merchandise|goods)(_|$)", norm):
        return BusinessRole.PRODUCT.value, [f"Column name '{name}' associates with product or catalog item"]
    if re.search(r"(^|_)(order|ord|invoice|receipt)(_|$)", norm):
        return BusinessRole.ORDER.value, [f"Column name '{name}' associates with purchase order entity"]
    if re.search(r"(^|_)(transaction|txn|payment|transfer)(_|$)", norm):
        return BusinessRole.TRANSACTION.value, [f"Column name '{name}' associates with financial transaction"]
    if re.search(r"(^|_)(employee|emp|staff|worker|agent|rep)(_|$)", norm):
        return BusinessRole.EMPLOYEE.value, [f"Column name '{name}' associates with employee or personnel entity"]
    if re.search(r"(^|_)(user|usr|account|login|profile)(_|$)", norm):
        return BusinessRole.USER.value, [f"Column name '{name}' associates with user account entity"]

    # 5. Geographic
    if semantic_type == SemanticType.GEOGRAPHIC.value:
        if re.search(r"(country|nation)", norm):
            return BusinessRole.COUNTRY.value, ["Geographic boundary: country entity"]
        if re.search(r"(state|province)", norm):
            return BusinessRole.STATE.value, ["Geographic boundary: state or province"]
        if re.search(r"(city|metro|town)", norm):
            return BusinessRole.CITY.value, ["Geographic boundary: city or municipality"]
        if re.search(r"(zip|postal)", norm):
            return BusinessRole.POSTAL_CODE.value, ["Postal or ZIP routing identifier"]
        return BusinessRole.LOCATION.value, ["Geographic dimension location"]
    if re.search(r"(region|territory|zone|area|district)", norm):
        return BusinessRole.REGION.value, [f"Column name '{name}' indicates regional or geographic zone"]

    # 6. Categorical / Organizational
    if re.search(r"(contract|plan|subscription)", norm):
        return BusinessRole.CONTRACT.value, [f"Column name '{name}' indicates service agreement or contract terms"]
    if re.search(r"(payment_?method|payment_?type|card_?type)", norm):
        return BusinessRole.PAYMENT_METHOD.value, [f"Column name '{name}' indicates billing or payment method"]
    if re.search(r"(tier|level|segment|grade)", norm):
        return BusinessRole.TIER.value, [f"Column name '{name}' indicates classification tier or segment"]
    if re.search(r"(status|state|active|flag)", norm):
        return BusinessRole.STATUS.value, [f"Column name '{name}' indicates operational status or state"]
    if re.search(r"(category|type|genre|department|dept|dept_code)", norm):
        return BusinessRole.CATEGORY.value, [f"Column name '{name}' indicates business grouping or category"]

    # 7. Fallback structural roles
    if role == SemanticRole.IDENTIFIER.value or semantic_type in (SemanticType.ID.value, SemanticType.IDENTIFIER.value):
        return BusinessRole.IDENTIFIER.value, ["Row or entity identifier key"]
    if role == SemanticRole.TIMESTAMP.value or semantic_type in (SemanticType.DATETIME.value, SemanticType.DATE.value):
        return BusinessRole.TIMESTAMP.value, ["Temporal tracking attribute"]

    return BusinessRole.UNKNOWN.value, []


def detect_target_candidate(
    name: str,
    s: pd.Series,
    semantic_type: str,
    role: str,
    business_role: str,
    n_unique: int,
    non_null_count: int,
) -> tuple[bool, Optional[str], float, list[str]]:
    """Determine whether a column is a viable ML target candidate.
    
    Returns (is_target, target_type, confidence, reasons).
    """
    # Exclude IDs, dates, free text, constant columns
    if role == SemanticRole.IDENTIFIER.value or semantic_type in (
        SemanticType.ID.value,
        SemanticType.IDENTIFIER.value,
        SemanticType.TEXT.value,
        SemanticType.DATETIME.value,
        SemanticType.EMAIL.value,
        SemanticType.PHONE.value,
    ):
        return False, None, 0.0, []

    if n_unique <= 1 or non_null_count < 10:
        return False, None, 0.0, []

    norm = _normalize_name(name)

    # 1. Binary classification target
    if n_unique == 2:
        if business_role in {
            BusinessRole.CHURN.value,
            BusinessRole.CONVERSION.value,
            BusinessRole.DEFAULT.value,
            BusinessRole.FRAUD.value,
            BusinessRole.TARGET.value,
        } or re.search(r"(^|_)(churn|converted|target|label|y|outcome|flag|is_churned|success|response)(_|$)", norm):
            return True, "classification", 0.94, [
                f"Binary outcome variable indicating '{business_role or 'outcome'}' event (ideal for binary classification)"
            ]
        return True, "classification", 0.72, [
            "Binary discrete indicator suitable as classification target"
        ]

    # 2. Multi-class classification target
    if 3 <= n_unique <= 8 and role == SemanticRole.DIMENSION.value:
        if business_role in {
            BusinessRole.CHURN.value,
            BusinessRole.STATUS.value,
            BusinessRole.CATEGORY.value,
            BusinessRole.TIER.value,
            BusinessRole.TARGET.value,
        } or re.search(r"(^|_)(status|stage|rating|outcome|level|target)(_|$)", norm):
            return True, "classification", 0.68, [
                f"Discrete categorical variable with {n_unique} classes suitable for multi-class classification"
            ]

    # 3. Continuous regression target
    if role == SemanticRole.MEASURE.value and business_role in {
        BusinessRole.REVENUE.value,
        BusinessRole.SALES.value,
        BusinessRole.PROFIT.value,
        BusinessRole.COST.value,
        BusinessRole.PRICE.value,
        BusinessRole.SCORE.value,
        BusinessRole.SALARY.value,
        BusinessRole.BALANCE.value,
    }:
        return True, "regression", 0.86, [
            f"Continuous quantitative {business_role} metric suitable for regression forecasting"
        ]
    if role == SemanticRole.MEASURE.value and re.search(r"(^|_)(target|y|outcome|forecast|loss|spend|amount)(_|$)", norm):
        return True, "regression", 0.82, [
            f"Quantitative metric with outcome naming suitable for regression modeling"
        ]

    return False, None, 0.0, []


def compute_distribution_summary(s: pd.Series, semantic_type: str, role: str) -> dict[str, Any]:
    """Calculate summary statistics without exposing individual raw data samples."""
    non_null = s.dropna()
    if len(non_null) == 0:
        return {}
    if pd.api.types.is_numeric_dtype(non_null) and not pd.api.types.is_bool_dtype(non_null):
        clean = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(clean) == 0:
            return {}
        return {
            "min": float(clean.min()),
            "max": float(clean.max()),
            "mean": round(float(clean.mean()), 3),
            "std": round(float(clean.std()), 3) if len(clean) > 1 else 0.0,
            "median": round(float(clean.median()), 3),
            "q25": round(float(clean.quantile(0.25)), 3),
            "q75": round(float(clean.quantile(0.75)), 3),
        }
    elif role == SemanticRole.TIMESTAMP.value or semantic_type == SemanticType.DATETIME.value:
        str_vals = non_null.astype(str)
        return {
            "earliest": str(str_vals.min()),
            "latest": str(str_vals.max()),
        }
    else:
        top_counts = non_null.astype(str).value_counts().head(5).to_dict()
        return {
            "top_categories": {str(k): int(v) for k, v in top_counts.items()}
        }


def detect_column_quality_issues(name: str, s: pd.Series, semantic_type: str, business_role: str) -> list[str]:
    """Audit column for hygiene anomalies, near-constant distributions, or impossible values."""
    issues = []
    n_rows = len(s)
    if n_rows == 0:
        return ["empty_column"]
    null_frac = float(s.isna().mean())
    if null_frac >= 0.30:
        issues.append("high_missing_density")
    non_null = s.dropna()
    n_unique = int(non_null.nunique()) if len(non_null) else 0
    if n_unique <= 1:
        issues.append("constant_or_empty")

    # Near-constant column check
    if len(non_null) >= 20 and n_unique > 1:
        top_freq = int(non_null.value_counts().iloc[0])
        if top_freq / len(non_null) >= 0.95:
            issues.append("near_constant")

    # Inconsistent categorical casing or leading/trailing whitespace
    if not pd.api.types.is_numeric_dtype(s) and len(non_null) > 0:
        raw_strings = non_null.astype(str)
        cleaned_strings = raw_strings.str.strip().str.lower()
        if cleaned_strings.nunique() < raw_strings.nunique():
            issues.append("inconsistent_categorical_casing_or_whitespace")

    # Impossible numeric values
    if pd.api.types.is_numeric_dtype(s) and len(non_null) > 0:
        num = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(num) > 0:
            if business_role in {BusinessRole.AGE.value} and (num < 0).any():
                issues.append("impossible_negative_age")
            if business_role in {BusinessRole.REVENUE.value, BusinessRole.SALES.value, BusinessRole.PRICE.value, BusinessRole.QUANTITY.value, BusinessRole.COST.value} and (num < 0).any():
                issues.append("impossible_negative_values")
            if business_role in {BusinessRole.PERCENTAGE.value} and ((num < 0).any() or (num > 100).any()):
                issues.append("out_of_bounds_percentage")

    return issues


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
    """Infer semantic type, role, business role, confidence, and explainable evidence for a column.

    If a user override is provided, user corrections take absolute precedence
    and are marked with `is_user_corrected = True`.
    """
    # 1. User corrections take absolute precedence
    if user_override and isinstance(user_override, dict):
        sem_type = user_override.get("semantic_type")
        role = user_override.get("role")
        brole = user_override.get("business_role")
        if sem_type or role or brole:
            resolved_type = str(sem_type or SemanticType.UNKNOWN.value)
            resolved_role = str(role or SemanticRole.DIMENSION.value)
            resolved_brole = str(brole or BusinessRole.UNKNOWN.value)
            notes = user_override.get("notes") or "User confirmed/corrected column meaning"
            is_id = (
                resolved_role == SemanticRole.IDENTIFIER.value
                or resolved_type in {SemanticType.ID.value, SemanticType.IDENTIFIER.value, SemanticType.ENTITY_ID.value}
            )
            return SemanticAnnotation(
                semantic_type=resolved_type,
                role=resolved_role,
                business_role=resolved_brole,
                confidence=1.0,
                confidence_type="user_assigned",
                evidence=[f"User override: {notes}"],
                is_user_corrected=True,
                is_identifier=is_id,
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
            business_role=BusinessRole.UNKNOWN.value,
            confidence=0.10,
            evidence=[
                "All values null or constant single value — zero variance",
                "Ambiguous/uninformative for semantic analysis",
            ],
            is_user_corrected=False,
            quality_issues=["constant_or_empty"],
        )

    # 4. Try high-precision detectors in priority order
    # (a) Datetime / Timestamps
    dt_res = detect_datetime(name, series, non_null_count)
    if dt_res and dt_res.confidence >= 0.85:
        res = dt_res

    # (b) Email addresses
    elif (email_res := detect_email(name, series, non_null_count)) and email_res.confidence >= 0.80:
        res = email_res

    # (c) Phone numbers
    elif (phone_res := detect_phone(name, series, non_null_count)) and phone_res.confidence >= 0.80:
        res = phone_res

    # (d) Currency & Monetary Values
    elif (curr_res := detect_currency(name, series, non_null_count)) and curr_res.confidence >= 0.80:
        res = curr_res

    # (e) Percentages
    elif (pct_res := detect_percentage(name, series, non_null_count)) and pct_res.confidence >= 0.80:
        res = pct_res

    # (f) Booleans / Binary Flags
    elif (bool_res := detect_boolean(name, series, n_unique, non_null_count)) and bool_res.confidence >= 0.80:
        res = bool_res

    # (e) Geographic Dimensions
    elif (geo_res := detect_geographic(name, series, n_unique, non_null_count)) and geo_res.confidence >= 0.80:
        res = geo_res

    # (f) Primary & Surrogate Identifiers
    elif (id_res := detect_identifier(name, series, total_rows, n_unique, non_null_count)) and id_res.confidence >= 0.80:
        res = id_res

    # (g) Free-form Unstructured Text
    elif (text_res := detect_text(name, series, n_unique, non_null_count)) and text_res.confidence >= 0.80:
        res = text_res

    # (h) Ordinal Dimensions
    elif (ord_res := detect_ordinal(name, series, n_unique, non_null_count)) and ord_res.confidence >= 0.85:
        res = ord_res

    # (i) Quantitative Measures
    elif (measure_res := detect_measure(name, series, n_unique, non_null_count)) and measure_res.confidence >= 0.80:
        res = measure_res

    # (j) Categorical Dimensions
    elif (cat_res := detect_categorical(name, series, n_unique, non_null_count)):
        res = cat_res

    # (k) Fallbacks: lower-confidence date or measure
    elif dt_res:
        res = dt_res
    elif measure_res:
        res = measure_res
    else:
        # Truly ambiguous / unknown
        res = SemanticAnnotation(
            semantic_type=SemanticType.UNKNOWN.value,
            role=SemanticRole.UNKNOWN.value,
            business_role=BusinessRole.UNKNOWN.value,
            confidence=0.30,
            evidence=[
                "No dominant date, geographic, identifier, or numeric pattern recognized",
                f"{n_unique} unique values among {non_null_count} rows ({n_unique / max(non_null_count, 1):.1%} uniqueness)",
                "Ambiguous distribution kept unknown for user clarification",
            ],
            is_user_corrected=False,
        )

    # 5. Enrich with business role inference
    if res.business_role == BusinessRole.UNKNOWN.value or not res.business_role:
        b_role, b_evidence = detect_business_role(
            name, series, res.semantic_type, res.role, n_unique, non_null_count
        )
        res.business_role = b_role
        for bev in b_evidence:
            if bev not in res.evidence:
                res.evidence.append(bev)

    # 6. Flag whether column is an identifier
    res.is_identifier = (
        res.role == SemanticRole.IDENTIFIER.value
        or res.semantic_type in {SemanticType.ID.value, SemanticType.IDENTIFIER.value, SemanticType.ENTITY_ID.value}
    )

    # 7. Evaluate target suitability
    is_tgt, tgt_type, tgt_conf, tgt_reasons = detect_target_candidate(
        name, series, res.semantic_type, res.role, res.business_role, n_unique, non_null_count
    )
    res.is_target_candidate = is_tgt
    res.target_type = tgt_type
    res.target_confidence = tgt_conf

    # 8. Compute distribution summary and audit quality issues
    res.distribution_summary = compute_distribution_summary(series, res.semantic_type, res.role)
    res.quality_issues = detect_column_quality_issues(name, series, res.semantic_type, res.business_role)

    return res
