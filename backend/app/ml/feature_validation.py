"""Strict feature payload validation before scoring.

Reject incomplete inputs instead of imputing zeros and returning a
confident-looking (often calm) score.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd


class FeatureValidationError(ValueError):
    """Raised when required scoring features are missing or empty."""

    def __init__(
        self,
        missing_features: Optional[list[str]] = None,
        empty_features: Optional[list[str]] = None,
        message: Optional[str] = None,
    ):
        self.missing_features = list(missing_features or [])
        self.empty_features = list(empty_features or [])
        if message:
            super().__init__(message)
        else:
            parts: list[str] = []
            if self.missing_features:
                parts.append(
                    "missing: " + ", ".join(self.missing_features[:20])
                    + ("…" if len(self.missing_features) > 20 else "")
                )
            if self.empty_features:
                parts.append(
                    "empty/null: " + ", ".join(self.empty_features[:20])
                    + ("…" if len(self.empty_features) > 20 else "")
                )
            super().__init__(
                "Required feature columns are incomplete ("
                + "; ".join(parts)
                + "). Provide all required features with non-empty values before scoring."
            )

    def as_detail(self) -> dict[str, Any]:
        return {
            "code": "missing_required_features",
            "message": str(self),
            "missing_features": self.missing_features,
            "empty_features": self.empty_features,
        }


def is_empty_feature_value(value: Any) -> bool:
    """True if a feature value cannot be used for scoring."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        # np.nan / pd.NA / NaT
        if pd.isna(value):
            return True
    except (ValueError, TypeError):
        pass
    return False


def validate_required_features(
    features: Optional[dict[str, Any]],
    required_columns: Sequence[str],
) -> None:
    """
    Ensure every required column is present and non-empty.

    Raises FeatureValidationError with explicit missing/empty lists.
    """
    required = [str(c) for c in (required_columns or []) if c is not None and str(c).strip()]
    if not required:
        raise FeatureValidationError(
            message="No required feature columns configured for this project/model."
        )

    feats = features if isinstance(features, dict) else {}
    missing: list[str] = []
    empty: list[str] = []
    for col in required:
        if col not in feats:
            missing.append(col)
        elif is_empty_feature_value(feats[col]):
            empty.append(col)

    if missing or empty:
        raise FeatureValidationError(missing_features=missing, empty_features=empty)


# Common identity column names across datasets (Telco, SaaS, etc.)
ENTITY_ID_CANDIDATES = (
    "customerID",
    "CustomerID",
    "customer_id",
    "customerId",
    "entity_id",
    "EntityId",
    "entityId",
    "account_id",
    "AccountID",
    "accountId",
    "user_id",
    "UserID",
    "userId",
    "EmployeeNumber",
    "employee_number",
    "employeeNumber",
    "EmployeeID",
    "employee_id",
    "id",
    "ID",
)


def resolve_entity_id(
    row: dict[str, Any],
    *,
    preferred_keys: Optional[Sequence[str]] = None,
    row_index: Optional[int] = None,
    fallback_prefix: str = "row",
) -> Optional[str]:
    """
    Pick a stable customer/row identifier from a raw data row.

    Prefer known ID columns (including non-feature ID cols present in the
    original CSV). Fall back to row-{index} when callers pass an index.
    """
    if not isinstance(row, dict):
        return f"{fallback_prefix}-{row_index}" if row_index is not None else None

    keys: list[str] = []
    if preferred_keys:
        keys.extend(str(k) for k in preferred_keys)
    keys.extend(ENTITY_ID_CANDIDATES)

    # exact
    for key in keys:
        if key in row and not is_empty_feature_value(row[key]):
            return str(row[key]).strip()

    # case-insensitive
    lower_map = {str(k).lower(): k for k in row.keys()}
    for key in keys:
        raw_key = lower_map.get(str(key).lower())
        if raw_key is None:
            continue
        if not is_empty_feature_value(row[raw_key]):
            return str(row[raw_key]).strip()

    if row_index is not None:
        return f"{fallback_prefix}-{row_index}"
    return None
