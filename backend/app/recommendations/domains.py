"""Domain detection for recommendation catalogs.

Telco churn stays the default catalog. HR attrition (and future domains)
select a different action set from project metadata / feature columns —
never a global override of Telco behavior.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

DOMAIN_TELCO = "telco"
DOMAIN_HR_ATTRITION = "hr_attrition"
DOMAIN_GENERIC = "generic"

DEFAULT_DOMAIN = DOMAIN_TELCO

_HR_FEATURE_MARKERS = {
    "overtime",
    "jobsatisfaction",
    "environmentsatisfaction",
    "worklifebalance",
    "yearsatcompany",
    "yearssincelastpromotion",
    "yearswithcurrmanager",
    "distancefromhome",
    "businesstravel",
    "monthlyincome",
    "numcompaniesworked",
    "stockoptionlevel",
    "trainingtimeslastyear",
    "jobinvolvement",
    "relationshipsatisfaction",
    "totalworkingyears",
    "percentsalaryhike",
}

_TELCO_FEATURE_MARKERS = {
    "monthlycharges",
    "totalcharges",
    "contract",
    "techsupport",
    "onlinesecurity",
    "internetservice",
    "streamingtv",
    "paymentmethod",
    "tenure",
    "phoneservice",
}


def _norm_token(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    s = s.lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "", s).strip("_")  # collapse for marker match


def detect_domain(
    *,
    feature_columns: Optional[Iterable[str]] = None,
    features: Optional[dict[str, Any]] = None,
    project_name: Optional[str] = None,
    target_column: Optional[str] = None,
    target_description: Optional[str] = None,
) -> str:
    """
    Infer recommendation domain.

    Preference order: strong name/target hints, then feature-set markers.
    Defaults to telco when ambiguous (preserves existing behavior).
    """
    text = " ".join(
        str(x or "").lower()
        for x in (project_name, target_column, target_description)
    )
    if any(
        tok in text
        for tok in (
            "attrition",
            "employee",
            "workforce",
            "hr ",
            " hr",
            "turnover",
            "resign",
        )
    ):
        return DOMAIN_HR_ATTRITION
    if any(tok in text for tok in ("churn", "telco", "telecom", "subscriber")):
        return DOMAIN_TELCO

    cols = list(feature_columns or [])
    if not cols and features:
        cols = list(features.keys())
    norms = {_norm_token(c) for c in cols}

    hr_hits = sum(1 for m in _HR_FEATURE_MARKERS if m in norms or any(m in n for n in norms))
    telco_hits = sum(
        1 for m in _TELCO_FEATURE_MARKERS if m in norms or any(m in n for n in norms)
    )

    if hr_hits >= 3 and hr_hits > telco_hits:
        return DOMAIN_HR_ATTRITION
    if telco_hits >= 2:
        return DOMAIN_TELCO
    if hr_hits >= 2:
        return DOMAIN_HR_ATTRITION
    return DEFAULT_DOMAIN
