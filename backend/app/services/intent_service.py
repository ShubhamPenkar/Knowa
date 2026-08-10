"""B1 Intent / smart onboarding — plain-language problem → draft project config.

Rules-first hybrid (no LLM required). Suggests only; human confirms via create.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.ml.dataset_profiler import (
    is_likely_id_column,
    name_suggests_id,
    suggest_positive_label,
)
from app.services.dataset_service import DatasetService


def _norm(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = s.replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", s).strip("_")


# Outcome concepts: keywords in user text → business label + column name hints
_OUTCOME_PATTERNS: list[dict[str, Any]] = [
    {
        "label": "churn",
        "problem_type": "binary_classification",
        "text": (
            "churn",
            "churned",
            "retention",
            "retain",
            "cancel",
            "cancellation",
            "unsubscribe",
            "leave the service",
            "telecom",
            "telco",
            "subscriber",
        ),
        "columns": ("churn", "churned", "exited", "exit", "cancelled", "canceled"),
    },
    {
        "label": "attrition",
        "problem_type": "binary_classification",
        "text": (
            "attrition",
            "attrit",
            "turnover",
            "resign",
            "resignation",
            "quit",
            "employee leave",
            "leave the company",
            "hr ",
            " workforce",
            "staff retention",
        ),
        "columns": ("attrition", "left", "terminated", "termination", "resign"),
    },
    {
        "label": "conversion",
        "problem_type": "binary_classification",
        "text": (
            "conversion",
            "convert",
            "purchase",
            "buy",
            "signup",
            "sign up",
            "subscribe",
            "lead",
        ),
        "columns": ("converted", "conversion", "purchased", "bought", "subscribed"),
    },
    {
        "label": "default",
        "problem_type": "binary_classification",
        "text": ("default", "delinquency", "credit risk", "loan default"),
        "columns": ("default", "defaulted", "delinquent"),
    },
    {
        "label": "fraud",
        "problem_type": "binary_classification",
        "text": ("fraud", "fraudulent", "chargeback"),
        "columns": ("fraud", "is_fraud", "fraudulent"),
    },
    {
        "label": "spend",
        "problem_type": "regression",
        "text": (
            "how much",
            "spend",
            "revenue",
            "amount",
            "ltv",
            "lifetime value",
            "sales amount",
            "predict the value",
        ),
        "columns": (
            "amount",
            "spend",
            "revenue",
            "sales",
            "ltv",
            "lifetimevalue",
            "monthlycharges",
            "totalcharges",
        ),
    },
]

_BINARY_HINTS = (
    "will they",
    "will it",
    "yes/no",
    "yes or no",
    "risk of",
    "likelihood",
    "probability",
    "classify",
    "predict whether",
)
_REGRESSION_HINTS = (
    "how much",
    "how many",
    "estimate",
    "forecast amount",
    "predict the number",
    "numeric",
)


class IntentService:
    def __init__(self, db: Session, org_id: str):
        self.db = db
        self.org_id = org_id
        self.datasets = DatasetService(db, org_id)

    def suggest_config(
        self,
        *,
        dataset_id: str,
        problem_description: str,
        project_name: Optional[str] = None,
    ) -> dict[str, Any]:
        dataset = self.datasets.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")

        problem_description = (problem_description or "").strip()
        if not problem_description:
            raise ValueError("Describe the decision problem in plain language.")

        text = " ".join(
            x for x in (problem_description, project_name or "", dataset.name or "") if x
        ).strip()

        df = self.datasets.load_dataframe(dataset_id)
        columns = list(df.columns)

        outcome = self._match_outcome(text)
        problem_type = self._infer_problem_type(text, outcome)
        target_description = (outcome or {}).get("label") or self._fallback_label(text)

        target_column, target_score, target_rationale = self._pick_target(
            df,
            columns,
            text=text,
            outcome=outcome,
            problem_type=problem_type,
        )
        if not target_column:
            raise ValueError(
                "Could not suggest a target column from this description. "
                "Pick the target manually below."
            )

        positive_label = None
        present_labels: list[str] = []
        if problem_type == "binary_classification":
            present_labels = (
                df[target_column].dropna().astype(str).value_counts().head(20).index.tolist()
            )
            positive_label = suggest_positive_label(present_labels) or (
                present_labels[0] if present_labels else "1"
            )

        feature_columns = self._pick_features(df, columns, target_column, problem_type)

        confidence = self._confidence(
            target_score=target_score,
            outcome=outcome,
            n_features=len(feature_columns),
            problem_type=problem_type,
        )

        rationale_bits = [
            target_rationale,
            f"Framed as a {'yes/no' if problem_type == 'binary_classification' else 'numeric'} question.",
            f"Business label set to “{target_description}” for insights and priorities.",
        ]
        if positive_label is not None:
            rationale_bits.append(f"Positive label suggested as “{positive_label}”.")
        if outcome:
            rationale_bits.append(
                f"Matched intent keywords for {outcome['label']}."
            )

        suggested_name = self._suggest_name(
            project_name, dataset.name, target_description, problem_type
        )

        return {
            "layer": "B1_intent_onboarding",
            "source": "rules",
            "dataset_id": dataset_id,
            "problem_description": problem_description,
            "suggested_name": suggested_name,
            "problem_type": problem_type,
            "target_column": target_column,
            "target_positive_label": positive_label,
            "target_description": target_description,
            "feature_columns": feature_columns,
            "present_target_labels": present_labels,
            "confidence": confidence,
            "rationale": " ".join(rationale_bits),
            "alternatives": self._alt_targets(
                df, columns, target_column, outcome, problem_type
            ),
        }

    def _match_outcome(self, text: str) -> Optional[dict[str, Any]]:
        low = text.lower()
        best = None
        best_hits = 0
        for pattern in _OUTCOME_PATTERNS:
            hits = sum(1 for kw in pattern["text"] if kw in low)
            if hits > best_hits:
                best_hits = hits
                best = pattern
        return best if best_hits else None

    def _infer_problem_type(
        self, text: str, outcome: Optional[dict[str, Any]]
    ) -> str:
        low = text.lower()
        if outcome and outcome.get("problem_type"):
            # Explicit regression/binary cues can override outcome default
            reg_hits = sum(1 for kw in _REGRESSION_HINTS if kw in low)
            bin_hits = sum(1 for kw in _BINARY_HINTS if kw in low)
            if reg_hits > bin_hits and outcome["problem_type"] != "regression":
                if any(k in low for k in ("how much", "amount", "spend", "revenue")):
                    return "regression"
            return str(outcome["problem_type"])
        if any(kw in low for kw in _REGRESSION_HINTS) and not any(
            kw in low for kw in ("churn", "attrition", "fraud", "default")
        ):
            return "regression"
        return "binary_classification"

    def _fallback_label(self, text: str) -> str:
        low = text.lower()
        for word in ("churn", "attrition", "conversion", "default", "fraud", "risk"):
            if word in low:
                return word
        return "outcome"

    def _column_score(
        self,
        col: str,
        *,
        text: str,
        outcome: Optional[dict[str, Any]],
        problem_type: str,
        series: pd.Series,
    ) -> float:
        n = _norm(col)
        score = 0.0
        text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))

        # Exact / substring match to outcome column hints
        if outcome:
            for hint in outcome.get("columns") or ():
                h = _norm(hint)
                if n == h:
                    score += 8.0
                elif h in n or n in h:
                    score += 5.0

        # Token overlap with free text
        col_tokens = set(n.split("_"))
        overlap = text_tokens & col_tokens
        score += 1.5 * len(overlap)

        # Prefer classic target names
        if n in {"churn", "attrition", "target", "label", "y", "outcome", "class"}:
            score += 3.0
        if n.endswith("_flag") or n.startswith("is_"):
            score += 1.0

        # Type fitness
        nunique = int(series.dropna().nunique())
        if problem_type == "binary_classification":
            if 2 <= nunique <= 6:
                score += 3.0
            elif nunique > 20:
                score -= 4.0
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().mean() >= 0.9 and nunique > 10:
                score += 3.0
            elif 2 <= nunique <= 5:
                score -= 2.0

        # Penalize ID-like
        try:
            if is_likely_id_column(col, series):
                score -= 10.0
        except Exception:
            pass

        return score

    def _pick_target(
        self,
        df: pd.DataFrame,
        columns: list[str],
        *,
        text: str,
        outcome: Optional[dict[str, Any]],
        problem_type: str,
    ) -> tuple[Optional[str], float, str]:
        ranked: list[tuple[float, str]] = []
        for col in columns:
            s = df[col]
            score = self._column_score(
                col,
                text=text,
                outcome=outcome,
                problem_type=problem_type,
                series=s,
            )
            ranked.append((score, col))
        ranked.sort(reverse=True)
        if not ranked or ranked[0][0] < 1.5:
            # Weak fall back: best binary-looking column
            for score, col in ranked:
                nunique = int(df[col].dropna().nunique())
                if problem_type == "binary_classification" and 2 <= nunique <= 4:
                    return (
                        col,
                        score,
                        f"Selected “{col}” as the most label-like column.",
                    )
            if ranked:
                col = ranked[0][1]
                return col, ranked[0][0], f"Best available guess for target is “{col}”."
            return None, 0.0, "No columns available."

        col = ranked[0][1]
        score = ranked[0][0]
        if outcome and any(_norm(h) in _norm(col) for h in outcome.get("columns") or ()):
            why = f"Matched “{col}” to your {outcome['label']} intent."
        else:
            why = f"Selected “{col}” from column names and value shape."
        return col, score, why

    def _pick_features(
        self,
        df: pd.DataFrame,
        columns: list[str],
        target_column: str,
        problem_type: str,
    ) -> list[str]:
        """Suggest features; drop name-ID / constant columns only.

        Avoid sequential-integer ID heuristics here — tenure-like columns are
        valid levers and get full profiler treatment at project create.
        """
        del problem_type
        cleaned = []
        for c in columns:
            if c == target_column:
                continue
            s = df[c]
            if name_suggests_id(c):
                continue
            if s.dropna().nunique() <= 1:
                continue
            cleaned.append(c)
        return cleaned

    def _alt_targets(
        self,
        df: pd.DataFrame,
        columns: list[str],
        chosen: str,
        outcome: Optional[dict[str, Any]],
        problem_type: str,
    ) -> list[dict[str, Any]]:
        alts = []
        for col in columns:
            if col == chosen:
                continue
            score = self._column_score(
                col,
                text="",
                outcome=outcome,
                problem_type=problem_type,
                series=df[col],
            )
            if score >= 3.0:
                alts.append({"column": col, "score": round(score, 2)})
        alts.sort(key=lambda x: -x["score"])
        return alts[:3]

    def _confidence(
        self,
        *,
        target_score: float,
        outcome: Optional[dict[str, Any]],
        n_features: int,
        problem_type: str,
    ) -> float:
        c = 0.35
        if outcome:
            c += 0.25
        c += min(0.3, max(0.0, target_score) / 20.0)
        if n_features >= 3:
            c += 0.1
        if problem_type in ("binary_classification", "regression"):
            c += 0.05
        return round(min(0.95, c), 2)

    def _suggest_name(
        self,
        project_name: Optional[str],
        dataset_name: Optional[str],
        target_description: str,
        problem_type: str,
    ) -> Optional[str]:
        if project_name and project_name.strip():
            return project_name.strip()
        label = (target_description or "outcome").replace("_", " ").strip().title()
        kind = "Risk" if problem_type == "binary_classification" else "Estimate"
        base = (dataset_name or "Project").strip()
        # Keep short
        name = f"{base} — {label} {kind}"
        return name[:100]
