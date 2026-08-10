"""Impact calculator for estimating action effects on outcome probability."""

from __future__ import annotations

from typing import Any, Optional

from app.recommendations.action_catalog import Action, _norm

# Only reshape catalog impact once we have a small sample of real outcomes
MIN_N_FOR_LEARNING = 3
# Cap how far history can pull catalog impact (0 = ignore history, 1 = history only)
MAX_HISTORY_BLEND = 0.4


class ImpactCalculator:
    """
    Expected impact of an action on positive-outcome probability.

    Uses base impact potential × driver relevance × risk band.
    Optional historical effectiveness (A7/B3 outcome log) blends in when
    enough cases are logged for that action.
    """

    def __init__(self, effectiveness_data: Optional[dict[str, dict]] = None):
        self.effectiveness_data = effectiveness_data or {}

    def calculate_impact(
        self,
        action: Action,
        features: dict[str, Any],
        probability: float,
        feature_importance: Optional[dict[str, float]] = None,
        *,
        soft_case: bool = False,
    ) -> dict[str, Any]:
        catalog_impact = float(action.impact_potential)
        base_impact = catalog_impact
        p = float(probability)

        learning_meta = self._learning_meta(action.code)
        if learning_meta.get("applied"):
            rate = float(learning_meta["effectiveness_rate"])
            n = int(learning_meta["n_outcomes"])
            # More evidence → slightly stronger pull, capped
            lam = min(MAX_HISTORY_BLEND, 0.12 * n)
            # Map success rate into an impact-like score (avoided bad event)
            historical_impact = max(0.05, min(0.95, rate))
            base_impact = (1.0 - lam) * catalog_impact + lam * historical_impact

        relevance_multiplier = self._relevance_multiplier(
            action, feature_importance or {}
        )
        risk_multiplier = self._risk_multiplier(p)

        # Soft / low-trust cases: de-scale aggressive interventions slightly
        uncertainty_factor = 0.85 if soft_case and action.category == "save" else 1.0
        # Monitor action impact is intentionally tiny
        if action.code in ("monitor_only", "monitor_attrition"):
            impact_score = 0.08
            expected_reduction = 0.0
            new_p = p
        else:
            impact_score = min(
                1.0, base_impact * relevance_multiplier * risk_multiplier * uncertainty_factor
            )
            # Cap expected reduction so we never claim magic
            max_possible_reduction = min(p * 0.55, 0.35)
            expected_reduction = max_possible_reduction * impact_score
            new_p = max(0.02, p - expected_reduction)

        return {
            "impact_score": round(impact_score, 4),
            "expected_probability_reduction": round(expected_reduction, 4),
            "new_probability_estimate": round(new_p, 4),
            "probability_reduction_percent": round(
                (expected_reduction / p) * 100, 1
            )
            if p > 1e-6
            else 0.0,
            "learning": learning_meta,
            "components": {
                "base_impact": round(base_impact, 4),
                "catalog_impact": round(catalog_impact, 4),
                "relevance_multiplier": round(relevance_multiplier, 4),
                "risk_multiplier": round(risk_multiplier, 4),
                "uncertainty_factor": uncertainty_factor,
                "learning_applied": bool(learning_meta.get("applied")),
            },
        }

    def _learning_meta(self, action_code: str) -> dict[str, Any]:
        hist = self.effectiveness_data.get(action_code) or {}
        n = int(hist.get("n") or hist.get("n_outcomes") or 0)
        rate = hist.get("effectiveness_rate", hist.get("success_rate"))
        # Derive rate from counts when only success_n/n are present
        if rate is None and n > 0 and hist.get("success_n") is not None:
            rate = float(hist["success_n"]) / float(n)
        if rate is None:
            return {
                "applied": False,
                "n_outcomes": n,
                "effectiveness_rate": None,
                "learning_note": "No logged outcomes for this action yet.",
            }
        rate_f = float(rate)
        if hist.get("success_n") is not None:
            success_n = int(hist["success_n"])
        else:
            success_n = int(round(rate_f * n))
        reliable = bool(hist.get("reliable", n >= MIN_N_FOR_LEARNING))
        if n < MIN_N_FOR_LEARNING or not reliable:
            return {
                "applied": False,
                "n_outcomes": n,
                "effectiveness_rate": round(rate_f, 4),
                "success_n": success_n,
                "learning_note": (
                    f"Logged {success_n}/{n} favorable so far — need {MIN_N_FOR_LEARNING}+ "
                    "before rankings shift."
                ),
            }
        if rate_f >= 0.65:
            note = (
                f"Favorable in {success_n}/{n} logged cases — ranking slightly boosted."
            )
        elif rate_f <= 0.35:
            note = (
                f"Only {success_n}/{n} logged cases went well — ranking tempered."
            )
        else:
            note = (
                f"Mixed results ({success_n}/{n} favorable) — mild ranking adjustment."
            )
        return {
            "applied": True,
            "n_outcomes": n,
            "effectiveness_rate": round(rate_f, 4),
            "success_n": success_n,
            "learning_note": note,
        }

    def _relevance_multiplier(
        self, action: Action, feature_importance: dict[str, float]
    ) -> float:
        if not feature_importance or not action.target_features:
            return 1.0

        # Only count *risk-raising* magnitude for addressable levers
        positives = {
            k: v for k, v in feature_importance.items() if float(v) > 0
        }
        total = sum(abs(v) for v in positives.values()) or sum(
            abs(v) for v in feature_importance.values()
        ) or 1.0

        target_imp = 0.0
        for t in action.target_features:
            tn = _norm(t)
            for k, v in feature_importance.items():
                kn = _norm(k)
                if kn == tn or tn in kn or kn in tn:
                    target_imp += abs(float(v))
                    break

        ratio = min(1.0, target_imp / total)
        # 0.55 … 1.45
        return 0.55 + 0.9 * ratio

    def _risk_multiplier(self, probability: float) -> float:
        if probability >= 0.85:
            return 0.75
        if probability >= 0.65:
            return 0.95
        if probability >= 0.4:
            return 1.05
        if probability >= 0.2:
            return 0.95
        return 0.7  # already calm — little "save" room
