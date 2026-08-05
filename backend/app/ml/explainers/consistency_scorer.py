"""Consistency scorer for comparing SHAP and LIME explanations."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


class ConsistencyScorer:
    """
    Compare SHAP vs LIME local explanations.

    High consistency → higher trust in the "why".
    Low consistency → surface a human-review flag (explanation may disagree).
    """

    def __init__(self, consistency_threshold: float = 0.7):
        self.consistency_threshold = consistency_threshold

    def calculate_consistency(
        self,
        shap_importance: dict[str, float],
        lime_importance: dict[str, float],
    ) -> dict[str, Any]:
        """
        Args:
            shap_importance / lime_importance: feature -> *signed* contribution
                (preferred) or absolute importance.
        """
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())
        if not common_features:
            # Try soft match on overlapping keys after stripping
            if not shap_importance or not lime_importance:
                return {
                    "consistency_score": 0.0,
                    "trust_level": "low",
                    "metrics": {},
                    "warning": "Missing SHAP or LIME contributions",
                    "plain": "We could not cross-check explanation methods for this case.",
                }
            # Union rank overlap only
            return self._rank_only_fallback(shap_importance, lime_importance)

        features = sorted(common_features)
        shap_signed = np.array([float(shap_importance[f]) for f in features])
        lime_signed = np.array([float(lime_importance[f]) for f in features])
        shap_abs = np.abs(shap_signed)
        lime_abs = np.abs(lime_signed)

        shap_norm = shap_abs / (np.linalg.norm(shap_abs) + 1e-10)
        lime_norm = lime_abs / (np.linalg.norm(lime_abs) + 1e-10)

        cosine_sim = float(
            cosine_similarity(shap_norm.reshape(1, -1), lime_norm.reshape(1, -1))[0, 0]
        )

        if len(features) > 2:
            rank_corr, p_value = spearmanr(shap_abs, lime_abs)
            rank_corr = float(rank_corr) if not np.isnan(rank_corr) else 0.0
        else:
            rank_corr = cosine_sim
            p_value = None

        k = min(5, len(features))
        shap_top_k = set(sorted(features, key=lambda f: abs(shap_importance[f]), reverse=True)[:k])
        lime_top_k = set(sorted(features, key=lambda f: abs(lime_importance[f]), reverse=True)[:k])
        top_k_overlap = len(shap_top_k & lime_top_k) / k

        direction_agreement = self._direction_agreement(shap_importance, lime_importance, features)

        consistency_score = (
            0.30 * max(0.0, cosine_sim)
            + 0.30 * max(0.0, (rank_corr + 1) / 2)
            + 0.20 * top_k_overlap
            + 0.20 * direction_agreement
        )

        trust_level = self.get_trust_level(consistency_score)
        plain = self._plain_language(consistency_score, trust_level, top_k_overlap)

        return {
            "consistency_score": round(consistency_score, 4),
            "trust_level": trust_level,
            "metrics": {
                "cosine_similarity": round(cosine_sim, 4),
                "rank_correlation": round(rank_corr, 4),
                "rank_correlation_pvalue": round(p_value, 4) if p_value is not None else None,
                "top_k_overlap": round(top_k_overlap, 4),
                "top_k": k,
                "direction_agreement": round(direction_agreement, 4),
            },
            "top_k_features": {
                "shap": list(shap_top_k),
                "lime": list(lime_top_k),
                "common": list(shap_top_k & lime_top_k),
            },
            "feature_count": len(features),
            "plain": plain,
            "should_flag": consistency_score < self.consistency_threshold,
        }

    def _rank_only_fallback(
        self, shap_importance: dict[str, float], lime_importance: dict[str, float]
    ) -> dict[str, Any]:
        k = 5
        shap_top = [f for f, _ in sorted(shap_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:k]]
        lime_top = [f for f, _ in sorted(lime_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:k]]
        overlap = len(set(shap_top) & set(lime_top)) / max(min(k, len(shap_top), len(lime_top)), 1)
        score = float(overlap)
        trust = self.get_trust_level(score)
        return {
            "consistency_score": round(score, 4),
            "trust_level": trust,
            "metrics": {"top_k_overlap": round(overlap, 4), "top_k": k},
            "top_k_features": {"shap": shap_top, "lime": lime_top, "common": list(set(shap_top) & set(lime_top))},
            "feature_count": 0,
            "plain": self._plain_language(score, trust, overlap),
            "should_flag": score < self.consistency_threshold,
            "warning": "Feature names did not fully align; used rank overlap only",
        }

    def _direction_agreement(
        self,
        shap_importance: dict[str, float],
        lime_importance: dict[str, float],
        features: list[str],
    ) -> float:
        agreements = 0
        total = 0
        for f in features:
            s = float(shap_importance[f])
            l = float(lime_importance[f])
            if abs(s) < 1e-12 and abs(l) < 1e-12:
                agreements += 1
            elif s * l > 0:
                agreements += 1
            total += 1
        return agreements / total if total else 0.0

    def get_trust_level(self, consistency_score: float) -> str:
        if consistency_score >= 0.8:
            return "high"
        if consistency_score >= self.consistency_threshold:
            return "medium"
        return "low"

    def _plain_language(self, score: float, trust: str, top_overlap: float) -> str:
        if trust == "high":
            return (
                "Two independent explanation methods largely agree on what is driving this case — "
                f"stronger trust in the “why” (score {score:.0%})."
            )
        if trust == "medium":
            return (
                "Explanation methods partially agree. Treat the main drivers as a useful guide, "
                f"but leave room for judgment (score {score:.0%}; top-factor overlap {top_overlap:.0%})."
            )
        return (
            "Explanation methods disagree more than usual on what matters here. "
            f"Confirm drivers with domain knowledge before costly action (score {score:.0%})."
        )

    def flag_low_consistency(
        self,
        shap_importance: dict[str, float],
        lime_importance: dict[str, float],
    ) -> dict[str, Any]:
        result = self.calculate_consistency(shap_importance, lime_importance)
        should_flag = result["consistency_score"] < self.consistency_threshold
        disagreements = []
        if result.get("trust_level") == "low":
            common = set(shap_importance) & set(lime_importance)
            shap_ranked = sorted(common, key=lambda f: abs(shap_importance[f]), reverse=True)
            lime_ranked = sorted(common, key=lambda f: abs(lime_importance[f]), reverse=True)
            for i, (sf, lf) in enumerate(zip(shap_ranked[:5], lime_ranked[:5])):
                if sf != lf:
                    disagreements.append({"rank": i + 1, "shap_feature": sf, "lime_feature": lf})
        return {
            "should_flag": should_flag,
            "consistency_score": result["consistency_score"],
            "trust_level": result["trust_level"],
            "disagreements": disagreements,
            "recommendation": result.get("plain")
            or (
                "Review explanation manually"
                if should_flag
                else "Explanation is reliable"
            ),
        }
