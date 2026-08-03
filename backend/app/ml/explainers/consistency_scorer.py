"""Consistency scorer for comparing SHAP and LIME explanations."""

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


class ConsistencyScorer:
    """
    Compares SHAP and LIME explanations to assess explanation reliability.
    
    High consistency = both methods agree = higher trust in explanation
    Low consistency = methods disagree = flag for human review
    """
    
    def __init__(self, consistency_threshold: float = 0.7):
        """
        Initialize consistency scorer.
        
        Args:
            consistency_threshold: Score below this = low trust
        """
        self.consistency_threshold = consistency_threshold
    
    def calculate_consistency(
        self,
        shap_importance: dict[str, float],
        lime_importance: dict[str, float]
    ) -> dict[str, Any]:
        """
        Calculate consistency between SHAP and LIME explanations.
        
        Uses multiple metrics:
        - Rank correlation (Spearman)
        - Cosine similarity of importance vectors
        - Top-k feature overlap
        
        Args:
            shap_importance: Feature -> SHAP importance
            lime_importance: Feature -> LIME importance
            
        Returns:
            Consistency metrics and overall score
        """
        # Get common features
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())
        
        if not common_features:
            return {
                "consistency_score": 0.0,
                "trust_level": "low",
                "metrics": {},
                "warning": "No common features between SHAP and LIME"
            }
        
        # Extract values for common features in same order
        features = sorted(common_features)
        shap_values = np.array([shap_importance[f] for f in features])
        lime_values = np.array([lime_importance[f] for f in features])
        
        # Normalize to unit vectors
        shap_norm = shap_values / (np.linalg.norm(shap_values) + 1e-10)
        lime_norm = lime_values / (np.linalg.norm(lime_values) + 1e-10)
        
        # 1. Cosine similarity
        cosine_sim = float(cosine_similarity(
            shap_norm.reshape(1, -1),
            lime_norm.reshape(1, -1)
        )[0, 0])
        
        # 2. Rank correlation (Spearman)
        if len(features) > 2:
            rank_corr, p_value = spearmanr(shap_values, lime_values)
            rank_corr = float(rank_corr) if not np.isnan(rank_corr) else 0.0
        else:
            rank_corr = cosine_sim  # Fallback for few features
            p_value = None
        
        # 3. Top-k overlap
        k = min(5, len(features))
        shap_top_k = set(sorted(features, key=lambda f: shap_importance[f], reverse=True)[:k])
        lime_top_k = set(sorted(features, key=lambda f: lime_importance[f], reverse=True)[:k])
        top_k_overlap = len(shap_top_k & lime_top_k) / k
        
        # 4. Direction agreement (do they agree on what increases/decreases risk?)
        # This requires signed values, so we'll check if signs match
        direction_agreement = self._calculate_direction_agreement(
            shap_importance, lime_importance, features
        )
        
        # Combined consistency score
        # Weighted average of metrics
        consistency_score = (
            0.30 * max(0, cosine_sim) +      # Cosine similarity (0-1)
            0.30 * max(0, (rank_corr + 1) / 2) +  # Rank correlation (-1 to 1 -> 0 to 1)
            0.20 * top_k_overlap +            # Top-k overlap (0-1)
            0.20 * direction_agreement        # Direction agreement (0-1)
        )
        
        # Determine trust level
        if consistency_score >= 0.8:
            trust_level = "high"
        elif consistency_score >= self.consistency_threshold:
            trust_level = "medium"
        else:
            trust_level = "low"
        
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
        }
    
    def _calculate_direction_agreement(
        self,
        shap_importance: dict[str, float],
        lime_importance: dict[str, float],
        features: list[str]
    ) -> float:
        """
        Calculate how often SHAP and LIME agree on direction.
        
        Note: This uses raw importance which may not include direction.
        If both use absolute values, this becomes less meaningful.
        """
        # For raw SHAP/LIME values (signed), check sign agreement
        # For absolute values, we can only check relative ranking
        agreements = 0
        total = 0
        
        for f in features:
            shap_val = shap_importance[f]
            lime_val = lime_importance[f]
            
            # If both have same sign (or both zero)
            if (shap_val >= 0 and lime_val >= 0) or (shap_val <= 0 and lime_val <= 0):
                agreements += 1
            total += 1
        
        return agreements / total if total > 0 else 0.0
    
    def get_trust_level(self, consistency_score: float) -> str:
        """Get trust level from consistency score."""
        if consistency_score >= 0.8:
            return "high"
        elif consistency_score >= self.consistency_threshold:
            return "medium"
        else:
            return "low"
    
    def flag_low_consistency(
        self,
        shap_importance: dict[str, float],
        lime_importance: dict[str, float]
    ) -> dict[str, Any]:
        """
        Check if explanation should be flagged for review.
        
        Returns flag status and specific disagreements.
        """
        result = self.calculate_consistency(shap_importance, lime_importance)
        
        should_flag = result["consistency_score"] < self.consistency_threshold
        
        disagreements = []
        if result["trust_level"] == "low":
            # Find specific feature disagreements
            common_features = set(shap_importance.keys()) & set(lime_importance.keys())
            
            shap_ranked = sorted(common_features, key=lambda f: shap_importance[f], reverse=True)
            lime_ranked = sorted(common_features, key=lambda f: lime_importance[f], reverse=True)
            
            for i, (shap_f, lime_f) in enumerate(zip(shap_ranked[:5], lime_ranked[:5])):
                if shap_f != lime_f:
                    disagreements.append({
                        "rank": i + 1,
                        "shap_feature": shap_f,
                        "lime_feature": lime_f,
                    })
        
        return {
            "should_flag": should_flag,
            "consistency_score": result["consistency_score"],
            "trust_level": result["trust_level"],
            "disagreements": disagreements,
            "recommendation": (
                "Review explanation manually - SHAP and LIME show different important features"
                if should_flag else
                "Explanation is reliable - SHAP and LIME largely agree"
            ),
        }
