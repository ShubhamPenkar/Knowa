"""B4 causal blindspot heuristics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.ml.blindspot import annotate_drivers, detect_blindspots


class TestBlindspotHeuristics(unittest.TestCase):
    def test_non_intervenable_gender(self):
        drivers = [
            {"feature": "gender", "impact": 0.4, "direction": "increases"},
            {"feature": "Contract", "impact": 0.2, "direction": "increases"},
        ]
        out = detect_blindspots(top_factors=drivers, outcome_label="churn")
        codes = {w["code"] for w in out["warnings"]}
        self.assertIn("non_intervenable", codes)
        self.assertEqual(out["driver_flags"]["gender"]["intervenability"], "low")
        self.assertTrue(out["driver_flags"]["gender"]["blindspot"])
        self.assertEqual(out["preferred_primary_feature"], "Contract")
        self.assertEqual(out["layer"], "B4_causal_blindspots")

    def test_context_not_dial_tenure(self):
        drivers = [
            {"feature": "tenure", "impact": 0.35, "direction": "increases"},
            {"feature": "MonthlyCharges", "impact": 0.22, "direction": "increases"},
        ]
        out = detect_blindspots(top_factors=drivers, outcome_label="churn")
        codes = {w["code"] for w in out["warnings"]}
        self.assertIn("context_not_dial", codes)
        self.assertEqual(out["preferred_primary_feature"], "MonthlyCharges")

    def test_missingness_indicator(self):
        drivers = [
            {"feature": "TotalCharges__is_missing", "impact": 0.5, "direction": "increases"},
        ]
        out = detect_blindspots(
            top_factors=drivers,
            feature_config={"TotalCharges__is_missing": {"derived": True, "type": "numeric"}},
        )
        codes = {w["code"] for w in out["warnings"]}
        self.assertIn("missingness_artifact", codes)

    def test_proxy_pair_from_training(self):
        rng = np.random.default_rng(0)
        n = 200
        a = rng.normal(size=n)
        b = a * 0.95 + rng.normal(scale=0.05, size=n)
        y = (a + rng.normal(scale=0.3, size=n) > 0).astype(int)
        df = pd.DataFrame({"feat_a": a, "feat_b": b, "Churn": y})
        drivers = [
            {"feature": "feat_a", "impact": 0.4, "direction": "increases"},
            {"feature": "feat_b", "impact": 0.35, "direction": "increases"},
        ]
        out = detect_blindspots(
            top_factors=drivers,
            training_data=df,
            target_column="Churn",
            target_positive_label="1",
            max_warnings=5,
        )
        codes = {w["code"] for w in out["warnings"]}
        self.assertIn("proxy_pair", codes)

    def test_segment_reversal(self):
        rows = []
        # Larger segment A (positive corr) so overall corr stays positive;
        # segment B reverses.
        for i in range(90):
            rows.append({"x": i / 90, "seg": "A", "y": 1 if i > 45 else 0})
        for i in range(50):
            rows.append({"x": i / 50, "seg": "B", "y": 0 if i > 25 else 1})
        df = pd.DataFrame(rows)
        drivers = [{"feature": "x", "impact": 0.5, "direction": "increases"}]
        out = detect_blindspots(
            top_factors=drivers,
            training_data=df,
            target_column="y",
            target_positive_label="1",
            max_warnings=5,
        )
        codes = {w["code"] for w in out["warnings"]}
        self.assertIn("segment_reversal", codes)

    def test_consistency_trap(self):
        drivers = [
            {"feature": "Age", "impact": 0.4, "direction": "increases"},
            {"feature": "OverTime", "impact": 0.3, "direction": "increases"},
        ]
        out = detect_blindspots(
            top_factors=drivers,
            consistency={"trust_level": "high", "score": 0.9},
            max_warnings=5,
        )
        codes = {w["code"] for w in out["warnings"]}
        self.assertIn("consistency_trap", codes)
        self.assertIn("non_intervenable", codes)

    def test_annotate_drivers(self):
        drivers = [
            {"feature": "gender", "impact": 0.2},
            {"feature": "Contract", "impact": 0.1},
        ]
        bs = detect_blindspots(top_factors=drivers)
        annotated = annotate_drivers(drivers, bs)
        by_feat = {d["feature"]: d for d in annotated}
        self.assertTrue(by_feat["gender"]["blindspot"])
        self.assertEqual(by_feat["gender"]["intervenability"], "low")

    def test_max_warnings_cap(self):
        drivers = [
            {"feature": "gender", "impact": 0.4, "direction": "increases"},
            {"feature": "Age", "impact": 0.3, "direction": "increases"},
            {"feature": "tenure", "impact": 0.25, "direction": "increases"},
            {"feature": "SeniorCitizen", "impact": 0.2, "direction": "increases"},
        ]
        out = detect_blindspots(
            top_factors=drivers,
            consistency={"trust_level": "high"},
            max_warnings=3,
        )
        self.assertLessEqual(len(out["warnings"]), 3)


if __name__ == "__main__":
    unittest.main()
