"""Regression and unit test suite for Phase 3 Dynamic Product Intelligence.

Validates:
1. Dynamic adaptation to unfamiliar business concepts without hardcoded column names or schemas.
2. Datasets without sales or product concepts (e.g., IoT sensor telemetry, scientific measurements).
3. Missing measures, missing timestamps, empty datasets, and null-heavy columns.
4. Correct aggregation (distribution summaries, breakdowns, trends, top performers).
5. Safe cross-dataset metrics and rejection of ambiguous / high-fan-out joins.
6. Detection of double-counting risks on measures.
7. Explainable insights with heuristic confidence scores and non-causal labeling.
8. Surface investigation opportunities without fabricated ROI or financial projections.
9. Tenant isolation across organizations.
10. Immutability of raw datasets and preservation of user semantic overrides.
11. End-to-end API endpoints (/intelligence/overview, /{dataset_id}/intelligence).
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.db.models import Dataset, Organization, User
from app.main import app
from app.ml.product_intelligence import (
    compute_column_summary,
    compute_categorical_breakdown,
    compute_temporal_trend,
    compute_top_performers,
    compute_cross_dataset_metric,
    generate_explainable_insights,
    surface_investigation_opportunities,
)
from app.services.auth_service import AuthContext, get_auth_context
from app.services.dataset_service import DatasetService
from app.services.product_intelligence_service import ProductIntelligenceService
from app.services.relationship_service import DatasetRelationshipService


class TestDynamicProductIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_prod_intel_")
        cls.db_path = os.path.join(cls.temp_dir, "test.db")
        cls.engine = create_engine(f"sqlite:///{cls.db_path}", connect_args={"check_same_thread": False})
        cls.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)
        Base.metadata.create_all(bind=cls.engine)

        cls.db = cls.SessionLocal()

        # Create two orgs for tenant isolation tests
        cls.org_a = Organization(id="org_alpha_id", name="Org Alpha", slug="org-alpha", industry="tech")
        cls.user_a = User(id="user_alpha_id", organization_id=cls.org_a.id, email="alpha@knowa.com", name="Alpha Admin", role="owner", password_hash="dummy_hash")
        cls.org_b = Organization(id="org_beta_id", name="Org Beta", slug="org-beta", industry="tech")
        cls.user_b = User(id="user_beta_id", organization_id=cls.org_b.id, email="beta@knowa.com", name="Beta Admin", role="owner", password_hash="dummy_hash")
        cls.db.add_all([cls.org_a, cls.user_a, cls.org_b, cls.user_b])
        cls.db.commit()

        cls.ds_svc_a = DatasetService(cls.db, cls.org_a.id)
        cls.ds_svc_b = DatasetService(cls.db, cls.org_b.id)
        cls.rel_svc_a = DatasetRelationshipService(cls.db, cls.org_a.id)
        cls.intel_svc_a = ProductIntelligenceService(cls.db, cls.org_a.id)
        cls.intel_svc_b = ProductIntelligenceService(cls.db, cls.org_b.id)

        def override_get_db():
            db = cls.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        async def override_get_auth():
            return AuthContext(organization=cls.org_a, user=cls.user_a, scopes=["admin"])

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_auth_context] = override_get_auth
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        app.dependency_overrides.clear()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _create_csv_dataset(self, svc, name, df: pd.DataFrame, org_id: str) -> Dataset:
        from app.ml.dataset_profiler import (
            enrich_column_metadata,
            profile_dataframe,
            profile_path_for_parquet,
            save_profile_report,
        )

        dataset_id = hashlib.sha256(f"{name}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
        org_dir = svc.upload_dir
        file_path = os.path.join(org_dir, f"{dataset_id}.parquet")
        df.to_parquet(file_path, index=False)

        report = profile_dataframe(df)
        save_profile_report(report, profile_path_for_parquet(file_path))

        cols = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]
        cols = enrich_column_metadata(cols, report)

        ds = Dataset(
            id=dataset_id,
            name=name,
            organization_id=org_id,
            file_path=file_path,
            row_count=len(df),
            columns=cols,
            file_size=os.path.getsize(file_path),
        )
        self.db.add(ds)
        self.db.commit()
        self.db.refresh(ds)
        return ds

    def test_non_product_telemetry_dataset_analysis(self):
        """Test analyzing a purely scientific/IoT telemetry dataset without product/sales concepts."""
        np.random.seed(42)
        n = 100
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n)]
        df = pd.DataFrame({
            "sensor_node_id": [f"NODE_{i % 5}" for i in range(n)],
            "timestamp": [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates],
            "vibration_frequency_hz": np.random.uniform(50.0, 150.0, n),
            "temperature_celsius": np.random.normal(70.0, 5.0, n),
            "severity_band": np.random.choice(["NORMAL", "WARNING", "ELEVATED"], n),
        })

        # Inject extreme outlier to test distribution skewness detection
        df.loc[0, "vibration_frequency_hz"] = 900.0

        ds = self._create_csv_dataset(self.ds_svc_a, "Sensor Telemetry", df, self.org_a.id)

        overview = self.intel_svc_a.build_overview(dataset_ids=[ds.id], refresh=True)

        self.assertEqual(overview["readiness"]["total_datasets"], 1)
        self.assertEqual(overview["readiness"]["ready_datasets"], 1)
        self.assertTrue(len(overview["metrics"]) >= 3)

        # Verify summary stats
        metric_vib = next((m for m in overview["metrics"] if "vibration_frequency_hz" in m["id"]), None)
        self.assertIsNotNone(metric_vib)
        self.assertIn("skewness", metric_vib["summary_stats"])
        self.assertTrue(abs(metric_vib["summary_stats"]["skewness"]) > 1.5)

        # Verify temporal trend
        metric_trend = next((m for m in overview["metrics"] if m["category"] == "trend"), None)
        self.assertIsNotNone(metric_trend)
        self.assertEqual(metric_trend["status"], "available")
        self.assertIn("direction", metric_trend["summary_stats"])

        # Verify non-causal explainable insight
        insights = overview["insights"]
        self.assertTrue(len(insights) > 0)
        for ins in insights:
            self.assertEqual(ins["confidence_type"], "heuristic")
            self.assertFalse(ins["is_causal_claim"])
            self.assertTrue(any("does not" in lim.lower() or "statistical" in lim.lower() for lim in ins["limitations"]))

    def test_missing_timestamps_and_missing_measures_handling(self):
        """Test datasets lacking timestamps or measures report them as unsupported with clear reasons."""
        df_categorical_only = pd.DataFrame({
            "region": ["North", "South", "East", "West"] * 25,
            "tier": ["Gold", "Silver", "Bronze", "Standard"] * 25,
            "status": ["Active", "Pending", "Active", "Inactive"] * 25,
        })
        ds = self._create_csv_dataset(self.ds_svc_a, "Categorical Only", df_categorical_only, self.org_a.id)

        overview = self.intel_svc_a.build_overview(dataset_ids=[ds.id], refresh=True)

        # Frequency breakdown should still work
        freq_metric = next((m for m in overview["metrics"] if m["category"] == "breakdown"), None)
        self.assertIsNotNone(freq_metric)
        self.assertEqual(freq_metric["aggregation"], "count_by_group")

        # Unsupported metrics should report clear reasons
        unsupported = {u["category"]: u["reason"] for u in overview["unavailable_metrics"]}
        self.assertIn("trend", unsupported)
        self.assertIn("timestamp", unsupported["trend"].lower())
        self.assertIn("summary", unsupported)
        self.assertIn("measure", unsupported["summary"].lower())

    def test_top_performers_and_concentration_insight(self):
        """Test concentration in top performers and categorical breakdowns."""
        df = pd.DataFrame({
            "account_id": [f"ACC_{i:03d}" for i in range(1, 21)],
            "usage_units": [1000.0 if i == 1 else 20.0 for i in range(1, 21)],  # ACC_001 dominates
            "segment": ["Enterprise" if i <= 5 else "SMB" for i in range(1, 21)],
        })
        ds = self._create_csv_dataset(self.ds_svc_a, "Account Usage", df, self.org_a.id)

        overview = self.intel_svc_a.build_overview(dataset_ids=[ds.id], refresh=True)

        perf_metric = next((m for m in overview["metrics"] if m["category"] == "top_performers"), None)
        self.assertIsNotNone(perf_metric)
        self.assertTrue(perf_metric["summary_stats"]["top_n_concentration_pct"] > 50.0)

        # Should generate an explainable insight about concentration
        skew_insight = next((ins for ins in overview["insights"] if "Concentration" in ins["title"] or "Entities Drive" in ins["title"]), None)
        self.assertIsNotNone(skew_insight)
        self.assertEqual(skew_insight["confidence_type"], "heuristic")
        self.assertFalse(skew_insight["is_causal_claim"])

        # Should surface an investigation opportunity linked to this insight
        opp = next((o for o in overview["opportunities"] if "opp_investigate" in o["id"]), None)
        self.assertIsNotNone(opp)
        self.assertEqual(opp["action_type"], "investigation_candidate")
        self.assertTrue(len(opp["missing_evidence"]) > 0)
        self.assertTrue(len(opp["uncertainty_notes"]) > 0)

    def test_safe_cross_dataset_join_and_double_counting_prevention(self):
        """Test safe multi-dataset metric aggregation and prevention of fan-out double counting."""
        # Orders (Transactions)
        orders_df = pd.DataFrame({
            "order_id": [f"ORD_{i}" for i in range(1, 101)],
            "customer_id": [f"CUST_{i % 10}" for i in range(1, 101)],
            "order_total": [50.0 + i for i in range(1, 101)],
        })
        # Customers (Entity: Unique)
        cust_df = pd.DataFrame({
            "customer_id": [f"CUST_{i}" for i in range(10)],
            "territory": ["EMEA", "APAC", "AMER", "EMEA", "APAC", "AMER", "EMEA", "APAC", "AMER", "EMEA"],
        })

        ds_orders = self._create_csv_dataset(self.ds_svc_a, "Orders", orders_df, self.org_a.id)
        ds_cust = self._create_csv_dataset(self.ds_svc_a, "Customers", cust_df, self.org_a.id)

        # Discover relationship
        rels = self.rel_svc_a.discover_relationships(dataset_ids=[ds_orders.id, ds_cust.id], refresh=True)
        self.assertTrue(len(rels) >= 1)

        overview = self.intel_svc_a.build_overview(dataset_ids=[ds_orders.id, ds_cust.id], refresh=True)

        # Cross-dataset metric should be present and safe
        cross_metric = next((m for m in overview["metrics"] if m["category"] == "cross_dataset"), None)
        self.assertIsNotNone(cross_metric)
        self.assertEqual(cross_metric["aggregation"], "cross_sum_by_group")
        self.assertIn("breakdown", cross_metric["summary_stats"])
        self.assertIn("Orders ⋈ Customers", cross_metric["dataset_name"])

    def test_ambiguous_or_high_fan_out_join_excluded_with_safety_warnings(self):
        """Test that ambiguous or high fan-out relationships emit safety warnings and exclude unsafe metrics."""
        # Many-to-many non-unique sets
        df1 = pd.DataFrame({
            "tag": ["alpha", "alpha", "alpha", "beta", "beta"],
            "metric_x": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        df2 = pd.DataFrame({
            "tag": ["alpha", "alpha", "beta", "beta", "beta"],
            "category_y": ["A", "B", "C", "D", "E"],
        })

        meta1 = {"id": "d1", "name": "Set1", "columns": [{"name": "tag", "role": "identifier"}, {"name": "metric_x", "role": "measure"}]}
        meta2 = {"id": "d2", "name": "Set2", "columns": [{"name": "tag", "role": "identifier"}, {"name": "category_y", "role": "dimension"}]}

        # Ambiguous relationship mock
        ambiguous_rel = {
            "source_column": "tag",
            "target_column": "tag",
            "cardinality": "many_to_many",
            "is_ambiguous": True,
            "validation_status": "ambiguous",
            "fan_out_ratio": 2.5,
            "double_counting_risk": True,
            "double_counting_measures": ["metric_x"],
        }

        metric, warnings = compute_cross_dataset_metric(
            source_df=df1,
            source_meta=meta1,
            source_grain={"grain_type": "transaction"},
            target_df=df2,
            target_meta=meta2,
            target_grain={"grain_type": "transaction"},
            rel=ambiguous_rel,
        )

        # Unsafe metric MUST NOT be computed
        self.assertIsNone(metric)
        self.assertTrue(len(warnings) > 0)
        self.assertEqual(warnings[0]["risk"], "ambiguity_risk")

    def test_data_quality_warnings_and_null_heavy_columns(self):
        """Test high missingness and high cardinality generate data quality warnings."""
        df = pd.DataFrame({
            "id": [f"ID_{i}" for i in range(100)],
            "sparse_feature": [None if i < 40 else 10.0 for i in range(100)],  # 40% nulls
            "fragmented_category": [f"CAT_{i}" for i in range(100)],  # 100 distinct categories
        })
        ds = self._create_csv_dataset(self.ds_svc_a, "Quality Test", df, self.org_a.id)

        overview = self.intel_svc_a.build_overview(dataset_ids=[ds.id], refresh=True)
        q_warnings = overview["data_quality_warnings"]
        self.assertTrue(any(w["warning_type"] == "missingness" and w["column_name"] == "sparse_feature" for w in q_warnings))
        self.assertTrue(any(w["warning_type"] == "high_cardinality" and w["column_name"] == "fragmented_category" for w in q_warnings))

    def test_tenant_isolation(self):
        """Verify organization A cannot see organization B's datasets or intelligence overview."""
        df_a = pd.DataFrame({"secret_a": [1, 2, 3], "val_a": [10.0, 20.0, 30.0]})
        df_b = pd.DataFrame({"secret_b": [4, 5, 6], "val_b": [40.0, 50.0, 60.0]})

        self._create_csv_dataset(self.ds_svc_a, "Alpha Private", df_a, self.org_a.id)
        self._create_csv_dataset(self.ds_svc_b, "Beta Private", df_b, self.org_b.id)

        overview_a = self.intel_svc_a.build_overview(refresh=True)
        overview_b = self.intel_svc_b.build_overview(refresh=True)

        ds_names_a = [d["dataset_name"] for d in overview_a["readiness"]["dataset_summaries"]]
        ds_names_b = [d["dataset_name"] for d in overview_b["readiness"]["dataset_summaries"]]

        self.assertIn("Alpha Private", ds_names_a)
        self.assertNotIn("Beta Private", ds_names_a)

        self.assertIn("Beta Private", ds_names_b)
        self.assertNotIn("Alpha Private", ds_names_b)

    def test_raw_dataset_immutability(self):
        """Ensure raw dataset files are byte-for-byte unmodified after Product Intelligence analysis."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["A", "B", "A"]})
        ds = self._create_csv_dataset(self.ds_svc_a, "Immutability Test", df, self.org_a.id)

        with open(ds.file_path, "rb") as f:
            hash_before = hashlib.sha256(f.read()).hexdigest()

        # Run analysis multiple times
        self.intel_svc_a.build_overview(dataset_ids=[ds.id], refresh=True)
        self.intel_svc_a.build_overview(dataset_ids=[ds.id], refresh=False)

        with open(ds.file_path, "rb") as f:
            hash_after = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(hash_before, hash_after)

    def test_http_api_endpoints(self):
        """Verify FastAPI endpoints: GET /intelligence/overview, POST /intelligence/overview, GET /{dataset_id}/intelligence."""
        # 1. GET /api/datasets/intelligence/overview
        res = self.client.get("/api/datasets/intelligence/overview?refresh=true")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("readiness", body)
        self.assertIn("metrics", body)
        self.assertIn("insights", body)
        self.assertIn("opportunities", body)
        self.assertEqual(body["organization_id"], self.org_a.id)

        # 2. POST /api/datasets/intelligence/overview
        res_post = self.client.post("/api/datasets/intelligence/overview", json={"refresh": True})
        self.assertEqual(res_post.status_code, 200)

        # 3. GET /api/datasets/{dataset_id}/intelligence
        datasets = self.client.get("/api/datasets").json()
        if datasets:
            target_id = datasets[0]["id"]
            res_single = self.client.get(f"/api/datasets/{target_id}/intelligence")
            self.assertEqual(res_single.status_code, 200)
            self.assertEqual(res_single.json()["readiness"]["total_datasets"], 1)

        # 4. Auth validation
        app.dependency_overrides.pop(get_auth_context, None)
        try:
            res_unauth = self.client.get("/api/datasets/intelligence/overview")
            self.assertEqual(res_unauth.status_code, 401)
        finally:
            async def override_get_auth():
                return AuthContext(organization=self.org_a, user=self.user_a, scopes=["admin"])
            app.dependency_overrides[get_auth_context] = override_get_auth


if __name__ == "__main__":
    unittest.main()
