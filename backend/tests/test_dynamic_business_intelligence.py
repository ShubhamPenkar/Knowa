"""Phase 3: Dynamic Business Intelligence Comprehensive Test Suite.

Validates:
1. Opportunity detection & relevance scoring based on semantic roles and grain.
2. Descriptive analysis: summary statistics, distributions, Pareto concentration.
3. Trend analysis: temporal trajectory, period-over-period change, direction.
4. Comparative analysis: measure by dimension, disparity ratios, target rates.
5. Correlation analysis: association coefficients, p-values, explicit non-causal warnings.
6. Anomaly detection: z-scores, IQR testing, baseline deviation, affected segments.
7. Segmentation: multivariate clustering excluding IDs, timestamps, and target leakage.
8. Grain awareness: pre-aggregation on joins, prevention of fan-out double-counting.
9. Insight synthesis: severity, confidence, evidence, deduplication, proactive ranking.
10. Explainability: retrieval of finding, metric calculation, data source, limitations.
11. Edge cases: empty data, single row, constant values, all-nulls.
12. End-to-end API routes and multi-tenant isolation.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.db.models import Dataset, Organization, User
from app.main import app
from app.ml.dataset_profiler import (
    enrich_column_metadata,
    profile_dataframe,
    profile_path_for_parquet,
    save_profile_report,
)
from app.ml.dynamic_intelligence import (
    AnalysisOpportunityEngine,
    AnalysisExecutionEngine,
    DynamicInsightEngine,
)
from app.services.auth_service import AuthContext, get_auth_context
from app.services.dataset_service import DatasetService
from app.services.dynamic_intelligence_service import DynamicIntelligenceService
from app.services.relationship_service import DatasetRelationshipService


class TestDynamicBusinessIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_dynamic_bi_")
        cls.db_path = os.path.join(cls.temp_dir, "test.db")
        cls.engine = create_engine(f"sqlite:///{cls.db_path}", connect_args={"check_same_thread": False})
        cls.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)
        Base.metadata.create_all(bind=cls.engine)

        cls.db = cls.SessionLocal()

        # Create two orgs for tenant isolation tests
        cls.org_a = Organization(id="org_bi_alpha", name="Alpha Corp", slug="alpha-corp", industry="saas")
        cls.user_a = User(id="user_bi_alpha", organization_id=cls.org_a.id, email="alpha@test.com", name="Alpha User", role="owner", password_hash="dummy")
        cls.org_b = Organization(id="org_bi_beta", name="Beta Corp", slug="beta-corp", industry="retail")
        cls.user_b = User(id="user_bi_beta", organization_id=cls.org_b.id, email="beta@test.com", name="Beta User", role="owner", password_hash="dummy")
        cls.db.add_all([cls.org_a, cls.user_a, cls.org_b, cls.user_b])
        cls.db.commit()

        cls.ds_svc_a = DatasetService(cls.db, cls.org_a.id)
        cls.ds_svc_b = DatasetService(cls.db, cls.org_b.id)
        cls.rel_svc_a = DatasetRelationshipService(cls.db, cls.org_a.id)
        cls.bi_svc_a = DynamicIntelligenceService(cls.db, cls.org_a.id)
        cls.bi_svc_b = DynamicIntelligenceService(cls.db, cls.org_b.id)

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

    def _create_dataset(self, svc, name: str, df: pd.DataFrame, org_id: str) -> Dataset:
        dataset_id = hashlib.sha256(f"{name}_{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:12]
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
        return ds

    # =================================================================
    # 1. OPPORTUNITY ENGINE & RELEVANCE SCORING
    # =================================================================

    def test_opportunity_discovery_for_saas_churn(self):
        """Customer churn data should yield churn rate comparison, tenure/spend correlations, and segmentation."""
        meta = {
            "id": "ds_churn",
            "name": "Customer Churn",
            "columns": [
                {"name": "customer_id", "role": "identifier", "semantic_type": "identifier", "distinct_count": 500},
                {"name": "plan", "role": "dimension", "business_role": "dimension", "semantic_type": "categorical", "distinct_count": 4},
                {"name": "monthly_spend", "role": "measure", "business_role": "revenue", "semantic_type": "currency", "distinct_count": 250},
                {"name": "tenure_months", "role": "measure", "business_role": "tenure", "semantic_type": "numeric_measure", "distinct_count": 48},
                {"name": "churn", "role": "dimension", "business_role": "churn", "is_target": True, "distinct_count": 2},
            ],
            "row_count": 500,
        }
        opps = AnalysisOpportunityEngine.discover_opportunities(meta)
        self.assertGreater(len(opps), 0)

        # Target comparison should have highest relevance
        top_opp = opps[0]
        self.assertEqual(top_opp["analysis_type"], "comparative")
        self.assertIn("Churn", top_opp["title"])
        self.assertGreaterEqual(top_opp["relevance_score"], 0.90)

        # Correlation between spend and tenure
        types = [o["analysis_type"] for o in opps]
        self.assertIn("correlation", types)
        self.assertIn("segmentation", types)
        self.assertIn("descriptive", types)

    def test_opportunity_discovery_for_retail_sales(self):
        """Retail sales dataset should yield revenue trend, regional breakdown, and concentration."""
        meta = {
            "id": "ds_sales",
            "name": "Retail Sales",
            "columns": [
                {"name": "order_id", "role": "identifier", "semantic_type": "identifier", "distinct_count": 1000},
                {"name": "order_date", "role": "timestamp", "semantic_type": "datetime", "distinct_count": 300},
                {"name": "region", "role": "dimension", "business_role": "region", "semantic_type": "geographic", "distinct_count": 5},
                {"name": "revenue", "role": "measure", "business_role": "revenue", "semantic_type": "currency", "distinct_count": 800},
                {"name": "quantity", "role": "measure", "business_role": "quantity", "semantic_type": "numeric_measure", "distinct_count": 50},
            ],
            "row_count": 1000,
        }
        opps = AnalysisOpportunityEngine.discover_opportunities(meta)
        types = [o["analysis_type"] for o in opps]
        self.assertIn("trend", types)
        self.assertIn("comparative", types)
        self.assertIn("anomaly", types)

        trend_opp = next(o for o in opps if o["analysis_type"] == "trend")
        self.assertEqual(trend_opp["primary_column"], "revenue")
        self.assertEqual(trend_opp["secondary_column"], "order_date")
        self.assertGreaterEqual(trend_opp["relevance_score"], 0.85)

    def test_opportunity_discovery_minimal_dataset(self):
        """Dataset with only identifiers and text should yield safe minimal candidates."""
        meta = {
            "id": "ds_minimal",
            "name": "Audit Log",
            "columns": [
                {"name": "log_id", "role": "identifier", "semantic_type": "identifier", "distinct_count": 100},
                {"name": "message", "role": "dimension", "semantic_type": "text", "distinct_count": 100},
            ],
            "row_count": 100,
        }
        opps = AnalysisOpportunityEngine.discover_opportunities(meta)
        # Should not invent measures or correlations
        types = [o["analysis_type"] for o in opps]
        self.assertNotIn("correlation", types)
        self.assertNotIn("segmentation", types)

    # =================================================================
    # 2. DESCRIPTIVE ANALYSIS
    # =================================================================

    def test_descriptive_execution_summary_and_pareto(self):
        """Descriptive analysis computes mean, median, IQR, and top 20% concentration."""
        np.random.seed(42)
        # Power law distribution: 80-20 rule
        data = np.concatenate([np.random.exponential(scale=10, size=80), np.random.exponential(scale=150, size=20)])
        df = pd.DataFrame({"revenue": data})
        opp = {
            "id": "opp_desc_rev",
            "analysis_type": "descriptive",
            "title": "Revenue Distribution",
            "primary_column": "revenue",
        }
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "completed")
        stats = res["summary_stats"]
        self.assertGreater(stats["top_20_pct_concentration"], 40.0)
        self.assertIsNotNone(stats["skewness"])
        self.assertEqual(len(res["visualization_data"]["histogram"]), 10)

    # =================================================================
    # 3. TREND ANALYSIS
    # =================================================================

    def test_trend_execution_upward_trajectory(self):
        """Trend execution captures growth, direction, and consecutive periods."""
        dates = pd.date_range("2026-01-01", periods=12, freq="W")
        values = [100 + i * 15 for i in range(12)]
        df = pd.DataFrame({"order_date": dates, "revenue": values})
        opp = {
            "id": "opp_trend_rev",
            "analysis_type": "trend",
            "title": "Revenue Trend",
            "primary_column": "revenue",
            "secondary_column": "order_date",
        }
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "completed")
        stats = res["summary_stats"]
        self.assertEqual(stats["direction"], "upward")
        self.assertGreater(stats["overall_change_pct"], 50.0)
        self.assertIn("series", res["visualization_data"])

    def test_trend_execution_insufficient_dates(self):
        """Trend analysis should handle datasets with fewer than 3 timestamps gracefully."""
        df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "revenue": [10, 20]})
        opp = {"id": "opp_t", "analysis_type": "trend", "title": "Trend", "primary_column": "revenue", "secondary_column": "date"}
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "unavailable")

    # =================================================================
    # 4. COMPARATIVE ANALYSIS
    # =================================================================

    def test_comparative_measure_by_dimension(self):
        """Comparative analysis evaluates group shares and segment disparity."""
        df = pd.DataFrame({
            "region": ["North"] * 50 + ["South"] * 50 + ["East"] * 50,
            "revenue": [500] * 50 + [100] * 50 + [250] * 50,
        })
        opp = {
            "id": "opp_comp_rev_reg",
            "analysis_type": "comparative",
            "title": "Revenue by Region",
            "primary_column": "revenue",
            "secondary_column": "region",
        }
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "completed")
        stats = res["summary_stats"]
        self.assertEqual(stats["top_segment"], "North")
        self.assertEqual(stats["bottom_segment"], "South")
        self.assertGreaterEqual(stats["disparity_ratio"], 4.5)

    def test_comparative_target_rate(self):
        """Comparative analysis calculates target event rates across groups."""
        df = pd.DataFrame({
            "tier": ["Free"] * 100 + ["Enterprise"] * 100,
            "churn": [1] * 35 + [0] * 65 + [1] * 5 + [0] * 95,
        })
        opp = {
            "id": "opp_comp_churn",
            "analysis_type": "comparative",
            "title": "Churn Rate by Tier",
            "primary_column": "churn",
            "secondary_column": "tier",
        }
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "completed")
        stats = res["summary_stats"]
        self.assertTrue(stats["is_target_rate"])
        self.assertEqual(stats["top_segment"], "Free")
        self.assertEqual(stats["top_segment_value"], 35.0)

    # =================================================================
    # 5. CORRELATION & NON-CAUSAL LABELS
    # =================================================================

    def test_correlation_positive_and_non_causal(self):
        """Correlation calculates Pearson r and explicitly enforces non-causal warnings."""
        x = np.linspace(10, 100, 50)
        y = 2.5 * x + np.random.normal(0, 5, 50)
        df = pd.DataFrame({"spend": x, "usage": y})
        opp = {
            "id": "opp_corr_spend_usage",
            "analysis_type": "correlation",
            "title": "Spend vs Usage",
            "primary_column": "spend",
            "secondary_column": "usage",
        }
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "completed")
        stats = res["summary_stats"]
        self.assertGreater(stats["correlation_coefficient"], 0.8)
        self.assertTrue(stats["is_significant"])

        # Must have non-causal warning
        warning_str = " ".join(res["warnings"]) + " " + " ".join(res["limitations"])
        self.assertIn("causation", warning_str.lower())

    def test_correlation_zero_variance(self):
        """Correlation with constant columns returns unavailable without crashing."""
        df = pd.DataFrame({"col_a": [5, 5, 5, 5, 5], "col_b": [1, 2, 3, 4, 5]})
        opp = {"id": "opp_c", "analysis_type": "correlation", "title": "Corr", "primary_column": "col_a", "secondary_column": "col_b"}
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "unavailable")

    # =================================================================
    # 6. ANOMALY DETECTION
    # =================================================================

    def test_anomaly_detection_with_spikes(self):
        """Anomaly detection finds records > 2.5 standard deviations from baseline."""
        baseline = [50] * 50
        baseline[10] = 500  # Obvious spike
        df = pd.DataFrame({"latency": baseline})
        opp = {
            "id": "opp_anom_latency",
            "analysis_type": "anomaly",
            "title": "Latency Anomaly Audit",
            "primary_column": "latency",
        }
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "completed")
        stats = res["summary_stats"]
        self.assertEqual(stats["anomaly_count"], 1)
        self.assertGreaterEqual(stats["max_z_score"], 2.5)

    def test_anomaly_detection_constant_values(self):
        """Constant column returns unavailable for anomalies."""
        df = pd.DataFrame({"val": [10] * 20})
        opp = {"id": "opp_a", "analysis_type": "anomaly", "title": "Anom", "primary_column": "val"}
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "unavailable")

    # =================================================================
    # 7. SEGMENTATION
    # =================================================================

    def test_segmentation_multivariate(self):
        """Segmentation clusters entities across continuous business features."""
        df = pd.DataFrame({
            "recency": [1, 2, 3, 50, 60, 70] * 5,
            "monetary": [1000, 1200, 1100, 100, 150, 120] * 5,
        })
        opp = {
            "id": "opp_seg",
            "analysis_type": "segmentation",
            "title": "Customer Segments",
            "primary_column": "recency",
            "secondary_column": "monetary",
        }
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["summary_stats"]["clusters_count"], 3)

    # =================================================================
    # 8. GRAIN-AWARE & CROSS-DATASET SAFEGUARDS
    # =================================================================

    def test_cross_dataset_safe_join_execution(self):
        """Cross-dataset analysis succeeds when relationship is validated and safe."""
        cust_df = pd.DataFrame({"cust_id": ["C1", "C2", "C3"], "region": ["US", "EU", "APAC"]})
        order_df = pd.DataFrame({"cust_id": ["C1", "C2", "C3"], "amount": [100.0, 200.0, 300.0]})
        rel = {
            "source_column": "cust_id",
            "target_column": "cust_id",
            "cardinality": "one_to_one",
            "fan_out_ratio": 1.0,
            "double_counting_risk": False,
            "validation_status": "validated",
        }
        opp = {
            "id": "opp_cross_safe",
            "analysis_type": "cross_dataset",
            "title": "Amount by Region",
            "primary_column": "cust_id",
            "secondary_column": "cust_id",
        }
        res = AnalysisExecutionEngine.execute_analysis(opp, order_df, secondary_df=cust_df, relationship=rel)
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["summary_stats"]["joined_records"], 3)

    def test_cross_dataset_blocked_when_fan_out_double_counting(self):
        """Cross-dataset analysis is blocked with explicit reason when fan-out risks double counting."""
        cust_df = pd.DataFrame({"cust_id": ["C1"], "lifetime_value": [1000.0]})
        order_df = pd.DataFrame({"cust_id": ["C1", "C1", "C1", "C1", "C1"], "amount": [10, 20, 30, 40, 50]})
        rel = {
            "source_column": "cust_id",
            "target_column": "cust_id",
            "cardinality": "one_to_many",
            "fan_out_ratio": 5.0,
            "double_counting_risk": True,
            "validation_status": "candidate",
        }
        opp = {
            "id": "opp_cross_unsafe",
            "analysis_type": "cross_dataset",
            "title": "LTV by Orders",
            "primary_column": "cust_id",
            "secondary_column": "cust_id",
        }
        res = AnalysisExecutionEngine.execute_analysis(opp, cust_df, secondary_df=order_df, relationship=rel)
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("fan-out", res["limitations"][0].lower())

    # =================================================================
    # 9. INSIGHT SYNTHESIS, DEDUPLICATION & RANKING
    # =================================================================

    def test_insight_synthesis_and_severity(self):
        """Insights derive correct severity levels (critical, warning, positive, info)."""
        # Critical churn growth
        trend_res = {
            "opportunity_id": "opp_churn_trend",
            "analysis_type": "trend",
            "title": "Customer Churn Over Time",
            "status": "completed",
            "summary_stats": {
                "overall_change_pct": 28.5,
                "direction": "upward",
                "period_type": "monthly",
                "start_period": "2026-01",
                "end_period": "2026-04",
                "start_value": 100,
                "end_value": 128.5,
                "average_pop_change_pct": 8.5,
            },
        }
        insights = DynamicInsightEngine.generate_insights_from_analysis(trend_res)
        self.assertEqual(len(insights), 1)
        self.assertEqual(insights[0]["severity"], "critical")
        self.assertFalse(insights[0]["is_causal_claim"])

    def test_insight_deduplication(self):
        """Deduplication combines redundant findings and merges unique evidence."""
        ins1 = {
            "id": "ins_1",
            "category": "trend",
            "metric": "revenue",
            "severity": "info",
            "confidence": 0.7,
            "evidence": ["Evidence A", "Evidence B"],
        }
        ins2 = {
            "id": "ins_2",
            "category": "trend",
            "metric": "revenue",
            "severity": "positive",
            "confidence": 0.9,
            "evidence": ["Evidence B", "Evidence C"],
        }
        deduped = DynamicInsightEngine.deduplicate_insights([ins1, ins2])
        self.assertEqual(len(deduped), 1)
        best = deduped[0]
        self.assertEqual(best["severity"], "positive")
        self.assertEqual(set(best["evidence"]), {"Evidence A", "Evidence B", "Evidence C"})

    def test_proactive_insight_ranking(self):
        """Insights are ranked by severity, magnitude, and confidence."""
        ins_list = [
            {"id": "1", "severity": "info", "confidence": 0.8, "category": "descriptive", "value": 5.0},
            {"id": "2", "severity": "critical", "confidence": 0.95, "category": "anomaly", "value": 50.0},
            {"id": "3", "severity": "warning", "confidence": 0.85, "category": "comparative", "value": 15.0},
        ]
        ranked = DynamicInsightEngine.rank_insights(ins_list, top_n=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["id"], "2")  # Critical first
        self.assertEqual(ranked[1]["id"], "3")  # Warning second

    # =================================================================
    # 10. EDGE CASES
    # =================================================================

    def test_edge_case_empty_and_single_row(self):
        """Engine handles empty and single-row data gracefully."""
        empty_df = pd.DataFrame()
        opp = {"id": "opp_e", "analysis_type": "descriptive", "title": "Empty", "primary_column": "x"}
        res = AnalysisExecutionEngine.execute_analysis(opp, empty_df)
        self.assertEqual(res["status"], "unavailable")

        single_df = pd.DataFrame({"x": [42]})
        res2 = AnalysisExecutionEngine.execute_analysis(opp, single_df)
        self.assertEqual(res2["status"], "unavailable")

    def test_edge_case_all_null_column(self):
        """All-null column returns unavailable without errors."""
        df = pd.DataFrame({"x": [None, None, None, None], "y": [1, 2, 3, 4]})
        opp = {"id": "opp_null", "analysis_type": "descriptive", "title": "Null", "primary_column": "x"}
        res = AnalysisExecutionEngine.execute_analysis(opp, df)
        self.assertEqual(res["status"], "unavailable")

    # =================================================================
    # 11. END-TO-END SERVICE & API ROUTES
    # =================================================================

    def test_end_to_end_service_and_api(self):
        """End-to-end test verifying API routes for opportunities, execution, and explainability."""
        df = pd.DataFrame({
            "order_id": [f"ORD_{i}" for i in range(100)],
            "region": ["North", "South", "East", "West"] * 25,
            "revenue": [50 + (i % 10) * 10 for i in range(100)],
            "created_at": pd.date_range("2026-01-01", periods=100, freq="D"),
        })
        ds = self._create_dataset(self.ds_svc_a, "Alpha Orders", df, self.org_a.id)

        # 1. GET /api/datasets/{id}/analysis-opportunities
        res = self.client.get(f"/api/datasets/{ds.id}/analysis-opportunities")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("opportunities", data)
        self.assertGreater(data["total_opportunities"], 0)

        # 2. POST /api/datasets/{id}/analyze
        run_res = self.client.post(f"/api/datasets/{ds.id}/analyze", json={"top_n": 3, "refresh": True})
        self.assertEqual(run_res.status_code, 200)
        run_data = run_res.json()
        self.assertIn("top_insights", run_data)
        self.assertIn("executed_analyses", run_data)
        self.assertEqual(len(run_data["executed_analyses"]), 3)

        # 3. GET /api/datasets/{id}/insights
        ins_res = self.client.get(f"/api/datasets/{ds.id}/insights")
        self.assertEqual(ins_res.status_code, 200)
        ins_data = ins_res.json()
        self.assertEqual(ins_data["dataset_id"], ds.id)

        # 4. GET /api/datasets/{id}/insights/{insight_id} (Explainability Detail)
        if run_data["top_insights"]:
            target_ins = run_data["top_insights"][0]
            det_res = self.client.get(f"/api/datasets/{ds.id}/insights/{target_ins['id']}")
            self.assertEqual(det_res.status_code, 200)
            det_data = det_res.json()
            self.assertEqual(det_data["finding"], target_ins["title"])
            self.assertIn("metric_calculation", det_data)
            self.assertIn("data_used", det_data)
            self.assertGreater(len(det_data["statistical_evidence"]), 0)

    def test_tenant_isolation(self):
        """Dataset belonging to Org B cannot be analyzed or read by Org A."""
        df = pd.DataFrame({"secret_metric": [10, 20, 30, 40, 50], "dim": ["A", "B", "C", "D", "E"]})
        ds_b = self._create_dataset(self.ds_svc_b, "Beta Secret Data", df, self.org_b.id)

        # Org A client queries Org B dataset
        res = self.client.get(f"/api/datasets/{ds_b.id}/analysis-opportunities")
        self.assertEqual(res.status_code, 404)

        res2 = self.client.get(f"/api/datasets/{ds_b.id}/insights")
        self.assertEqual(res2.status_code, 404)
