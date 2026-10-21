"""Comprehensive Phase 1 Tests for Semantic Data Understanding.

Covers:
1. Physical and Semantic data types: integer, float, string, boolean, date, datetime,
   identifier, measure, dimension, percentage, currency, text, unknown, email, phone.
2. Evidence-based business role inference (e.g. uninformative column names like 'X17').
3. Edge cases:
   - Empty dataset
   - Single-column dataset
   - Constant column
   - High-cardinality text column
   - Mostly-null column
   - Duplicate rows
   - Mixed-type column
   - Weird/unnamed column names
4. Confidence bounds and deterministic evidence.
5. Potential target detection & ranking.
6. Data quality scoring, outlier detection, duplicate rows calculation.
7. API Integration (GET /semantic-profile, POST /profile, PATCH semantic with business_role).
"""

import os
import uuid
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.database import Base, get_db
from app.main import app
from app.ml.dataset_profiler import profile_dataframe, profile_column
from app.ml.semantic_inferencer import (
    BusinessRole,
    SemanticRole,
    SemanticType,
    infer_column_semantics,
)
from app.db.models import Dataset, Organization, User


class TestSemanticDataTypesAndRoles:
    """Validate data types and evidence-based semantic inference."""

    def test_integer_age_inferred_from_distribution_without_age_name(self):
        """Even with an uninformative column name like 'X17', ages 20-80 should be inferred as age."""
        np.random.seed(42)
        values = np.random.randint(18, 75, size=200)
        s = pd.Series(values, name="X17")

        ann = infer_column_semantics("X17", s, n_rows=200)
        assert 0.0 <= ann.confidence <= 1.0
        assert ann.semantic_type in (SemanticType.MEASURE.value, SemanticType.NUMERIC_MEASURE.value)
        assert ann.business_role == BusinessRole.AGE.value
        assert any("age" in ev.lower() for ev in ann.evidence)

    def test_rating_inferred_from_distribution_1_to_5(self):
        """Values between 1.0 and 5.0 with low cardinality suggest a rating/score."""
        s = pd.Series([1.0, 2.0, 3.5, 4.0, 5.0, 4.5, 3.0] * 30, name="metric_score_unlabeled")
        ann = infer_column_semantics("metric_score_unlabeled", s, n_rows=len(s))
        assert ann.business_role == BusinessRole.RATING.value
        assert ann.semantic_type in (SemanticType.MEASURE.value, SemanticType.NUMERIC_MEASURE.value)

    def test_currency_and_monetary_measure(self):
        """Currency formatted strings or high-variance monetary values are classified as currency/measure."""
        currency_strings = pd.Series(["$120.50", "$45.00", "$1,230.99", "$99.95"] * 25, name="unlabeled_money")
        ann = infer_column_semantics("unlabeled_money", currency_strings, n_rows=100)
        assert ann.semantic_type == SemanticType.CURRENCY.value
        assert ann.role == SemanticRole.MEASURE.value
        assert ann.confidence >= 0.85
        assert len(ann.evidence) > 0

    def test_percentage_inference(self):
        """Percentages in [0, 1] or with % signs are identified as percentage."""
        pct_strings = pd.Series(["12.5%", "95.0%", "4.2%", "50.0%"] * 25, name="rate_val")
        ann = infer_column_semantics("rate_val", pct_strings, n_rows=100)
        assert ann.semantic_type == SemanticType.PERCENTAGE.value
        assert ann.role == SemanticRole.MEASURE.value

    def test_email_and_phone_patterns(self):
        """Email and phone number patterns are detected as contact attributes."""
        emails = pd.Series(["user1@example.com", "sales@corp.org", "info@domain.net", "ceo@company.io"] * 20, name="contact_col")
        ann_email = infer_column_semantics("contact_col", emails, n_rows=80)
        assert ann_email.semantic_type == SemanticType.EMAIL.value

        phones = pd.Series(["+1 (555) 234-5678", "555-876-5432", "+44 20 7946 0958"] * 25, name="digits_col")
        ann_phone = infer_column_semantics("digits_col", phones, n_rows=75)
        assert ann_phone.semantic_type == SemanticType.PHONE.value

    def test_date_and_datetime_inference(self):
        """ISO strings and timestamp patterns are classified as datetime/date."""
        dates = pd.Series(["2023-01-15", "2023-02-20", "2023-03-25", "2023-04-30"] * 25, name="event_time")
        ann = infer_column_semantics("event_time", dates, n_rows=100)
        assert ann.semantic_type in (SemanticType.DATETIME.value, SemanticType.DATE.value)
        assert ann.role == SemanticRole.TIMESTAMP.value

    def test_boolean_inference(self):
        """True/False, Yes/No, or 0/1 indicator flags."""
        flags = pd.Series([True, False, True, True, False] * 20, name="active_flag")
        ann = infer_column_semantics("active_flag", flags, n_rows=100)
        assert ann.semantic_type == SemanticType.BOOLEAN.value

    def test_unstructured_text(self):
        """Long sentences with varied vocabulary are classified as unstructured text."""
        paragraphs = pd.Series([
            "Customer contacted support regarding their invoice and expressed high satisfaction with the speedy resolution.",
            "Account cancellation requested due to relocation to an unsupported region. Requested data export.",
            "User experienced intermittent latency when loading the analytics dashboard during peak morning hours.",
            "Inquiry concerning enterprise SLA agreements and custom single sign-on integration requirements.",
        ] * 25, name="feedback_notes")
        ann = infer_column_semantics("feedback_notes", paragraphs, n_rows=100)
        assert ann.semantic_type == SemanticType.TEXT.value
        assert ann.role == SemanticRole.DIMENSION.value

    def test_unknown_when_evidence_is_insufficient(self):
        """Sparse, ambiguous data with no recognizable pattern defaults cleanly to unknown."""
        weird = pd.Series(["---", "???", None, "N/A"] * 10, name="mystery_column")
        ann = infer_column_semantics("mystery_column", weird, n_rows=40)
        assert ann.semantic_type in (SemanticType.UNKNOWN.value, SemanticType.CATEGORICAL.value)


class TestSemanticEdgeCases:
    """Verify graceful handling of all edge-case dataset structures."""

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        report = profile_dataframe(df)
        assert report.blocking is True
        assert report.n_rows == 0
        assert report.n_columns == 0
        assert len(report.columns) == 0
        assert any(issue.get("code") == "empty_dataframe" for issue in report.blocking_issues)

    def test_single_column_dataframe(self):
        df = pd.DataFrame({"revenue": [100.0, 250.0, 300.0, 450.0, 500.0]})
        report = profile_dataframe(df)
        assert report.n_rows == 5
        assert report.n_columns == 1
        assert len(report.columns) == 1
        assert report.columns[0].column_name == "revenue"
        assert report.columns[0].business_role == BusinessRole.REVENUE.value

    def test_constant_column(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "constant_feature": ["SAME", "SAME", "SAME", "SAME", "SAME"],
        })
        report = profile_dataframe(df)
        assert "constant_feature" in report.dropped_as_constant
        const_col = report.column_map()["constant_feature"]
        assert const_col.action_taken == "drop_constant"
        assert "constant_or_empty" in const_col.issues

    def test_extremely_high_cardinality_categorical(self):
        uuids = [str(uuid.uuid4()) for _ in range(100)]
        df = pd.DataFrame({
            "random_hash": uuids,
            "feature": range(100),
        })
        report = profile_dataframe(df)
        col = report.column_map()["random_hash"]
        assert col.unique_ratio == 1.0
        assert (
            "high_cardinality_categorical" in col.issues
            or "likely_identifier" in col.issues
            or col.action_taken == "exclude_as_id"
        )

    def test_mostly_null_column(self):
        values = [None] * 95 + [10.0, 20.0, 30.0, 40.0, 50.0]
        df = pd.DataFrame({"sparse_data": values})
        report = profile_dataframe(df)
        col = report.column_map()["sparse_data"]
        assert col.missing_ratio >= 0.90
        assert "high_missing_density" in col.issues

    def test_duplicate_rows_detection(self):
        df = pd.DataFrame({
            "col_a": [1, 1, 2, 2, 3],
            "col_b": ["A", "A", "B", "B", "C"],
        })
        report = profile_dataframe(df)
        assert report.duplicate_rows_count == 2
        assert report.duplicate_rows_ratio == 0.4
        assert any(w.get("code") == "duplicate_rows_detected" for w in report.warnings)

    def test_mixed_type_column(self):
        mixed = pd.Series([100, "two hundred", 300, None, "error", 500], name="mixed_col")
        col_prof = profile_column("mixed_col", mixed, n_rows=6)
        assert col_prof.column_name == "mixed_col"
        assert 0.0 <= col_prof.confidence <= 1.0

    def test_weird_and_unnamed_column_names(self):
        df = pd.DataFrame({
            "Unnamed: 0": range(10),
            "___$$$###": [10.5, 20.2, 30.1, 40.8, 50.3] * 2,
            "12345": ["A", "B", "C", "D", "E"] * 2,
        })
        report = profile_dataframe(df)
        assert report.n_columns == 3
        # Should profile each column without crashing
        assert len(report.columns) == 3


class TestConfidenceAndEvidence:
    """Verify confidence score calibration and explainability contracts."""

    def test_confidence_is_bounded_and_deterministic(self):
        s = pd.Series([10.5, 20.0, 30.2, 40.1, 55.4] * 20, name="annual_spend")
        ann1 = infer_column_semantics("annual_spend", s, n_rows=100)
        ann2 = infer_column_semantics("annual_spend", s, n_rows=100)

        assert 0.0 <= ann1.confidence <= 1.0
        assert ann1.confidence == ann2.confidence
        assert ann1.semantic_type == ann2.semantic_type
        assert ann1.business_role == ann2.business_role

    def test_every_inference_has_non_empty_evidence(self):
        s = pd.Series(["cust_001", "cust_002", "cust_003", "cust_004"] * 10, name="customer_id")
        ann = infer_column_semantics("customer_id", s, n_rows=40)
        assert len(ann.evidence) > 0
        assert isinstance(ann.evidence, list)
        for ev in ann.evidence:
            assert isinstance(ev, str)
            assert len(ev.strip()) > 0


class TestTargetDetectionAndDataQuality:
    """Verify target ranking, relationships, and dataset quality scoring."""

    def test_target_candidate_ranking(self):
        df = pd.DataFrame({
            "customer_id": [f"ID_{i}" for i in range(100)],
            "monthly_charges": np.random.uniform(20.0, 100.0, 100),
            "churn": [0, 1] * 50,
            "sentiment_score": np.random.uniform(1.0, 5.0, 100),
        })
        report = profile_dataframe(df)
        assert len(report.potential_targets) >= 1
        top_target = report.potential_targets[0]
        assert top_target["column_name"] == "churn"
        assert top_target["target_type"] in ("classification", "binary_classification")
        assert top_target["confidence"] >= 0.8

    def test_high_correlation_relationship_detection(self):
        base = np.linspace(10, 100, 100)
        noisy = base * 2.0 + np.random.normal(0, 0.5, 100)
        df = pd.DataFrame({
            "feature_x": base,
            "feature_y": noisy,
            "feature_z": np.random.uniform(0, 1, 100),
        })
        report = profile_dataframe(df)
        corrs = [r for r in report.column_relationships if r.get("type") == "high_correlation"]
        assert len(corrs) >= 1
        pair = corrs[0]["columns"]
        assert "feature_x" in pair and "feature_y" in pair
        assert corrs[0]["strength"] >= 0.90

    def test_dataset_quality_score_and_summary(self):
        clean_df = pd.DataFrame({
            "id": range(100),
            "feature_a": np.random.normal(50, 10, 100),
            "feature_b": ["Cat", "Dog"] * 50,
        })
        clean_report = profile_dataframe(clean_df)
        assert clean_report.quality_score >= 85.0
        assert clean_report.quality_summary["grade"] in ("Excellent", "Good")

        dirty_df = pd.DataFrame({
            "dup_id": [1, 1, 1, 1, 1] * 20,
            "all_null": [None] * 100,
            "mostly_null": [None] * 90 + [1.0] * 10,
        })
        dirty_report = profile_dataframe(dirty_df)
        assert dirty_report.quality_score < clean_report.quality_score
        assert dirty_report.duplicate_rows_count > 0


class TestSemanticApiIntegration:
    """Test API endpoints for semantic profile retrieval and user overrides."""

    @pytest.fixture(autouse=True)
    def setup_api(self, tmp_path):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from app.database import Base, get_db
        from app.db.models import Dataset, Organization, User
        from app.ml.dataset_profiler import enrich_column_metadata, profile_dataframe
        from app.services.auth_service import AuthContext, get_auth_context

        self.tmp_dir = str(tmp_path)
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db = self.Session()

        self.org = Organization(name="Semantic Org", slug="semantic-org")
        self.db.add(self.org)
        self.db.flush()

        self.user = User(
            organization_id=self.org.id,
            email="semantic@knowa.ai",
            password_hash="pw",
            name="Semantic Admin",
            role="admin",
        )
        self.db.add(self.user)
        self.db.flush()

        # Seed sample dataset
        csv_file = tmp_path / "business_data.parquet"
        df = pd.DataFrame({
            "cust_id": [f"CUST_{i:04d}" for i in range(50)],
            "annual_revenue": np.random.uniform(5000, 50000, 50),
            "customer_churn": [0, 1] * 25,
            "tenure_months": np.random.randint(1, 72, 50),
        })
        df.to_parquet(str(csv_file), index=False)

        report = profile_dataframe(df)
        base_columns = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]
        columns = enrich_column_metadata(base_columns, report)

        self.dataset = Dataset(
            organization_id=self.org.id,
            name="Business Customers",
            file_path=str(csv_file),
            file_size=os.path.getsize(str(csv_file)),
            row_count=50,
            columns=columns,
        )
        self.db.add(self.dataset)
        self.db.commit()

        def _get_db_override():
            yield self.db

        async def _get_auth_override():
            return AuthContext(organization=self.org, user=self.user, scopes=["admin"])

        app.dependency_overrides[get_db] = _get_db_override
        app.dependency_overrides[get_auth_context] = _get_auth_override
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer test-token"}

        yield

        app.dependency_overrides.clear()
        self.db.close()
        self.engine.dispose()

    def test_get_semantic_profile_endpoint(self):
        """GET /api/datasets/{id}/semantic-profile returns structured semantic profile."""
        res = self.client.get(
            f"/api/datasets/{self.dataset.id}/semantic-profile",
            headers=self.headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert "dataset" in data
        assert data["dataset"]["row_count"] == 50
        assert "quality_score" in data["dataset"]
        assert "columns" in data
        assert len(data["columns"]) == 4

        # Verify annual_revenue column profile
        rev_col = next(c for c in data["columns"] if c["name"] == "annual_revenue")
        assert rev_col["role"] == "measure"
        assert rev_col["business_role"] == "revenue"
        assert rev_col["confidence"] >= 0.8
        assert len(rev_col["evidence"]) > 0

        # Verify target candidate
        assert len(data["potential_targets"]) >= 1
        target = data["potential_targets"][0]
        assert target["column_name"] == "customer_churn"

    def test_post_trigger_profile_endpoint(self):
        """POST /api/datasets/{id}/profile forces re-profiling."""
        res = self.client.post(
            f"/api/datasets/{self.dataset.id}/profile",
            headers=self.headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert data["n_rows"] == 50
        assert "quality_score" in data

    def test_patch_column_semantic_with_business_role(self):
        """PATCH /api/datasets/{id}/columns/{col}/semantic persists business_role and user overrides."""
        res = self.client.patch(
            f"/api/datasets/{self.dataset.id}/columns/annual_revenue/semantic",
            headers=self.headers,
            json={
                "role": "measure",
                "semantic_type": "currency",
                "business_role": "profit",
                "notes": "User designated this column as operating profit",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["column"] == "annual_revenue"
        updated = data["updated"]
        assert updated["role"] == "measure"
        assert updated["semantic_type"] == "currency"
        assert updated["business_role"] == "profit"
        assert updated["is_user_corrected"] is True
        assert updated["confidence"] == 1.0
        assert updated["confidence_type"] == "user_assigned"
