"""Automated tests for Phase 1: Semantic Data Understanding.

Validates:
1. Primary and surrogate identifiers (and low-cardinality ID exception)
2. Quantitative measures and metrics (including Superstore continuous sales)
3. Dates, timestamps, and epoch integers
4. Booleans and binary indicator flags
5. Geographic fields (countries, states, zip codes, coordinates)
6. Categorical dimensions and unstructured text
7. Ambiguous/unknown columns with calibrated low confidence
8. Explainable evidence generation
9. User corrections and persistence across re-profiling runs
10. Non-destructive raw data preservation and backward compatibility
"""

import numpy as np
import pandas as pd
import pytest

from app.ml.dataset_profiler import enrich_column_metadata, profile_dataframe
from app.ml.semantic_inferencer import (
    SemanticRole,
    SemanticType,
    infer_column_semantics,
)


class TestSemanticInferenceUnit:
    """Unit tests for semantic understanding inference rules."""

    def test_uuid_identifier(self):
        uuids = [
            "123e4567-e89b-12d3-a456-426614174000",
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
            "fa99d21e-1d37-4d98-89c7-5bc148f3e27f",
            "7e4d8e5e-6b9d-4e92-9599-232152341255",
            "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        ] * 4
        s = pd.Series(uuids)
        ann = infer_column_semantics("user_token", s)
        assert ann.semantic_type == SemanticType.ID.value
        assert ann.role == SemanticRole.IDENTIFIER.value
        assert ann.confidence >= 0.95
        assert any("UUID" in ev for ev in ann.evidence)

    def test_sequential_surrogate_key_identifier(self):
        s = pd.Series(list(range(1, 101)))
        ann = infer_column_semantics("row_idx", s)
        assert ann.semantic_type == SemanticType.ID.value
        assert ann.role == SemanticRole.IDENTIFIER.value
        assert ann.confidence >= 0.85
        assert any("Sequential" in ev for ev in ann.evidence)

    def test_low_cardinality_id_classified_as_dimension_not_identifier(self):
        # E.g. 'category_id' with only 4 categories is a dimension, NOT a row identifier
        s = pd.Series(["cat_1", "cat_2", "cat_3", "cat_4"] * 25)
        ann = infer_column_semantics("category_id", s)
        assert ann.role == SemanticRole.DIMENSION.value
        assert ann.semantic_type == SemanticType.CATEGORICAL.value
        assert ann.confidence >= 0.85

    def test_continuous_measure_with_near_unique_sales(self):
        # Superstore sales: near-unique continuous numbers must stay a measure, not an ID
        sales = [100.50 + i * 1.37 for i in range(100)]
        s = pd.Series(sales)
        ann = infer_column_semantics("Store_Sales", s)
        assert ann.semantic_type == SemanticType.NUMERIC_MEASURE.value
        assert ann.role == SemanticRole.MEASURE.value
        assert ann.confidence >= 0.90
        assert any("metric" in ev.lower() or "numeric" in ev.lower() for ev in ann.evidence)

    def test_financial_and_metric_names(self):
        s = pd.Series(np.random.uniform(10.0, 500.0, size=50))
        for col_name in ["monthly_charges", "total_revenue", "salary", "account_balance"]:
            ann = infer_column_semantics(col_name, s)
            assert ann.semantic_type == SemanticType.NUMERIC_MEASURE.value
            assert ann.role == SemanticRole.MEASURE.value
            assert ann.confidence >= 0.90

    def test_iso_datetime_strings(self):
        dates = [f"2023-01-{i:02d} 14:30:00" for i in range(1, 29)]
        s = pd.Series(dates)
        ann = infer_column_semantics("created_at", s)
        assert ann.semantic_type == SemanticType.DATETIME.value
        assert ann.role == SemanticRole.TIMESTAMP.value
        assert ann.confidence >= 0.90
        assert any("date" in ev.lower() for ev in ann.evidence)

    def test_slash_dates(self):
        dates = [f"05/{i:02d}/2022" for i in range(1, 25)]
        s = pd.Series(dates)
        ann = infer_column_semantics("Order Date", s)
        assert ann.semantic_type == SemanticType.DATETIME.value
        assert ann.role == SemanticRole.TIMESTAMP.value
        assert ann.confidence >= 0.90

    def test_native_datetime_series(self):
        s = pd.Series(pd.date_range("2023-01-01", periods=30, freq="D"))
        ann = infer_column_semantics("event_time", s)
        assert ann.semantic_type == SemanticType.DATETIME.value
        assert ann.role == SemanticRole.TIMESTAMP.value
        assert ann.confidence >= 0.98

    def test_epoch_timestamp(self):
        # Seconds ~ 1.7e9 (current decade)
        epochs = [1700000000 + i * 3600 for i in range(20)]
        s = pd.Series(epochs)
        ann = infer_column_semantics("event_timestamp", s)
        assert ann.semantic_type == SemanticType.DATETIME.value
        assert ann.role == SemanticRole.TIMESTAMP.value
        assert ann.confidence >= 0.85

    def test_boolean_yes_no(self):
        s = pd.Series(["Yes", "No", "Yes", "Yes", "No"] * 10)
        ann = infer_column_semantics("churn", s)
        assert ann.semantic_type == SemanticType.BOOLEAN.value
        assert ann.role == SemanticRole.DIMENSION.value
        assert ann.confidence >= 0.90

    def test_boolean_binary_flags_with_is_prefix(self):
        s = pd.Series([1, 0, 1, 1, 0] * 10)
        ann = infer_column_semantics("is_active_member", s)
        assert ann.semantic_type == SemanticType.BOOLEAN.value
        assert ann.role == SemanticRole.DIMENSION.value
        assert ann.confidence >= 0.90

    def test_geographic_country_names_and_codes(self):
        s1 = pd.Series(["US", "CA", "GB", "DE", "FR"] * 10)
        ann1 = infer_column_semantics("country_code", s1)
        assert ann1.semantic_type == SemanticType.GEOGRAPHIC.value
        assert ann1.role == SemanticRole.DIMENSION.value
        assert ann1.confidence >= 0.90

        s2 = pd.Series(["United States", "Canada", "Germany", "India"] * 10)
        ann2 = infer_column_semantics("country", s2)
        assert ann2.semantic_type == SemanticType.GEOGRAPHIC.value
        assert ann2.role == SemanticRole.DIMENSION.value

    def test_geographic_coordinates(self):
        lats = pd.Series([37.7749, 40.7128, 34.0522, 51.5074] * 5)
        ann_lat = infer_column_semantics("latitude", lats)
        assert ann_lat.semantic_type == SemanticType.GEOGRAPHIC.value
        assert ann_lat.role == SemanticRole.DIMENSION.value
        assert ann_lat.confidence >= 0.95

        lons = pd.Series([-122.4194, -74.0060, -118.2437, -0.1278] * 5)
        ann_lon = infer_column_semantics("longitude", lons)
        assert ann_lon.semantic_type == SemanticType.GEOGRAPHIC.value
        assert ann_lon.role == SemanticRole.DIMENSION.value
        assert ann_lon.confidence >= 0.95

    def test_geographic_zip_codes(self):
        zips = pd.Series(["90210", "10001", "94105", "30301", "60601"] * 5)
        ann = infer_column_semantics("zip_code", zips)
        assert ann.semantic_type == SemanticType.GEOGRAPHIC.value
        assert ann.role == SemanticRole.DIMENSION.value
        assert ann.confidence >= 0.90

    def test_unstructured_text_descriptions(self):
        texts = [
            "Customer contacted support regarding unexpected roaming charges on their monthly invoice.",
            "Account upgraded to enterprise tier following successful pilot and technical review.",
            "User reported intermittent latency issues during peak business hours in Europe.",
        ] * 10
        s = pd.Series(texts)
        ann = infer_column_semantics("feedback_notes", s)
        assert ann.semantic_type == SemanticType.TEXT.value
        assert ann.role == SemanticRole.DIMENSION.value
        assert ann.confidence >= 0.85

    def test_categorical_dimensions(self):
        s = pd.Series(["Month-to-month", "One year", "Two year"] * 20)
        ann = infer_column_semantics("contract_type", s)
        assert ann.semantic_type == SemanticType.CATEGORICAL.value
        assert ann.role == SemanticRole.DIMENSION.value
        assert ann.confidence >= 0.90

    def test_ambiguous_column_remains_unknown_with_low_confidence(self):
        # Arbitrary random strings without clear date, ID, or geo structure
        vals = ["xq9", "m2b", "7pk", "zr4", "99a", "pl0"] * 10
        s = pd.Series(vals)
        ann = infer_column_semantics("mystery_column", s)
        # Should be kept unknown rather than confidently misclassified
        assert ann.semantic_type in (SemanticType.UNKNOWN.value, SemanticType.CATEGORICAL.value)
        if ann.semantic_type == SemanticType.UNKNOWN.value:
            assert ann.role == SemanticRole.UNKNOWN.value
            assert ann.confidence <= 0.50
            assert any("ambiguous" in ev.lower() or "no dominant" in ev.lower() for ev in ann.evidence)

    def test_all_null_or_constant_remains_unknown(self):
        s = pd.Series([None] * 50)
        ann = infer_column_semantics("empty_col", s)
        assert ann.semantic_type == SemanticType.UNKNOWN.value
        assert ann.role == SemanticRole.UNKNOWN.value
        assert ann.confidence <= 0.20

    def test_user_correction_takes_precedence_with_certainty(self):
        # Inferred naturally as numeric measure
        s = pd.Series([101, 102, 103, 104, 105] * 10)
        override = {
            "semantic_type": SemanticType.CATEGORICAL.value,
            "role": SemanticRole.DIMENSION.value,
            "notes": "Department business unit codes, not numbers",
        }
        ann = infer_column_semantics("dept_code", s, user_override=override)
        assert ann.is_user_corrected is True
        assert ann.semantic_type == SemanticType.CATEGORICAL.value
        assert ann.role == SemanticRole.DIMENSION.value
        assert ann.confidence == 1.0
        assert any("User override" in ev for ev in ann.evidence)


class TestSemanticProfilingIntegration:
    """End-to-end integration tests using profile_dataframe and metadata enrichment."""

    @pytest.fixture
    def sample_customer_df(self):
        n = 100
        return pd.DataFrame({
            "customer_id": [f"CUST-{1000 + i}" for i in range(n)],
            "signup_date": [f"2023-01-{(i % 28) + 1:02d}" for i in range(n)],
            "country": ["US", "CA", "GB", "DE"] * (n // 4),
            "monthly_charges": [29.99 + (i * 0.75) for i in range(n)],
            "contract": ["month-to-month", "one_year", "two_year"] * (n // 3) + ["month-to-month"],
            "is_churned": [0, 1] * (n // 2),
            "support_notes": ["Regular routine check-in with client account manager."] * n,
        })

    def test_profile_dataframe_attaches_semantic_annotations(self, sample_customer_df):
        report = profile_dataframe(sample_customer_df)
        cmap = report.column_map()

        # Customer ID
        assert cmap["customer_id"].semantic_type == SemanticType.ID.value
        assert cmap["customer_id"].role == SemanticRole.IDENTIFIER.value
        assert cmap["customer_id"].confidence >= 0.90

        # Signup Date
        assert cmap["signup_date"].semantic_type == SemanticType.DATETIME.value
        assert cmap["signup_date"].role == SemanticRole.TIMESTAMP.value

        # Country
        assert cmap["country"].semantic_type == SemanticType.GEOGRAPHIC.value
        assert cmap["country"].role == SemanticRole.DIMENSION.value

        # Monthly charges
        assert cmap["monthly_charges"].semantic_type == SemanticType.NUMERIC_MEASURE.value
        assert cmap["monthly_charges"].role == SemanticRole.MEASURE.value

        # Contract
        assert cmap["contract"].semantic_type == SemanticType.CATEGORICAL.value
        assert cmap["contract"].role == SemanticRole.DIMENSION.value

        # Is churned
        assert cmap["is_churned"].semantic_type == SemanticType.BOOLEAN.value
        assert cmap["is_churned"].role == SemanticRole.DIMENSION.value

    def test_user_overrides_persist_across_reprofiling(self, sample_customer_df):
        user_overrides = {
            "country": {
                "semantic_type": SemanticType.CATEGORICAL.value,
                "role": SemanticRole.DIMENSION.value,
                "notes": "Treat country purely as generic category",
            }
        }
        report = profile_dataframe(sample_customer_df, user_overrides=user_overrides)
        cmap = report.column_map()

        country_prof = cmap["country"]
        assert country_prof.is_user_corrected is True
        assert country_prof.semantic_type == SemanticType.CATEGORICAL.value
        assert country_prof.role == SemanticRole.DIMENSION.value
        assert country_prof.confidence == 1.0

        # Verify serializability in report.to_dict()
        rep_dict = report.to_dict()
        assert "user_overrides" in rep_dict
        assert "country" in rep_dict["user_overrides"]

    def test_enrich_column_metadata_includes_semantic_fields(self, sample_customer_df):
        report = profile_dataframe(sample_customer_df)
        base_columns = [
            {"name": col, "dtype": str(sample_customer_df[col].dtype)}
            for col in sample_customer_df.columns
        ]
        enriched = enrich_column_metadata(base_columns, report)

        for col in enriched:
            assert "semantic_type" in col
            assert "role" in col
            assert "confidence" in col
            assert "evidence" in col
            assert "is_user_corrected" in col
            assert isinstance(col["evidence"], list)
            assert 0.0 <= col["confidence"] <= 1.0

    def test_raw_data_remains_unmutated(self, sample_customer_df):
        original_df = sample_customer_df.copy(deep=True)
        report = profile_dataframe(sample_customer_df)
        pd.testing.assert_frame_equal(sample_customer_df, original_df)


class TestSemanticApiEndpoints:
    """Integration tests verifying HTTP endpoints, input validation, and persistence."""

    @pytest.fixture(autouse=True)
    def setup_api(self, tmp_path):
        import hashlib
        import os
        from fastapi.testclient import TestClient
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from app.database import Base, get_db
        from app.db.models import Dataset, Organization, User
        from app.main import app
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

        self.org = Organization(name="Semantic Test Org", slug="semantic-test-org")
        self.db.add(self.org)
        self.db.flush()

        self.user = User(
            organization_id=self.org.id,
            email="test@example.com",
            password_hash="pw",
            name="Test User",
            role="admin",
        )
        self.db.add(self.user)
        self.db.flush()

        df = pd.DataFrame({
            "cust_id": ["C1", "C2", "C3", "C4", "C5"],
            "spend": [10.0, 20.0, 30.0, 40.0, 50.0],
            "city": ["NYC", "LA", "SF", "CHI", "MIA"],
        })
        self.parquet_path = os.path.join(self.tmp_dir, "sample.parquet")
        df.to_parquet(self.parquet_path, index=False)

        with open(self.parquet_path, "rb") as f:
            self.baseline_sha256 = hashlib.sha256(f.read()).hexdigest()

        report = profile_dataframe(df)
        base_columns = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]
        columns = enrich_column_metadata(base_columns, report)

        self.dataset = Dataset(
            organization_id=self.org.id,
            name="Sample Dataset",
            file_path=self.parquet_path,
            file_size=os.path.getsize(self.parquet_path),
            row_count=len(df),
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

        yield

        app.dependency_overrides.clear()
        self.db.close()
        self.engine.dispose()

    def test_patch_single_column_semantic_success(self):
        import hashlib

        url = f"/api/datasets/{self.dataset.id}/columns/spend/semantic"
        res = self.client.patch(
            url,
            json={
                "semantic_type": "numeric_measure",
                "role": "measure",
                "notes": "Verified revenue metric",
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["column"] == "spend"
        updated = data["updated"]
        assert updated["semantic_type"] == "numeric_measure"
        assert updated["role"] == "measure"
        assert updated["is_user_corrected"] is True
        assert updated["confidence"] == 1.0

        # Verify raw data file was NOT modified
        with open(self.parquet_path, "rb") as f:
            current_sha256 = hashlib.sha256(f.read()).hexdigest()
        assert current_sha256 == self.baseline_sha256

    def test_patch_invalid_payload_rejected_at_boundary(self):
        url = f"/api/datasets/{self.dataset.id}/columns/spend/semantic"
        # Invalid role should be rejected with 422 by Pydantic validation
        res = self.client.patch(
            url,
            json={
                "semantic_type": "numeric_measure",
                "role": "invalid_superhero_role",
            },
        )
        assert res.status_code == 422

    def test_patch_nonexistent_dataset_returns_404(self):
        url = "/api/datasets/00000000-0000-0000-0000-000000000000/columns/spend/semantic"
        res = self.client.patch(
            url,
            json={"semantic_type": "numeric_measure", "role": "measure"},
        )
        assert res.status_code == 404

    def test_patch_nonexistent_column_returns_400(self):
        url = f"/api/datasets/{self.dataset.id}/columns/nonexistent_col/semantic"
        res = self.client.patch(
            url,
            json={"semantic_type": "numeric_measure", "role": "measure"},
        )
        assert res.status_code == 400
        assert "not found" in res.json()["detail"].lower()

    def test_batch_put_semantics_success(self):
        url = f"/api/datasets/{self.dataset.id}/semantics"
        res = self.client.put(
            url,
            json={
                "corrections": {
                    "spend": {"semantic_type": "numeric_measure", "role": "measure"},
                    "city": {"semantic_type": "geographic", "role": "dimension"},
                }
            },
        )
        assert res.status_code == 200, res.text
        columns = res.json()["columns"]
        spend_col = next(c for c in columns if c["name"] == "spend")
        city_col = next(c for c in columns if c["name"] == "city")
        assert spend_col["is_user_corrected"] is True
        assert city_col["is_user_corrected"] is True
        assert city_col["semantic_type"] == "geographic"

    def test_batch_put_nonexistent_column_returns_400(self):
        url = f"/api/datasets/{self.dataset.id}/semantics"
        res = self.client.put(
            url,
            json={
                "corrections": {
                    "bad_column_name": {"semantic_type": "id", "role": "identifier"},
                }
            },
        )
        assert res.status_code == 400
        assert "not found" in res.json()["detail"].lower()

    def test_persistence_across_restart_and_reprofiling(self):
        import os
        from app.ml.dataset_profiler import profile_path_for_parquet
        from app.services.dataset_service import DatasetService

        # Apply correction via service
        service = DatasetService(self.db, self.org.id)
        service.update_column_semantic(
            self.dataset.id,
            column_name="city",
            semantic_type="geographic",
            role="dimension",
            notes="Business HQ metro area",
        )

        # Simulate restart: delete profile sidecar cache on disk
        sidecar = profile_path_for_parquet(self.dataset.file_path)
        if os.path.exists(sidecar):
            os.remove(sidecar)

        # Create brand-new service instance with new session
        new_db = self.Session()
        try:
            fresh_service = DatasetService(new_db, self.org.id)
            recomputed = fresh_service.get_profile(self.dataset.id)
            assert recomputed is not None

            # Verify city override was preserved even after sidecar deletion
            city_entry = next(c for c in recomputed["columns"] if c["column_name"] == "city")
            assert city_entry["is_user_corrected"] is True
            assert city_entry["semantic_type"] == "geographic"
            assert city_entry["role"] == "dimension"
            assert city_entry["confidence"] == 1.0

            # Verify sidecar was recreated with user_overrides
            assert os.path.exists(sidecar)
            assert "city" in recomputed.get("user_overrides", {})
        finally:
            new_db.close()


class TestSemanticPrivacyAndHeuristicLabeling:
    """Tests verifying evidence privacy hardening and heuristic confidence labeling."""

    def test_evidence_excludes_raw_data_sample_values(self):
        # 1. Confidential categories must not appear in evidence
        secret_cat1 = "TopSecretDiagnosisAlpha"
        secret_cat2 = "RestrictedInternalCodeBeta"
        s_cat = pd.Series([secret_cat1, secret_cat2] * 20)
        ann_cat = infer_column_semantics("diagnosis_dept", s_cat)
        assert ann_cat.semantic_type == SemanticType.CATEGORICAL.value
        for ev in ann_cat.evidence:
            assert secret_cat1 not in ev
            assert secret_cat2 not in ev
        assert any("discrete" in ev.lower() or "categories" in ev.lower() for ev in ann_cat.evidence)

        # 2. Raw UUID values must not appear in evidence
        secret_uuid = "e3b0c442-98fc-1c14-9afb-f4c59620740e"
        s_uuid = pd.Series([secret_uuid, "c56a4180-65aa-42ec-a945-5fd21dec75d2"] * 10)
        ann_uuid = infer_column_semantics("secret_token", s_uuid)
        assert ann_uuid.semantic_type == SemanticType.ID.value
        for ev in ann_uuid.evidence:
            assert secret_uuid not in ev
        assert any("UUID" in ev for ev in ann_uuid.evidence)

        # 3. Raw date strings must not appear in evidence
        raw_date = "2024-03-15 09:30:00"
        s_date = pd.Series([raw_date, "2024-03-16 10:45:00"] * 10)
        ann_dt = infer_column_semantics("timestamp_col", s_date)
        for ev in ann_dt.evidence:
            assert raw_date not in ev
        assert any("date" in ev.lower() or "timestamp" in ev.lower() for ev in ann_dt.evidence)

        # 4. Raw ZIP codes must not appear in evidence
        raw_zip = "90210"
        s_zip = pd.Series([raw_zip, "94105", "10001"] * 10)
        ann_zip = infer_column_semantics("postal_code", s_zip)
        for ev in ann_zip.evidence:
            assert raw_zip not in ev

    def test_confidence_labeled_as_heuristic_score(self):
        s = pd.Series(["Active", "Inactive"] * 25)
        ann = infer_column_semantics("status_flag", s)
        assert ann.confidence_type == "heuristic"
        assert 0.0 <= ann.confidence <= 1.0

        # User override is explicitly user_assigned
        override = {"semantic_type": "categorical", "role": "dimension", "notes": "manual"}
        ann_override = infer_column_semantics("status_flag", s, user_override=override)
        assert ann_override.confidence_type == "user_assigned"
        assert ann_override.confidence == 1.0

    def test_column_profile_and_enrichment_propagate_confidence_type(self):
        df = pd.DataFrame({
            "code": ["A1", "A2", "A3", "A4", "A5"] * 10,
            "val": [1.5, 2.5, 3.5, 4.5, 5.5] * 10,
        })
        report = profile_dataframe(df)
        col_map = report.column_map()
        assert col_map["code"].confidence_type == "heuristic"
        assert col_map["val"].confidence_type == "heuristic"

        base_cols = [{"name": "code", "dtype": "object"}, {"name": "val", "dtype": "float64"}]
        enriched = enrich_column_metadata(base_cols, report)
        for c in enriched:
            assert c["confidence_type"] == "heuristic"
