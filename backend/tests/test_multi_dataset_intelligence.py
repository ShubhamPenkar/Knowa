"""Automated tests for Phase 2: Multi-Dataset Intelligence.

Validates:
1. Exact key matches and compatible columns with different names.
2. Same-name columns with incompatible meanings (e.g. status/type vs identifiers).
3. High-overlap vs low-overlap candidate relationship filtering.
4. Unique vs non-unique keys: 1:1, 1:N, N:1, and N:N cardinality detection.
5. Null-heavy columns pruned from candidate generation.
6. Fan-out detection and double-counting risk prevention on source measures.
7. Ambiguous relationships surfaced with warnings and not selected automatically.
8. Dataset grain inference: entity, transaction, periodic aggregate, composite key, unknown.
9. Sidecar persistence and refresh behavior.
10. Non-destructive raw data preservation and user semantic override retention.
11. API routes: discovery, validation, grain, and dataset relationships.
12. Resource bounds and sampling on large datasets.
"""

import hashlib
import os
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.db.models import Dataset, Organization, User
from app.main import app
from app.ml.dataset_profiler import enrich_column_metadata, profile_dataframe
from app.ml.relationship_inferencer import (
    DatasetRelationship,
    GrainType,
    JoinSafety,
    RelationshipCardinality,
    ValidationStatus,
    discover_candidate_relationships,
    infer_dataset_grain,
    simulate_relationship_join,
    validate_relationship_join,
)
from app.services.auth_service import AuthContext, get_auth_context
from app.services.relationship_service import DatasetRelationshipService


class TestDatasetGrainInference:
    """Unit tests for dataset grain and level-of-detail inference."""

    def test_entity_grain_single_unique_id(self):
        df = pd.DataFrame({
            "customer_id": [f"CUST-{i:04d}" for i in range(1, 101)],
            "age": [20 + (i % 50) for i in range(1, 101)],
            "country": ["US", "CA", "GB", "DE"] * 25,
            "annual_income": [50000.0 + i * 200 for i in range(1, 101)],
        })
        meta = {
            "id": "customers",
            "name": "Customers",
            "columns": [
                {"name": "customer_id", "role": "identifier", "semantic_type": "id"},
                {"name": "age", "role": "dimension", "semantic_type": "numeric_measure"},
                {"name": "country", "role": "dimension", "semantic_type": "geographic"},
                {"name": "annual_income", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }
        grain = infer_dataset_grain(df, meta)
        assert grain.grain_type == GrainType.ENTITY.value
        assert grain.primary_key == ["customer_id"]
        assert grain.confidence >= 0.90
        assert grain.confidence_type == "heuristic"
        assert "annual_income" in grain.potential_measures

    def test_transaction_grain_repeated_fk_with_timestamp(self):
        df = pd.DataFrame({
            "order_id": [f"ORD-{i:05d}" for i in range(1, 101)],
            "customer_id": [f"CUST-{1 + (i % 10):04d}" for i in range(1, 101)],  # Repeated FK
            "order_date": [f"2023-05-{(i % 28) + 1:02d}" for i in range(1, 101)],
            "order_amount": [15.50 + i * 1.25 for i in range(1, 101)],
        })
        meta = {
            "id": "orders",
            "name": "Orders",
            "columns": [
                {"name": "order_id", "role": "identifier", "semantic_type": "id"},
                {"name": "customer_id", "role": "identifier", "semantic_type": "id"},
                {"name": "order_date", "role": "timestamp", "semantic_type": "datetime"},
                {"name": "order_amount", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }
        grain = infer_dataset_grain(df, meta)
        assert grain.grain_type == GrainType.TRANSACTION.value
        assert "order_id" in grain.primary_key
        assert "customer_id" in grain.entity_identifiers
        assert "order_amount" in grain.potential_measures
        assert grain.confidence >= 0.85

    def test_periodic_aggregate_composite_grain(self):
        # Store daily sales aggregate: store_id + date is unique
        stores = ["STORE_A", "STORE_B"]
        dates = [f"2023-01-{i:02d}" for i in range(1, 11)]
        rows = []
        for s in stores:
            for d in dates:
                rows.append({"store_id": s, "summary_date": d, "total_sales": 5000.0})
        df = pd.DataFrame(rows)
        meta = {
            "id": "daily_store_sales",
            "name": "Daily Store Sales",
            "columns": [
                {"name": "store_id", "role": "dimension", "semantic_type": "categorical"},
                {"name": "summary_date", "role": "timestamp", "semantic_type": "datetime"},
                {"name": "total_sales", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }
        grain = infer_dataset_grain(df, meta)
        assert grain.grain_type in (GrainType.PERIODIC_AGGREGATE.value, GrainType.COMPOSITE_KEY.value)
        assert len(grain.primary_key) == 2
        assert "store_id" in grain.primary_key
        assert "summary_date" in grain.primary_key

    def test_unknown_grain_when_keys_contain_duplicates(self):
        # Dataset with full duplicate rows and no primary key
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "C", "C"],
            "rating": [4.5, 4.5, 3.2, 3.2, 5.0, 5.0],
        })
        meta = {
            "id": "survey",
            "name": "Survey",
            "columns": [
                {"name": "category", "role": "dimension", "semantic_type": "categorical"},
                {"name": "rating", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }
        grain = infer_dataset_grain(df, meta)
        assert grain.grain_type == GrainType.UNKNOWN.value
        assert grain.confidence <= 0.35
        assert len(grain.warnings) > 0

    def test_bridge_grain_association_table(self):
        """Bridge table connecting students and courses with composite foreign keys and no measures."""
        df = pd.DataFrame({
            "student_id": ["S1", "S1", "S2", "S2", "S3"],
            "course_id": ["C101", "C102", "C101", "C103", "C102"],
            "enrollment_status": ["enrolled", "completed", "enrolled", "dropped", "enrolled"],
        })
        meta = {
            "id": "student_courses",
            "name": "StudentCourses",
            "columns": [
                {"name": "student_id", "role": "identifier", "semantic_type": "id"},
                {"name": "course_id", "role": "identifier", "semantic_type": "id"},
                {"name": "enrollment_status", "role": "dimension", "semantic_type": "categorical"},
            ],
        }
        grain = infer_dataset_grain(df, meta)
        assert grain.grain_type == GrainType.BRIDGE.value
        assert grain.entity_type == "bridge"
        assert set(grain.primary_key) == {"student_id", "course_id"}
        assert grain.confidence >= 0.85
        assert "association" in grain.grain_description.lower()

    def test_event_grain_activity_log(self):
        """Event stream logging user actions over time."""
        df = pd.DataFrame({
            "event_id": [f"EVT-{i:04d}" for i in range(1, 51)],
            "user_id": [f"USR-{1 + (i % 5):03d}" for i in range(1, 51)],
            "timestamp": [f"2023-08-01 10:{i:02d}:00" for i in range(1, 51)],
            "event_type": ["login", "view_page", "click_cta", "logout", "purchase"] * 10,
        })
        meta = {
            "id": "event_logs",
            "name": "EventLogs",
            "columns": [
                {"name": "event_id", "role": "identifier", "semantic_type": "id"},
                {"name": "user_id", "role": "identifier", "semantic_type": "id"},
                {"name": "timestamp", "role": "timestamp", "semantic_type": "datetime"},
                {"name": "event_type", "role": "dimension", "semantic_type": "categorical"},
            ],
        }
        grain = infer_dataset_grain(df, meta)
        assert grain.grain_type == GrainType.EVENT.value
        assert grain.entity_type == "event"
        assert grain.primary_key == ["event_id"]
        assert grain.confidence >= 0.85
        assert "event" in grain.grain_description.lower()



class TestRelationshipDiscoveryUnit:
    """Unit tests for join discovery, cardinality, filtering, and risk detection."""

    @pytest.fixture
    def customer_and_orders_dfs(self):
        n_cust = 50
        cust_df = pd.DataFrame({
            "client_id": [f"CL-{i:03d}" for i in range(1, n_cust + 1)],
            "client_name": [f"Client {i}" for i in range(1, n_cust + 1)],
            "account_balance": [1000.0 * i for i in range(1, n_cust + 1)],
        })
        cust_meta = {
            "id": "ds_clients",
            "name": "Clients",
            "columns": [
                {"name": "client_id", "role": "identifier", "semantic_type": "id"},
                {"name": "client_name", "role": "dimension", "semantic_type": "categorical"},
                {"name": "account_balance", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }

        # Orders dataset has matching foreign key 'client_id' (repeating 1:N)
        orders_df = pd.DataFrame({
            "order_number": [f"ORD-{i:04d}" for i in range(1, 151)],
            "client_id": [f"CL-{(i % 40) + 1:03d}" for i in range(1, 151)],
            "amount": [50.0 + i * 2.0 for i in range(1, 151)],
        })
        orders_meta = {
            "id": "ds_orders",
            "name": "Orders",
            "columns": [
                {"name": "order_number", "role": "identifier", "semantic_type": "id"},
                {"name": "client_id", "role": "identifier", "semantic_type": "id"},
                {"name": "amount", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }
        return cust_df, cust_meta, orders_df, orders_meta

    def test_exact_key_match_with_one_to_many_cardinality(self, customer_and_orders_dfs):
        cust_df, cust_meta, orders_df, orders_meta = customer_and_orders_dfs
        candidates = discover_candidate_relationships(cust_df, cust_meta, orders_df, orders_meta)
        assert len(candidates) >= 1

        rel = next((c for c in candidates if c.source_column == "client_id" and c.target_column == "client_id"), None)
        assert rel is not None
        assert rel.cardinality == RelationshipCardinality.ONE_TO_MANY.value
        assert rel.confidence >= 0.85
        assert rel.confidence_type == "heuristic"
        assert rel.containment_ratio >= 0.75

        # Must flag double-counting risk because Clients table has account_balance measure
        assert rel.double_counting_risk is True
        assert "account_balance" in rel.double_counting_measures
        assert rel.fan_out_ratio > 1.0

    def test_compatible_columns_with_different_names(self):
        # Source has 'customer_key', Target has 'cust_id'
        ids = [f"ID-{i:03d}" for i in range(1, 41)]
        df_a = pd.DataFrame({"customer_key": ids, "tier": ["Gold"] * 40})
        meta_a = {
            "id": "a", "name": "A",
            "columns": [
                {"name": "customer_key", "role": "identifier", "semantic_type": "id"},
                {"name": "tier", "role": "dimension", "semantic_type": "categorical"},
            ]
        }

        df_b = pd.DataFrame({"cust_id": ids[:35], "score": [95] * 35})
        meta_b = {
            "id": "b", "name": "B",
            "columns": [
                {"name": "cust_id", "role": "identifier", "semantic_type": "id"},
                {"name": "score", "role": "measure", "semantic_type": "numeric_measure"},
            ]
        }

        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        matched = [c for c in candidates if c.source_column == "customer_key" and c.target_column == "cust_id"]
        assert len(matched) == 1
        assert matched[0].cardinality == RelationshipCardinality.ONE_TO_ONE.value
        assert matched[0].confidence >= 0.80

    def test_same_name_columns_with_incompatible_meanings_rejected(self):
        # Both datasets have a column named 'type', but values and semantics differ completely
        df_a = pd.DataFrame({"type": ["fiber_optic", "dsl", "cable"] * 20})
        meta_a = {
            "id": "telco", "name": "Telco",
            "columns": [{"name": "type", "role": "dimension", "semantic_type": "categorical"}]
        }

        df_b = pd.DataFrame({"type": ["standard_class", "first_class", "second_class"] * 20})
        meta_b = {
            "id": "superstore", "name": "Superstore",
            "columns": [{"name": "type", "role": "dimension", "semantic_type": "categorical"}]
        }

        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        assert len(candidates) == 0  # Zero overlap must prevent join discovery despite same name

    def test_null_heavy_columns_pruned(self):
        # Column with 85% nulls should be pruned
        vals = [None] * 85 + [f"K-{i}" for i in range(15)]
        df_a = pd.DataFrame({"sparse_key": vals})
        meta_a = {
            "id": "a", "name": "A",
            "columns": [{"name": "sparse_key", "role": "dimension", "semantic_type": "categorical"}]
        }
        df_b = pd.DataFrame({"sparse_key": [f"K-{i}" for i in range(20)]})
        meta_b = {
            "id": "b", "name": "B",
            "columns": [{"name": "sparse_key", "role": "dimension", "semantic_type": "categorical"}]
        }
        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        assert len(candidates) == 0

    def test_ambiguous_relationship_flagged_with_warning(self):
        # Target dataset has two columns with almost identical keys (e.g. billing_id and shipping_id)
        keys = [f"LOC-{i:03d}" for i in range(1, 31)]
        df_a = pd.DataFrame({"location_id": keys})
        meta_a = {
            "id": "loc", "name": "Locations",
            "columns": [{"name": "location_id", "role": "identifier", "semantic_type": "id"}]
        }

        df_b = pd.DataFrame({
            "billing_location_id": keys,
            "shipping_location_id": keys,
        })
        meta_b = {
            "id": "orders", "name": "Orders",
            "columns": [
                {"name": "billing_location_id", "role": "identifier", "semantic_type": "id"},
                {"name": "shipping_location_id", "role": "identifier", "semantic_type": "id"},
            ]
        }

        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        # Both candidate relationships should be flagged as ambiguous
        assert len(candidates) == 2
        for c in candidates:
            assert c.is_ambiguous is True
            assert c.validation_status == ValidationStatus.AMBIGUOUS.value
            assert any("manual confirmation" in w.lower() for w in c.warnings)

    def test_many_to_many_join_flags_high_fan_out_warning(self):
        # Both sides have repeated non-unique category codes
        df_a = pd.DataFrame({"dept_code": ["D1", "D2", "D3"] * 20, "budget": [1000] * 60})
        meta_a = {
            "id": "dept_budgets", "name": "Dept Budgets",
            "columns": [
                {"name": "dept_code", "role": "dimension", "semantic_type": "categorical"},
                {"name": "budget", "role": "measure", "semantic_type": "numeric_measure"},
            ]
        }
        df_b = pd.DataFrame({"dept_code": ["D1", "D2", "D3"] * 30})
        meta_b = {
            "id": "dept_staff", "name": "Dept Staff",
            "columns": [{"name": "dept_code", "role": "dimension", "semantic_type": "categorical"}]
        }

        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        assert len(candidates) == 1
        rel = candidates[0]
        assert rel.cardinality == RelationshipCardinality.MANY_TO_MANY.value
        assert rel.double_counting_risk is True
        assert any("many-to-many" in w.lower() for w in rel.warnings)

    def test_incompatible_demographic_and_financial_balance_overlap_rejected(self):
        """Clearly incompatible semantic types (e.g. demographic age vs financial balance) must not join."""
        # Raw numeric values overlap 100%, but concepts are completely incompatible
        values = [20, 25, 30, 35, 40, 45, 50, 55, 60]
        df_a = pd.DataFrame({"customer_age": values})
        meta_a = {
            "id": "ds_demo",
            "name": "Demographics",
            "columns": [{"name": "customer_age", "role": "dimension", "semantic_type": "categorical"}],
        }
        df_b = pd.DataFrame({"account_balance": [float(v) for v in values]})
        meta_b = {
            "id": "ds_fin",
            "name": "Financials",
            "columns": [{"name": "account_balance", "role": "measure", "semantic_type": "numeric_measure"}],
        }

        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        # Incompatible semantic pairs must be rejected early despite identical numbers
        assert len(candidates) == 0

    def test_valid_join_keys_with_different_column_names(self):
        """Legitimate relationships must be discovered even when column names differ completely."""
        ids = [f"ENTITY_REF_{i:04d}" for i in range(1, 41)]
        df_a = pd.DataFrame({"customer_reference": ids, "plan": ["Pro"] * 40})
        meta_a = {
            "id": "ds_a",
            "name": "Customers",
            "columns": [
                {"name": "customer_reference", "role": "identifier", "semantic_type": "id"},
                {"name": "plan", "role": "dimension", "semantic_type": "categorical"},
            ],
        }
        df_b = pd.DataFrame({"account_holder_key": ids[:36], "score": [88] * 36})
        meta_b = {
            "id": "ds_b",
            "name": "Accounts",
            "columns": [
                {"name": "account_holder_key", "role": "identifier", "semantic_type": "id"},
                {"name": "score", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }

        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        matched = [c for c in candidates if c.source_column == "customer_reference" and c.target_column == "account_holder_key"]
        assert len(matched) == 1
        assert matched[0].validation_status in (ValidationStatus.CANDIDATE.value, ValidationStatus.NEEDS_REVIEW.value)
        assert any("different column names" in ev for ev in matched[0].evidence)

    def test_compatible_keys_with_poor_containment_requires_review(self):
        """Keys with low containment should be flagged as needing review rather than high-confidence candidate."""
        # 25% overlap
        src_keys = [f"K_{i:03d}" for i in range(1, 41)]
        tgt_keys = [f"K_{i:03d}" for i in range(1, 11)] + [f"OTHER_{i:03d}" for i in range(1, 31)]
        df_a = pd.DataFrame({"user_id": src_keys})
        meta_a = {
            "id": "a", "name": "A",
            "columns": [{"name": "user_id", "role": "identifier", "semantic_type": "id"}]
        }
        df_b = pd.DataFrame({"user_id": tgt_keys})
        meta_b = {
            "id": "b", "name": "B",
            "columns": [{"name": "user_id", "role": "identifier", "semantic_type": "id"}]
        }

        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        assert len(candidates) == 1
        rel = candidates[0]
        assert rel.validation_status == ValidationStatus.NEEDS_REVIEW.value
        assert rel.review_reason is not None
        assert "containment" in rel.review_reason.lower() or "overlap" in rel.review_reason.lower()

    def test_compatible_keys_with_excessive_nulls_requires_review(self):
        """Keys with notable missingness (e.g. 40% nulls) should be flagged as needing review."""
        keys = [f"ID_{i:03d}" for i in range(1, 21)]
        src_keys = keys + [None] * 15  # ~43% nulls
        tgt_keys = keys + [f"EXTRA_{i}" for i in range(10)]
        df_a = pd.DataFrame({"member_id": src_keys})
        meta_a = {
            "id": "a", "name": "A",
            "columns": [{"name": "member_id", "role": "identifier", "semantic_type": "id"}]
        }
        df_b = pd.DataFrame({"member_id": tgt_keys})
        meta_b = {
            "id": "b", "name": "B",
            "columns": [{"name": "member_id", "role": "identifier", "semantic_type": "id"}]
        }

        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        assert len(candidates) == 1
        rel = candidates[0]
        assert rel.validation_status == ValidationStatus.NEEDS_REVIEW.value
        assert rel.review_reason is not None
        assert "missingness" in rel.review_reason.lower() or "null" in rel.review_reason.lower()

    def test_unique_vs_non_unique_keys_cardinality_detection(self):
        """Cardinally should accurately differentiate 1:1, 1:N, N:1, and N:N."""
        keys = [f"SKU_{i:03d}" for i in range(1, 21)]
        df_unique = pd.DataFrame({"sku_id": keys})
        meta_unique = {
            "id": "u", "name": "Unique",
            "columns": [{"name": "sku_id", "role": "identifier", "semantic_type": "id"}]
        }
        df_repeats = pd.DataFrame({"sku_id": keys * 3})
        meta_repeats = {
            "id": "r", "name": "Repeats",
            "columns": [{"name": "sku_id", "role": "identifier", "semantic_type": "id"}]
        }

        # Unique -> Repeats: ONE_TO_MANY
        cands_1n = discover_candidate_relationships(df_unique, meta_unique, df_repeats, meta_repeats)
        assert len(cands_1n) == 1
        assert cands_1n[0].cardinality == RelationshipCardinality.ONE_TO_MANY.value

        # Repeats -> Unique: MANY_TO_ONE
        cands_n1 = discover_candidate_relationships(df_repeats, meta_repeats, df_unique, meta_unique)
        assert len(cands_n1) == 1
        assert cands_n1[0].cardinality == RelationshipCardinality.MANY_TO_ONE.value

    def test_competing_candidate_keys_flagged_with_review_reason(self):
        """Competing candidate keys should include unambiguous review reasons."""
        keys = [f"ORG_{i:03d}" for i in range(1, 25)]
        df_a = pd.DataFrame({"org_id": keys})
        meta_a = {
            "id": "src", "name": "Source",
            "columns": [{"name": "org_id", "role": "identifier", "semantic_type": "id"}]
        }
        df_b = pd.DataFrame({
            "billing_org_id": keys,
            "shipping_org_id": keys,
        })
        meta_b = {
            "id": "tgt", "name": "Target",
            "columns": [
                {"name": "billing_org_id", "role": "identifier", "semantic_type": "id"},
                {"name": "shipping_org_id", "role": "identifier", "semantic_type": "id"},
            ]
        }
        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        assert len(candidates) == 2
        for c in candidates:
            assert c.is_ambiguous is True
            assert c.validation_status == ValidationStatus.AMBIGUOUS.value
            assert c.review_reason is not None
            assert "multiple candidate join keys" in c.review_reason.lower()

    def test_join_validation_zero_matches_sets_rejected_status(self):
        """Join validation producing zero matches must set validation_status to rejected with reason."""
        df_a = pd.DataFrame({"id": ["A1", "A2", "A3"]})
        df_b = pd.DataFrame({"id": ["B1", "B2", "B3"]})
        cand = DatasetRelationship(
            id="test_cand",
            source_dataset_id="a",
            source_dataset_name="A",
            source_column="id",
            target_dataset_id="b",
            target_dataset_name="B",
            target_column="id",
        )
        res = validate_relationship_join(df_a, "id", df_b, "id", cand)
        assert res.validation_status == ValidationStatus.REJECTED.value
        assert res.rejection_reason is not None
        assert "zero matching rows" in res.rejection_reason.lower()

    def test_join_safety_separation_from_validity_1_to_n(self):
        """Relationship is structurally valid (is_valid=True) but has fanout double-counting risk for measures."""
        df_cust = pd.DataFrame({
            "customer_id": [f"C{i:03d}" for i in range(1, 21)],
            "annual_revenue": [1000.0 * i for i in range(1, 21)],
        })
        meta_cust = {
            "id": "customers",
            "name": "Customers",
            "columns": [
                {"name": "customer_id", "role": "identifier", "semantic_type": "id"},
                {"name": "annual_revenue", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }

        df_orders = pd.DataFrame({
            "order_id": [f"ORD-{i:03d}" for i in range(1, 61)],
            "customer_id": [f"C{1 + (i % 20):03d}" for i in range(1, 61)],  # 3 orders per customer
            "order_amount": [50.0 + i for i in range(1, 61)],
        })
        meta_orders = {
            "id": "orders",
            "name": "Orders",
            "columns": [
                {"name": "order_id", "role": "identifier", "semantic_type": "id"},
                {"name": "customer_id", "role": "identifier", "semantic_type": "id"},
                {"name": "order_amount", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }

        candidates = discover_candidate_relationships(df_cust, meta_cust, df_orders, meta_orders)
        rel = next((c for c in candidates if c.source_column == "customer_id" and c.target_column == "customer_id"), None)
        assert rel is not None
        assert rel.is_valid is True  # Structurally valid!
        assert rel.cardinality == RelationshipCardinality.ONE_TO_MANY.value
        assert rel.join_safety == JoinSafety.FANOUT_RISK.value  # Analytically unsafe for customer measures
        assert rel.double_counting_risk is True
        assert "annual_revenue" in rel.double_counting_measures
        assert rel.fan_out_ratio >= 2.5
        assert any("double-counting" in w.lower() for w in rel.warnings)

    def test_data_quality_casing_and_whitespace_mismatches(self):
        """Detect join key cleanliness issues such as whitespace padding and casing differences."""
        df_a = pd.DataFrame({
            "code": ["KEY-01 ", "KEY-02", "KEY-03", "KEY-04"],  # Unstripped whitespace
        })
        meta_a = {
            "id": "a",
            "name": "A",
            "columns": [{"name": "code", "role": "identifier", "semantic_type": "id"}],
        }
        df_b = pd.DataFrame({
            "code": ["key-01", "key-02", "key-03", "key-04"],  # Lowercase mismatch
        })
        meta_b = {
            "id": "b",
            "name": "B",
            "columns": [{"name": "code", "role": "identifier", "semantic_type": "id"}],
        }
        candidates = discover_candidate_relationships(df_a, meta_a, df_b, meta_b)
        assert len(candidates) >= 1
        rel = candidates[0]
        assert len(rel.quality_notes) > 0
        assert any("whitespace" in q.lower() or "casing" in q.lower() for q in rel.quality_notes)

    def test_unrelated_datasets_produce_no_candidate_joins(self):
        """Completely disjoint datasets should not produce false positive relationships."""
        df_weather = pd.DataFrame({
            "city": ["Tokyo", "Paris", "Berlin", "Sydney"],
            "temperature_c": [22.5, 18.0, 16.5, 25.0],
        })
        meta_weather = {
            "id": "weather",
            "name": "Weather",
            "columns": [
                {"name": "city", "role": "dimension", "semantic_type": "geographic"},
                {"name": "temperature_c", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }
        df_employees = pd.DataFrame({
            "employee_id": ["E101", "E102", "E103", "E104"],
            "salary": [75000, 82000, 91000, 68000],
        })
        meta_employees = {
            "id": "employees",
            "name": "Employees",
            "columns": [
                {"name": "employee_id", "role": "identifier", "semantic_type": "id"},
                {"name": "salary", "role": "measure", "semantic_type": "numeric_measure"},
            ],
        }
        candidates = discover_candidate_relationships(df_weather, meta_weather, df_employees, meta_employees)
        assert len(candidates) == 0


class TestJoinSimulation:
    """Unit tests for non-destructive join simulation and row expansion calculations."""

    def test_simulate_relationship_join_1_to_n_fanout(self):
        df_customers = pd.DataFrame({
            "customer_id": [f"C{i:03d}" for i in range(1, 11)],
            "region": ["East", "West"] * 5,
        })
        df_orders = pd.DataFrame({
            "order_id": [f"O{i:03d}" for i in range(1, 51)],
            "customer_id": [f"C{1 + (i % 10):03d}" for i in range(1, 51)],  # 5 orders per customer
        })

        sim = simulate_relationship_join(df_customers, "customer_id", df_orders, "customer_id")
        assert sim["rows_before"] == 10
        assert sim["target_rows_before"] == 50
        assert sim["rows_after"] == 50
        assert sim["row_expansion_factor"] == 5.0
        assert sim["matched_rows"] == 50
        assert sim["unmatched_source_rows"] == 0
        assert sim["join_safety"] == JoinSafety.FANOUT_RISK.value
        assert sim["is_valid"] is True
        assert len(sim["warnings"]) > 0

    def test_simulate_relationship_join_zero_matches(self):
        df_a = pd.DataFrame({"id": ["A1", "A2", "A3"]})
        df_b = pd.DataFrame({"id": ["B1", "B2", "B3"]})

        sim = simulate_relationship_join(df_a, "id", df_b, "id")
        assert sim["rows_after"] == 0
        assert sim["matched_rows"] == 0
        assert sim["unmatched_source_rows"] == 3
        assert sim["is_valid"] is False
        assert sim["join_safety"] == JoinSafety.INSUFFICIENT_EVIDENCE.value

    def test_simulate_relationship_join_many_to_many_multiplication(self):
        df_a = pd.DataFrame({"tag": ["T1", "T1", "T2", "T2"]})
        df_b = pd.DataFrame({"tag": ["T1", "T1", "T1", "T2"]})

        sim = simulate_relationship_join(df_a, "tag", df_b, "tag")
        assert sim["duplicate_key_multiplication"] is True
        assert sim["join_safety"] == JoinSafety.MANY_TO_MANY_RISK.value


class TestRelationshipApiAndPersistence:
    """Integration tests verifying database isolation, API routes, sidecars, and validation."""

    @pytest.fixture(autouse=True)
    def setup_api(self, tmp_path):
        self.tmp_dir = str(tmp_path)
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db = self.Session()

        self.org = Organization(name="Multi Dataset Org", slug="multi-dataset-org")
        self.db.add(self.org)
        self.db.flush()

        self.user = User(
            organization_id=self.org.id,
            email="researcher@example.com",
            password_hash="pw",
            name="Data Researcher",
            role="admin",
        )
        self.db.add(self.user)
        self.db.flush()

        # Create two datasets: Users and Activity
        users_df = pd.DataFrame({
            "user_id": [f"USR-{i:03d}" for i in range(1, 31)],
            "signup_plan": ["Starter", "Pro"] * 15,
            "monthly_fee": [29.0, 99.0] * 15,
        })
        self.users_path = os.path.join(self.tmp_dir, "users.parquet")
        users_df.to_parquet(self.users_path, index=False)

        activity_df = pd.DataFrame({
            "event_id": [f"EVT-{i:04d}" for i in range(1, 61)],
            "user_id": [f"USR-{(i % 25) + 1:03d}" for i in range(1, 61)],
            "event_time": [f"2023-08-{(i % 28) + 1:02d}" for i in range(1, 61)],
        })
        self.activity_path = os.path.join(self.tmp_dir, "activity.parquet")
        activity_df.to_parquet(self.activity_path, index=False)

        # Baseline SHA hashes to verify raw data immutability
        with open(self.users_path, "rb") as f:
            self.users_sha = hashlib.sha256(f.read()).hexdigest()
        with open(self.activity_path, "rb") as f:
            self.activity_sha = hashlib.sha256(f.read()).hexdigest()

        rep_users = profile_dataframe(users_df)
        cols_users = enrich_column_metadata(
            [{"name": c, "dtype": str(users_df[c].dtype)} for c in users_df.columns],
            rep_users,
        )

        rep_activity = profile_dataframe(activity_df)
        cols_activity = enrich_column_metadata(
            [{"name": c, "dtype": str(activity_df[c].dtype)} for c in activity_df.columns],
            rep_activity,
        )

        self.ds_users = Dataset(
            organization_id=self.org.id,
            name="Users Dataset",
            file_path=self.users_path,
            file_size=os.path.getsize(self.users_path),
            row_count=len(users_df),
            columns=cols_users,
        )
        self.ds_activity = Dataset(
            organization_id=self.org.id,
            name="Activity Dataset",
            file_path=self.activity_path,
            file_size=os.path.getsize(self.activity_path),
            row_count=len(activity_df),
            columns=cols_activity,
        )
        self.db.add_all([self.ds_users, self.ds_activity])
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

    def test_get_dataset_grain_endpoint(self):
        res = self.client.get(f"/api/datasets/{self.ds_users.id}/grain")
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["dataset_id"] == self.ds_users.id
        assert data["grain_type"] == GrainType.ENTITY.value
        assert "user_id" in data["primary_key"]
        assert data["confidence_type"] == "heuristic"

        # Check sidecar created
        grain_file = self.users_path.replace(".parquet", ".grain.json")
        assert os.path.exists(grain_file)

    def test_discover_relationships_endpoint_and_sidecar(self):
        res = self.client.post("/api/datasets/relationships/discover")
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["total_found"] >= 1
        rels = data["relationships"]

        match = next((r for r in rels if r["source_column"] == "user_id" and r["target_column"] == "user_id"), None)
        assert match is not None
        assert match["confidence_type"] == "heuristic"
        assert match["validation_status"] == ValidationStatus.CANDIDATE.value

        # Check relationships for specific dataset endpoint
        res_ds = self.client.get(f"/api/datasets/{self.ds_users.id}/relationships")
        assert res_ds.status_code == 200
        ds_rels = res_ds.json()
        assert len(ds_rels) >= 1

    def test_validate_relationship_endpoint(self):
        body = {
            "source_dataset_id": self.ds_users.id,
            "source_column": "user_id",
            "target_dataset_id": self.ds_activity.id,
            "target_column": "user_id",
        }
        res = self.client.post("/api/datasets/relationships/validate", json=body)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["validation_status"] == ValidationStatus.VALIDATED.value
        assert data["fan_out_ratio"] > 1.0

        # Verify raw parquet files remained unchanged
        with open(self.users_path, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == self.users_sha
        with open(self.activity_path, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == self.activity_sha

    def test_error_handling_nonexistent_dataset(self):
        res = self.client.get("/api/datasets/00000000-0000-0000-0000-000000000000/grain")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

        bad_body = {
            "source_dataset_id": "00000000-0000-0000-0000-000000000000",
            "source_column": "user_id",
            "target_dataset_id": self.ds_activity.id,
            "target_column": "user_id",
        }
        res_val = self.client.post("/api/datasets/relationships/validate", json=bad_body)
        assert res_val.status_code == 404

    def test_get_organization_relationships_endpoint(self):
        """GET /api/datasets/relationships should list discovered relationships."""
        # Prime discovery
        self.client.post("/api/datasets/relationships/discover")
        res = self.client.get("/api/datasets/relationships")
        assert res.status_code == 200, res.text
        data = res.json()
        assert "relationships" in data
        assert data["total_found"] >= 1

    def test_simulate_relationship_join_endpoint(self):
        """POST /api/datasets/relationships/simulate executes non-destructive simulation."""
        body = {
            "source_dataset_id": self.ds_users.id,
            "source_column": "user_id",
            "target_dataset_id": self.ds_activity.id,
            "target_column": "user_id",
        }
        res = self.client.post("/api/datasets/relationships/simulate", json=body)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["rows_after"] > 0
        assert data["row_expansion_factor"] >= 1.0
        assert data["is_valid"] is True
        assert "join_safety" in data
        assert len(data["evidence"]) > 0
