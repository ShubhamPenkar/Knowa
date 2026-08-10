"""B1 Intent onboarding — rules-first suggest-config."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.db.models import Dataset, Organization
from app.services.intent_service import IntentService


class IntentFixture:
    def __init__(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db = self.Session()
        self.tmp = tempfile.TemporaryDirectory()
        self.org = Organization(name="Intent Org", slug="intent-org")
        self.db.add(self.org)
        self.db.commit()

    def close(self):
        self.db.close()
        self.engine.dispose()
        self.tmp.cleanup()

    def add_dataset(self, name: str, df: pd.DataFrame) -> Dataset:
        path = Path(self.tmp.name) / f"{name}.parquet"
        df.to_parquet(path, index=False)
        columns = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]
        ds = Dataset(
            organization_id=self.org.id,
            name=name,
            file_path=str(path),
            file_size=path.stat().st_size,
            row_count=len(df),
            columns=columns,
        )
        self.db.add(ds)
        self.db.commit()
        self.db.refresh(ds)
        return ds


class TestIntentSuggest(unittest.TestCase):
    def setUp(self):
        self.fx = IntentFixture()
        self.svc = IntentService(self.fx.db, self.fx.org.id)

    def tearDown(self):
        self.fx.close()

    def test_telco_churn_intent(self):
        df = pd.DataFrame(
            {
                "customerID": [f"C{i}" for i in range(40)],
                "tenure": list(range(40)),
                "MonthlyCharges": [30 + i for i in range(40)],
                "Contract": ["Month-to-month", "One year"] * 20,
                "Churn": ["Yes", "No"] * 20,
            }
        )
        ds = self.fx.add_dataset("telco", df)
        out = self.svc.suggest_config(
            dataset_id=ds.id,
            problem_description="Find telecom customers likely to churn so we can retain them",
            project_name="Telco retention",
        )
        self.assertEqual(out["layer"], "B1_intent_onboarding")
        self.assertEqual(out["problem_type"], "binary_classification")
        self.assertEqual(out["target_column"], "Churn")
        self.assertEqual(out["target_description"], "churn")
        self.assertIn(out["target_positive_label"], ("Yes", "yes"))
        self.assertNotIn("customerID", out["feature_columns"])
        self.assertIn("tenure", out["feature_columns"])
        self.assertGreaterEqual(out["confidence"], 0.5)

    def test_hr_attrition_intent(self):
        df = pd.DataFrame(
            {
                "EmployeeNumber": list(range(50)),
                "OverTime": ["Yes", "No"] * 25,
                "MonthlyIncome": [3000 + i * 10 for i in range(50)],
                "JobSatisfaction": [1, 2, 3, 4] * 12 + [1, 2],
                "Attrition": ["Yes", "No"] * 25,
            }
        )
        ds = self.fx.add_dataset("hr", df)
        out = self.svc.suggest_config(
            dataset_id=ds.id,
            problem_description="Which employees are at attrition risk so HR can intervene?",
        )
        self.assertEqual(out["target_column"], "Attrition")
        self.assertEqual(out["target_description"], "attrition")
        self.assertEqual(out["problem_type"], "binary_classification")
        self.assertNotIn("EmployeeNumber", out["feature_columns"])

    def test_regression_spend_intent(self):
        df = pd.DataFrame(
            {
                "user_id": list(range(30)),
                "sessions": list(range(30)),
                "spend": [10.0 + i for i in range(30)],
            }
        )
        ds = self.fx.add_dataset("spend", df)
        out = self.svc.suggest_config(
            dataset_id=ds.id,
            problem_description="How much will each customer spend next month?",
        )
        self.assertEqual(out["problem_type"], "regression")
        self.assertEqual(out["target_column"], "spend")
        self.assertEqual(out["target_description"], "spend")
        self.assertIsNone(out["target_positive_label"])

    def test_missing_description_raises(self):
        df = pd.DataFrame({"a": [1, 2], "b": [0, 1]})
        ds = self.fx.add_dataset("tiny", df)
        with self.assertRaises(ValueError):
            self.svc.suggest_config(dataset_id=ds.id, problem_description="   ")

    def test_missing_dataset_raises(self):
        with self.assertRaises(ValueError):
            self.svc.suggest_config(
                dataset_id="does-not-exist",
                problem_description="churn risk",
            )

    def test_conversion_intent(self):
        df = pd.DataFrame(
            {
                "lead_id": list(range(40)),
                "visits": list(range(40)),
                "Converted": ["Yes", "No"] * 20,
            }
        )
        ds = self.fx.add_dataset("leads", df)
        out = self.svc.suggest_config(
            dataset_id=ds.id,
            problem_description="Predict lead conversion / signup likelihood",
        )
        self.assertEqual(out["target_description"], "conversion")
        self.assertEqual(out["target_column"], "Converted")
        self.assertNotIn("lead_id", out["feature_columns"])

    def test_off_topic_bananas_rejected(self):
        df = pd.DataFrame(
            {
                "customerID": [f"C{i}" for i in range(40)],
                "tenure": list(range(40)),
                "MonthlyCharges": [30 + i for i in range(40)],
                "Churn": ["Yes", "No"] * 20,
            }
        )
        ds = self.fx.add_dataset("telco-bananas", df)
        with self.assertRaises(ValueError) as ctx:
            self.svc.suggest_config(
                dataset_id=ds.id,
                problem_description="help me with bananas",
            )
        msg = str(ctx.exception).lower()
        self.assertIn("doesn't match", msg)
        self.assertIn("manually", msg)

    def test_gibberish_rejected_even_with_classic_target_column(self):
        df = pd.DataFrame(
            {
                "x1": list(range(20)),
                "Attrition": ["Yes", "No"] * 10,
            }
        )
        ds = self.fx.add_dataset("hr-gibberish", df)
        with self.assertRaises(ValueError):
            self.svc.suggest_config(
                dataset_id=ds.id,
                problem_description="asdf qwerty hello world",
            )

    def test_named_column_without_catalog_keyword_still_ok(self):
        """User names the column + decision language → accept without catalog hit."""
        df = pd.DataFrame(
            {
                "visits": list(range(30)),
                "Upgraded": ["Yes", "No"] * 15,
            }
        )
        ds = self.fx.add_dataset("upgrade", df)
        out = self.svc.suggest_config(
            dataset_id=ds.id,
            problem_description="Predict whether customers Upgraded based on visits",
        )
        self.assertEqual(out["target_column"], "Upgraded")
        self.assertGreaterEqual(out["confidence"], 0.35)


if __name__ == "__main__":
    unittest.main()
