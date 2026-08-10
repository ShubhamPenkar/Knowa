"""Assignees, activity log, and audit export for the decision ledger."""

from __future__ import annotations

from datetime import datetime, timedelta
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.db.models import Dataset, Decision, Organization, Project, User
from app.services.decision_service import DecisionService


def _now() -> datetime:
    return datetime.utcnow()


class AssignExportFixture:
    def __init__(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db = self.Session()

        self.org = Organization(name="Assign Org", slug="assign-org")
        self.db.add(self.org)
        self.db.flush()

        self.user_a = User(
            organization_id=self.org.id,
            email="a@example.com",
            password_hash="x",
            name="Alice",
            role="admin",
        )
        self.user_b = User(
            organization_id=self.org.id,
            email="b@example.com",
            password_hash="x",
            name="Bob",
            role="member",
        )
        self.db.add_all([self.user_a, self.user_b])
        self.db.flush()

        self.ds = Dataset(
            organization_id=self.org.id,
            name="DS",
            file_path="/tmp/a.csv",
            file_size=10,
            row_count=10,
            columns=[{"name": "Churn", "dtype": "object"}],
        )
        self.db.add(self.ds)
        self.db.flush()

        self.project = Project(
            organization_id=self.org.id,
            dataset_id=self.ds.id,
            name="Telco",
            target_column="Churn",
            target_description="churn",
            feature_columns=["tenure"],
            status="ready",
        )
        self.db.add(self.project)
        self.db.commit()

        self.svc = DecisionService(self.db, self.org.id)

    def close(self):
        self.db.close()
        self.engine.dispose()


class TestAssignAndExport(unittest.TestCase):
    def setUp(self):
        self.fx = AssignExportFixture()

    def tearDown(self):
        self.fx.close()

    def test_create_defaults_assignee_to_actor(self):
        out = self.fx.svc.create_from_case(
            self.fx.project.id,
            action_code="engagement_campaign",
            action_name="Re-engage",
            expected_lift=-0.1,
            actor_user_id=self.fx.user_a.id,
        )
        self.assertEqual(out["assignee_user_id"], self.fx.user_a.id)
        self.assertEqual(out["assignee"]["email"], "a@example.com")
        self.assertEqual(out["committed_by_user_id"], self.fx.user_a.id)
        acts = out.get("activities") or []
        # create_from_case returns without include_activity — fetch via get
        detail = self.fx.svc.get_decision(self.fx.project.id, out["id"])
        events = [a["event"] for a in detail["activities"]]
        self.assertIn("created", events)
        self.assertIn("assigned", events)

    def test_assign_and_mine_filter(self):
        created = self.fx.svc.create_from_case(
            self.fx.project.id,
            action_code="engagement_campaign",
            action_name="Mine",
            recheck_interval_days=30,
            actor_user_id=self.fx.user_a.id,
        )
        # Reassign to Bob
        assigned = self.fx.svc.assign(
            self.fx.project.id,
            created["id"],
            assignee_user_id=self.fx.user_b.id,
            actor_user_id=self.fx.user_a.id,
        )
        self.assertEqual(assigned["assignee_user_id"], self.fx.user_b.id)

        port_bob = self.fx.svc.list_portfolio(assignee_user_id=self.fx.user_b.id)
        all_ids = [
            d["id"]
            for bucket in ("overdue", "due_now", "upcoming", "closed_recent")
            for d in port_bob.get(bucket, [])
        ]
        self.assertIn(created["id"], all_ids)

        port_alice = self.fx.svc.list_portfolio(assignee_user_id=self.fx.user_a.id)
        alice_ids = [
            d["id"]
            for bucket in ("overdue", "due_now", "upcoming", "closed_recent")
            for d in port_alice.get(bucket, [])
        ]
        self.assertNotIn(created["id"], alice_ids)

    def test_check_in_writes_activity(self):
        created = self.fx.svc.create_from_case(
            self.fx.project.id,
            action_code="discount_10",
            action_name="Discount",
            actor_user_id=self.fx.user_a.id,
        )
        # Force due
        d = self.fx.db.query(Decision).filter(Decision.id == created["id"]).one()
        d.recheck_at = _now() - timedelta(days=2)
        self.fx.db.commit()

        out = self.fx.svc.check_in(
            self.fx.project.id,
            created["id"],
            actual_outcome="no",
            notes="Worked",
            close=True,
            actor_user_id=self.fx.user_b.id,
        )
        self.assertEqual(out["status"], "closed")
        events = [a["event"] for a in out["activities"]]
        self.assertIn("closed", events)

    def test_export_json_and_csv(self):
        self.fx.svc.create_from_case(
            self.fx.project.id,
            action_code="engagement_campaign",
            action_name="Export me",
            actor_user_id=self.fx.user_a.id,
        )
        payload = self.fx.svc.export_decisions(include_activity=True)
        self.assertEqual(payload["layer"], "decision_export")
        self.assertGreaterEqual(payload["n"], 1)
        self.assertTrue(payload["decisions"][0].get("activities"))

        csv_text = self.fx.svc.export_csv()
        self.assertIn("action_name", csv_text)
        self.assertIn("Export me", csv_text)

    def test_reject_foreign_assignee(self):
        other = Organization(name="Other", slug="other-org")
        self.fx.db.add(other)
        self.fx.db.flush()
        foreign = User(
            organization_id=other.id,
            email="x@other.com",
            password_hash="x",
            name="X",
        )
        self.fx.db.add(foreign)
        self.fx.db.commit()

        created = self.fx.svc.create_from_case(
            self.fx.project.id,
            action_code="engagement_campaign",
            action_name="No foreign",
            actor_user_id=self.fx.user_a.id,
        )
        with self.assertRaises(ValueError):
            self.fx.svc.assign(
                self.fx.project.id,
                created["id"],
                assignee_user_id=foreign.id,
                actor_user_id=self.fx.user_a.id,
            )

    def test_http_export_members_assign_not_shadowed(self):
        from fastapi.testclient import TestClient

        from app.main import app
        from app.database import get_db
        from app.services.auth_service import AuthContext, get_auth_context

        created = self.fx.svc.create_from_case(
            self.fx.project.id,
            action_code="engagement_campaign",
            action_name="HTTP Export",
            actor_user_id=self.fx.user_a.id,
        )

        def _db():
            try:
                yield self.fx.db
            finally:
                pass

        async def _auth():
            return AuthContext(
                organization=self.fx.org, user=self.fx.user_a, scopes=["admin"]
            )

        app.dependency_overrides[get_db] = _db
        app.dependency_overrides[get_auth_context] = _auth
        try:
            client = TestClient(app)
            res = client.get("/api/projects/decisions/export?format=json")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["layer"], "decision_export")

            res = client.get("/api/projects/decisions/export?format=csv")
            self.assertEqual(res.status_code, 200)
            self.assertIn("action_name", res.text)

            res = client.get("/api/auth/members")
            self.assertEqual(res.status_code, 200)
            self.assertGreaterEqual(res.json()["n"], 1)

            res = client.get("/api/projects/decisions/portfolio?mine=true")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["assignee_user_id"], self.fx.user_a.id)

            res = client.patch(
                f"/api/projects/{self.fx.project.id}/decisions/{created['id']}/assign",
                json={"assignee_user_id": self.fx.user_b.id},
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["assignee_user_id"], self.fx.user_b.id)

            res = client.get(
                f"/api/projects/{self.fx.project.id}/decisions/{created['id']}/activity"
            )
            self.assertEqual(res.status_code, 200)
            self.assertGreaterEqual(res.json()["n"], 1)
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
