"""B3 scheduled recheck sweep (service-level; no Redis required)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.db.models import Dataset, Decision, Organization, Project
from app.services.decision_service import DecisionService, flag_due_rechecks


def _now() -> datetime:
    return datetime.utcnow()


class RecheckFixture:
    def __init__(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db = self.Session()
        self.org = Organization(name="Recheck Org", slug="recheck-org")
        self.org_b = Organization(name="Other Org", slug="recheck-other")
        self.db.add_all([self.org, self.org_b])
        self.db.flush()
        self.ds = Dataset(
            organization_id=self.org.id,
            name="DS",
            file_path="/tmp/r.csv",
            file_size=1,
            row_count=1,
            columns=[{"name": "Churn"}],
        )
        self.ds_b = Dataset(
            organization_id=self.org_b.id,
            name="DSB",
            file_path="/tmp/rb.csv",
            file_size=1,
            row_count=1,
            columns=[{"name": "Churn"}],
        )
        self.db.add_all([self.ds, self.ds_b])
        self.db.flush()
        self.proj = Project(
            organization_id=self.org.id,
            dataset_id=self.ds.id,
            name="P",
            target_column="Churn",
            target_description="churn",
            feature_columns=["tenure"],
            status="ready",
        )
        self.proj_b = Project(
            organization_id=self.org_b.id,
            dataset_id=self.ds_b.id,
            name="PB",
            target_column="Churn",
            target_description="churn",
            feature_columns=["tenure"],
            status="ready",
        )
        self.db.add_all([self.proj, self.proj_b])
        self.db.commit()
        self.svc = DecisionService(self.db, self.org.id)

    def close(self):
        self.db.close()
        self.engine.dispose()

    def add(
        self,
        project: Project,
        *,
        name: str,
        status: str = "committed",
        recheck_at: Optional[datetime] = None,
    ) -> Decision:
        d = Decision(
            organization_id=project.organization_id,
            project_id=project.id,
            status=status,
            action_code="engagement_campaign",
            action_name=name,
            recheck_interval_days=30,
            recheck_at=recheck_at,
            committed_at=_now() - timedelta(days=10),
            case_snapshot={},
        )
        self.db.add(d)
        self.db.commit()
        self.db.refresh(d)
        return d


class TestFlagDueRechecks(unittest.TestCase):
    def setUp(self):
        self.fx = RecheckFixture()

    def tearDown(self):
        self.fx.close()

    def test_flags_due_committed(self):
        due = self.fx.add(
            self.fx.proj,
            name="Due One",
            recheck_at=_now() - timedelta(days=2),
        )
        future = self.fx.add(
            self.fx.proj,
            name="Future",
            recheck_at=_now() + timedelta(days=10),
        )
        out = self.fx.svc.flag_due_rechecks()
        self.assertEqual(out["layer"], "B3_scheduled_rechecks")
        self.assertEqual(out["flagged"], 1)
        self.assertIn(due.id, out["decision_ids"])
        self.fx.db.refresh(due)
        self.fx.db.refresh(future)
        self.assertEqual(due.status, "checking")
        self.assertEqual(future.status, "committed")
        self.assertIn("Scheduled recheck due", due.outcome_notes or "")

    def test_idempotent_second_run(self):
        d = self.fx.add(
            self.fx.proj,
            name="Once",
            recheck_at=_now() - timedelta(hours=5),
        )
        first = self.fx.svc.flag_due_rechecks()
        second = self.fx.svc.flag_due_rechecks()
        self.assertEqual(first["flagged"], 1)
        self.assertEqual(second["flagged"], 0)
        self.assertEqual(second["already_checking"], 1)
        self.fx.db.refresh(d)
        # note stamped once
        self.assertEqual((d.outcome_notes or "").count("Scheduled recheck due"), 1)

    def test_null_recheck_treated_due(self):
        d = self.fx.add(self.fx.proj, name="No Date", recheck_at=None)
        out = self.fx.svc.flag_due_rechecks()
        self.assertEqual(out["flagged"], 1)
        self.fx.db.refresh(d)
        self.assertEqual(d.status, "checking")

    def test_skips_closed_and_cancelled(self):
        self.fx.add(
            self.fx.proj,
            name="Closed",
            status="closed",
            recheck_at=_now() - timedelta(days=1),
        )
        self.fx.add(
            self.fx.proj,
            name="Cancelled",
            status="cancelled",
            recheck_at=_now() - timedelta(days=1),
        )
        out = self.fx.svc.flag_due_rechecks()
        self.assertEqual(out["flagged"], 0)

    def test_org_isolation(self):
        a = self.fx.add(
            self.fx.proj,
            name="OrgA",
            recheck_at=_now() - timedelta(days=1),
        )
        b = self.fx.add(
            self.fx.proj_b,
            name="OrgB",
            recheck_at=_now() - timedelta(days=1),
        )
        out_a = self.fx.svc.flag_due_rechecks()
        self.assertEqual(out_a["flagged"], 1)
        self.assertIn(a.id, out_a["decision_ids"])
        self.fx.db.refresh(b)
        self.assertEqual(b.status, "committed")

        out_all = flag_due_rechecks(self.fx.db, org_id=None, limit=50)
        self.assertEqual(out_all["flagged"], 1)
        self.fx.db.refresh(b)
        self.assertEqual(b.status, "checking")


class TestHttpRecheckSweep(unittest.TestCase):
    """POST /api/projects/decisions/recheck-sweep (no Redis)."""

    def setUp(self):
        self.fx = RecheckFixture()
        from fastapi.testclient import TestClient
        from app.main import app
        from app.database import get_db
        from app.services.auth_service import AuthContext, get_auth_context

        self.app = app
        self.fx.add(
            self.fx.proj,
            name="HTTP Due",
            recheck_at=_now() - timedelta(days=1),
        )

        def _override_db():
            try:
                yield self.fx.db
            finally:
                pass

        async def _override_auth():
            return AuthContext(organization=self.fx.org, scopes=["admin"])

        self.app.dependency_overrides[get_db] = _override_db
        self.app.dependency_overrides[get_auth_context] = _override_auth
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        self.fx.close()

    def test_sweep_flags_due(self):
        res = self.client.post("/api/projects/decisions/recheck-sweep")
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["layer"], "B3_scheduled_rechecks")
        self.assertEqual(body["flagged"], 1)
        self.assertIn("plain_summary", body)

    def test_sweep_not_shadowed_by_project_id(self):
        res = self.client.post("/api/projects/decisions/recheck-sweep")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["layer"], "B3_scheduled_rechecks")

    def test_unauthenticated_rejected(self):
        from app.services.auth_service import get_auth_context

        self.app.dependency_overrides.pop(get_auth_context, None)
        res = self.client.post("/api/projects/decisions/recheck-sweep")
        self.assertIn(res.status_code, (401, 403))


if __name__ == "__main__":
    unittest.main()
