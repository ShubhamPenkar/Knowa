"""Thorough coverage for B3 follow-up portfolio board + decision ledger edges."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.db.models import Dataset, Decision, Organization, Project
from app.services.decision_service import DecisionService


def _now() -> datetime:
    return datetime.utcnow()


class PortfolioFixture:
    def __init__(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db = self.Session()

        self.org_a = Organization(name="Org A", slug="org-a")
        self.org_b = Organization(name="Org B", slug="org-b")
        self.db.add_all([self.org_a, self.org_b])
        self.db.flush()

        self.ds_a = Dataset(
            organization_id=self.org_a.id,
            name="DS A",
            file_path="/tmp/a.csv",
            file_size=10,
            row_count=10,
            columns=[{"name": "Churn", "dtype": "object"}],
        )
        self.ds_b = Dataset(
            organization_id=self.org_b.id,
            name="DS B",
            file_path="/tmp/b.csv",
            file_size=10,
            row_count=10,
            columns=[{"name": "Churn", "dtype": "object"}],
        )
        self.db.add_all([self.ds_a, self.ds_b])
        self.db.flush()

        self.proj_a1 = Project(
            organization_id=self.org_a.id,
            dataset_id=self.ds_a.id,
            name="Telco Alpha",
            target_column="Churn",
            target_description="churn",
            feature_columns=["tenure", "MonthlyCharges"],
            status="ready",
        )
        self.proj_a2 = Project(
            organization_id=self.org_a.id,
            dataset_id=self.ds_a.id,
            name="HR Beta",
            target_column="Attrition",
            target_description="attrition",
            feature_columns=["OverTime", "JobSatisfaction"],
            status="ready",
        )
        self.proj_b1 = Project(
            organization_id=self.org_b.id,
            dataset_id=self.ds_b.id,
            name="Other Org Project",
            target_column="Churn",
            target_description="churn",
            feature_columns=["tenure"],
            status="ready",
        )
        self.db.add_all([self.proj_a1, self.proj_a2, self.proj_b1])
        self.db.commit()

        self.svc_a = DecisionService(self.db, self.org_a.id)
        self.svc_b = DecisionService(self.db, self.org_b.id)

    def close(self):
        self.db.close()
        self.engine.dispose()

    def add_decision(
        self,
        project: Project,
        *,
        action_name: str,
        status: str = "committed",
        recheck_at: Optional[datetime] = None,
        closed_at: Optional[datetime] = None,
        expected_lift: Optional[float] = None,
        action_code: str = "engagement_campaign",
        committed_at: Optional[datetime] = None,
        actual_outcome: Optional[str] = None,
    ) -> Decision:
        d = Decision(
            organization_id=project.organization_id,
            project_id=project.id,
            status=status,
            action_code=action_code,
            action_name=action_name,
            action_description="test",
            expected_lift=expected_lift,
            recheck_interval_days=30,
            recheck_at=recheck_at,
            closed_at=closed_at,
            committed_at=committed_at or _now(),
            actual_outcome=actual_outcome,
            decision_summary=f"Test {action_name}",
            case_snapshot={},
        )
        self.db.add(d)
        self.db.commit()
        self.db.refresh(d)
        return d


class TestPortfolioBuckets(unittest.TestCase):
    def setUp(self):
        self.fx = PortfolioFixture()

    def tearDown(self):
        self.fx.close()

    def test_empty_portfolio(self):
        port = self.fx.svc_a.list_portfolio()
        self.assertEqual(port["counts"]["overdue"], 0)
        self.assertEqual(port["counts"]["due_now"], 0)
        self.assertEqual(port["counts"]["upcoming"], 0)
        self.assertEqual(port["counts"]["closed_recent"], 0)
        self.assertIn("No open follow-ups", port["plain_summary"])
        self.assertEqual(port["layer"], "B3_followup_portfolio")

    def test_overdue_bucket(self):
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Overdue Action",
            recheck_at=_now() - timedelta(days=5),
            expected_lift=-0.15,
        )
        port = self.fx.svc_a.list_portfolio()
        self.assertEqual(port["counts"]["overdue"], 1)
        self.assertEqual(port["counts"]["due_now"], 0)
        item = port["overdue"][0]
        self.assertEqual(item["action_name"], "Overdue Action")
        self.assertEqual(item["project_name"], "Telco Alpha")
        self.assertTrue(item["due_for_recheck"])
        self.assertIn("lower chance", item["impact_hint"])

    def test_due_now_within_last_day(self):
        # Past but within 24h → due_now (not overdue)
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Due Hours Ago",
            recheck_at=_now() - timedelta(hours=6),
        )
        port = self.fx.svc_a.list_portfolio()
        self.assertEqual(port["counts"]["due_now"], 1)
        self.assertEqual(port["counts"]["overdue"], 0)
        self.assertEqual(port["due_now"][0]["action_name"], "Due Hours Ago")

    def test_null_recheck_goes_due_now(self):
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="No Recheck Date",
            recheck_at=None,
        )
        port = self.fx.svc_a.list_portfolio()
        self.assertEqual(port["counts"]["due_now"], 1)
        self.assertEqual(port["due_now"][0]["action_name"], "No Recheck Date")

    def test_upcoming_bucket(self):
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Future Check",
            recheck_at=_now() + timedelta(days=20),
            expected_lift=0.08,
        )
        port = self.fx.svc_a.list_portfolio()
        self.assertEqual(port["counts"]["upcoming"], 1)
        self.assertEqual(port["upcoming"][0]["action_name"], "Future Check")
        self.assertIn("up ~", port["upcoming"][0]["impact_hint"])

    def test_closed_recent_vs_old(self):
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Closed Recent",
            status="closed",
            recheck_at=None,
            closed_at=_now() - timedelta(days=2),
            expected_lift=-0.1,
        )
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Closed Old",
            status="closed",
            recheck_at=None,
            closed_at=_now() - timedelta(days=60),
        )
        port = self.fx.svc_a.list_portfolio(closed_days=30)
        self.assertEqual(port["counts"]["closed_recent"], 1)
        self.assertEqual(port["closed_recent"][0]["action_name"], "Closed Recent")

        # Wider window includes old closed
        port_wide = self.fx.svc_a.list_portfolio(closed_days=90)
        names = {x["action_name"] for x in port_wide["closed_recent"]}
        self.assertIn("Closed Recent", names)
        self.assertIn("Closed Old", names)

    def test_cancelled_and_proposed_excluded(self):
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Cancelled",
            status="cancelled",
            recheck_at=_now() - timedelta(days=10),
        )
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Proposed",
            status="proposed",
            recheck_at=_now() - timedelta(days=10),
        )
        port = self.fx.svc_a.list_portfolio()
        self.assertEqual(sum(port["counts"].values()), 0)

    def test_checking_status_still_open(self):
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="In Progress Check",
            status="checking",
            recheck_at=_now() - timedelta(days=4),
        )
        port = self.fx.svc_a.list_portfolio()
        self.assertEqual(port["counts"]["overdue"], 1)
        self.assertEqual(port["overdue"][0]["status"], "checking")

    def test_all_buckets_together(self):
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="O1",
            recheck_at=_now() - timedelta(days=10),
            expected_lift=-0.2,
        )
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="D1",
            recheck_at=_now() - timedelta(hours=2),
        )
        self.fx.add_decision(
            self.fx.proj_a2,
            action_name="U1",
            recheck_at=_now() + timedelta(days=5),
        )
        self.fx.add_decision(
            self.fx.proj_a2,
            action_name="U_far",
            recheck_at=_now() + timedelta(days=40),
        )
        self.fx.add_decision(
            self.fx.proj_a2,
            action_name="C1",
            status="closed",
            closed_at=_now() - timedelta(days=1),
        )
        port = self.fx.svc_a.list_portfolio(due_soon_days=7)
        self.assertEqual(port["counts"]["overdue"], 1)
        self.assertEqual(port["counts"]["due_now"], 1)
        self.assertEqual(port["counts"]["upcoming"], 2)
        self.assertEqual(port["counts"]["due_soon"], 1)
        self.assertEqual(port["counts"]["closed_recent"], 1)
        self.assertIn("1 overdue", port["plain_summary"])
        self.assertIn("1 due now", port["plain_summary"])
        self.assertIn("upcoming", port["plain_summary"])


class TestPortfolioFiltersAndIsolation(unittest.TestCase):
    def setUp(self):
        self.fx = PortfolioFixture()

    def tearDown(self):
        self.fx.close()

    def test_project_filter(self):
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="On A1",
            recheck_at=_now() - timedelta(days=3),
        )
        self.fx.add_decision(
            self.fx.proj_a2,
            action_name="On A2",
            recheck_at=_now() - timedelta(days=3),
        )
        port = self.fx.svc_a.list_portfolio(project_id=self.fx.proj_a1.id)
        self.assertEqual(port["counts"]["overdue"], 1)
        self.assertEqual(port["overdue"][0]["action_name"], "On A1")
        self.assertEqual(port["overdue"][0]["project_id"], self.fx.proj_a1.id)

    def test_bad_project_filter_raises(self):
        with self.assertRaises(ValueError):
            self.fx.svc_a.list_portfolio(project_id="does-not-exist")

    def test_org_isolation(self):
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="OrgA Only",
            recheck_at=_now() - timedelta(days=3),
        )
        self.fx.add_decision(
            self.fx.proj_b1,
            action_name="OrgB Only",
            recheck_at=_now() - timedelta(days=3),
        )
        port_a = self.fx.svc_a.list_portfolio()
        port_b = self.fx.svc_b.list_portfolio()
        self.assertEqual(port_a["counts"]["overdue"], 1)
        self.assertEqual(port_a["overdue"][0]["action_name"], "OrgA Only")
        self.assertEqual(port_b["counts"]["overdue"], 1)
        self.assertEqual(port_b["overdue"][0]["action_name"], "OrgB Only")


class TestImpactHintsAndSorting(unittest.TestCase):
    def setUp(self):
        self.fx = PortfolioFixture()

    def tearDown(self):
        self.fx.close()

    def test_impact_hint_variants(self):
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Big Down",
            recheck_at=_now() - timedelta(days=2),
            expected_lift=-0.12,
        )
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Big Up",
            recheck_at=_now() - timedelta(days=2),
            expected_lift=0.09,
        )
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Tiny",
            recheck_at=_now() - timedelta(days=2),
            expected_lift=0.001,
        )
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="NoLift",
            recheck_at=_now() - timedelta(days=2),
            expected_lift=None,
        )
        port = self.fx.svc_a.list_portfolio()
        by_name = {x["action_name"]: x["impact_hint"] for x in port["overdue"]}
        self.assertIn("lower chance", by_name["Big Down"])
        self.assertIn("up ~", by_name["Big Up"])
        self.assertEqual(by_name["Tiny"], "Small expected change")
        self.assertIsNone(by_name["NoLift"])

    def test_open_sorted_by_recheck_then_impact(self):
        earlier = _now() - timedelta(days=8)
        later = _now() - timedelta(days=3)
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Later Small",
            recheck_at=later,
            expected_lift=-0.01,
        )
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Earlier Big",
            recheck_at=earlier,
            expected_lift=-0.25,
        )
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Earlier Small",
            recheck_at=earlier,
            expected_lift=-0.02,
        )
        port = self.fx.svc_a.list_portfolio()
        names = [x["action_name"] for x in port["overdue"]]
        # Same earlier date: higher |lift| first, then later date last
        self.assertEqual(names[0], "Earlier Big")
        self.assertEqual(names[1], "Earlier Small")
        self.assertEqual(names[2], "Later Small")


class TestCreateCheckInAffectsPortfolio(unittest.TestCase):
    def setUp(self):
        self.fx = PortfolioFixture()

    def tearDown(self):
        self.fx.close()

    def test_create_from_case_lands_upcoming(self):
        item = self.fx.svc_a.create_from_case(
            self.fx.proj_a1.id,
            action_code="engagement_campaign",
            action_name="Fresh Commit",
            probability=0.4,
            expected_lift=-0.05,
            recheck_interval_days=30,
        )
        self.assertEqual(item["status"], "committed")
        self.assertIsNotNone(item["recheck_at"])
        port = self.fx.svc_a.list_portfolio()
        self.assertEqual(port["counts"]["upcoming"], 1)
        self.assertEqual(port["upcoming"][0]["id"], item["id"])

    def test_check_in_close_moves_to_closed_recent(self):
        item = self.fx.svc_a.create_from_case(
            self.fx.proj_a1.id,
            action_code="engagement_campaign",
            action_name="Will Close",
            probability=0.5,
            expected_lift=-0.1,
        )
        # Force overdue then close
        d = self.fx.db.query(Decision).filter(Decision.id == item["id"]).one()
        d.recheck_at = _now() - timedelta(days=4)
        self.fx.db.commit()

        before = self.fx.svc_a.list_portfolio()
        self.assertEqual(before["counts"]["overdue"], 1)

        closed = self.fx.svc_a.check_in(
            self.fx.proj_a1.id,
            item["id"],
            actual_outcome="yes",
            notes="Worked",
            close=True,
        )
        self.assertEqual(closed["status"], "closed")
        after = self.fx.svc_a.list_portfolio()
        self.assertEqual(after["counts"]["overdue"], 0)
        self.assertEqual(after["counts"]["closed_recent"], 1)
        self.assertEqual(after["closed_recent"][0]["id"], item["id"])

    def test_check_in_reschedule_moves_to_upcoming(self):
        item = self.fx.svc_a.create_from_case(
            self.fx.proj_a1.id,
            action_code="discount_10",
            action_name="Reschedule Me",
            probability=0.6,
            expected_lift=-0.08,
        )
        d = self.fx.db.query(Decision).filter(Decision.id == item["id"]).one()
        d.recheck_at = _now() - timedelta(days=5)
        self.fx.db.commit()

        self.fx.svc_a.check_in(
            self.fx.proj_a1.id,
            item["id"],
            notes="Still watching",
            close=False,
            schedule_next=True,
            recheck_interval_days=60,
        )
        port = self.fx.svc_a.list_portfolio()
        self.assertEqual(port["counts"]["overdue"], 0)
        self.assertEqual(port["counts"]["upcoming"], 1)
        self.assertEqual(port["upcoming"][0]["recheck_interval_days"], 60)

    def test_check_in_keep_same_date_stays_due(self):
        item = self.fx.svc_a.create_from_case(
            self.fx.proj_a1.id,
            action_code="engagement_campaign",
            action_name="Keep Date",
            probability=0.55,
        )
        d = self.fx.db.query(Decision).filter(Decision.id == item["id"]).one()
        due = _now() - timedelta(hours=3)
        d.recheck_at = due
        self.fx.db.commit()

        self.fx.svc_a.check_in(
            self.fx.proj_a1.id,
            item["id"],
            notes="Partial",
            close=False,
            schedule_next=False,
        )
        port = self.fx.svc_a.list_portfolio()
        self.assertEqual(port["counts"]["due_now"], 1)
        # recheck date should remain roughly the same
        kept = port["due_now"][0]["recheck_at"]
        self.assertIsNotNone(kept)

    def test_invalid_check_in_on_closed(self):
        item = self.fx.svc_a.create_from_case(
            self.fx.proj_a1.id,
            action_code="engagement_campaign",
            action_name="Already Done",
        )
        self.fx.svc_a.check_in(
            self.fx.proj_a1.id,
            item["id"],
            close=True,
            actual_outcome="no",
        )
        with self.assertRaises(ValueError):
            self.fx.svc_a.check_in(
                self.fx.proj_a1.id,
                item["id"],
                notes="again",
            )

    def test_list_decisions_due_count(self):
        item = self.fx.svc_a.create_from_case(
            self.fx.proj_a1.id,
            action_code="engagement_campaign",
            action_name="Due List",
        )
        d = self.fx.db.query(Decision).filter(Decision.id == item["id"]).one()
        d.recheck_at = _now() - timedelta(days=2)
        self.fx.db.commit()
        listed = self.fx.svc_a.list_decisions(self.fx.proj_a1.id)
        self.assertEqual(listed["due_for_recheck"], 1)
        self.assertTrue(listed["decisions"][0]["due_for_recheck"])


class TestPortfolioLimits(unittest.TestCase):
    def setUp(self):
        self.fx = PortfolioFixture()

    def tearDown(self):
        self.fx.close()

    def test_cap_lists_preserve_true_counts(self):
        for i in range(12):
            self.fx.add_decision(
                self.fx.proj_a1,
                action_name=f"O{i}",
                recheck_at=_now() - timedelta(days=2 + i),
                expected_lift=-0.01 * (i + 1),
            )
        port = self.fx.svc_a.list_portfolio(limit=5)
        # Cap = min(max(limit,10), 50) → 10
        self.assertEqual(len(port["overdue"]), 10)
        self.assertEqual(port["counts"]["overdue"], 12)
        self.assertTrue(port["truncated"]["overdue"])

    def test_overdue_not_starved_by_newer_commits(self):
        """Older overdue must surface even when many recent upcoming exist."""
        old = self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Ancient Overdue",
            recheck_at=_now() - timedelta(days=40),
            committed_at=_now() - timedelta(days=70),
            expected_lift=-0.2,
        )
        for i in range(25):
            self.fx.add_decision(
                self.fx.proj_a1,
                action_name=f"Fresh Upcoming {i}",
                recheck_at=_now() + timedelta(days=10 + i),
                committed_at=_now() - timedelta(minutes=i),
            )
        port = self.fx.svc_a.list_portfolio(limit=10)
        overdue_ids = {x["id"] for x in port["overdue"]}
        self.assertIn(old.id, overdue_ids)
        self.assertGreaterEqual(port["counts"]["overdue"], 1)


class TestHttpPortfolioRoute(unittest.TestCase):
    """FastAPI route wiring + auth + query params."""

    def setUp(self):
        self.fx = PortfolioFixture()
        from fastapi.testclient import TestClient
        from app.main import app
        from app.database import get_db
        from app.services.auth_service import AuthContext, get_auth_context

        self.app = app
        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="HTTP Overdue",
            recheck_at=_now() - timedelta(days=4),
            expected_lift=-0.11,
        )

        def _override_db():
            try:
                yield self.fx.db
            finally:
                pass

        async def _override_auth():
            return AuthContext(organization=self.fx.org_a, scopes=["admin"])

        self.app.dependency_overrides[get_db] = _override_db
        self.app.dependency_overrides[get_auth_context] = _override_auth
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        self.fx.close()

    def test_portfolio_endpoint_ok(self):
        res = self.client.get("/api/projects/decisions/portfolio")
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["layer"], "B3_followup_portfolio")
        self.assertEqual(body["counts"]["overdue"], 1)
        self.assertEqual(body["overdue"][0]["action_name"], "HTTP Overdue")
        self.assertEqual(body["overdue"][0]["project_name"], "Telco Alpha")

    def test_portfolio_project_filter(self):
        self.fx.add_decision(
            self.fx.proj_a2,
            action_name="Other Project",
            recheck_at=_now() - timedelta(days=3),
        )
        res = self.client.get(
            f"/api/projects/decisions/portfolio?project_id={self.fx.proj_a1.id}"
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["counts"]["overdue"], 1)
        self.assertEqual(body["overdue"][0]["action_name"], "HTTP Overdue")

    def test_portfolio_bad_project_400(self):
        res = self.client.get("/api/projects/decisions/portfolio?project_id=missing")
        self.assertEqual(res.status_code, 400)

    def test_portfolio_not_shadowed_by_project_id_route(self):
        # Critical: /decisions/portfolio must not be captured by /{project_id}
        res = self.client.get("/api/projects/decisions/portfolio")
        self.assertEqual(res.status_code, 200)
        self.assertIn("counts", res.json())

    def test_unauthenticated_rejected(self):
        from app.database import get_db
        from app.services.auth_service import get_auth_context

        self.app.dependency_overrides.pop(get_auth_context, None)

        # Keep db override so we don't need real DB for auth failure path
        res = self.client.get("/api/projects/decisions/portfolio")
        self.assertIn(res.status_code, (401, 403))


class TestPortfolioIntelligence(unittest.TestCase):
    def setUp(self):
        self.fx = PortfolioFixture()

    def tearDown(self):
        self.fx.close()

    def test_empty_intel(self):
        intel = self.fx.svc_a.portfolio_intelligence()
        self.assertEqual(intel["layer"], "portfolio_intelligence")
        self.assertEqual(intel["counts"]["actions"], 0)
        self.assertEqual(intel["actions"], [])
        self.assertIn("No committed actions", intel["plain_summary"])

    def test_roi_thin_then_reliable(self):
        # 2 favorable outcomes → thin; add a 3rd → reliable
        for i in range(2):
            self.fx.add_decision(
                self.fx.proj_a1,
                action_name="Engagement",
                action_code="engagement_campaign",
                status="closed",
                closed_at=_now(),
                recheck_at=None,
                expected_lift=-0.08,
                actual_outcome="retained",
            )
        intel = self.fx.svc_a.portfolio_intelligence()
        self.assertEqual(intel["counts"]["actions"], 1)
        row = intel["actions"][0]
        self.assertEqual(row["outcome_n"], 2)
        self.assertEqual(row["favorable_n"], 2)
        self.assertEqual(row["evidence"], "thin")
        self.assertFalse(row["reliable"])
        self.assertAlmostEqual(row["avg_expected_lift_pp"], -8.0, places=1)

        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Engagement",
            action_code="engagement_campaign",
            status="closed",
            closed_at=_now(),
            recheck_at=None,
            expected_lift=-0.08,
            actual_outcome="churned",
        )
        intel2 = self.fx.svc_a.portfolio_intelligence()
        row2 = intel2["actions"][0]
        self.assertEqual(row2["outcome_n"], 3)
        self.assertEqual(row2["favorable_n"], 2)
        self.assertEqual(row2["evidence"], "reliable")
        self.assertTrue(row2["reliable"])
        self.assertAlmostEqual(row2["favorable_rate"], 2 / 3, places=3)

    def test_capacity_alert_high_cost_action(self):
        # discount_20 → high cost → capacity 5
        for i in range(6):
            self.fx.add_decision(
                self.fx.proj_a1,
                action_name="20% discount",
                action_code="discount_20",
                recheck_at=_now() + timedelta(days=10),
                expected_lift=-0.12,
            )
        intel = self.fx.svc_a.portfolio_intelligence()
        row = next(a for a in intel["actions"] if a["action_code"] == "discount_20")
        self.assertEqual(row["open_n"], 6)
        self.assertEqual(row["capacity_open"], 5)
        self.assertTrue(row["over_capacity"])
        self.assertEqual(intel["counts"]["over_capacity"], 1)
        self.assertEqual(len(intel["capacity_alerts"]), 1)
        self.assertIn("can't afford", intel["capacity_alerts"][0]["plain"].lower())

    def test_org_isolation(self):
        self.fx.add_decision(
            self.fx.proj_b1,
            action_name="Other Org",
            action_code="discount_20",
            recheck_at=_now() + timedelta(days=5),
        )
        intel = self.fx.svc_a.portfolio_intelligence()
        self.assertEqual(intel["counts"]["actions"], 0)

    def test_http_portfolio_intel_not_shadowed(self):
        from fastapi.testclient import TestClient

        from app.main import app
        from app.database import get_db
        from app.services.auth_service import AuthContext, get_auth_context

        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="HTTP Intel",
            action_code="engagement_campaign",
            recheck_at=_now() + timedelta(days=5),
            expected_lift=-0.05,
        )

        def _db():
            try:
                yield self.fx.db
            finally:
                pass

        async def _auth():
            return AuthContext(organization=self.fx.org_a, scopes=["admin"])

        app.dependency_overrides[get_db] = _db
        app.dependency_overrides[get_auth_context] = _auth
        try:
            client = TestClient(app)
            res = client.get("/api/projects/decisions/portfolio-intel")
            self.assertEqual(res.status_code, 200)
            body = res.json()
            self.assertEqual(body["layer"], "portfolio_intelligence")
            self.assertGreaterEqual(body["counts"]["actions"], 1)
        finally:
            app.dependency_overrides.clear()

    def test_http_org_health_not_shadowed(self):
        from fastapi.testclient import TestClient

        from app.main import app
        from app.database import get_db
        from app.services.auth_service import AuthContext, get_auth_context

        def _db():
            try:
                yield self.fx.db
            finally:
                pass

        async def _auth():
            return AuthContext(organization=self.fx.org_a, scopes=["admin"])

        app.dependency_overrides[get_db] = _db
        app.dependency_overrides[get_auth_context] = _auth
        try:
            client = TestClient(app)
            res = client.get("/api/projects/org-health")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["layer"], "org_health")
        finally:
            app.dependency_overrides.clear()


class TestOrgHealth(unittest.TestCase):
    def setUp(self):
        self.fx = PortfolioFixture()

    def tearDown(self):
        self.fx.close()

    def test_org_health_due_and_dont_act_counts(self):
        from app.db.models import ProjectPrediction
        from app.services.project_service import ProjectService

        self.fx.add_decision(
            self.fx.proj_a1,
            action_name="Due",
            recheck_at=_now() - timedelta(days=3),
        )
        pred = ProjectPrediction(
            project_id=self.fx.proj_a1.id,
            model_version="test-v1",
            features={"tenure": 1},
            probability=0.5,
            low_confidence=True,
        )
        self.fx.db.add(pred)
        self.fx.db.commit()

        health = ProjectService(self.fx.db, self.fx.org_a.id).org_health()
        self.assertEqual(health["layer"], "org_health")
        self.assertEqual(health["counts"]["due_attention"], 1)
        self.assertEqual(health["counts"]["soft_cases"], 1)
        self.assertEqual(health["counts"]["ready_projects"], 2)
        self.assertIn("don't-act", health["plain_summary"].lower())


if __name__ == "__main__":
    unittest.main()
