"""B3 Decision ledger — commit actions as durable, recheckable decisions."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import case
from sqlalchemy.orm import Session

from app.db.models import Decision, DecisionActivity, Project, ProjectPrediction, User
from app.recommendations.action_catalog import get_action
from app.recommendations.domains import detect_domain

logger = logging.getLogger(__name__)


VALID_STATUSES = {"proposed", "committed", "checking", "closed", "cancelled"}
VALID_INTERVALS = {30, 60, 90}
VALID_ACTIVITY_EVENTS = {
    "created",
    "assigned",
    "checked_in",
    "closed",
    "rescheduled",
    "note",
}


def flag_due_rechecks(
    db: Session,
    *,
    org_id: Optional[str] = None,
    limit: int = 200,
) -> dict[str, Any]:
    """
    System/org sweep: flip due committed decisions to checking.

    Used by Celery Beat and the manual recheck-sweep API. Does not invent
    outcomes or move recheck_at — portfolio/UI surfaces the follow-up.
    """
    now = datetime.utcnow()
    lim = min(max(int(limit), 1), 1000)

    due_filter = (Decision.recheck_at.is_(None)) | (Decision.recheck_at <= now)

    already_q = db.query(Decision).filter(
        Decision.status == "checking",
        due_filter,
    )
    if org_id:
        already_q = already_q.filter(Decision.organization_id == org_id)
    already_checking = int(already_q.count())

    null_first = case((Decision.recheck_at.is_(None), 0), else_=1)
    q = (
        db.query(Decision)
        .filter(
            Decision.status == "committed",
            due_filter,
        )
        .order_by(null_first, Decision.recheck_at.asc(), Decision.committed_at.asc())
    )
    if org_id:
        q = q.filter(Decision.organization_id == org_id)

    rows = q.limit(lim).all()
    flagged_ids: list[str] = []
    stamp = (
        f"[{now.isoformat()}Z] Scheduled recheck due — open this follow-up "
        f"and log what happened."
    )
    for d in rows:
        d.status = "checking"
        d.updated_at = now
        notes = (d.outcome_notes or "").strip()
        # Avoid duplicate scheduled stamps on rare race retries
        if stamp not in notes:
            d.outcome_notes = f"{notes}\n---\n{stamp}" if notes else stamp
        flagged_ids.append(d.id)

    if flagged_ids:
        db.commit()

    plain = (
        f"Flagged {len(flagged_ids)} follow-up(s) for check-in"
        + (f"; {already_checking} already in review." if already_checking else ".")
    )
    return {
        "layer": "B3_scheduled_rechecks",
        "flagged": len(flagged_ids),
        "already_checking": already_checking,
        "decision_ids": flagged_ids,
        "limit": lim,
        "org_id": org_id,
        "plain_summary": plain,
        "ran_at": now.isoformat(),
    }


class DecisionService:
    def __init__(self, db: Session, org_id: str):
        self.db = db
        self.org_id = org_id

    def _get_project(self, project_id: str) -> Project:
        project = (
            self.db.query(Project)
            .filter(
                Project.id == project_id,
                Project.organization_id == self.org_id,
            )
            .first()
        )
        if not project:
            raise ValueError("Project not found")
        return project

    def create_from_case(
        self,
        project_id: str,
        *,
        action_code: str,
        prediction_id: Optional[str] = None,
        action_name: Optional[str] = None,
        action_description: Optional[str] = None,
        entity_id: Optional[str] = None,
        probability: Optional[float] = None,
        risk_level: Optional[str] = None,
        expected_probability_after: Optional[float] = None,
        expected_lift: Optional[float] = None,
        decision_summary: Optional[str] = None,
        case_snapshot: Optional[dict[str, Any]] = None,
        recheck_interval_days: int = 30,
        status: str = "committed",
        assignee_user_id: Optional[str] = None,
        actor_user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Open a ledger decision from a scored case (and optional prediction row)."""
        project = self._get_project(project_id)
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'")
        interval = int(recheck_interval_days or 30)
        if interval not in VALID_INTERVALS:
            raise ValueError("recheck_interval_days must be 30, 60, or 90")

        code = (action_code or "").strip()
        if not code:
            raise ValueError("action_code is required")

        assignee_id = self._validate_org_user(assignee_user_id) if assignee_user_id else None
        # Default assignee to the person who committed, when signed in
        if not assignee_id and actor_user_id:
            assignee_id = self._validate_org_user(actor_user_id)

        domain = detect_domain(
            feature_columns=project.feature_columns,
            project_name=project.name,
            target_column=project.target_column,
            target_description=project.target_description,
        )
        catalog_action = get_action(code, domain=domain)

        name = (action_name or "").strip() or (
            catalog_action.name if catalog_action else code
        )
        desc = action_description or (
            catalog_action.description if catalog_action else None
        )

        pred: Optional[ProjectPrediction] = None
        if prediction_id:
            pred = (
                self.db.query(ProjectPrediction)
                .filter(
                    ProjectPrediction.id == prediction_id,
                    ProjectPrediction.project_id == project_id,
                )
                .first()
            )
            if not pred:
                raise ValueError("Prediction not found for this project")

        # Prefer live payload; fall back to stored prediction
        p = probability
        if p is None and pred is not None:
            p = pred.probability
        risk = risk_level or (pred.risk_level if pred else None)
        ent = entity_id or (pred.entity_id if pred else None)

        lift = expected_lift
        new_p = expected_probability_after
        if lift is None and new_p is not None and p is not None:
            lift = float(new_p) - float(p)
        if new_p is None and lift is not None and p is not None:
            new_p = max(0.0, min(1.0, float(p) + float(lift)))

        snapshot = dict(case_snapshot or {})
        if pred is not None:
            snapshot.setdefault("features", pred.features)
            snapshot.setdefault("top_factors", pred.top_factors)
            snapshot.setdefault("recommendations", pred.recommendations)
        snapshot["domain"] = domain
        snapshot["layer"] = "B3_decision_ledger"

        now = datetime.utcnow()
        decision = Decision(
            organization_id=self.org_id,
            project_id=project_id,
            prediction_id=pred.id if pred else None,
            entity_id=ent,
            status=status,
            action_code=code,
            action_name=name,
            action_description=desc,
            probability_at_commit=float(p) if p is not None else None,
            risk_level_at_commit=risk,
            expected_probability_after=float(new_p) if new_p is not None else None,
            expected_lift=float(lift) if lift is not None else None,
            decision_summary=decision_summary
            or f"Committed “{name}” with a {interval}-day recheck.",
            case_snapshot=snapshot,
            recheck_interval_days=interval,
            recheck_at=now + timedelta(days=interval),
            committed_at=now,
            assignee_user_id=assignee_id,
            committed_by_user_id=actor_user_id,
        )
        self.db.add(decision)
        self.db.flush()

        self._log_activity(
            decision,
            event="created",
            actor_user_id=actor_user_id,
            payload={
                "action_code": code,
                "action_name": name,
                "recheck_interval_days": interval,
                "assignee_user_id": assignee_id,
            },
            commit=False,
        )
        if assignee_id:
            self._log_activity(
                decision,
                event="assigned",
                actor_user_id=actor_user_id,
                payload={"assignee_user_id": assignee_id, "from": None},
                commit=False,
            )

        # Light A7 bridge: stamp action on prediction if present and empty
        if pred is not None and not pred.action_taken:
            pred.action_taken = code

        self.db.commit()
        self.db.refresh(decision)
        return self._format(decision, project)

    def list_decisions(
        self,
        project_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        project = self._get_project(project_id)
        q = self.db.query(Decision).filter(Decision.project_id == project_id)
        if status:
            q = q.filter(Decision.status == status)
        rows = q.order_by(Decision.committed_at.desc()).limit(min(limit, 200)).all()

        now = datetime.utcnow()
        due = [
            d
            for d in rows
            if d.recheck_at
            and d.recheck_at <= now
            and d.status in ("committed", "checking")
        ]
        items = [self._format(d, project, include_activity=True) for d in rows]
        plain = (
            f"{len(items)} follow-up(s) on this project"
            + (f"; {len(due)} due for a check-in." if due else ".")
        )
        return {
            "project_id": project_id,
            "layer": "B3_decision_ledger",
            "n": len(items),
            "due_for_recheck": len(due),
            "decisions": items,
            "plain_summary": plain,
        }

    def list_portfolio(
        self,
        *,
        project_id: Optional[str] = None,
        limit: int = 100,
        closed_days: int = 30,
        due_soon_days: int = 7,
        assignee_user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Org-wide follow-up board: overdue / due now / upcoming / recently closed.

        Counts are true totals; list payloads are capped for the UI.
        Open rows are fetched by recheck urgency (not commit recency) so old
        overdue items are never starved by newer commits.
        """
        now = datetime.utcnow()
        closed_after = now - timedelta(days=max(1, int(closed_days)))
        soon_horizon = now + timedelta(days=max(0, int(due_soon_days)))
        overdue_cutoff = now - timedelta(days=1)
        list_cap = min(max(int(limit), 10), 50)
        open_fetch_cap = 500

        base = self.db.query(Decision).filter(Decision.organization_id == self.org_id)
        if project_id:
            self._get_project(project_id)
            base = base.filter(Decision.project_id == project_id)
        if assignee_user_id:
            base = base.filter(Decision.assignee_user_id == assignee_user_id)

        # Null recheck_at first (treat as due), then earliest recheck
        null_first = case((Decision.recheck_at.is_(None), 0), else_=1)
        open_rows = (
            base.filter(Decision.status.in_(["committed", "checking"]))
            .order_by(null_first, Decision.recheck_at.asc(), Decision.committed_at.desc())
            .limit(open_fetch_cap)
            .all()
        )
        closed_rows = (
            base.filter(
                Decision.status == "closed",
                Decision.closed_at.isnot(None),
                Decision.closed_at >= closed_after,
            )
            .order_by(Decision.closed_at.desc())
            .limit(max(list_cap, 20))
            .all()
        )

        needed_ids = {d.project_id for d in open_rows} | {d.project_id for d in closed_rows}
        projects = {}
        if needed_ids:
            projects = {
                p.id: p
                for p in self.db.query(Project)
                .filter(
                    Project.organization_id == self.org_id,
                    Project.id.in_(list(needed_ids)),
                )
                .all()
            }

        overdue: list[dict[str, Any]] = []
        due_now: list[dict[str, Any]] = []
        upcoming: list[dict[str, Any]] = []
        closed_recent: list[dict[str, Any]] = []

        def _impact_key(item: dict[str, Any]) -> float:
            lift = item.get("expected_lift")
            try:
                return abs(float(lift)) if lift is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        def _enrich(d: Decision) -> Optional[dict[str, Any]]:
            project = projects.get(d.project_id)
            if not project:
                return None
            item = self._format(d, project, include_project_name=True)
            lift = d.expected_lift
            if lift is not None:
                pp = float(lift) * 100
                if abs(pp) >= 0.5:
                    item["impact_hint"] = (
                        f"Hoped to lower chance by ~{abs(pp):.0f} pp"
                        if pp < 0
                        else f"Expected chance up ~{pp:.0f} pp"
                    )
                else:
                    item["impact_hint"] = "Small expected change"
            else:
                item["impact_hint"] = None
            return item

        n_due_soon = 0
        for d in open_rows:
            item = _enrich(d)
            if not item:
                continue
            if d.recheck_at is None:
                due_now.append(item)
            elif d.recheck_at <= overdue_cutoff:
                overdue.append(item)
            elif d.recheck_at <= now:
                due_now.append(item)
            else:
                upcoming.append(item)
                if d.recheck_at <= soon_horizon:
                    n_due_soon += 1

        for d in closed_rows:
            item = _enrich(d)
            if item:
                closed_recent.append(item)

        def _sort_open(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(
                items,
                key=lambda x: (
                    x.get("recheck_at") or "",
                    -_impact_key(x),
                ),
            )

        overdue = _sort_open(overdue)
        due_now = _sort_open(due_now)
        upcoming = _sort_open(upcoming)
        # closed_rows already newest-first

        n_overdue, n_due, n_up = len(overdue), len(due_now), len(upcoming)
        n_closed = len(closed_recent)

        overdue_out = overdue[:list_cap]
        due_now_out = due_now[:list_cap]
        upcoming_out = upcoming[:list_cap]
        closed_out = closed_recent[: min(20, list_cap)]

        bits = []
        if n_overdue:
            bits.append(f"{n_overdue} overdue")
        if n_due:
            bits.append(f"{n_due} due now")
        if n_up:
            if n_due_soon and n_due_soon < n_up:
                bits.append(f"{n_up} upcoming ({n_due_soon} in {due_soon_days}d)")
            else:
                bits.append(f"{n_up} upcoming")
        if not bits:
            plain = "No open follow-ups — save one from a case when you commit to an action."
        else:
            plain = "Follow-ups across your projects: " + "; ".join(bits) + "."

        return {
            "layer": "B3_followup_portfolio",
            "counts": {
                "overdue": n_overdue,
                "due_now": n_due,
                "upcoming": n_up,
                "due_soon": n_due_soon,
                "closed_recent": n_closed,
            },
            "truncated": {
                "overdue": n_overdue > len(overdue_out),
                "due_now": n_due > len(due_now_out),
                "upcoming": n_up > len(upcoming_out),
                "closed_recent": n_closed > len(closed_out),
            },
            "overdue": overdue_out,
            "due_now": due_now_out,
            "upcoming": upcoming_out,
            "closed_recent": closed_out,
            "plain_summary": plain,
            "due_soon_days": due_soon_days,
            "closed_days": closed_days,
            "assignee_user_id": assignee_user_id,
        }

    @staticmethod
    def _default_open_capacity(base_cost: float) -> int:
        """Concurrent open-follow-up budget by catalog cost (org defaults)."""
        c = float(base_cost or 0.25)
        if c >= 0.45:
            return 5
        if c >= 0.2:
            return 10
        return 20

    def portfolio_intelligence(
        self,
        *,
        project_id: Optional[str] = None,
        min_n: int = 3,
    ) -> dict[str, Any]:
        """
        ROI-by-action + capacity view over the decision ledger.

        Favorable = bad event avoided (negative binary outcome), same as A5 learning.
        Capacity = default concurrent open budget by action cost; warn when over.
        """
        min_n = max(1, int(min_n))
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)

        base = self.db.query(Decision).filter(Decision.organization_id == self.org_id)
        if project_id:
            self._get_project(project_id)
            base = base.filter(Decision.project_id == project_id)

        rows = (
            base.filter(Decision.status.in_(["committed", "checking", "closed"]))
            .order_by(Decision.committed_at.desc())
            .limit(2000)
            .all()
        )
        if not rows:
            return {
                "layer": "portfolio_intelligence",
                "actions": [],
                "capacity_alerts": [],
                "counts": {
                    "actions": 0,
                    "open": 0,
                    "with_outcomes": 0,
                    "over_capacity": 0,
                    "reliable_actions": 0,
                },
                "plain_summary": (
                    "No committed actions yet — save a follow-up from a case to start "
                    "tracking ROI and capacity."
                ),
                "min_n": min_n,
            }

        project_ids = {d.project_id for d in rows}
        projects = {
            p.id: p
            for p in self.db.query(Project)
            .filter(
                Project.organization_id == self.org_id,
                Project.id.in_(list(project_ids)),
            )
            .all()
        }

        # action_code -> aggregates
        buckets: dict[str, dict[str, Any]] = {}
        for d in rows:
            project = projects.get(d.project_id)
            if not project:
                continue
            code = (d.action_code or "").strip()
            if not code:
                continue
            b = buckets.setdefault(
                code,
                {
                    "action_code": code,
                    "action_name": d.action_name or code,
                    "open_n": 0,
                    "committed_this_week": 0,
                    "outcome_n": 0,
                    "favorable_n": 0,
                    "unknown_n": 0,
                    "lift_sum": 0.0,
                    "lift_n": 0,
                },
            )
            # Prefer latest non-empty name
            if d.action_name:
                b["action_name"] = d.action_name

            open_status = d.status in ("committed", "checking")
            if open_status:
                b["open_n"] += 1
            if d.committed_at and d.committed_at >= week_ago:
                b["committed_this_week"] += 1

            if d.expected_lift is not None:
                try:
                    b["lift_sum"] += float(d.expected_lift)
                    b["lift_n"] += 1
                except (TypeError, ValueError):
                    pass

            if not d.actual_outcome:
                continue
            kind = self._normalize_binary_outcome(d.actual_outcome, project)
            if kind == "unknown":
                b["unknown_n"] += 1
                continue
            if kind not in ("positive", "negative"):
                continue
            b["outcome_n"] += 1
            # Favorable = event avoided
            if kind == "negative":
                b["favorable_n"] += 1

        actions_out: list[dict[str, Any]] = []
        capacity_alerts: list[dict[str, Any]] = []

        for code, b in buckets.items():
            # Domain from any project using this action (first match)
            domain = None
            for d in rows:
                if d.action_code == code and d.project_id in projects:
                    p = projects[d.project_id]
                    domain = detect_domain(
                        feature_columns=p.feature_columns,
                        project_name=p.name,
                        target_column=p.target_column,
                        target_description=p.target_description,
                    )
                    break
            catalog = get_action(code, domain=domain)
            base_cost = float(catalog.base_cost) if catalog else 0.25
            capacity = self._default_open_capacity(base_cost)
            open_n = int(b["open_n"])
            over = open_n > capacity
            outcome_n = int(b["outcome_n"])
            favorable_n = int(b["favorable_n"])
            fav_rate = round(favorable_n / outcome_n, 4) if outcome_n else None
            avg_lift = (
                round(b["lift_sum"] / b["lift_n"], 4) if b["lift_n"] else None
            )
            avg_lift_pp = round(avg_lift * 100, 1) if avg_lift is not None else None
            reliable = outcome_n >= min_n
            evidence = "reliable" if reliable else ("thin" if outcome_n > 0 else "none")

            if avg_lift_pp is not None and abs(avg_lift_pp) >= 0.5:
                expected_bit = (
                    f"hoped ~{abs(avg_lift_pp):.0f}pp lower chance"
                    if avg_lift_pp < 0
                    else f"hoped ~{avg_lift_pp:.0f}pp higher chance"
                )
            else:
                expected_bit = "small expected move"

            if outcome_n == 0:
                autopsy = (
                    f"{b['action_name']}: {open_n} open, no logged outcomes yet — "
                    f"{expected_bit} at commit."
                )
            elif not reliable:
                autopsy = (
                    f"{b['action_name']}: n={outcome_n}, favorable "
                    f"{favorable_n}/{outcome_n} ({fav_rate:.0%}) — thin evidence; "
                    f"{expected_bit}."
                )
            else:
                autopsy = (
                    f"{b['action_name']}: n={outcome_n}, favorable "
                    f"{fav_rate:.0%}, {expected_bit}."
                )

            cost_label = (
                "high" if base_cost >= 0.45 else "medium" if base_cost >= 0.2 else "low"
            )
            item = {
                "action_code": code,
                "action_name": b["action_name"],
                "open_n": open_n,
                "committed_this_week": int(b["committed_this_week"]),
                "outcome_n": outcome_n,
                "favorable_n": favorable_n,
                "favorable_rate": fav_rate,
                "unknown_n": int(b["unknown_n"]),
                "avg_expected_lift": avg_lift,
                "avg_expected_lift_pp": avg_lift_pp,
                "capacity_open": capacity,
                "over_capacity": over,
                "capacity_headroom": max(0, capacity - open_n),
                "cost_label": cost_label,
                "evidence": evidence,
                "reliable": reliable,
                "autopsy_strip": autopsy,
            }
            actions_out.append(item)
            if over:
                capacity_alerts.append(
                    {
                        "action_code": code,
                        "action_name": b["action_name"],
                        "open_n": open_n,
                        "capacity_open": capacity,
                        "plain": (
                            f"You can't afford {open_n} open \"{b['action_name']}\" "
                            f"follow-ups at once (budget {capacity} for {cost_label}-cost actions)."
                        ),
                    }
                )

        # Rank: over-capacity first, then reliable outcomes, then open volume
        actions_out.sort(
            key=lambda x: (
                0 if x["over_capacity"] else 1,
                0 if x["reliable"] else 1,
                -x["outcome_n"],
                -x["open_n"],
            )
        )

        n_open = sum(a["open_n"] for a in actions_out)
        n_outcomes = sum(a["outcome_n"] for a in actions_out)
        n_reliable = sum(1 for a in actions_out if a["reliable"])
        n_over = len(capacity_alerts)

        bits: list[str] = []
        if n_over:
            bits.append(f"{n_over} action(s) over capacity")
        if n_reliable:
            bits.append(f"{n_reliable} action(s) with enough outcomes to read ROI")
        elif n_outcomes:
            bits.append(
                f"{n_outcomes} outcome(s) logged — need {min_n}+ per action for firm ROI"
            )
        else:
            bits.append("check in on follow-ups to unlock ROI by action")
        plain = "Portfolio: " + "; ".join(bits) + "."

        return {
            "layer": "portfolio_intelligence",
            "actions": actions_out,
            "capacity_alerts": capacity_alerts,
            "counts": {
                "actions": len(actions_out),
                "open": n_open,
                "with_outcomes": n_outcomes,
                "over_capacity": n_over,
                "reliable_actions": n_reliable,
            },
            "plain_summary": plain,
            "min_n": min_n,
        }

    def flag_due_rechecks(self, *, limit: int = 200) -> dict[str, Any]:
        """Mark due committed follow-ups as checking (scheduled recheck sweep).

        Does not invent outcomes or change recheck_at — humans still check in.
        Idempotent: already-checking rows are counted but not re-noted.
        """
        return flag_due_rechecks(
            self.db,
            org_id=self.org_id,
            limit=limit,
        )

    def get_decision(self, project_id: str, decision_id: str) -> dict[str, Any]:
        project = self._get_project(project_id)
        d = (
            self.db.query(Decision)
            .filter(
                Decision.id == decision_id,
                Decision.project_id == project_id,
            )
            .first()
        )
        if not d:
            raise ValueError("Decision not found")
        return self._format(d, project, include_snapshot=True, include_activity=True)

    def assign(
        self,
        project_id: str,
        decision_id: str,
        *,
        assignee_user_id: Optional[str],
        actor_user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Set or clear the follow-up assignee (must be an active org member)."""
        project = self._get_project(project_id)
        d = (
            self.db.query(Decision)
            .filter(
                Decision.id == decision_id,
                Decision.project_id == project_id,
                Decision.organization_id == self.org_id,
            )
            .first()
        )
        if not d:
            raise ValueError("Decision not found")

        previous = d.assignee_user_id
        new_id = None
        if assignee_user_id:
            new_id = self._validate_org_user(assignee_user_id)
        if previous == new_id:
            return self._format(d, project, include_activity=True)

        d.assignee_user_id = new_id
        d.updated_at = datetime.utcnow()
        self._log_activity(
            d,
            event="assigned",
            actor_user_id=actor_user_id,
            payload={"from": previous, "assignee_user_id": new_id},
            commit=False,
        )
        self.db.commit()
        self.db.refresh(d)
        return self._format(d, project, include_activity=True)

    def list_activities(
        self, project_id: str, decision_id: str, *, limit: int = 50
    ) -> dict[str, Any]:
        self._get_project(project_id)
        d = (
            self.db.query(Decision)
            .filter(
                Decision.id == decision_id,
                Decision.project_id == project_id,
                Decision.organization_id == self.org_id,
            )
            .first()
        )
        if not d:
            raise ValueError("Decision not found")
        rows = (
            self.db.query(DecisionActivity)
            .filter(DecisionActivity.decision_id == decision_id)
            .order_by(DecisionActivity.created_at.desc())
            .limit(min(max(int(limit), 1), 200))
            .all()
        )
        return {
            "layer": "decision_activity",
            "decision_id": decision_id,
            "n": len(rows),
            "activities": [self._format_activity(a) for a in rows],
        }

    def export_decisions(
        self,
        *,
        project_id: Optional[str] = None,
        assignee_user_id: Optional[str] = None,
        status: Optional[str] = None,
        include_activity: bool = True,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Org-scoped audit export of decisions (+ optional activity rows)."""
        lim = min(max(int(limit), 1), 2000)
        q = self.db.query(Decision).filter(Decision.organization_id == self.org_id)
        if project_id:
            self._get_project(project_id)
            q = q.filter(Decision.project_id == project_id)
        if assignee_user_id:
            q = q.filter(Decision.assignee_user_id == assignee_user_id)
        if status:
            q = q.filter(Decision.status == status)
        rows = q.order_by(Decision.committed_at.desc()).limit(lim).all()

        project_ids = {d.project_id for d in rows}
        projects = {}
        if project_ids:
            projects = {
                p.id: p
                for p in self.db.query(Project)
                .filter(
                    Project.organization_id == self.org_id,
                    Project.id.in_(list(project_ids)),
                )
                .all()
            }

        activities_by_decision: dict[str, list[dict[str, Any]]] = {}
        if include_activity and rows:
            acts = (
                self.db.query(DecisionActivity)
                .filter(
                    DecisionActivity.organization_id == self.org_id,
                    DecisionActivity.decision_id.in_([d.id for d in rows]),
                )
                .order_by(DecisionActivity.created_at.asc())
                .all()
            )
            for a in acts:
                activities_by_decision.setdefault(a.decision_id, []).append(
                    self._format_activity(a)
                )

        items = []
        for d in rows:
            project = projects.get(d.project_id)
            if not project:
                continue
            item = self._format(d, project)
            if include_activity:
                item["activities"] = activities_by_decision.get(d.id, [])
            items.append(item)

        return {
            "layer": "decision_export",
            "n": len(items),
            "include_activity": include_activity,
            "decisions": items,
            "plain_summary": f"Exported {len(items)} decision(s) for audit.",
            "exported_at": datetime.utcnow().isoformat(),
        }

    def export_csv(
        self,
        *,
        project_id: Optional[str] = None,
        assignee_user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 500,
    ) -> str:
        import csv
        import io

        payload = self.export_decisions(
            project_id=project_id,
            assignee_user_id=assignee_user_id,
            status=status,
            include_activity=False,
            limit=limit,
        )
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=[
                "id",
                "project_id",
                "project_name",
                "entity_id",
                "action_code",
                "action_name",
                "status",
                "assignee_user_id",
                "assignee_email",
                "assignee_name",
                "committed_by_user_id",
                "probability_at_commit",
                "expected_lift",
                "actual_outcome",
                "checkin_count",
                "recheck_at",
                "committed_at",
                "closed_at",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for d in payload["decisions"]:
            assignee = d.get("assignee") or {}
            writer.writerow(
                {
                    "id": d.get("id"),
                    "project_id": d.get("project_id"),
                    "project_name": d.get("project_name"),
                    "entity_id": d.get("entity_id"),
                    "action_code": d.get("action_code"),
                    "action_name": d.get("action_name"),
                    "status": d.get("status"),
                    "assignee_user_id": assignee.get("id"),
                    "assignee_email": assignee.get("email"),
                    "assignee_name": assignee.get("name"),
                    "committed_by_user_id": d.get("committed_by_user_id"),
                    "probability_at_commit": d.get("probability_at_commit"),
                    "expected_lift": d.get("expected_lift"),
                    "actual_outcome": d.get("actual_outcome"),
                    "checkin_count": d.get("checkin_count"),
                    "recheck_at": d.get("recheck_at"),
                    "committed_at": d.get("committed_at"),
                    "closed_at": d.get("closed_at"),
                }
            )
        return buf.getvalue()

    def _validate_org_user(self, user_id: str) -> str:
        uid = (user_id or "").strip()
        if not uid:
            raise ValueError("assignee_user_id is required")
        user = (
            self.db.query(User)
            .filter(
                User.id == uid,
                User.organization_id == self.org_id,
                User.is_active == True,  # noqa: E712
            )
            .first()
        )
        if not user:
            raise ValueError("Assignee must be an active member of this organization")
        return user.id

    def _user_brief(self, user_id: Optional[str]) -> Optional[dict[str, Any]]:
        if not user_id:
            return None
        user = (
            self.db.query(User)
            .filter(User.id == user_id, User.organization_id == self.org_id)
            .first()
        )
        if not user:
            return {"id": user_id, "name": None, "email": None}
        return {"id": user.id, "name": user.name, "email": user.email}

    def _log_activity(
        self,
        decision: Decision,
        *,
        event: str,
        actor_user_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        commit: bool = True,
    ) -> DecisionActivity:
        if event not in VALID_ACTIVITY_EVENTS:
            raise ValueError(f"Invalid activity event '{event}'")
        row = DecisionActivity(
            organization_id=self.org_id,
            decision_id=decision.id,
            actor_user_id=actor_user_id,
            event=event,
            payload=payload or {},
        )
        self.db.add(row)
        if commit:
            self.db.commit()
            self.db.refresh(row)
        return row

    def _format_activity(self, a: DecisionActivity) -> dict[str, Any]:
        actor = self._user_brief(a.actor_user_id)
        return {
            "id": a.id,
            "decision_id": a.decision_id,
            "event": a.event,
            "actor": actor,
            "payload": a.payload or {},
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }

    def check_in(
        self,
        project_id: str,
        decision_id: str,
        *,
        actual_outcome: Optional[str] = None,
        notes: Optional[str] = None,
        close: bool = False,
        schedule_next: bool = True,
        recheck_interval_days: Optional[int] = None,
        actor_user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record a 30/60/90 check-in; optionally close or schedule next."""
        project = self._get_project(project_id)
        d = (
            self.db.query(Decision)
            .filter(
                Decision.id == decision_id,
                Decision.project_id == project_id,
            )
            .first()
        )
        if not d:
            raise ValueError("Decision not found")
        if d.status == "cancelled":
            raise ValueError("Cannot check in a cancelled decision")
        if d.status == "closed":
            raise ValueError("Decision is already closed")

        now = datetime.utcnow()
        d.last_checkin_at = now
        d.checkin_count = int(d.checkin_count or 0) + 1
        d.status = "checking"

        if actual_outcome:
            d.actual_outcome = str(actual_outcome).strip().lower()
        if notes:
            d.outcome_notes = (d.outcome_notes or "")
            if d.outcome_notes:
                d.outcome_notes += "\n---\n"
            d.outcome_notes += f"[{now.isoformat()}Z] {notes.strip()}"

        event = "checked_in"
        if close:
            d.status = "closed"
            d.closed_at = now
            d.recheck_at = None
            event = "closed"
        elif schedule_next:
            if recheck_interval_days is not None:
                interval = int(recheck_interval_days)
                if interval not in VALID_INTERVALS:
                    raise ValueError("recheck_interval_days must be 30, 60, or 90")
                d.recheck_interval_days = interval
            d.status = "committed"
            d.recheck_at = now + timedelta(days=int(d.recheck_interval_days or 30))
            event = "rescheduled"
        else:
            # Keep open without bumping the next recheck date
            d.status = "committed"

        d.autopsy_narrative = self._build_autopsy(d, project)
        d.updated_at = now

        self._log_activity(
            d,
            event=event,
            actor_user_id=actor_user_id,
            payload={
                "actual_outcome": d.actual_outcome,
                "notes": (notes or "").strip() or None,
                "close": bool(close),
                "recheck_at": d.recheck_at.isoformat() if d.recheck_at else None,
                "recheck_interval_days": d.recheck_interval_days,
            },
            commit=False,
        )

        self.db.commit()
        self.db.refresh(d)

        payload = self._format(d, project, include_snapshot=True, include_activity=True)

        # Bridge to A7: stamp prediction when we have a known yes/no (don't wipe with unknown)
        feedback_synced: Optional[bool] = None
        feedback_sync_error: Optional[str] = None
        if d.actual_outcome:
            from app.services.project_service import ProjectService

            ps = ProjectService(self.db, self.org_id)
            kind = self._normalize_binary_outcome(d.actual_outcome, project)
            if d.prediction_id and kind in ("positive", "negative"):
                try:
                    ps.record_feedback(
                        d.prediction_id,
                        kind,
                        action_taken=d.action_code,
                        project_id=project_id,
                    )
                    feedback_synced = True
                except Exception as exc:
                    # Check-in already saved; surface sync failure for the client
                    feedback_synced = False
                    feedback_sync_error = str(exc)[:240]
                    logger.warning(
                        "A7 feedback sync failed for decision %s: %s",
                        d.id,
                        feedback_sync_error,
                    )
            else:
                # Decision-only outcomes still affect effectiveness blends
                ps._invalidate_effectiveness_cache(project_id)

        if feedback_synced is not None:
            payload["feedback_synced"] = feedback_synced
        if feedback_sync_error:
            payload["feedback_sync_error"] = feedback_sync_error
        return payload

    def _normalize_binary_outcome(self, raw: Optional[str], project: Project) -> Optional[str]:
        """Map free-text outcome via the shared A7 normalizer (positive/negative/unknown)."""
        if not raw:
            return None
        from app.services.project_service import ProjectService

        is_regression = project.problem_type == "regression"
        norm = ProjectService(self.db, self.org_id)._normalize_outcome(
            str(raw), project, is_regression=is_regression
        )
        if not norm:
            return "unknown"
        if norm.get("kind") == "unknown":
            return "unknown"
        stored = norm.get("stored")
        if stored in ("positive", "negative"):
            return stored
        return "unknown"

    def _outcome_label(self, project: Project) -> str:
        """Prefer a real business label; ignore placeholder descriptions like 'outcome'."""
        desc = (project.target_description or "").strip()
        col = (project.target_column or "").strip()
        generic = {"", "outcome", "the outcome", "target", "label", "y", "result"}
        if desc and desc.lower() not in generic:
            return desc
        return col or desc or "the outcome"

    def _build_autopsy(self, d: Decision, project: Project) -> str:
        """What we did / what we expected / what happened — business narrative."""
        outcome_label = self._outcome_label(project)
        parts = [
            f"Check-in #{int(d.checkin_count or 0)} for “{d.action_name}”.",
        ]
        if d.probability_at_commit is not None:
            parts.append(
                f"When this was saved, chance of {outcome_label} was "
                f"{float(d.probability_at_commit):.0%}."
            )
        if d.expected_lift is not None:
            lift_pp = float(d.expected_lift) * 100
            direction = "down" if lift_pp < 0 else "up" if lift_pp > 0 else "unchanged"
            parts.append(
                f"Playbook expectation was an illustrative {abs(lift_pp):.0f} pp move {direction}."
            )
        elif d.expected_probability_after is not None:
            parts.append(
                f"Playbook expected chance after action around "
                f"{float(d.expected_probability_after):.0%}."
            )

        kind = self._normalize_binary_outcome(d.actual_outcome, project)
        if kind == "positive":
            parts.append(f"Logged result: {outcome_label} did occur.")
            if d.expected_lift is not None and float(d.expected_lift) < 0:
                parts.append(
                    "That goes against the hoped-for improvement — treat this action as "
                    "not clearly effective for similar cases until more evidence accumulates."
                )
            elif d.expected_lift is not None and float(d.expected_lift) > 0:
                parts.append(
                    "Result aligns with an action that was expected to raise risk — "
                    "review whether this follow-up was the right lever."
                )
            else:
                parts.append("Compare this result with similar open cases before scaling the action.")
        elif kind == "negative":
            parts.append(f"Logged result: {outcome_label} did not occur.")
            if d.expected_lift is not None and float(d.expected_lift) < 0:
                parts.append(
                    "Outcome matches the hoped-for direction (risk reduced / event avoided). "
                    "This is weak evidence the action helped — keep logging to confirm."
                )
            else:
                parts.append(
                    "Favorable non-event so far; keep watching if the case was already low risk."
                )
        elif kind == "unknown":
            parts.append("Outcome still marked unknown — close later when the result is clear.")
        else:
            parts.append("Outcome still open — keep watching until the next recheck.")

        if d.status == "closed":
            parts.append("Follow-up closed.")
        return " ".join(parts)

    def _format(
        self,
        d: Decision,
        project: Project,
        *,
        include_snapshot: bool = False,
        include_project_name: bool = False,
        include_activity: bool = False,
    ) -> dict[str, Any]:
        now = datetime.utcnow()
        due = bool(
            d.recheck_at
            and d.recheck_at <= now
            and d.status in ("committed", "checking")
        )
        assignee = self._user_brief(getattr(d, "assignee_user_id", None))
        payload = {
            "id": d.id,
            "project_id": d.project_id,
            "project_name": project.name,
            "prediction_id": d.prediction_id,
            "entity_id": d.entity_id,
            "status": d.status,
            "action_code": d.action_code,
            "action_name": d.action_name,
            "action_description": d.action_description,
            "probability_at_commit": d.probability_at_commit,
            "risk_level_at_commit": d.risk_level_at_commit,
            "expected_probability_after": d.expected_probability_after,
            "expected_lift": d.expected_lift,
            "decision_summary": d.decision_summary,
            "recheck_interval_days": d.recheck_interval_days,
            "recheck_at": d.recheck_at.isoformat() if d.recheck_at else None,
            "due_for_recheck": due,
            "last_checkin_at": d.last_checkin_at.isoformat() if d.last_checkin_at else None,
            "checkin_count": d.checkin_count,
            "actual_outcome": d.actual_outcome,
            "outcome_notes": d.outcome_notes,
            "autopsy_narrative": d.autopsy_narrative,
            "assignee_user_id": getattr(d, "assignee_user_id", None),
            "assignee": assignee,
            "committed_by_user_id": getattr(d, "committed_by_user_id", None),
            "committed_at": d.committed_at.isoformat() if d.committed_at else None,
            "closed_at": d.closed_at.isoformat() if d.closed_at else None,
            "layer": "B3_decision_ledger",
            "plain_summary": d.decision_summary
            or f"Follow-up “{d.action_name}” ({d.status}).",
        }
        if include_snapshot:
            payload["case_snapshot"] = d.case_snapshot
        if include_activity:
            acts = (
                self.db.query(DecisionActivity)
                .filter(DecisionActivity.decision_id == d.id)
                .order_by(DecisionActivity.created_at.desc())
                .limit(8)
                .all()
            )
            payload["activities"] = [self._format_activity(a) for a in acts]
        if not include_project_name and not include_snapshot:
            # Keep project_name always — useful for deep links; cheap field
            pass
        return payload
