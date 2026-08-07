"""B3 Decision ledger — commit actions as durable, recheckable decisions."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.db.models import Decision, Project, ProjectPrediction
from app.recommendations.action_catalog import get_action
from app.recommendations.domains import detect_domain


VALID_STATUSES = {"proposed", "committed", "checking", "closed", "cancelled"}
VALID_INTERVALS = {30, 60, 90}


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
        )
        self.db.add(decision)

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
        items = [self._format(d, project) for d in rows]
        plain = (
            f"{len(items)} decision(s) on the ledger"
            + (f"; {len(due)} due for recheck." if due else ".")
        )
        return {
            "project_id": project_id,
            "layer": "B3_decision_ledger",
            "n": len(items),
            "due_for_recheck": len(due),
            "decisions": items,
            "plain_summary": plain,
        }

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
        return self._format(d, project, include_snapshot=True)

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

        if close:
            d.status = "closed"
            d.closed_at = now
            d.recheck_at = None
        elif schedule_next:
            if recheck_interval_days is not None:
                interval = int(recheck_interval_days)
                if interval not in VALID_INTERVALS:
                    raise ValueError("recheck_interval_days must be 30, 60, or 90")
                d.recheck_interval_days = interval
            d.status = "committed"
            d.recheck_at = now + timedelta(days=int(d.recheck_interval_days or 30))
        else:
            # Keep open without bumping the next recheck date
            d.status = "committed"

        d.autopsy_narrative = self._build_autopsy(d, project)
        d.updated_at = now
        self.db.commit()
        self.db.refresh(d)

        # Bridge to A7: stamp prediction when we have a known yes/no (don't wipe with unknown)
        if d.prediction_id and d.actual_outcome:
            kind = self._normalize_binary_outcome(d.actual_outcome, project)
            if kind in ("positive", "negative"):
                try:
                    from app.services.project_service import ProjectService

                    ProjectService(self.db, self.org_id).record_feedback(
                        d.prediction_id,
                        d.actual_outcome,
                        action_taken=d.action_code,
                        project_id=project_id,
                    )
                except Exception:
                    # Check-in already saved; A7 sync is best-effort
                    pass

        return self._format(d, project, include_snapshot=True)

    def _normalize_binary_outcome(self, raw: Optional[str], project: Project) -> Optional[str]:
        """Map free-text outcome to positive / negative / unknown (classification)."""
        if not raw:
            return None
        low = str(raw).strip().lower()
        if not low:
            return None
        pos = str(project.target_positive_label or "1").strip().lower()
        if low in ("unknown", "unk", "n/a", "na", "pending"):
            return "unknown"
        if low in ("positive", "yes", "1", "true", "y") or low == pos:
            return "positive"
        if low in ("negative", "no", "0", "false", "n"):
            return "negative"
        # Non-positive label text often means the event did not happen
        if pos and low != pos:
            return "negative"
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
    ) -> dict[str, Any]:
        now = datetime.utcnow()
        due = bool(
            d.recheck_at
            and d.recheck_at <= now
            and d.status in ("committed", "checking")
        )
        payload = {
            "id": d.id,
            "project_id": d.project_id,
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
            "committed_at": d.committed_at.isoformat() if d.committed_at else None,
            "closed_at": d.closed_at.isoformat() if d.closed_at else None,
            "layer": "B3_decision_ledger",
            "plain_summary": d.decision_summary
            or f"Decision “{d.action_name}” ({d.status}).",
        }
        if include_snapshot:
            payload["case_snapshot"] = d.case_snapshot
            payload["project_name"] = project.name
        return payload
