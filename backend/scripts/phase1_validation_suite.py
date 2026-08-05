#!/usr/bin/env python3
"""Phase-1 prediction/confidence validation suite (5 checks).

1. Spot-check style metrics on the full held-out test set
2. Conformal residual coverage: share of y ∈ [lower, upper]
3. Accuracy split by low_confidence vs not
4. Accuracy vs majority-class baseline
5. Same metrics across 3 independent random train/calib/test splits

Run from backend/:
  .venv311/bin/python scripts/phase1_validation_suite.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Allow `python scripts/...` from backend root
BACKEND_ROOT = Path(__file__).resolve().parents[1]
os.chdir(BACKEND_ROOT)
sys.path.insert(0, str(BACKEND_ROOT))

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.config import get_settings
from app.database import SessionLocal
from app.db.models import Project
from app.ml.model_loader import build_model_for_strategy, load_routed_model
from app.ml.pipelines.feature_pipeline import FeatureTransformer
from app.ml.router import route_training
from app.ml.soft_range import interval_is_soft
from app.ml.threshold_tuning import get_decision_threshold, tune_decision_threshold
from app.services.dataset_service import DatasetService
from app.services.project_service import ProjectService


@dataclass
class EvalBundle:
    label: str
    n: int
    n_pos: int
    n_neg: int
    majority_class: int
    majority_baseline_accuracy: float
    accuracy: float
    lift_vs_majority_pp: float
    agree_rate: float  # same as accuracy at 0.5 for binary
    auc: Optional[float]
    brier: Optional[float]
    high_risk_n: int
    high_risk_precision: Optional[float]
    low_risk_n: int
    low_risk_true_neg_rate: Optional[float]
    soft_share: float
    # Conformal residual coverage: y ∈ [lo, hi]
    conformal_target: float
    conformal_coverage: float
    conformal_coverage_gap: float  # coverage - target (positive = over-cover)
    mean_interval_width: float
    median_interval_width: float
    conformal_quantile: Optional[float]
    # Confidence splits
    n_low_conf: int
    n_high_conf: int
    acc_low_conf: Optional[float]
    acc_high_conf: Optional[float]
    soft_low_conf_share: Optional[float]
    soft_high_conf_share: Optional[float]

    def verdict_bits(self) -> dict[str, str]:
        bits = {}
        bits["beats_majority"] = (
            "pass" if self.accuracy > self.majority_baseline_accuracy + 0.02 else
            "borderline" if self.accuracy > self.majority_baseline_accuracy else
            "fail"
        )
        # Residual conformal is often wide → coverage should be ≥ target
        if self.conformal_coverage + 0.02 >= self.conformal_target:
            bits["coverage"] = "pass"
        elif self.conformal_coverage + 0.08 >= self.conformal_target:
            bits["coverage"] = "borderline"
        else:
            bits["coverage"] = "fail"
        # High-conf should not be *worse* than low-conf when both have samples
        if self.acc_high_conf is None or self.acc_low_conf is None:
            bits["conf_split"] = "n/a"
        elif self.acc_high_conf + 0.01 >= self.acc_low_conf:
            bits["conf_split"] = "pass"
        elif self.acc_high_conf + 0.05 >= self.acc_low_conf:
            bits["conf_split"] = "borderline"
        else:
            bits["conf_split"] = "fail"
        bits["soft_not_all"] = "pass" if self.soft_share < 0.95 else "fail"
        return bits


def _score_frame(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    label: str,
    decision_threshold: float = 0.5,
) -> EvalBundle:
    y = np.asarray(y, dtype=int).ravel()
    n = len(y)
    if n == 0:
        raise ValueError("Empty evaluation frame")

    # Batch point + intervals
    p = np.asarray(model.predict_proba(X), dtype=float).ravel()
    lower, upper, width = model.calibrator.predict_interval(p)

    # Rebuild low_confidence with same policy as production
    disagreement = np.zeros(n, dtype=float)
    if hasattr(model, "_point_disagreement"):
        disagreement = np.asarray(model._point_disagreement(X), dtype=float).ravel()
    elif hasattr(model, "_base_disagreement"):
        try:
            disagreement = np.asarray(model._base_disagreement(X), dtype=float).ravel()
        except Exception:
            disagreement = np.zeros(n)

    low_conf = np.zeros(n, dtype=bool)
    for i in range(n):
        u = model.calibrator.evaluate_uncertainty(
            y_pred=float(p[i]),
            disagreement=float(disagreement[i]) if i < len(disagreement) else 0.0,
        )
        # Prefer recomputed interval from batch for consistency
        low_conf[i] = bool(u.low_confidence)

    # Soft range (aligned with UI/spot-check)
    soft = np.array(
        [
            interval_is_soft(
                point=float(p[i]),
                lower=float(lower[i]),
                upper=float(upper[i]),
                low_confidence=bool(low_conf[i]),
                is_regression=False,
            )["is_soft"]
            for i in range(n)
        ],
        dtype=bool,
    )

    pred = (p >= decision_threshold).astype(int)
    acc = float(accuracy_score(y, pred))

    # Majority baseline on *this* eval set
    n_pos = int(y.sum())
    n_neg = n - n_pos
    maj = 1 if n_pos >= n_neg else 0
    maj_acc = max(n_pos, n_neg) / n

    # Residual conformal coverage: binary y in interval
    covered = (y.astype(float) >= lower) & (y.astype(float) <= upper)
    coverage = float(covered.mean())
    target = float(getattr(model.calibrator, "coverage_level", 0.9))

    # Risk buckets
    high_mask = p >= 0.6
    low_mask = p < 0.4
    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    high_prec = float(y[high_mask].mean()) if n_high else None
    low_tnr = float((y[low_mask] == 0).mean()) if n_low else None

    auc = None
    brier = None
    if len(np.unique(y)) > 1:
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = None
        brier = float(brier_score_loss(y, p))

    lc = low_conf
    hc = ~low_conf
    n_lc = int(lc.sum())
    n_hc = int(hc.sum())
    acc_lc = float(accuracy_score(y[lc], pred[lc])) if n_lc else None
    acc_hc = float(accuracy_score(y[hc], pred[hc])) if n_hc else None
    soft_lc = float(soft[lc].mean()) if n_lc else None
    soft_hc = float(soft[hc].mean()) if n_hc else None

    q = getattr(model.calibrator, "quantile", None)

    return EvalBundle(
        label=label,
        n=n,
        n_pos=n_pos,
        n_neg=n_neg,
        majority_class=maj,
        majority_baseline_accuracy=round(maj_acc, 4),
        accuracy=round(acc, 4),
        lift_vs_majority_pp=round((acc - maj_acc) * 100, 2),
        agree_rate=round(acc, 4),
        auc=round(auc, 4) if auc is not None else None,
        brier=round(brier, 4) if brier is not None else None,
        high_risk_n=n_high,
        high_risk_precision=round(high_prec, 4) if high_prec is not None else None,
        low_risk_n=n_low,
        low_risk_true_neg_rate=round(low_tnr, 4) if low_tnr is not None else None,
        soft_share=round(float(soft.mean()), 4),
        conformal_target=round(target, 4),
        conformal_coverage=round(coverage, 4),
        conformal_coverage_gap=round(coverage - target, 4),
        mean_interval_width=round(float(np.mean(width)), 4),
        median_interval_width=round(float(np.median(width)), 4),
        conformal_quantile=round(float(q), 4) if q is not None else None,
        n_low_conf=n_lc,
        n_high_conf=n_hc,
        acc_low_conf=round(acc_lc, 4) if acc_lc is not None else None,
        acc_high_conf=round(acc_hc, 4) if acc_hc is not None else None,
        soft_low_conf_share=round(soft_lc, 4) if soft_lc is not None else None,
        soft_high_conf_share=round(soft_hc, 4) if soft_hc is not None else None,
    )


def _prepare_project_xy(db, project: Project) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    ds = DatasetService(db, project.organization_id)
    df = ds.load_dataframe(project.dataset_id)
    X_raw = df[project.feature_columns].copy()
    target_values = df[project.target_column].astype(str)
    pos = str(project.target_positive_label)
    y = (target_values == pos).astype(int)
    return X_raw, y, df


def eval_live_model(db, project: Project) -> EvalBundle:
    svc = ProjectService(db, project.organization_id)
    trained = svc.get_active_model(project.id)
    if not trained:
        raise RuntimeError("No active trained model")
    model = load_routed_model(trained.model_path, problem_type=project.problem_type)

    test_path = os.path.join(trained.model_path, "test_data.parquet")
    if not os.path.exists(test_path):
        raise RuntimeError(f"Missing test_data.parquet at {test_path}")

    test_df = pd.read_parquet(test_path)
    # Encode with training transformer if present
    from app.ml.pipelines.feature_pipeline import FeatureTransformer

    ft_path = trained.model_path
    transformer = FeatureTransformer()
    if os.path.exists(os.path.join(ft_path, "feature_transformer.joblib")) or os.path.exists(
        os.path.join(ft_path, "feature_config.json")
    ):
        try:
            transformer.load(ft_path)
            X = transformer.transform(test_df[project.feature_columns])
        except Exception:
            # Fallback: prepare per-row via service (slower)
            X = None
    else:
        X = None

    pos = str(project.target_positive_label)
    y = (test_df[project.target_column].astype(str) == pos).astype(int).to_numpy()

    if X is None:
        # Align features to model.feature_names when possible
        rows = []
        for _, row in test_df.iterrows():
            features = {c: row.get(c) for c in (project.feature_columns or [])}
            frame = svc._prepare_feature_frame(project, trained, features, model)
            rows.append(frame.iloc[0])
        X = pd.DataFrame(rows)

    # Ensure column order
    if getattr(model, "feature_names", None):
        for c in model.feature_names:
            if c not in X.columns:
                X[c] = 0
        X = X[model.feature_names]

    return _score_frame(
        model,
        X,
        y,
        label=f"live_full_test (n={len(y)})",
        decision_threshold=get_decision_threshold(project),
    )


def eval_random_split(
    X_raw: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    settings,
) -> EvalBundle:
    """Train from scratch on a random 60/16/20-ish split and evaluate on its test set."""
    strat = y if y.nunique() > 1 else None
    X_temp, X_test_raw, y_temp, y_test = train_test_split(
        X_raw,
        y,
        test_size=settings.test_size,
        random_state=seed,
        stratify=strat,
    )
    calib_frac = min(0.5, max(0.05, settings.calib_size / max(1e-6, (1.0 - settings.test_size))))
    try:
        X_train_raw, X_calib_raw, y_train, y_calib = train_test_split(
            X_temp,
            y_temp,
            test_size=calib_frac,
            random_state=seed,
            stratify=y_temp if y_temp.nunique() > 1 else None,
        )
    except ValueError:
        X_train_raw, X_calib_raw, y_train, y_calib = train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,
            random_state=seed,
        )

    transformer = FeatureTransformer(
        drop_leakage=settings.drop_leakage_columns,
        add_missing_indicators=settings.add_missing_indicators,
    )
    X_train = transformer.fit_transform(X_train_raw, y_train, protected_columns=list(X_raw.columns))
    X_calib = transformer.transform(X_calib_raw)
    X_test = transformer.transform(X_test_raw)

    force = None
    if settings.routing_mode in ("foundation_model", "ensemble"):
        force = settings.routing_mode
    decision = route_training(
        X_train,
        max_foundation_rows=settings.foundation_max_rows,
        max_foundation_features=settings.foundation_max_features,
        force_strategy=force,
    )
    model = build_model_for_strategy(decision.strategy, problem_type="binary_classification")
    # Keep multi-seed runs practical (Optuna thrashing not the goal of this suite)
    if hasattr(model, "enable_optuna"):
        model.enable_optuna = False
    if hasattr(model, "random_state"):
        model.random_state = seed

    t0 = time.time()
    model.train(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        calibration_data=(X_calib, y_calib),
    )
    train_s = time.time() - t0
    print(
        f"  seed={seed} strategy={decision.strategy} backend={getattr(model, 'backend', '?')} "
        f"train={train_s:.1f}s n_test={len(y_test)}"
    )

    pos_rate = float(y_calib.mean())
    thr_metric = (
        "accuracy"
        if pos_rate < 0.35 or pos_rate > 0.65
        else "balanced_accuracy"
    )
    thr, _ = tune_decision_threshold(
        y_calib.to_numpy(),
        model.predict_proba(X_calib),
        metric=thr_metric,
    )

    return _score_frame(
        model,
        X_test,
        y_test.to_numpy(),
        label=f"split_seed_{seed}",
        decision_threshold=thr,
    )


def _fmt_bundle(b: EvalBundle) -> str:
    lines = [
        f"### {b.label}",
        f"  n={b.n}  pos={b.n_pos} ({b.n_pos/b.n:.1%})  neg={b.n_neg}",
        f"  accuracy={b.accuracy:.1%}  majority_baseline={b.majority_baseline_accuracy:.1%}  "
        f"lift={b.lift_vs_majority_pp:+.1f} pp",
        f"  auc={b.auc}  brier={b.brier}",
        f"  high-risk@0.6: n={b.high_risk_n} precision={b.high_risk_precision}",
        f"  low-risk@0.4:  n={b.low_risk_n} true-neg={b.low_risk_true_neg_rate}",
        f"  soft_share={b.soft_share:.1%}",
        f"  conformal: target={b.conformal_target:.0%} coverage={b.conformal_coverage:.1%} "
        f"gap={b.conformal_coverage_gap:+.1%}  median_width={b.median_interval_width:.3f} "
        f"q={b.conformal_quantile}",
        f"  conf split: low_conf n={b.n_low_conf} acc={b.acc_low_conf} | "
        f"high_conf n={b.n_high_conf} acc={b.acc_high_conf}",
        f"  gates: {b.verdict_bits()}",
    ]
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Phase-1 validation suite")
    parser.add_argument(
        "--project-id",
        default=None,
        help="Project UUID (prefix ok). Default: first trained classification project.",
    )
    args = parser.parse_args()

    settings = get_settings()
    out_dir = BACKEND_ROOT / "data" / "eval_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        q = (
            db.query(Project)
            .filter(Project.status.in_(["trained", "ready"]))
            .filter(Project.problem_type != "regression")
        )
        if args.project_id:
            pid = args.project_id
            projects = [p for p in q.all() if p.id == pid or p.id.startswith(pid)]
        else:
            projects = q.all()
        if not projects:
            print("No trained classification projects found.")
            return 1
        project = projects[0]
        print(f"Project: {project.name} ({project.id})")
        print(f"Target: {project.target_column} positive={project.target_positive_label}")

        report: dict[str, Any] = {
            "project_id": project.id,
            "project_name": project.name,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checks": {},
        }

        # --- Live full held-out (checks 1–4) ---
        print("\n=== LIVE MODEL — full held-out test ===")
        live = eval_live_model(db, project)
        print(_fmt_bundle(live))
        report["checks"]["1_large_spot_check"] = asdict(live)
        report["checks"]["2_conformal_coverage"] = {
            "target": live.conformal_target,
            "observed": live.conformal_coverage,
            "gap": live.conformal_coverage_gap,
            "mean_width": live.mean_interval_width,
            "median_width": live.median_interval_width,
            "quantile": live.conformal_quantile,
            "note": "Coverage = share of true binary labels y inside residual interval [p−q, p+q]∩[0,1]",
        }
        report["checks"]["3_accuracy_by_confidence"] = {
            "n_low_conf": live.n_low_conf,
            "acc_low_conf": live.acc_low_conf,
            "n_high_conf": live.n_high_conf,
            "acc_high_conf": live.acc_high_conf,
        }
        report["checks"]["4_vs_majority_baseline"] = {
            "accuracy": live.accuracy,
            "majority_baseline": live.majority_baseline_accuracy,
            "majority_class": live.majority_class,
            "lift_pp": live.lift_vs_majority_pp,
        }

        # --- 3 random splits (check 5) ---
        print("\n=== THREE RANDOM SPLITS (retrain + eval) ===")
        X_raw, y, _ = _prepare_project_xy(db, project)
        print(f"Full data: n={len(y)} pos_rate={y.mean():.3f}")
        seeds = [7, 21, 99]
        split_results: list[EvalBundle] = []
        for seed in seeds:
            print(f"\n--- seed {seed} ---")
            b = eval_random_split(X_raw, y, seed=seed, settings=settings)
            print(_fmt_bundle(b))
            split_results.append(b)

        report["checks"]["5_three_random_splits"] = [asdict(b) for b in split_results]

        # Stability summary
        accs = [b.accuracy for b in split_results]
        covs = [b.conformal_coverage for b in split_results]
        lifts = [b.lift_vs_majority_pp for b in split_results]
        softs = [b.soft_share for b in split_results]
        report["summary"] = {
            "live": asdict(live),
            "splits": {
                "accuracy_mean": round(float(np.mean(accs)), 4),
                "accuracy_std": round(float(np.std(accs)), 4),
                "accuracy_min": round(float(np.min(accs)), 4),
                "coverage_mean": round(float(np.mean(covs)), 4),
                "coverage_min": round(float(np.min(covs)), 4),
                "lift_pp_mean": round(float(np.mean(lifts)), 2),
                "lift_pp_min": round(float(np.min(lifts)), 2),
                "soft_share_mean": round(float(np.mean(softs)), 4),
            },
            "verdict_live": live.verdict_bits(),
            "verdict_splits": [b.verdict_bits() for b in split_results],
        }

        # Overall pass heuristic
        live_ok = all(
            live.verdict_bits()[k] in ("pass", "borderline", "n/a")
            for k in ("beats_majority", "coverage", "conf_split", "soft_not_all")
        )
        splits_ok = all(
            all(v[k] in ("pass", "borderline", "n/a") for k in ("beats_majority", "coverage", "soft_not_all"))
            for v in report["summary"]["verdict_splits"]
        )
        splits_stable = float(np.std(accs)) < 0.05  # <5 pp std
        report["overall"] = {
            "live_checks_ok": live_ok,
            "splits_checks_ok": splits_ok,
            "splits_stable_acc": splits_stable,
            "recommendation": (
                "solid_enough_for_phase2"
                if live_ok and splits_ok and splits_stable
                else "investigate_before_phase2"
            ),
        }

        out_path = out_dir / f"phase1_validation_{project.id[:8]}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n=== OVERALL: {report['overall']['recommendation']} ===")
        print(f"Wrote {out_path}")
        return 0 if report["overall"]["recommendation"] == "solid_enough_for_phase2" else 2
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
