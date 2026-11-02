import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "temporal_mapping"
    / "xd_temporal_scores_218.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_VALIDATION_RECALL = 0.98
RANDOM_SEED = 42


def score_metrics(labels, scores):
    return {
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "roc_auc": float(
            roc_auc_score(labels, scores)
        ),
        "brier_score": float(
            brier_score_loss(labels, scores)
        ),
    }


def retention_metrics(labels, retain):
    labels = np.asarray(labels, dtype=int)
    retain = np.asarray(retain, dtype=int)

    positives = int(labels.sum())
    retained = int(retain.sum())

    true_retained = int(
        ((labels == 1) & (retain == 1)).sum()
    )

    false_retained = int(
        ((labels == 0) & (retain == 1)).sum()
    )

    missed = int(
        ((labels == 1) & (retain == 0)).sum()
    )

    return {
        "temporal_evidence_recall": (
            true_retained / positives
            if positives > 0
            else 0.0
        ),
        "retention_ratio": retained / len(labels),
        "retained_unit_percent": (
            100.0 * retained / len(labels)
        ),
        "storage_reduction_percent": (
            100.0 * (1.0 - retained / len(labels))
        ),
        "retained_temporal_units": retained,
        "true_retained_units": true_retained,
        "false_retained_units": false_retained,
        "missed_anomalous_units": missed,
        "retention_precision": float(
            precision_score(
                labels,
                retain,
                zero_division=0,
            )
        ),
    }


def select_threshold(validation_df):
    probabilities = validation_df[
        "event_probability"
    ].to_numpy()

    labels = validation_df[
        "anomaly_label"
    ].to_numpy(dtype=int)

    thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.0, 1.0]),
                probabilities,
            ]
        )
    )

    curve = []

    for threshold in thresholds:
        retain = (probabilities >= threshold).astype(int)

        metrics = retention_metrics(
            labels,
            retain,
        )

        curve.append(
            {
                "threshold": float(threshold),
                **metrics,
            }
        )

    valid = [
        row
        for row in curve
        if row["temporal_evidence_recall"]
        >= TARGET_VALIDATION_RECALL
    ]

    if not valid:
        raise RuntimeError(
            "No threshold meets the validation recall target."
        )

    selected = sorted(
        valid,
        key=lambda row: (
            row["retention_ratio"],
            -row["threshold"],
        ),
    )[0]

    return selected, curve


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing temporal score file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "split",
        "offline_raw_score",
        "anomaly_label",
        "video_id",
        "temporal_index",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    evaluation_df = df[
        df["split"] == "evaluation"
    ].copy()

    if validation_df.empty or evaluation_df.empty:
        raise RuntimeError(
            "Validation or evaluation subset is empty."
        )

    x_validation = validation_df[
        ["offline_raw_score"]
    ].to_numpy()

    y_validation = validation_df[
        "anomaly_label"
    ].to_numpy(dtype=int)

    x_evaluation = evaluation_df[
        ["offline_raw_score"]
    ].to_numpy()

    y_evaluation = evaluation_df[
        "anomaly_label"
    ].to_numpy(dtype=int)

    calibrator = LogisticRegression(
        random_state=RANDOM_SEED,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(
        x_validation,
        y_validation,
    )

    validation_df["event_probability"] = (
        calibrator.predict_proba(x_validation)[:, 1]
    )

    evaluation_df["event_probability"] = (
        calibrator.predict_proba(x_evaluation)[:, 1]
    )

    validation_raw = score_metrics(
        y_validation,
        validation_df["offline_raw_score"].to_numpy(),
    )

    validation_calibrated = score_metrics(
        y_validation,
        validation_df["event_probability"].to_numpy(),
    )

    evaluation_raw = score_metrics(
        y_evaluation,
        evaluation_df["offline_raw_score"].to_numpy(),
    )

    evaluation_calibrated = score_metrics(
        y_evaluation,
        evaluation_df["event_probability"].to_numpy(),
    )

    selected, threshold_curve = select_threshold(
        validation_df
    )

    threshold = selected["threshold"]

    c0_retain = np.ones(
        len(evaluation_df),
        dtype=int,
    )

    c2_retain = (
        evaluation_df["event_probability"].to_numpy()
        >= threshold
    ).astype(int)

    rng = np.random.default_rng(RANDOM_SEED)

    random_retain = np.zeros(
        len(evaluation_df),
        dtype=int,
    )

    random_indices = rng.choice(
        len(evaluation_df),
        size=int(c2_retain.sum()),
        replace=False,
    )

    random_retain[random_indices] = 1

    evaluation_df["c0_continuous_retain"] = c0_retain
    evaluation_df["c2_score_triggered_retain"] = c2_retain
    evaluation_df["c6_random_matched_retain"] = random_retain

    decision_path = (
        OUTPUT_DIR
        / "xd_temporal_retention_pilot_decisions.csv"
    )

    evaluation_df.to_csv(
        decision_path,
        index=False,
    )

    result_rows = [
        {
            "policy": "C0_continuous_retention",
            "threshold": 0.0,
            **retention_metrics(
                y_evaluation,
                c0_retain,
            ),
        },
        {
            "policy": "C2_temporal_score_triggered",
            "threshold": threshold,
            **retention_metrics(
                y_evaluation,
                c2_retain,
            ),
        },
        {
            "policy": "C6_random_duration_matched",
            "threshold": None,
            **retention_metrics(
                y_evaluation,
                random_retain,
            ),
        },
    ]

    result_path = (
        OUTPUT_DIR
        / "xd_temporal_retention_pilot_results.csv"
    )

    pd.DataFrame(result_rows).to_csv(
        result_path,
        index=False,
    )

    curve_path = (
        OUTPUT_DIR
        / "xd_temporal_retention_threshold_curve.csv"
    )

    pd.DataFrame(threshold_curve).to_csv(
        curve_path,
        index=False,
    )

    calibrated_path = (
        OUTPUT_DIR
        / "xd_temporal_calibrated_scores.csv"
    )

    pd.concat(
        [validation_df, evaluation_df],
        ignore_index=True,
    ).to_csv(
        calibrated_path,
        index=False,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "experiment_type": "PILOT_ONLY",
        "validation_video_count": int(
            validation_df["video_id"].nunique()
        ),
        "evaluation_video_count": int(
            evaluation_df["video_id"].nunique()
        ),
        "validation_temporal_units": int(
            len(validation_df)
        ),
        "evaluation_temporal_units": int(
            len(evaluation_df)
        ),
        "calibration_method": (
            "Logistic regression / Platt scaling"
        ),
        "threshold_target_validation_recall": (
            TARGET_VALIDATION_RECALL
        ),
        "selected_threshold": threshold,
        "selected_validation_policy_metrics": selected,
        "validation_raw_metrics": validation_raw,
        "validation_calibrated_metrics": (
            validation_calibrated
        ),
        "evaluation_raw_metrics": evaluation_raw,
        "evaluation_calibrated_metrics": (
            evaluation_calibrated
        ),
        "evaluation_retention_results": result_rows,
        "scope_note": (
            "This is a feature-level temporal retention pilot. "
            "It does not use raw videos, context-conditioned VAS, "
            "temporal clip merging, encryption, IPFS, or Fabric."
        ),
        "files": {
            "calibrated_scores": str(calibrated_path),
            "decisions": str(decision_path),
            "results": str(result_path),
            "threshold_curve": str(curve_path),
        },
    }

    report_path = (
        LOG_DIR
        / "xd_temporal_retention_pilot.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    c2_result = result_rows[1]
    random_result = result_rows[2]

    print("Temporal retention pilot: PASS")
    print(
        f"Selected threshold: "
        f"{threshold:.6f}"
    )
    print()
    print(
        "Evaluation temporal AP, raw: "
        f"{evaluation_raw['average_precision']:.6f}"
    )
    print(
        "Evaluation temporal AP, calibrated: "
        f"{evaluation_calibrated['average_precision']:.6f}"
    )
    print()
    print(
        "C2 temporal evidence recall: "
        f"{c2_result['temporal_evidence_recall']:.6f}"
    )
    print(
        "C2 retained temporal units: "
        f"{c2_result['retained_unit_percent']:.2f}%"
    )
    print(
        "C2 storage reduction: "
        f"{c2_result['storage_reduction_percent']:.2f}%"
    )
    print(
        "C2 missed anomalous units: "
        f"{c2_result['missed_anomalous_units']}"
    )
    print()
    print(
        "Random matched evidence recall: "
        f"{random_result['temporal_evidence_recall']:.6f}"
    )
    print()
    print(f"Results: {result_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()