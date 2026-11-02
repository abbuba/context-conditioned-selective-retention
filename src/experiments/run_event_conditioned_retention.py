import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "scores"
    / "xdviodel_offline_calibrated_event_probabilities.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_VALIDATION_EVIDENCE_RECALL = 0.98
RANDOM_SEED = 42


def calculate_policy_metrics(labels, retain):
    labels = np.asarray(labels, dtype=int)
    retain = np.asarray(retain, dtype=int)

    positive_count = int(labels.sum())
    retained_count = int(retain.sum())

    evidence_recall = (
        float(((retain == 1) & (labels == 1)).sum() / positive_count)
        if positive_count > 0
        else 0.0
    )

    retention_ratio = float(retain.mean())
    storage_reduction = float(100.0 * (1.0 - retention_ratio))

    false_preserved = int(
        ((retain == 1) & (labels == 0)).sum()
    )

    missed_events = int(
        ((retain == 0) & (labels == 1)).sum()
    )

    precision = float(
        precision_score(
            labels,
            retain,
            zero_division=0,
        )
    )

    return {
        "evidence_recall": evidence_recall,
        "retention_ratio": retention_ratio,
        "retained_video_percent": 100.0 * retention_ratio,
        "storage_reduction_percent": storage_reduction,
        "retained_video_count": retained_count,
        "false_preserved_video_count": false_preserved,
        "missed_event_count": missed_events,
        "preservation_precision": precision,
    }


def select_threshold(validation_df):
    thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.0, 1.0]),
                validation_df["event_probability"].to_numpy(),
            ]
        )
    )

    candidates = []

    labels = validation_df["weak_label"].to_numpy(dtype=int)
    probabilities = validation_df["event_probability"].to_numpy()

    for threshold in thresholds:
        retain = (probabilities >= threshold).astype(int)

        metrics = calculate_policy_metrics(
            labels,
            retain,
        )

        candidates.append(
            {
                "threshold": float(threshold),
                **metrics,
            }
        )

    valid_candidates = [
        item
        for item in candidates
        if item["evidence_recall"]
        >= TARGET_VALIDATION_EVIDENCE_RECALL
    ]

    if not valid_candidates:
        raise RuntimeError(
            "No validation threshold achieved the required "
            "evidence-recall target."
        )

    selected = sorted(
        valid_candidates,
        key=lambda item: (
            item["retention_ratio"],
            -item["threshold"],
        ),
    )[0]

    return selected, candidates


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing calibrated score file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    validation_df = df[df["split"] == "validation"].copy()
    evaluation_df = df[df["split"] == "evaluation"].copy()

    if len(validation_df) != 160 or len(evaluation_df) != 640:
        raise RuntimeError(
            "Expected 160 validation and 640 evaluation rows."
        )

    selected, curve_rows = select_threshold(validation_df)

    threshold = selected["threshold"]

    evaluation_labels = evaluation_df["weak_label"].to_numpy(dtype=int)
    evaluation_probability = (
        evaluation_df["event_probability"].to_numpy(dtype=float)
    )

    c0_retain = np.ones(
        len(evaluation_df),
        dtype=int,
    )

    c2_retain = (
        evaluation_probability >= threshold
    ).astype(int)

    rng = np.random.default_rng(RANDOM_SEED)

    random_retain = np.zeros(
        len(evaluation_df),
        dtype=int,
    )

    retained_count = int(c2_retain.sum())

    random_indices = rng.choice(
        len(evaluation_df),
        size=retained_count,
        replace=False,
    )

    random_retain[random_indices] = 1

    c0_metrics = calculate_policy_metrics(
        evaluation_labels,
        c0_retain,
    )

    c2_metrics = calculate_policy_metrics(
        evaluation_labels,
        c2_retain,
    )

    random_metrics = calculate_policy_metrics(
        evaluation_labels,
        random_retain,
    )

    evaluation_df["c0_continuous_retain"] = c0_retain
    evaluation_df["c2_event_conditioned_retain"] = c2_retain
    evaluation_df["c6_random_matched_retain"] = random_retain

    decision_path = (
        OUTPUT_DIR
        / "xd_event_conditioned_retention_decisions.csv"
    )

    evaluation_df.to_csv(
        decision_path,
        index=False,
    )

    curve_path = (
        OUTPUT_DIR
        / "xd_validation_retention_threshold_curve.csv"
    )

    pd.DataFrame(curve_rows).to_csv(
        curve_path,
        index=False,
    )

    result_rows = [
        {
            "policy": "C0_continuous_retention",
            "threshold": 0.0,
            **c0_metrics,
        },
        {
            "policy": "C2_audio_video_event_conditioned",
            "threshold": threshold,
            **c2_metrics,
        },
        {
            "policy": "C6_random_duration_matched",
            "threshold": None,
            **random_metrics,
        },
    ]

    result_path = (
        OUTPUT_DIR
        / "xd_event_conditioned_retention_results.csv"
    )

    pd.DataFrame(result_rows).to_csv(
        result_path,
        index=False,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "score_input": str(INPUT_FILE),
        "validation_videos": int(len(validation_df)),
        "evaluation_videos": int(len(evaluation_df)),
        "threshold_selection": {
            "target_validation_evidence_recall":
                TARGET_VALIDATION_EVIDENCE_RECALL,
            "selected_threshold": threshold,
            "selected_validation_metrics": selected,
        },
        "evaluation_results": {
            "C0_continuous_retention": c0_metrics,
            "C2_audio_video_event_conditioned": c2_metrics,
            "C6_random_duration_matched": random_metrics,
        },
        "random_seed": RANDOM_SEED,
        "scope_note": (
            "This is a video-level retention simulation based on "
            "feature-level event probabilities. It does not yet represent "
            "temporal raw-video clip extraction."
        ),
        "decision_file": str(decision_path),
        "threshold_curve_file": str(curve_path),
        "results_file": str(result_path),
    }

    report_path = (
        LOG_DIR
        / "xd_event_conditioned_retention_experiment.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Event-conditioned retention: PASS")
    print(
        "Selected threshold: "
        f"{threshold:.6f}"
    )
    print()
    print(
        "C2 evidence recall: "
        f"{c2_metrics['evidence_recall']:.6f}"
    )
    print(
        "C2 retained video: "
        f"{c2_metrics['retained_video_percent']:.2f}%"
    )
    print(
        "C2 storage reduction: "
        f"{c2_metrics['storage_reduction_percent']:.2f}%"
    )
    print(
        "C2 missed events: "
        f"{c2_metrics['missed_event_count']}"
    )
    print()
    print(
        "Random matched evidence recall: "
        f"{random_metrics['evidence_recall']:.6f}"
    )
    print(f"Results: {result_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()