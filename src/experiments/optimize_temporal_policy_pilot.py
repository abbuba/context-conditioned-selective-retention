import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_temporal_calibrated_scores.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_VALIDATION_UNIT_RECALL = 0.98
EVENT_COVERAGE_THRESHOLD = 0.80

SMOOTHING_WINDOWS = [1, 3, 5]
MIN_TRIGGER_UNITS = [1, 2]
PRE_BUFFER_UNITS = [0, 1, 2]
POST_BUFFER_UNITS = [0, 1, 2]

THRESHOLDS = np.linspace(
    0.01,
    0.99,
    99,
)


def smooth_scores(scores, window):
    if window == 1:
        return scores.copy()

    kernel = np.ones(window) / window

    return np.convolve(
        scores,
        kernel,
        mode="same",
    )


def remove_short_runs(trigger, minimum_length):
    result = trigger.copy()
    start = None

    for index in range(len(trigger) + 1):
        current = trigger[index] if index < len(trigger) else 0

        if current == 1 and start is None:
            start = index

        if current == 0 and start is not None:
            length = index - start

            if length < minimum_length:
                result[start:index] = 0

            start = None

    return result


def add_buffers(trigger, pre_buffer, post_buffer):
    result = trigger.copy()

    trigger_indices = np.where(trigger == 1)[0]

    for index in trigger_indices:
        start = max(0, index - pre_buffer)
        end = min(len(trigger), index + post_buffer + 1)

        result[start:end] = 1

    return result


def contiguous_events(labels):
    events = []
    start = None

    for index in range(len(labels) + 1):
        current = labels[index] if index < len(labels) else 0

        if current == 1 and start is None:
            start = index

        if current == 0 and start is not None:
            events.append((start, index - 1))
            start = None

    return events


def evaluate_policy(df, threshold, smoothing, minimum_trigger, pre_buffer, post_buffer):
    total_positive_units = 0
    retained_positive_units = 0
    total_units = 0
    retained_units = 0
    total_events = 0
    preserved_events = 0
    false_retained_units = 0

    for _, video_df in df.groupby("video_id", sort=False):
        video_df = video_df.sort_values("temporal_index")

        scores = video_df["event_probability"].to_numpy(
            dtype=float
        )

        labels = video_df["anomaly_label"].to_numpy(
            dtype=int
        )

        smoothed = smooth_scores(
            scores,
            smoothing,
        )

        trigger = (smoothed >= threshold).astype(int)

        trigger = remove_short_runs(
            trigger,
            minimum_trigger,
        )

        retain = add_buffers(
            trigger,
            pre_buffer,
            post_buffer,
        )

        total_positive_units += int(labels.sum())

        retained_positive_units += int(
            ((labels == 1) & (retain == 1)).sum()
        )

        false_retained_units += int(
            ((labels == 0) & (retain == 1)).sum()
        )

        total_units += len(labels)
        retained_units += int(retain.sum())

        events = contiguous_events(labels)

        total_events += len(events)

        for event_start, event_end in events:
            event_length = event_end - event_start + 1

            retained_length = int(
                retain[event_start:event_end + 1].sum()
            )

            coverage = retained_length / event_length

            if coverage >= EVENT_COVERAGE_THRESHOLD:
                preserved_events += 1

    unit_recall = (
        retained_positive_units / total_positive_units
        if total_positive_units > 0
        else 0.0
    )

    retention_ratio = (
        retained_units / total_units
        if total_units > 0
        else 0.0
    )

    event_recall = (
        preserved_events / total_events
        if total_events > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "smoothing_window": smoothing,
        "minimum_trigger_units": minimum_trigger,
        "pre_buffer_units": pre_buffer,
        "post_buffer_units": post_buffer,
        "unit_evidence_recall": unit_recall,
        "event_evidence_recall": event_recall,
        "retention_ratio": retention_ratio,
        "retained_unit_percent": 100.0 * retention_ratio,
        "storage_reduction_percent": 100.0 * (
            1.0 - retention_ratio
        ),
        "retained_units": retained_units,
        "false_retained_units": false_retained_units,
        "missed_positive_units": (
            total_positive_units - retained_positive_units
        ),
        "total_events": total_events,
        "preserved_events": preserved_events,
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing calibrated temporal file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    evaluation_df = df[
        df["split"] == "evaluation"
    ].copy()

    if validation_df.empty or evaluation_df.empty:
        raise RuntimeError(
            "Validation or evaluation data is empty."
        )

    validation_results = []

    for smoothing in SMOOTHING_WINDOWS:
        for minimum_trigger in MIN_TRIGGER_UNITS:
            for pre_buffer in PRE_BUFFER_UNITS:
                for post_buffer in POST_BUFFER_UNITS:
                    for threshold in THRESHOLDS:
                        result = evaluate_policy(
                            validation_df,
                            threshold,
                            smoothing,
                            minimum_trigger,
                            pre_buffer,
                            post_buffer,
                        )

                        validation_results.append(result)

    valid = [
        row
        for row in validation_results
        if row["unit_evidence_recall"]
        >= TARGET_VALIDATION_UNIT_RECALL
    ]

    if not valid:
        raise RuntimeError(
            "No temporal policy met the validation recall target."
        )

    selected = sorted(
        valid,
        key=lambda row: (
            row["retention_ratio"],
            -row["event_evidence_recall"],
            -row["threshold"],
        ),
    )[0]

    evaluation_result = evaluate_policy(
        evaluation_df,
        selected["threshold"],
        selected["smoothing_window"],
        selected["minimum_trigger_units"],
        selected["pre_buffer_units"],
        selected["post_buffer_units"],
    )

    baseline_result = evaluate_policy(
        evaluation_df,
        threshold=0.119894,
        smoothing=1,
        minimum_trigger=1,
        pre_buffer=0,
        post_buffer=0,
    )

    validation_curve_path = (
        OUTPUT_DIR
        / "xd_temporal_policy_optimization_validation.csv"
    )

    pd.DataFrame(validation_results).to_csv(
        validation_curve_path,
        index=False,
    )

    result_rows = [
        {
            "policy": "simple_threshold_baseline",
            **baseline_result,
        },
        {
            "policy": "optimized_smoothing_buffer_policy",
            **evaluation_result,
        },
    ]

    result_path = (
        OUTPUT_DIR
        / "xd_temporal_policy_optimization_results.csv"
    )

    pd.DataFrame(result_rows).to_csv(
        result_path,
        index=False,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "experiment_type": "PILOT_ONLY",
        "validation_recall_target": (
            TARGET_VALIDATION_UNIT_RECALL
        ),
        "event_coverage_threshold": (
            EVENT_COVERAGE_THRESHOLD
        ),
        "selected_policy_on_validation": selected,
        "evaluation_result": evaluation_result,
        "simple_threshold_baseline": baseline_result,
        "files": {
            "validation_grid": str(
                validation_curve_path
            ),
            "results": str(result_path),
        },
        "scope_note": (
            "This is feature-level policy optimization. It does not "
            "represent final raw-video retention or final context VAS."
        ),
    }

    report_path = (
        LOG_DIR
        / "xd_temporal_policy_optimization.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Temporal policy optimization: PASS")
    print()
    print("Selected validation policy:")
    print(
        f"Threshold: {selected['threshold']:.6f}"
    )
    print(
        f"Smoothing window: "
        f"{selected['smoothing_window']}"
    )
    print(
        f"Minimum trigger units: "
        f"{selected['minimum_trigger_units']}"
    )
    print(
        f"Pre-buffer units: "
        f"{selected['pre_buffer_units']}"
    )
    print(
        f"Post-buffer units: "
        f"{selected['post_buffer_units']}"
    )
    print()
    print("Evaluation result:")
    print(
        f"Unit recall: "
        f"{evaluation_result['unit_evidence_recall']:.6f}"
    )
    print(
        f"Event recall: "
        f"{evaluation_result['event_evidence_recall']:.6f}"
    )
    print(
        f"Retained units: "
        f"{evaluation_result['retained_unit_percent']:.2f}%"
    )
    print(
        f"Storage reduction: "
        f"{evaluation_result['storage_reduction_percent']:.2f}%"
    )
    print(
        f"Missed units: "
        f"{evaluation_result['missed_positive_units']}"
    )
    print()
    print(f"Results: {result_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()