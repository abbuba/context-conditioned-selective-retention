import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_GRID = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_temporal_policy_optimization_validation.csv"
)

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

TARGET_EVENT_RECALLS = [0.90, 0.95, 0.97, 0.98]
EVENT_COVERAGE_THRESHOLD = 0.80


def smooth_scores(scores, window):
    if window == 1:
        return scores.copy()

    return (
        pd.Series(scores)
        .rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )


def remove_short_runs(trigger, minimum_length):
    result = trigger.copy()
    start = None

    for index in range(len(trigger) + 1):
        value = trigger[index] if index < len(trigger) else 0

        if value == 1 and start is None:
            start = index

        if value == 0 and start is not None:
            if index - start < minimum_length:
                result[start:index] = 0

            start = None

    return result


def add_buffers(trigger, pre_buffer, post_buffer):
    result = trigger.copy()

    for index in range(len(trigger)):
        if trigger[index] == 1:
            start = max(0, index - pre_buffer)
            end = min(
                len(trigger),
                index + post_buffer + 1,
            )
            result[start:end] = 1

    return result


def contiguous_events(labels):
    events = []
    start = None

    for index in range(len(labels) + 1):
        value = labels[index] if index < len(labels) else 0

        if value == 1 and start is None:
            start = index

        if value == 0 and start is not None:
            events.append((start, index - 1))
            start = None

    return events


def evaluate_policy(
    evaluation_df,
    threshold,
    smoothing_window,
    minimum_trigger_units,
    pre_buffer_units,
    post_buffer_units,
):
    total_units = 0
    retained_units = 0
    positive_units = 0
    retained_positive_units = 0
    total_events = 0
    preserved_events = 0

    for _, video_df in evaluation_df.groupby(
        "video_id",
        sort=False,
    ):
        video_df = video_df.sort_values(
            "temporal_index"
        )

        scores = video_df[
            "event_probability"
        ].to_numpy()

        labels = video_df[
            "anomaly_label"
        ].to_numpy(dtype=int)

        smoothed = smooth_scores(
            scores,
            smoothing_window,
        )

        trigger = (
            smoothed >= threshold
        ).astype(int)

        trigger = remove_short_runs(
            trigger,
            minimum_trigger_units,
        )

        retain = add_buffers(
            trigger,
            pre_buffer_units,
            post_buffer_units,
        )

        total_units += len(labels)
        retained_units += int(retain.sum())

        positive_units += int(labels.sum())

        retained_positive_units += int(
            ((labels == 1) & (retain == 1)).sum()
        )

        for start, end in contiguous_events(labels):
            total_events += 1

            event_length = end - start + 1

            coverage = (
                retain[start:end + 1].sum()
                / event_length
            )

            if coverage >= EVENT_COVERAGE_THRESHOLD:
                preserved_events += 1

    return {
        "evaluation_event_recall": (
            preserved_events / total_events
            if total_events > 0
            else 0.0
        ),
        "evaluation_unit_recall": (
            retained_positive_units / positive_units
            if positive_units > 0
            else 0.0
        ),
        "retention_ratio": retained_units / total_units,
        "retained_unit_percent": (
            100.0 * retained_units / total_units
        ),
        "storage_reduction_percent": (
            100.0
            * (
                1.0
                - retained_units / total_units
            )
        ),
        "total_events": total_events,
        "preserved_events": preserved_events,
        "missed_events": total_events - preserved_events,
        "total_positive_units": positive_units,
        "missed_positive_units": (
            positive_units - retained_positive_units
        ),
    }


def main():
    if not VALIDATION_GRID.exists():
        raise FileNotFoundError(
            f"Validation grid missing: {VALIDATION_GRID}"
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Calibrated score file missing: {INPUT_FILE}"
        )

    grid_df = pd.read_csv(VALIDATION_GRID)

    df = pd.read_csv(INPUT_FILE)

    evaluation_df = df[
        df["split"] == "evaluation"
    ].copy()

    result_rows = []

    for target in TARGET_EVENT_RECALLS:
        candidates = grid_df[
            grid_df["event_evidence_recall"] >= target
        ].copy()

        if candidates.empty:
            result_rows.append(
                {
                    "target_validation_event_recall": target,
                    "status": "NO_POLICY_FOUND",
                }
            )
            continue

        selected = candidates.sort_values(
            by=[
                "retention_ratio",
                "threshold",
            ],
            ascending=[
                True,
                False,
            ],
        ).iloc[0]

        evaluation = evaluate_policy(
            evaluation_df=evaluation_df,
            threshold=float(selected["threshold"]),
            smoothing_window=int(
                selected["smoothing_window"]
            ),
            minimum_trigger_units=int(
                selected["minimum_trigger_units"]
            ),
            pre_buffer_units=int(
                selected["pre_buffer_units"]
            ),
            post_buffer_units=int(
                selected["post_buffer_units"]
            ),
        )

        result_rows.append(
            {
                "target_validation_event_recall": target,
                "status": "PASS",
                "selected_threshold": float(
                    selected["threshold"]
                ),
                "smoothing_window": int(
                    selected["smoothing_window"]
                ),
                "minimum_trigger_units": int(
                    selected["minimum_trigger_units"]
                ),
                "pre_buffer_units": int(
                    selected["pre_buffer_units"]
                ),
                "post_buffer_units": int(
                    selected["post_buffer_units"]
                ),
                "validation_event_recall": float(
                    selected["event_evidence_recall"]
                ),
                "validation_unit_recall": float(
                    selected["unit_evidence_recall"]
                ),
                "validation_retained_percent": float(
                    selected["retained_unit_percent"]
                ),
                **evaluation,
            }
        )

    result_path = (
        OUTPUT_DIR
        / "xd_temporal_event_recall_operating_points.csv"
    )

    pd.DataFrame(result_rows).to_csv(
        result_path,
        index=False,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "experiment_type": "PILOT_ONLY",
        "event_coverage_threshold": (
            EVENT_COVERAGE_THRESHOLD
        ),
        "target_event_recalls": TARGET_EVENT_RECALLS,
        "results_file": str(result_path),
        "results": result_rows,
        "scope_note": (
            "This evaluates feature-level temporal policy operating "
            "points. It is not raw-video or context-conditioned "
            "evidence retention."
        ),
    }

    report_path = (
        LOG_DIR
        / "xd_temporal_event_recall_operating_points.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Event-recall operating points: PASS")

    for row in result_rows:
        print()

        print(
            "Target validation event recall:",
            row["target_validation_event_recall"],
        )

        if row["status"] != "PASS":
            print("No valid policy found.")
            continue

        print(
            "Evaluation event recall: "
            f"{row['evaluation_event_recall']:.6f}"
        )

        print(
            "Evaluation retained units: "
            f"{row['retained_unit_percent']:.2f}%"
        )

        print(
            "Evaluation storage reduction: "
            f"{row['storage_reduction_percent']:.2f}%"
        )

        print(
            "Evaluation missed events: "
            f"{row['missed_events']}"
        )

    print()
    print(f"Results: {result_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()