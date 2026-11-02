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

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "labels"
    / "official"
    / "annotations.txt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FPS = 24.0
FRAMES_PER_SCORE = 16
EVENT_COVERAGE_THRESHOLD = 0.80

TARGET_EVENT_RECALLS = [0.90, 0.95, 0.97, 0.98]

SMOOTHING_WINDOWS = [1, 3, 5]
MIN_TRIGGER_UNITS = [1, 2]
PRE_BUFFER_UNITS = [0, 1, 2]
POST_BUFFER_UNITS = [0, 1, 2]

THRESHOLDS = np.linspace(0.01, 0.99, 99)


def read_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_annotations():
    annotations = {}

    for line in read_lines(ANNOTATION_FILE):
        tokens = line.split()
        video_id = tokens[0].removeprefix("v=")

        values = [
            int(value)
            for value in tokens[1:]
        ]

        if len(values) % 2 != 0:
            continue

        intervals = []

        for index in range(0, len(values), 2):
            intervals.append(
                (
                    values[index] / FPS,
                    (values[index + 1] + 1) / FPS,
                )
            )

        annotations[video_id] = intervals

    return annotations


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


def retained_intervals(retain):
    intervals = []
    start = None

    for index in range(len(retain) + 1):
        value = retain[index] if index < len(retain) else 0

        if value == 1 and start is None:
            start = index

        if value == 0 and start is not None:
            intervals.append(
                (
                    start * FRAMES_PER_SCORE / FPS,
                    index * FRAMES_PER_SCORE / FPS,
                )
            )
            start = None

    return intervals


def overlap_duration(event_start, event_end, retained):
    overlaps = []

    for retain_start, retain_end in retained:
        start = max(event_start, retain_start)
        end = min(event_end, retain_end)

        if end > start:
            overlaps.append((start, end))

    if not overlaps:
        return 0.0

    overlaps.sort()

    merged = []
    current_start, current_end = overlaps[0]

    for start, end in overlaps[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append(
                (current_start, current_end)
            )
            current_start, current_end = start, end

    merged.append((current_start, current_end))

    return sum(
        end - start
        for start, end in merged
    )


def evaluate_policy(
    df,
    annotations,
    threshold,
    smoothing_window,
    minimum_trigger_units,
    pre_buffer_units,
    post_buffer_units,
):
    total_units = 0
    retained_units = 0

    total_events = 0
    preserved_events = 0

    total_event_duration = 0.0
    covered_event_duration = 0.0

    for video_id, video_df in df.groupby(
        "video_id",
        sort=False,
    ):
        if video_id not in annotations:
            continue

        video_df = video_df.sort_values(
            "temporal_index"
        )

        scores = video_df[
            "event_probability"
        ].to_numpy()

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

        retained = retained_intervals(retain)

        total_units += len(retain)
        retained_units += int(retain.sum())

        for event_start, event_end in annotations[video_id]:
            event_duration = event_end - event_start

            covered_duration = overlap_duration(
                event_start,
                event_end,
                retained,
            )

            coverage = (
                covered_duration / event_duration
                if event_duration > 0
                else 0.0
            )

            total_events += 1
            total_event_duration += event_duration
            covered_event_duration += covered_duration

            if coverage >= EVENT_COVERAGE_THRESHOLD:
                preserved_events += 1

    return {
        "official_event_recall": (
            preserved_events / total_events
            if total_events > 0
            else 0.0
        ),
        "event_duration_coverage": (
            covered_event_duration
            / total_event_duration
            if total_event_duration > 0
            else 0.0
        ),
        "retention_ratio": (
            retained_units / total_units
            if total_units > 0
            else 0.0
        ),
        "retained_unit_percent": (
            100.0 * retained_units / total_units
            if total_units > 0
            else 0.0
        ),
        "storage_reduction_percent": (
            100.0
            * (
                1.0
                - retained_units / total_units
            )
            if total_units > 0
            else 0.0
        ),
        "total_official_events": total_events,
        "preserved_official_events": preserved_events,
        "missed_official_events": (
            total_events - preserved_events
        ),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing calibrated scores: {INPUT_FILE}"
        )

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing annotations: {ANNOTATION_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)
    annotations = parse_annotations()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    evaluation_df = df[
        df["split"] == "evaluation"
    ].copy()

    validation_grid = []

    for smoothing in SMOOTHING_WINDOWS:
        for minimum_trigger in MIN_TRIGGER_UNITS:
            for pre_buffer in PRE_BUFFER_UNITS:
                for post_buffer in POST_BUFFER_UNITS:
                    for threshold in THRESHOLDS:
                        result = evaluate_policy(
                            validation_df,
                            annotations,
                            threshold,
                            smoothing,
                            minimum_trigger,
                            pre_buffer,
                            post_buffer,
                        )

                        validation_grid.append(
                            {
                                "threshold": float(threshold),
                                "smoothing_window": smoothing,
                                "minimum_trigger_units": (
                                    minimum_trigger
                                ),
                                "pre_buffer_units": pre_buffer,
                                "post_buffer_units": post_buffer,
                                **result,
                            }
                        )

    grid_df = pd.DataFrame(validation_grid)

    result_rows = []

    for target_recall in TARGET_EVENT_RECALLS:
        candidates = grid_df[
            grid_df["official_event_recall"]
            >= target_recall
        ].copy()

        if candidates.empty:
            result_rows.append(
                {
                    "target_validation_event_recall": (
                        target_recall
                    ),
                    "status": "NO_VALID_POLICY",
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
            evaluation_df,
            annotations,
            float(selected["threshold"]),
            int(selected["smoothing_window"]),
            int(selected["minimum_trigger_units"]),
            int(selected["pre_buffer_units"]),
            int(selected["post_buffer_units"]),
        )

        result_rows.append(
            {
                "target_validation_event_recall": (
                    target_recall
                ),
                "status": "PASS",
                "selected_threshold": float(
                    selected["threshold"]
                ),
                "smoothing_window": int(
                    selected["smoothing_window"]
                ),
                "minimum_trigger_units": int(
                    selected[
                        "minimum_trigger_units"
                    ]
                ),
                "pre_buffer_units": int(
                    selected["pre_buffer_units"]
                ),
                "post_buffer_units": int(
                    selected["post_buffer_units"]
                ),
                "validation_official_event_recall": float(
                    selected["official_event_recall"]
                ),
                "validation_retained_unit_percent": float(
                    selected["retained_unit_percent"]
                ),
                **evaluation,
            }
        )

    validation_path = (
        OUTPUT_DIR
        / "xd_official_event_policy_validation_grid.csv"
    )

    result_path = (
        OUTPUT_DIR
        / "xd_official_event_policy_operating_points.csv"
    )

    grid_df.to_csv(
        validation_path,
        index=False,
    )

    pd.DataFrame(result_rows).to_csv(
        result_path,
        index=False,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "event_definition": (
            "Official XD-Violence annotation frame-interval pairs"
        ),
        "event_coverage_threshold": (
            EVENT_COVERAGE_THRESHOLD
        ),
        "target_validation_event_recalls": (
            TARGET_EVENT_RECALLS
        ),
        "operating_points": result_rows,
        "validation_grid": str(validation_path),
        "results_file": str(result_path),
        "scope_note": (
            "This corrects prior contiguous-unit event grouping. "
            "Official annotation intervals are now the event unit."
        ),
    }

    report_path = (
        LOG_DIR
        / "xd_official_event_policy_optimization.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Official-event policy optimization: PASS")

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
            "Evaluation official event recall: "
            f"{row['official_event_recall']:.6f}"
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
            "Evaluation missed official events: "
            f"{row['missed_official_events']}"
        )

    print()
    print(f"Results: {result_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()