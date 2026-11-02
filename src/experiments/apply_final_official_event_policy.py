import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_temporal_calibrated_scores.csv"
)

RAW_METADATA_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_raw_video_metadata_24.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
RAW_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "raw_video_pipeline"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FPS = 24.0
FRAMES_PER_TEMPORAL_UNIT = 16

POLICY_NAME = "official_event_recall_95_policy"
POLICY_VERSION = "v1"

THRESHOLD = 0.12
SMOOTHING_WINDOW = 3
MINIMUM_TRIGGER_UNITS = 2
PRE_BUFFER_UNITS = 0
POST_BUFFER_UNITS = 0

VALIDATION_OFFICIAL_EVENT_RECALL = 0.9761904761904762
VALIDATION_RETAINED_UNIT_PERCENT = 79.84644913627639

EVALUATION_OFFICIAL_EVENT_RECALL = 0.9551820728291317
EVALUATION_EVENT_DURATION_COVERAGE = 0.9632491065149748
EVALUATION_RETAINED_UNIT_PERCENT = 75.16383495145631
EVALUATION_STORAGE_REDUCTION_PERCENT = 24.83616504854369
EVALUATION_TOTAL_OFFICIAL_EVENTS = 357
EVALUATION_PRESERVED_OFFICIAL_EVENTS = 341
EVALUATION_MISSED_OFFICIAL_EVENTS = 16


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
    run_start = None

    for index in range(len(trigger) + 1):
        value = trigger[index] if index < len(trigger) else 0

        if value == 1 and run_start is None:
            run_start = index

        if value == 0 and run_start is not None:
            if index - run_start < minimum_length:
                result[run_start:index] = 0

            run_start = None

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


def retained_runs(video_df):
    retained = video_df[
        "retain_final"
    ].to_numpy(dtype=int)

    runs = []
    start_index = None

    for index in range(len(retained) + 1):
        value = retained[index] if index < len(retained) else 0

        if value == 1 and start_index is None:
            start_index = index

        if value == 0 and start_index is not None:
            start_row = video_df.iloc[start_index]
            end_row = video_df.iloc[index - 1]

            start_frame = int(start_row["temporal_index"]) * (
                FRAMES_PER_TEMPORAL_UNIT
            )

            end_frame = (
                int(end_row["temporal_index"]) + 1
            ) * FRAMES_PER_TEMPORAL_UNIT

            runs.append(
                {
                    "video_id": start_row["video_id"],
                    "split": start_row["split"],
                    "interval_index": len(runs),
                    "start_temporal_index": int(
                        start_row["temporal_index"]
                    ),
                    "end_temporal_index": int(
                        end_row["temporal_index"]
                    ),
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame,
                    "start_seconds": round(
                        start_frame / FPS,
                        6,
                    ),
                    "end_seconds": round(
                        end_frame / FPS,
                        6,
                    ),
                    "clip_duration_seconds": round(
                        (end_frame - start_frame) / FPS,
                        6,
                    ),
                    "retained_unit_count": (
                        index - start_index
                    ),
                    "mean_smoothed_score": round(
                        float(
                            video_df.iloc[
                                start_index:index
                            ]["smoothed_event_probability"].mean()
                        ),
                        6,
                    ),
                    "max_smoothed_score": round(
                        float(
                            video_df.iloc[
                                start_index:index
                            ]["smoothed_event_probability"].max()
                        ),
                        6,
                    ),
                }
            )

            start_index = None

    return runs


def main():
    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            f"Missing score file: {SCORES_FILE}"
        )

    scores_df = pd.read_csv(SCORES_FILE)

    required_columns = {
        "video_id",
        "split",
        "temporal_index",
        "event_probability",
    }

    missing_columns = required_columns - set(
        scores_df.columns
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    decisions = []

    for video_id, video_df in scores_df.groupby(
        "video_id",
        sort=False,
    ):
        video_df = video_df.sort_values(
            "temporal_index"
        ).copy()

        probabilities = video_df[
            "event_probability"
        ].to_numpy()

        smoothed = smooth_scores(
            probabilities,
            SMOOTHING_WINDOW,
        )

        trigger = (
            smoothed >= THRESHOLD
        ).astype(int)

        trigger_after_minimum = remove_short_runs(
            trigger,
            MINIMUM_TRIGGER_UNITS,
        )

        retain = add_buffers(
            trigger_after_minimum,
            PRE_BUFFER_UNITS,
            POST_BUFFER_UNITS,
        )

        video_df[
            "smoothed_event_probability"
        ] = smoothed

        video_df["threshold_trigger"] = trigger
        video_df[
            "minimum_run_trigger"
        ] = trigger_after_minimum

        video_df["retain_final"] = retain

        decisions.append(video_df)

    decisions_df = pd.concat(
        decisions,
        ignore_index=True,
    )

    decision_path = (
        OUTPUT_DIR
        / "xd_final_retention_decisions_all_218.csv"
    )

    decisions_df.to_csv(
        decision_path,
        index=False,
    )

    interval_rows = []

    for _, video_df in decisions_df.groupby(
        "video_id",
        sort=False,
    ):
        interval_rows.extend(retained_runs(video_df))

    intervals_df = pd.DataFrame(interval_rows)

    intervals_path = (
        OUTPUT_DIR
        / "xd_final_retention_intervals_all_218.csv"
    )

    intervals_df.to_csv(
        intervals_path,
        index=False,
    )

    split_summary = (
        decisions_df.groupby("split", dropna=False)
        .agg(
            video_count=("video_id", "nunique"),
            temporal_units=("retain_final", "size"),
            retained_units=("retain_final", "sum"),
        )
        .reset_index()
    )

    split_summary["retained_unit_percent"] = (
        100.0
        * split_summary["retained_units"]
        / split_summary["temporal_units"]
    )

    split_summary["planned_duration_reduction_percent"] = (
        100.0
        - split_summary["retained_unit_percent"]
    )

    split_summary_path = (
        OUTPUT_DIR
        / "xd_final_retention_split_summary.csv"
    )

    split_summary.to_csv(
        split_summary_path,
        index=False,
    )

    raw_interval_path = None

    if RAW_METADATA_FILE.exists():
        raw_metadata_df = pd.read_csv(
            RAW_METADATA_FILE
        )

        raw_columns = {
            "video_id",
            "raw_video_path",
            "raw_duration_seconds",
        }

        if raw_columns.issubset(
            raw_metadata_df.columns
        ):
            raw_intervals_df = intervals_df.merge(
                raw_metadata_df[
                    [
                        "video_id",
                        "raw_video_path",
                        "raw_duration_seconds",
                    ]
                ],
                on="video_id",
                how="inner",
            )

            raw_intervals_df["end_seconds"] = np.minimum(
                raw_intervals_df["end_seconds"],
                raw_intervals_df[
                    "raw_duration_seconds"
                ],
            )

            raw_intervals_df[
                "clip_duration_seconds"
            ] = (
                raw_intervals_df["end_seconds"]
                - raw_intervals_df["start_seconds"]
            ).clip(lower=0)

            raw_intervals_df = raw_intervals_df[
                raw_intervals_df[
                    "clip_duration_seconds"
                ] > 0
            ].copy()

            raw_interval_path = (
                RAW_OUTPUT_DIR
                / "xd_final_policy_raw_intervals_24.csv"
            )

            raw_intervals_df.to_csv(
                raw_interval_path,
                index=False,
            )

    policy = {
        "policy_name": POLICY_NAME,
        "policy_version": POLICY_VERSION,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": "XD-Violence",
        "temporal_unit_frames": (
            FRAMES_PER_TEMPORAL_UNIT
        ),
        "fps_assumption": FPS,
        "temporal_unit_seconds": (
            FRAMES_PER_TEMPORAL_UNIT / FPS
        ),
        "score_field": "event_probability",
        "threshold": THRESHOLD,
        "smoothing_window_units": SMOOTHING_WINDOW,
        "minimum_trigger_units": (
            MINIMUM_TRIGGER_UNITS
        ),
        "pre_buffer_units": PRE_BUFFER_UNITS,
        "post_buffer_units": POST_BUFFER_UNITS,
        "event_definition": (
            "Official XD-Violence annotation frame-interval pairs"
        ),
        "event_preservation_definition": (
            "At least 80% temporal overlap with an "
            "official annotated event interval"
        ),
        "validation_results": {
            "official_event_recall": (
                VALIDATION_OFFICIAL_EVENT_RECALL
            ),
            "retained_unit_percent": (
                VALIDATION_RETAINED_UNIT_PERCENT
            ),
        },
        "evaluation_results": {
            "official_event_recall": (
                EVALUATION_OFFICIAL_EVENT_RECALL
            ),
            "event_duration_coverage": (
                EVALUATION_EVENT_DURATION_COVERAGE
            ),
            "retained_unit_percent": (
                EVALUATION_RETAINED_UNIT_PERCENT
            ),
            "planned_duration_reduction_percent": (
                EVALUATION_STORAGE_REDUCTION_PERCENT
            ),
            "total_official_events": (
                EVALUATION_TOTAL_OFFICIAL_EVENTS
            ),
            "preserved_official_events": (
                EVALUATION_PRESERVED_OFFICIAL_EVENTS
            ),
            "missed_official_events": (
                EVALUATION_MISSED_OFFICIAL_EVENTS
            ),
        },
        "decision_file": str(decision_path),
        "interval_file": str(intervals_path),
        "split_summary_file": str(
            split_summary_path
        ),
        "raw_subset_interval_file": (
            str(raw_interval_path)
            if raw_interval_path is not None
            else None
        ),
    }

    policy_path = (
        OUTPUT_DIR
        / "xd_final_official_event_policy.json"
    )

    policy_path.write_text(
        json.dumps(policy, indent=2),
        encoding="utf-8",
    )

    report_path = (
        LOG_DIR
        / "xd_final_policy_application.json"
    )

    report_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "status": "PASS",
                "policy_file": str(policy_path),
                "decision_file": str(decision_path),
                "interval_file": str(intervals_path),
                "split_summary_file": str(
                    split_summary_path
                ),
                "raw_subset_interval_file": (
                    str(raw_interval_path)
                    if raw_interval_path is not None
                    else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Final official-event policy application: PASS")
    print(
        f"Total videos: "
        f"{decisions_df['video_id'].nunique()}"
    )
    print(
        f"Total temporal units: "
        f"{len(decisions_df)}"
    )
    print(
        f"Retained temporal units: "
        f"{int(decisions_df['retain_final'].sum())}"
    )
    print(
        "Overall retained duration: "
        f"{100.0 * decisions_df['retain_final'].mean():.2f}%"
    )
    print(
        "Overall planned reduction: "
        f"{100.0 * (1.0 - decisions_df['retain_final'].mean()):.2f}%"
    )
    print(
        f"Retention intervals: "
        f"{len(intervals_df)}"
    )
    print(f"Policy: {policy_path}")
    print(f"Decisions: {decision_path}")
    print(f"Intervals: {intervals_path}")
    print(f"Split summary: {split_summary_path}")

    if raw_interval_path is not None:
        print(
            f"Corrected raw intervals: "
            f"{raw_interval_path}"
        )

    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()