import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMPORAL_SCORES = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_temporal_calibrated_scores.csv"
)

OPERATING_POINTS = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_temporal_event_recall_operating_points.csv"
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

RGB_LIST = (
    PROJECT_ROOT
    / "external"
    / "XDVioDet"
    / "list"
    / "rgb_test.list"
)

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FPS = 24.0
FRAMES_PER_SCORE = 16
EVENT_COVERAGE_THRESHOLD = 0.80
TARGET_VALIDATION_EVENT_RECALL = 0.95


def read_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def video_id_from_rgb_path(path_text):
    return Path(path_text).name.rsplit("__", 1)[0]


def parse_annotations():
    annotations = {}

    for line in read_lines(ANNOTATION_FILE):
        tokens = line.split()
        video_id = tokens[0].removeprefix("v=")
        values = [int(value) for value in tokens[1:]]

        if len(values) % 2 != 0:
            continue

        intervals = []

        for index in range(0, len(values), 2):
            intervals.append(
                (
                    values[index],
                    values[index + 1],
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


def video_event_recall(labels, retain):
    events = contiguous_events(labels)

    if not events:
        return None, 0, 0

    preserved = 0

    for start, end in events:
        event_length = end - start + 1
        coverage = retain[start:end + 1].sum() / event_length

        if coverage >= EVENT_COVERAGE_THRESHOLD:
            preserved += 1

    return preserved / len(events), preserved, len(events)


def get_policy():
    operating_points = pd.read_csv(OPERATING_POINTS)

    rows = operating_points[
        operating_points[
            "target_validation_event_recall"
        ] == TARGET_VALIDATION_EVENT_RECALL
    ]

    if rows.empty:
        raise RuntimeError(
            "Could not find 95% operating point."
        )

    row = rows.iloc[0]

    if row["status"] != "PASS":
        raise RuntimeError(
            "The selected 95% operating point did not pass."
        )

    return {
        "threshold": float(row["selected_threshold"]),
        "smoothing_window": int(row["smoothing_window"]),
        "minimum_trigger_units": int(
            row["minimum_trigger_units"]
        ),
        "pre_buffer_units": int(row["pre_buffer_units"]),
        "post_buffer_units": int(row["post_buffer_units"]),
    }


def main():
    for path in [
        TEMPORAL_SCORES,
        OPERATING_POINTS,
        ANNOTATION_FILE,
        RGB_LIST,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(TEMPORAL_SCORES)

    annotations = parse_annotations()
    policy = get_policy()

    evaluation_df = df[
        df["split"] == "evaluation"
    ].copy()

    records = []

    for video_id, video_df in evaluation_df.groupby(
        "video_id",
        sort=False,
    ):
        video_df = video_df.sort_values("temporal_index")

        scores = video_df[
            "event_probability"
        ].to_numpy()

        labels = video_df[
            "anomaly_label"
        ].to_numpy(dtype=int)

        smoothed = smooth_scores(
            scores,
            policy["smoothing_window"],
        )

        trigger = (
            smoothed >= policy["threshold"]
        ).astype(int)

        trigger = remove_short_runs(
            trigger,
            policy["minimum_trigger_units"],
        )

        retain = add_buffers(
            trigger,
            policy["pre_buffer_units"],
            policy["post_buffer_units"],
        )

        event_recall, preserved, total_events = (
            video_event_recall(labels, retain)
        )

        retained_percent = float(
            100.0 * retain.mean()
        )

        mean_distance_to_threshold = float(
            np.abs(
                scores - policy["threshold"]
            ).mean()
        )

        records.append(
            {
                "video_id": video_id,
                "source_type": "annotated_anomaly",
                "event_recall": event_recall,
                "preserved_event_count": preserved,
                "event_count": total_events,
                "retained_percent": retained_percent,
                "mean_probability": float(scores.mean()),
                "max_probability": float(scores.max()),
                "mean_distance_to_threshold": (
                    mean_distance_to_threshold
                ),
                "annotation_intervals": json.dumps(
                    annotations.get(video_id, [])
                ),
            }
        )

    annotated_df = pd.DataFrame(records)

    retained_anomalies = annotated_df[
        annotated_df["event_recall"] >= 1.0
    ].sort_values(
        "retained_percent"
    ).head(8)

    missed_anomalies = annotated_df[
        annotated_df["event_recall"] < 1.0
    ].sort_values(
        "event_recall"
    ).head(4)

    near_threshold_anomalies = annotated_df[
        ~annotated_df["video_id"].isin(
            retained_anomalies["video_id"]
        )
        & ~annotated_df["video_id"].isin(
            missed_anomalies["video_id"]
        )
    ].sort_values(
        "mean_distance_to_threshold"
    ).head(4)

    rgb_entries = read_lines(RGB_LIST)

    all_video_ids = [
        video_id_from_rgb_path(
            rgb_entries[index * 5]
        )
        for index in range(800)
    ]

    normal_ids = [
        video_id
        for video_id in all_video_ids
        if "_label_A" in video_id
        and video_id not in annotations
    ]

    normal_rows = pd.DataFrame(
        {
            "video_id": normal_ids,
            "source_type": "normal",
            "event_recall": None,
            "preserved_event_count": 0,
            "event_count": 0,
            "retained_percent": None,
            "mean_probability": None,
            "max_probability": None,
            "mean_distance_to_threshold": None,
            "annotation_intervals": "[]",
        }
    ).head(8)

    subset_df = pd.concat(
        [
            retained_anomalies.assign(
                subset_group="retained_anomaly"
            ),
            missed_anomalies.assign(
                subset_group="missed_anomaly"
            ),
            near_threshold_anomalies.assign(
                subset_group="near_threshold_anomaly"
            ),
            normal_rows.assign(
                subset_group="normal"
            ),
        ],
        ignore_index=True,
    )

    if len(subset_df) != 24:
        raise RuntimeError(
            "Expected 24 videos in the subset, found "
            f"{len(subset_df)}."
        )

    subset_df.insert(
        0,
        "subset_index",
        range(1, len(subset_df) + 1),
    )

    subset_df["required_raw_video_filename"] = (
        subset_df["video_id"] + ".mp4"
    )

    subset_df["selection_policy"] = (
        "95_percent_validation_event_recall_operating_point"
    )

    subset_path = (
        METADATA_DIR
        / "xd_raw_video_pipeline_subset_24.csv"
    )

    subset_df.to_csv(
        subset_path,
        index=False,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "purpose": (
            "Fixed 24-video raw evidence pipeline proof subset."
        ),
        "policy": policy,
        "subset_counts": {
            "retained_anomaly": 8,
            "missed_anomaly": 4,
            "near_threshold_anomaly": 4,
            "normal": 8,
            "total": 24,
        },
        "subset_manifest": str(subset_path),
        "scope_note": (
            "This subset is for end-to-end raw-video pipeline proof, "
            "not final statistical performance reporting."
        ),
    }

    report_path = (
        LOG_DIR
        / "xd_raw_video_pipeline_subset_selection.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Raw-video pipeline subset selection: PASS")
    print("Total selected videos: 24")
    print("Retained anomaly videos: 8")
    print("Missed anomaly videos: 4")
    print("Near-threshold anomaly videos: 4")
    print("Normal videos: 8")
    print(f"Manifest: {subset_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()