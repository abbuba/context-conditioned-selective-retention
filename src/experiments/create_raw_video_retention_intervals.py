import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "external" / "XDVioDet"

CHECKPOINT = REPO_ROOT / "ckpt" / "wsanodet_mix2.pkl"
RGB_LIST = REPO_ROOT / "list" / "rgb_test.list"
AUDIO_LIST = REPO_ROOT / "list" / "audio_test.list"

SUBSET_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "xd_raw_video_pipeline_subset_24.csv"
)

RAW_METADATA = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_raw_video_metadata_24.csv"
)

TEMPORAL_CALIBRATED = (
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "raw_video_pipeline"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

from dataset import Dataset
from model import Model
import option


FPS = 24.0
FRAMES_PER_SCORE = 16
TARGET_EVENT_RECALL = 0.95
RANDOM_SEED = 42


def read_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def video_id_from_rgb_path(path_text):
    return Path(path_text).name.rsplit("__", 1)[0]


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


def merge_retained_units(retain):
    intervals = []
    start = None

    for index in range(len(retain) + 1):
        value = retain[index] if index < len(retain) else 0

        if value == 1 and start is None:
            start = index

        if value == 0 and start is not None:
            intervals.append((start, index - 1))
            start = None

    return intervals


def load_checkpoint(model):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    checkpoint = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(
        checkpoint,
        strict=True,
    )


def load_fixed_policy():
    df = pd.read_csv(OPERATING_POINTS)

    row = df[
        df["target_validation_event_recall"]
        == TARGET_EVENT_RECALL
    ]

    if row.empty:
        raise RuntimeError(
            "95% operating point not found."
        )

    row = row.iloc[0]

    return {
        "threshold": float(row["selected_threshold"]),
        "smoothing_window": int(
            row["smoothing_window"]
        ),
        "minimum_trigger_units": int(
            row["minimum_trigger_units"]
        ),
        "pre_buffer_units": int(
            row["pre_buffer_units"]
        ),
        "post_buffer_units": int(
            row["post_buffer_units"]
        ),
    }


def fit_temporal_calibrator():
    df = pd.read_csv(TEMPORAL_CALIBRATED)

    validation_df = df[
        df["split"] == "validation"
    ]

    x = validation_df[
        ["offline_raw_score"]
    ].to_numpy()

    y = validation_df[
        "anomaly_label"
    ].to_numpy(dtype=int)

    calibrator = LogisticRegression(
        random_state=RANDOM_SEED,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(x, y)

    return calibrator


def main():
    for path in [
        CHECKPOINT,
        RGB_LIST,
        AUDIO_LIST,
        SUBSET_MANIFEST,
        RAW_METADATA,
        TEMPORAL_CALIBRATED,
        OPERATING_POINTS,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    subset_df = pd.read_csv(SUBSET_MANIFEST)
    metadata_df = pd.read_csv(RAW_METADATA)

    metadata_map = {
        row["video_id"]: row
        for _, row in metadata_df.iterrows()
        if row["status"] == "PASS"
    }

    policy = load_fixed_policy()
    calibrator = fit_temporal_calibrator()

    rgb_entries = read_lines(RGB_LIST)

    video_index_map = {
        video_id_from_rgb_path(
            rgb_entries[index * 5]
        ): index
        for index in range(800)
    }

    args = option.parser.parse_args([])

    args.modality = "MIX2"
    args.feature_size = 1152
    args.rgb_list = str(RGB_LIST)
    args.audio_list = str(AUDIO_LIST)
    args.test_rgb_list = str(RGB_LIST)
    args.test_audio_list = str(AUDIO_LIST)
    args.gpus = "-1"
    args.workers = 0

    device = torch.device("cpu")

    dataset = Dataset(
        args,
        test_mode=True,
    )

    model = Model(args).to(device)
    load_checkpoint(model)
    model.eval()

    interval_rows = []
    video_rows = []

    with torch.no_grad():
        for count, (_, item) in enumerate(
            subset_df.iterrows(),
            start=1,
        ):
            video_id = item["video_id"]

            if video_id not in video_index_map:
                raise RuntimeError(
                    f"Feature index missing for {video_id}"
                )

            if video_id not in metadata_map:
                raise RuntimeError(
                    f"Raw metadata missing for {video_id}"
                )

            video_index = video_index_map[video_id]

            crops = [
                torch.as_tensor(
                    dataset[video_index * 5 + crop_index],
                    dtype=torch.float32,
                )
                for crop_index in range(5)
            ]

            features = torch.stack(
                crops,
                dim=0,
            ).float().to(device)

            offline_logits, _ = model(
                inputs=features,
                seq_len=None,
            )

            raw_scores = torch.sigmoid(
                offline_logits
            ).squeeze(-1).mean(dim=0).cpu().numpy()

            probabilities = calibrator.predict_proba(
                raw_scores.reshape(-1, 1)
            )[:, 1]

            smoothed = smooth_scores(
                probabilities,
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

            temporal_intervals = merge_retained_units(
                retain
            )

            duration_seconds = float(
                metadata_map[video_id]["duration_seconds"]
            )

            retained_duration = 0.0

            for interval_index, (
                start_unit,
                end_unit,
            ) in enumerate(temporal_intervals):
                start_seconds = (
                    start_unit
                    * FRAMES_PER_SCORE
                    / FPS
                )

                end_seconds = min(
                    (end_unit + 1)
                    * FRAMES_PER_SCORE
                    / FPS,
                    duration_seconds,
                )

                clip_duration = max(
                    0.0,
                    end_seconds - start_seconds,
                )

                retained_duration += clip_duration

                interval_rows.append(
                    {
                        "video_id": video_id,
                        "subset_group": item["subset_group"],
                        "video_index": video_index,
                        "interval_index": interval_index,
                        "start_temporal_unit": start_unit,
                        "end_temporal_unit": end_unit,
                        "start_seconds": round(
                            start_seconds,
                            6,
                        ),
                        "end_seconds": round(
                            end_seconds,
                            6,
                        ),
                        "clip_duration_seconds": round(
                            clip_duration,
                            6,
                        ),
                        "raw_video_path": metadata_map[
                            video_id
                        ]["local_video_path"],
                    }
                )

            video_rows.append(
                {
                    "video_id": video_id,
                    "subset_group": item["subset_group"],
                    "video_index": video_index,
                    "raw_video_path": metadata_map[
                        video_id
                    ]["local_video_path"],
                    "raw_duration_seconds": duration_seconds,
                    "raw_size_bytes": int(
                        metadata_map[video_id][
                            "file_size_bytes"
                        ]
                    ),
                    "temporal_score_count": len(
                        probabilities
                    ),
                    "retained_temporal_units": int(
                        retain.sum()
                    ),
                    "retained_duration_seconds": round(
                        retained_duration,
                        6,
                    ),
                    "retention_ratio_duration": round(
                        retained_duration / duration_seconds,
                        6,
                    )
                    if duration_seconds > 0
                    else None,
                    "retained_interval_count": len(
                        temporal_intervals
                    ),
                    "policy_threshold": policy["threshold"],
                    "smoothing_window": policy[
                        "smoothing_window"
                    ],
                    "minimum_trigger_units": policy[
                        "minimum_trigger_units"
                    ],
                    "pre_buffer_units": policy[
                        "pre_buffer_units"
                    ],
                    "post_buffer_units": policy[
                        "post_buffer_units"
                    ],
                }
            )

            print(
                f"Generated intervals {count}/"
                f"{len(subset_df)}: {video_id}"
            )

    interval_path = (
        OUTPUT_DIR
        / "xd_raw_video_retention_intervals_24.csv"
    )

    video_path = (
        OUTPUT_DIR
        / "xd_raw_video_retention_video_summary_24.csv"
    )

    pd.DataFrame(interval_rows).to_csv(
        interval_path,
        index=False,
    )

    pd.DataFrame(video_rows).to_csv(
        video_path,
        index=False,
    )

    total_raw_duration = sum(
        row["raw_duration_seconds"]
        for row in video_rows
    )

    total_retained_duration = sum(
        row["retained_duration_seconds"]
        for row in video_rows
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "experiment_type": "RAW_VIDEO_PIPELINE_PROOF",
        "policy": policy,
        "selected_videos": len(video_rows),
        "retained_intervals": len(interval_rows),
        "total_raw_duration_seconds": total_raw_duration,
        "total_planned_retained_duration_seconds": (
            total_retained_duration
        ),
        "planned_duration_retention_ratio": (
            total_retained_duration
            / total_raw_duration
        ),
        "planned_duration_storage_reduction_percent": (
            100.0
            * (
                1.0
                - total_retained_duration
                / total_raw_duration
            )
        ),
        "interval_manifest": str(interval_path),
        "video_summary": str(video_path),
        "scope_note": (
            "Intervals are derived using the fixed 95% pilot policy. "
            "No FFmpeg clips have been created in this step."
        ),
    }

    report_path = (
        LOG_DIR
        / "xd_raw_video_retention_intervals.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("Raw-video interval generation: PASS")
    print(f"Videos: {len(video_rows)}")
    print(f"Intervals: {len(interval_rows)}")
    print(
        "Planned retained duration: "
        f"{total_retained_duration:.2f} s"
    )
    print(
        "Planned duration retention: "
        f"{100 * report['planned_duration_retention_ratio']:.2f}%"
    )
    print(
        "Planned duration reduction: "
        f"{report['planned_duration_storage_reduction_percent']:.2f}%"
    )
    print(f"Intervals: {interval_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()