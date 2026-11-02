import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "external" / "XDVioDet"

CHECKPOINT = REPO_ROOT / "ckpt" / "wsanodet_mix2.pkl"
RGB_LIST = REPO_ROOT / "list" / "rgb_test.list"
AUDIO_LIST = REPO_ROOT / "list" / "audio_test.list"

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "labels"
    / "official"
    / "annotations.txt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "temporal_mapping"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

from dataset import Dataset
from model import Model
import option


FPS = 24.0
FRAMES_PER_SCORE = 16
RANDOM_SEED = 42
VALIDATION_VIDEO_COUNT = 44


def read_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def video_id_from_rgb_path(path_text):
    return Path(path_text).name.rsplit("__", 1)[0]


def parse_annotation_file():
    annotations = {}

    for line in read_lines(ANNOTATION_FILE):
        tokens = line.split()

        video_id = tokens[0]

        if video_id.startswith("v="):
            video_id = video_id[2:]

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


def load_checkpoint(model):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    cleaned = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(
        cleaned,
        strict=True,
    )


def score_overlaps_anomaly(start_frame, end_frame, intervals):
    for anomaly_start, anomaly_end in intervals:
        if (
            start_frame <= anomaly_end
            and end_frame >= anomaly_start
        ):
            return 1

    return 0


def main():
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file missing: {ANNOTATION_FILE}"
        )

    annotation_map = parse_annotation_file()

    rgb_entries = read_lines(RGB_LIST)

    if len(rgb_entries) != 4000:
        raise RuntimeError(
            f"Expected 4000 RGB entries; found {len(rgb_entries)}."
        )

    all_video_ids = [
        video_id_from_rgb_path(
            rgb_entries[video_index * 5]
        )
        for video_index in range(800)
    ]

    matched_video_indices = [
        video_index
        for video_index, video_id in enumerate(all_video_ids)
        if video_id in annotation_map
    ]

    if len(matched_video_indices) != 218:
        print(
            "WARNING: Expected 218 annotation-matched videos, found "
            f"{len(matched_video_indices)}."
        )

    rng = random.Random(RANDOM_SEED)

    shuffled_indices = matched_video_indices.copy()
    rng.shuffle(shuffled_indices)

    validation_indices = set(
        shuffled_indices[:VALIDATION_VIDEO_COUNT]
    )

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

    temporal_rows = []
    split_rows = []
    summary_rows = []

    with torch.no_grad():
        for counter, video_index in enumerate(
            matched_video_indices,
            start=1,
        ):
            video_id = all_video_ids[video_index]
            intervals = annotation_map[video_id]

            crop_features = []

            for crop_index in range(5):
                crop_features.append(
                    torch.as_tensor(
                        dataset[video_index * 5 + crop_index],
                        dtype=torch.float32,
                    )
                )

            features = torch.stack(
                crop_features,
                dim=0,
            ).float().to(device)

            offline_logits, online_logits = model(
                inputs=features,
                seq_len=None,
            )

            offline_scores = torch.sigmoid(
                offline_logits
            ).squeeze(-1).mean(dim=0).cpu().numpy()

            online_scores = torch.sigmoid(
                online_logits
            ).squeeze(-1).mean(dim=0).cpu().numpy()

            split = (
                "validation"
                if video_index in validation_indices
                else "evaluation"
            )

            positive_units = 0

            for temporal_index in range(len(offline_scores)):
                start_frame = (
                    temporal_index * FRAMES_PER_SCORE
                )

                end_frame = (
                    (temporal_index + 1)
                    * FRAMES_PER_SCORE
                    - 1
                )

                anomaly_label = score_overlaps_anomaly(
                    start_frame,
                    end_frame,
                    intervals,
                )

                positive_units += anomaly_label

                temporal_rows.append(
                    {
                        "video_id": video_id,
                        "video_index": video_index,
                        "split": split,
                        "temporal_index": temporal_index,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "start_time_seconds": round(
                            start_frame / FPS,
                            6,
                        ),
                        "end_time_seconds": round(
                            (end_frame + 1) / FPS,
                            6,
                        ),
                        "offline_raw_score": float(
                            offline_scores[temporal_index]
                        ),
                        "online_raw_score": float(
                            online_scores[temporal_index]
                        ),
                        "anomaly_label": anomaly_label,
                    }
                )

            split_rows.append(
                {
                    "video_id": video_id,
                    "video_index": video_index,
                    "split": split,
                    "annotation_interval_count": len(intervals),
                    "temporal_score_count": len(offline_scores),
                }
            )

            summary_rows.append(
                {
                    "video_id": video_id,
                    "video_index": video_index,
                    "split": split,
                    "temporal_score_count": len(offline_scores),
                    "positive_temporal_units": positive_units,
                    "annotation_interval_count": len(intervals),
                }
            )

            if counter % 10 == 0 or counter == len(matched_video_indices):
                print(
                    f"Processed {counter}/"
                    f"{len(matched_video_indices)} videos"
                )

    temporal_csv = (
        OUTPUT_DIR
        / "xd_temporal_scores_218.csv"
    )

    with temporal_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(temporal_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(temporal_rows)

    split_csv = (
        METADATA_DIR
        / "xd_temporal_video_split.csv"
    )

    with split_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(split_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(split_rows)

    summary_csv = (
        OUTPUT_DIR
        / "xd_temporal_video_summary_218.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    total_positive_units = sum(
        row["anomaly_label"]
        for row in temporal_rows
    )

    total_units = len(temporal_rows)

    validation_video_total = sum(
        row["split"] == "validation"
        for row in split_rows
    )

    evaluation_video_total = sum(
        row["split"] == "evaluation"
        for row in split_rows
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "purpose": (
            "Pilot extraction of temporal pretrained MIX2 scores for "
            "annotation-matched XD-Violence videos."
        ),
        "checkpoint": str(CHECKPOINT),
        "modality": "MIX2",
        "fps_for_feature_time_estimation": FPS,
        "frames_per_temporal_score": FRAMES_PER_SCORE,
        "temporal_duration_seconds": (
            FRAMES_PER_SCORE / FPS
        ),
        "matched_annotation_videos": len(
            matched_video_indices
        ),
        "validation_videos": validation_video_total,
        "evaluation_videos": evaluation_video_total,
        "total_temporal_units": total_units,
        "positive_temporal_units": total_positive_units,
        "negative_temporal_units": (
            total_units - total_positive_units
        ),
        "random_seed": RANDOM_SEED,
        "temporal_score_csv": str(temporal_csv),
        "split_csv": str(split_csv),
        "summary_csv": str(summary_csv),
        "scope_note": (
            "This is a temporal score extraction and pilot split only. "
            "No retention threshold, VAS, context, raw-video extraction, "
            "IPFS, or Hyperledger Fabric operation is performed."
        ),
    }

    report_path = (
        LOG_DIR
        / "xd_temporal_score_extraction.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("Temporal extraction: PASS")
    print(
        f"Annotation-matched videos: "
        f"{len(matched_video_indices)}"
    )
    print(
        f"Validation videos: "
        f"{validation_video_total}"
    )
    print(
        f"Evaluation videos: "
        f"{evaluation_video_total}"
    )
    print(
        f"Temporal units: "
        f"{total_units}"
    )
    print(
        f"Anomalous temporal units: "
        f"{total_positive_units}"
    )
    print(f"Scores: {temporal_csv}")
    print(f"Split: {split_csv}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()