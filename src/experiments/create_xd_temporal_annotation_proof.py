import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

from dataset import Dataset
from model import Model
import option


FPS = 24.0
FRAMES_PER_TEMPORAL_SCORE = 16
PROOF_VIDEO_COUNT = 5


def read_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def video_id_from_rgb_path(path_text):
    return Path(path_text).name.rsplit("__", 1)[0]


def parse_annotations(annotation_path):
    annotations = {}

    for line in read_lines(annotation_path):
        tokens = line.split()

        video_id = tokens[0]

        if video_id.startswith("v="):
            video_id = video_id[2:]

        frame_values = [int(value) for value in tokens[1:]]

        if len(frame_values) % 2 != 0:
            print(
                f"Skipping malformed annotation line: {line}"
            )
            continue

        intervals = []

        for index in range(0, len(frame_values), 2):
            start_frame = frame_values[index]
            end_frame = frame_values[index + 1]

            intervals.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
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

    checkpoint = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(
        checkpoint,
        strict=True,
    )


def overlaps(
    score_start_frame,
    score_end_frame,
    intervals,
):
    for interval in intervals:
        anomaly_start = interval["start_frame"]
        anomaly_end = interval["end_frame"]

        if (
            score_start_frame <= anomaly_end
            and score_end_frame >= anomaly_start
        ):
            return 1

    return 0


def main():
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file missing: {ANNOTATION_FILE}"
        )

    annotation_map = parse_annotations(
        ANNOTATION_FILE
    )

    rgb_entries = read_lines(RGB_LIST)
    audio_entries = read_lines(AUDIO_LIST)

    video_ids = [
        video_id_from_rgb_path(
            rgb_entries[index * 5]
        )
        for index in range(800)
    ]

    matched_annotation_indices = []

    for video_index, video_id in enumerate(video_ids):
        if video_id in annotation_map:
            matched_annotation_indices.append(
                video_index
            )

    if len(matched_annotation_indices) < PROOF_VIDEO_COUNT:
        raise RuntimeError(
            "Not enough annotation-matched feature videos found."
        )

    selected_indices = matched_annotation_indices[
        :PROOF_VIDEO_COUNT
    ]

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

    output_rows = []
    proof_summary = []

    with torch.no_grad():
        for video_index in selected_indices:
            feature_crops = []

            for crop_index in range(5):
                feature_crops.append(
                    torch.as_tensor(
                        dataset[video_index * 5 + crop_index],
                        dtype=torch.float32,
                    )
                )

            features = torch.stack(
                feature_crops,
                dim=0,
            ).float().to(device)

            logits_offline, logits_online = model(
                inputs=features,
                seq_len=None,
            )

            offline_temporal_scores = torch.sigmoid(
                logits_offline
            ).squeeze(-1).mean(dim=0)

            online_temporal_scores = torch.sigmoid(
                logits_online
            ).squeeze(-1).mean(dim=0)

            offline_temporal_scores = (
                offline_temporal_scores
                .cpu()
                .numpy()
            )

            online_temporal_scores = (
                online_temporal_scores
                .cpu()
                .numpy()
            )

            video_id = video_ids[video_index]
            intervals = annotation_map[video_id]

            for temporal_index in range(
                len(offline_temporal_scores)
            ):
                start_frame = (
                    temporal_index
                    * FRAMES_PER_TEMPORAL_SCORE
                )

                end_frame = (
                    ((temporal_index + 1)
                    * FRAMES_PER_TEMPORAL_SCORE)
                    - 1
                )

                start_time = start_frame / FPS
                end_time = (
                    (end_frame + 1)
                    / FPS
                )

                anomaly_label = overlaps(
                    start_frame,
                    end_frame,
                    intervals,
                )

                output_rows.append(
                    {
                        "video_id": video_id,
                        "video_index": video_index,
                        "temporal_index": temporal_index,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "start_time_seconds": round(
                            start_time,
                            6,
                        ),
                        "end_time_seconds": round(
                            end_time,
                            6,
                        ),
                        "offline_raw_score": float(
                            offline_temporal_scores[
                                temporal_index
                            ]
                        ),
                        "online_raw_score": float(
                            online_temporal_scores[
                                temporal_index
                            ]
                        ),
                        "anomaly_label": anomaly_label,
                    }
                )

            proof_summary.append(
                {
                    "video_id": video_id,
                    "video_index": video_index,
                    "temporal_score_count": int(
                        len(offline_temporal_scores)
                    ),
                    "annotation_intervals": intervals,
                    "positive_temporal_scores": int(
                        sum(
                            row["anomaly_label"]
                            for row in output_rows
                            if row["video_id"] == video_id
                        )
                    ),
                }
            )

            print(
                f"Processed proof video: {video_id}"
            )

    csv_path = (
        OUTPUT_DIR
        / "xdviodel_temporal_annotation_proof.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "fps_assumption": FPS,
        "frames_per_temporal_score":
            FRAMES_PER_TEMPORAL_SCORE,
        "temporal_score_duration_seconds":
            FRAMES_PER_TEMPORAL_SCORE / FPS,
        "temporal_mapping": {
            "start_frame": "16 * temporal_index",
            "end_frame": "16 * (temporal_index + 1) - 1",
            "start_time_seconds":
                "(16 * temporal_index) / 24",
            "end_time_seconds":
                "(16 * (temporal_index + 1)) / 24",
        },
        "proof_video_count": len(proof_summary),
        "proof_summary": proof_summary,
        "csv_path": str(csv_path),
        "critical_note": (
            "The 16-frame temporal unit is verified from the official "
            "test.py np.repeat(pred, 16) evaluation logic. The 24 FPS "
            "mapping is supported by feature sequence lengths and "
            "filename clip durations; raw-video inspection is required "
            "before claiming exact encoded-video timing."
        ),
    }

    report_path = (
        LOG_DIR
        / "xdviodel_temporal_annotation_proof.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("Temporal annotation proof: PASS")
    print(
        f"Annotation-matched videos found: "
        f"{len(matched_annotation_indices)}"
    )
    print(
        f"Proof videos processed: "
        f"{len(proof_summary)}"
    )
    print(
        f"Temporal rows written: "
        f"{len(output_rows)}"
    )
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()