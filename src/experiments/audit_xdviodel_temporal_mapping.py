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


SUBSET_VIDEO_COUNT = 5


def read_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def video_id_from_rgb_path(path_text):
    return Path(path_text).name.rsplit("__", 1)[0]


def video_id_from_audio_path(path_text):
    return Path(path_text).name.removesuffix("__vggish.npy")


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


def inspect_annotation_file():
    if not ANNOTATION_FILE.exists():
        return {
            "exists": False,
            "line_count": 0,
            "sample_lines": [],
        }

    lines = read_lines(ANNOTATION_FILE)

    return {
        "exists": True,
        "line_count": len(lines),
        "sample_lines": lines[:20],
    }


def main():
    rgb_entries = read_lines(RGB_LIST)
    audio_entries = read_lines(AUDIO_LIST)

    if len(rgb_entries) != 4000:
        raise RuntimeError(
            f"Expected 4000 RGB entries, found {len(rgb_entries)}."
        )

    if len(audio_entries) != 800:
        raise RuntimeError(
            f"Expected 800 audio entries, found {len(audio_entries)}."
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

    loader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = Model(args).to(device)
    load_checkpoint(model)
    model.eval()

    subset_records = []
    temporal_score_rows = []

    annotation_info = inspect_annotation_file()

    with torch.no_grad():
        for batch_index, processed_features in enumerate(loader):
            if batch_index >= SUBSET_VIDEO_COUNT:
                break

            rgb_paths = rgb_entries[
                batch_index * 5:(batch_index + 1) * 5
            ]

            audio_path = audio_entries[batch_index]

            video_id = video_id_from_rgb_path(rgb_paths[0])

            original_rgb = np.load(rgb_paths[0])
            original_audio = np.load(audio_path)

            processed_features = processed_features.float().to(device)

            logits_offline, logits_online = model(
                inputs=processed_features,
                seq_len=None,
            )

            offline_probability = torch.sigmoid(
                logits_offline
            ).detach().cpu().numpy()

            online_probability = torch.sigmoid(
                logits_online
            ).detach().cpu().numpy()

            record = {
                "video_index": batch_index,
                "video_id": video_id,
                "rgb_crop_paths": rgb_paths,
                "audio_path": audio_path,
                "original_rgb_shape": list(original_rgb.shape),
                "original_audio_shape": list(original_audio.shape),
                "processed_input_shape": list(
                    processed_features.shape
                ),
                "offline_logits_shape": list(
                    logits_offline.shape
                ),
                "online_logits_shape": list(
                    logits_online.shape
                ),
                "offline_probability_shape": list(
                    offline_probability.shape
                ),
                "online_probability_shape": list(
                    online_probability.shape
                ),
                "max_seqlen": getattr(
                    args,
                    "max_seqlen",
                    None,
                ),
            }

            subset_records.append(record)

            offline_array = np.asarray(offline_probability)
            online_array = np.asarray(online_probability)

            if offline_array.ndim >= 3 and offline_array.shape[-1] == 1:
                offline_array = np.squeeze(offline_array, axis=-1)

            if online_array.ndim >= 3 and online_array.shape[-1] == 1:
                online_array = np.squeeze(online_array, axis=-1)

            if offline_array.ndim > 1:
                offline_array = np.mean(offline_array, axis=0)

            if online_array.ndim > 1:
                online_array = np.mean(online_array, axis=0)

            offline_array = np.atleast_1d(offline_array)
            online_array = np.atleast_1d(online_array)

            temporal_length = max(
                len(offline_array),
                len(online_array),
            )

            for temporal_index in range(temporal_length):
                offline_score = (
                    float(offline_array[temporal_index])
                    if temporal_index < len(offline_array)
                    else None
                )

                online_score = (
                    float(online_array[temporal_index])
                    if temporal_index < len(online_array)
                    else None
                )

                temporal_score_rows.append(
                    {
                        "video_id": video_id,
                        "video_index": batch_index,
                        "temporal_index": temporal_index,
                        "offline_raw_score": offline_score,
                        "online_raw_score": online_score,
                        "start_time": None,
                        "end_time": None,
                        "anomaly_label": None,
                        "time_mapping_status": (
                            "UNRESOLVED_PENDING_FEATURE_EXTRACTION_MAPPING"
                        ),
                    }
                )

            print(
                f"Audited video {batch_index + 1}/"
                f"{SUBSET_VIDEO_COUNT}: {video_id}"
            )

    temporal_csv = (
        OUTPUT_DIR
        / "xdviodel_temporal_score_probe.csv"
    )

    with temporal_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=temporal_score_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(temporal_score_rows)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "purpose": (
            "Expose existing pretrained temporal model outputs before "
            "official score averaging. No model weights, architecture, "
            "features, or scoring formulation were changed."
        ),
        "subset_video_count": SUBSET_VIDEO_COUNT,
        "checkpoint": str(CHECKPOINT),
        "modality": "MIX2",
        "feature_dimension": 1152,
        "dataset_max_seqlen": getattr(
            args,
            "max_seqlen",
            None,
        ),
        "subset_records": subset_records,
        "annotation_file": annotation_info,
        "temporal_score_csv": str(temporal_csv),
        "critical_note": (
            "start_time, end_time, and anomaly_label remain blank until "
            "the exact original-feature-to-processed-sequence mapping and "
            "annotation frame convention are verified."
        ),
    }

    report_path = (
        LOG_DIR
        / "xdviodel_temporal_mapping_audit.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("Temporal mapping probe: PASS")
    print(f"Videos inspected: {len(subset_records)}")
    print(f"Temporal score rows: {len(temporal_score_rows)}")
    print(f"Temporal CSV: {temporal_csv}")
    print(f"Audit report: {report_path}")
    print()
    print("IMPORTANT:")
    print(
        "Time fields are intentionally blank until the exact mapping "
        "is proven."
    )


if __name__ == "__main__":
    main()