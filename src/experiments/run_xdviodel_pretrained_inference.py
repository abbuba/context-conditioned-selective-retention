import csv
import hashlib
import json
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scores"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

from dataset import Dataset
from model import Model
import option


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def video_id_from_rgb_path(path_text):
    return Path(path_text).name.rsplit("__", 1)[0]


def label_from_video_id(video_id):
    return 0 if "_label_A" in video_id else 1


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint does not contain a valid state dictionary. "
            f"Found type: {type(checkpoint).__name__}"
        )

    cleaned_checkpoint = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }

    load_result = model.load_state_dict(
        cleaned_checkpoint,
        strict=True,
    )

    return {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "state_dict_key_count": len(cleaned_checkpoint),
    }


def read_nonempty_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not RGB_LIST.exists():
        raise FileNotFoundError(f"RGB list not found: {RGB_LIST}")

    if not AUDIO_LIST.exists():
        raise FileNotFoundError(f"Audio list not found: {AUDIO_LIST}")

    args = option.parser.parse_args([])

    args.modality = "MIX2"
    args.feature_size = 1152
    args.rgb_list = str(RGB_LIST)
    args.audio_list = str(AUDIO_LIST)
    args.test_rgb_list = str(RGB_LIST)
    args.test_audio_list = str(AUDIO_LIST)
    args.gpus = "-1"
    args.workers = 0

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Checkpoint: {CHECKPOINT}")

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

    checkpoint_info = load_checkpoint(
        model,
        CHECKPOINT,
    )

    model.eval()

    rgb_entries = read_nonempty_lines(RGB_LIST)
    audio_entries = read_nonempty_lines(AUDIO_LIST)

    if len(rgb_entries) != 4000:
        raise RuntimeError(
            f"Expected 4,000 RGB entries, found {len(rgb_entries)}."
        )

    if len(audio_entries) != 800:
        raise RuntimeError(
            f"Expected 800 audio entries, found {len(audio_entries)}."
        )

    expected_batches = 800

    if len(loader) != expected_batches:
        raise RuntimeError(
            f"Expected {expected_batches} batches, found {len(loader)}."
        )

    results = []

    with torch.no_grad():
        for batch_index, features in enumerate(loader):
            features = features.float().to(device)

            logits_offline, logits_online = model(
                inputs=features,
                seq_len=None,
            )

            offline_scores = torch.sigmoid(logits_offline)
            online_scores = torch.sigmoid(logits_online)

            offline_score = offline_scores.squeeze().mean().item()
            online_score = online_scores.squeeze().mean().item()

            first_rgb_path = rgb_entries[batch_index * 5]
            video_id = video_id_from_rgb_path(first_rgb_path)

            results.append(
                {
                    "video_index": batch_index,
                    "video_id": video_id,
                    "weak_label": label_from_video_id(video_id),
                    "offline_raw_score": float(offline_score),
                    "online_raw_score": float(online_score),
                    "rgb_crop_count": 5,
                }
            )

            completed = batch_index + 1

            if completed % 25 == 0 or completed == len(loader):
                print(
                    f"Processed {completed}/{len(loader)} videos"
                )

    if len(results) != 800:
        raise RuntimeError(
            f"Expected 800 video scores, generated {len(results)}."
        )

    csv_path = OUTPUT_DIR / "xdviodel_mix2_raw_video_scores.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    offline_array = np.array(
        [row["offline_raw_score"] for row in results],
        dtype=np.float32,
    )

    online_array = np.array(
        [row["online_raw_score"] for row in results],
        dtype=np.float32,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "device": str(device),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_loading": checkpoint_info,
        "repository_commit": "f846d8cfc454943ec165bd2489b71e7dc0a1064d",
        "modality": "MIX2",
        "feature_dimension": 1152,
        "video_count": len(results),
        "rgb_entries": len(rgb_entries),
        "audio_entries": len(audio_entries),
        "batch_size": 5,
        "score_type": (
            "Mean sigmoid output over five RGB crops for each video"
        ),
        "offline_score_summary": {
            "minimum": float(offline_array.min()),
            "maximum": float(offline_array.max()),
            "mean": float(offline_array.mean()),
            "standard_deviation": float(offline_array.std()),
        },
        "online_score_summary": {
            "minimum": float(online_array.min()),
            "maximum": float(online_array.max()),
            "mean": float(online_array.mean()),
            "standard_deviation": float(online_array.std()),
        },
        "output_csv": str(csv_path),
    }

    report_path = LOG_DIR / "xdviodel_pretrained_inference.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nInference complete: PASS")
    print(f"Videos scored: {len(results)}")
    print(
        f"Offline score mean: "
        f"{report['offline_score_summary']['mean']:.6f}"
    )
    print(
        f"Online score mean: "
        f"{report['online_score_summary']['mean']:.6f}"
    )
    print(f"Scores CSV: {csv_path}")
    print(f"Run report: {report_path}")


if __name__ == "__main__":
    main()