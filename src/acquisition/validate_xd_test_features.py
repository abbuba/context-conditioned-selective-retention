import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RGB_DIR = (
    PROJECT_ROOT / "data" / "raw" / "xd_violence" / "official_features" / "i3d_rgb_test"
)

AUDIO_DIR = (
    PROJECT_ROOT / "data" / "raw" / "xd_violence" / "official_features" / "vggish_audio_test"
)

ANNOTATIONS = (
    PROJECT_ROOT / "data" / "raw" / "xd_violence" / "labels" / "official" / "annotations.txt"
)

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalise_rgb_name(name):
    return re.sub(r"__([0-4])\.npy$", "", name)


def normalise_audio_name(name):
    return re.sub(r"__vggish\.npy$", "", name)


def inspect_array(path):
    array = np.load(path, mmap_mode="r")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def main():
    rgb_paths = sorted(RGB_DIR.glob("*.npy"))
    audio_paths = sorted(AUDIO_DIR.glob("*.npy"))

    rgb_groups = defaultdict(list)
    for path in rgb_paths:
        rgb_groups[normalise_rgb_name(path.name)].append(path)

    audio_groups = {
        normalise_audio_name(path.name): path
        for path in audio_paths
    }

    rgb_ids = set(rgb_groups)
    audio_ids = set(audio_groups)

    matched_ids = sorted(rgb_ids & audio_ids)
    rgb_only = sorted(rgb_ids - audio_ids)
    audio_only = sorted(audio_ids - rgb_ids)

    malformed_rgb = {
        video_id: sorted(path.name for path in paths)
        for video_id, paths in rgb_groups.items()
        if len(paths) != 5
    }

    sample_records = []

    for video_id in matched_ids[:10]:
        rgb_crop_zero = RGB_DIR / f"{video_id}__0.npy"
        audio_path = audio_groups[video_id]

        sample_records.append(
            {
                "video_id": video_id,
                "rgb_crop_0": rgb_crop_zero.name,
                "rgb": inspect_array(rgb_crop_zero),
                "audio": inspect_array(audio_path),
            }
        )

    annotation_lines = []
    if ANNOTATIONS.exists():
        annotation_lines = [
            line.strip()
            for line in ANNOTATIONS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rgb_file_count": len(rgb_paths),
        "audio_file_count": len(audio_paths),
        "rgb_video_count": len(rgb_ids),
        "audio_video_count": len(audio_ids),
        "matched_video_count": len(matched_ids),
        "rgb_only_video_count": len(rgb_only),
        "audio_only_video_count": len(audio_only),
        "rgb_videos_not_exactly_five_crops": len(malformed_rgb),
        "annotation_file_exists": ANNOTATIONS.exists(),
        "annotation_line_count": len(annotation_lines),
        "sample_arrays": sample_records,
        "validation_status": (
            "PASS"
            if len(matched_ids) == 800
            and len(rgb_only) == 0
            and len(audio_only) == 0
            and len(malformed_rgb) == 0
            else "REVIEW_REQUIRED"
        ),
    }

    manifest_rows = [
        "video_id,rgb_crop_0_path,audio_path,rgb_crop_count",
    ]

    for video_id in matched_ids:
        rgb_crop_zero = RGB_DIR / f"{video_id}__0.npy"
        audio_path = audio_groups[video_id]

        manifest_rows.append(
            f'"{video_id}","{rgb_crop_zero}","{audio_path}",{len(rgb_groups[video_id])}'
        )

    manifest_path = METADATA_DIR / "xd_violence_test_feature_manifest.csv"
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    report_path = OUTPUT_DIR / "xd_test_feature_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validation status: {report['validation_status']}")
    print(f"RGB files: {report['rgb_file_count']}")
    print(f"Audio files: {report['audio_file_count']}")
    print(f"Matched test videos: {report['matched_video_count']}")
    print(f"Malformed RGB crop groups: {report['rgb_videos_not_exactly_five_crops']}")
    print(f"Annotation lines: {report['annotation_line_count']}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()