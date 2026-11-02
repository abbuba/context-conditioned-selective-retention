import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RGB_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "official_features"
    / "i3d_rgb_test"
)

AUDIO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "official_features"
    / "vggish_audio_test"
)

REPO_LIST_DIR = PROJECT_ROOT / "external" / "XDVioDet" / "list"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"

REPO_LIST_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rgb_files = sorted(RGB_DIR.glob("*.npy"))
    audio_files = sorted(AUDIO_DIR.glob("*.npy"))

    rgb_groups = {}

    for path in rgb_files:
        video_id = path.name.rsplit("__", 1)[0]
        rgb_groups.setdefault(video_id, []).append(path)

    valid_video_ids = sorted(
        video_id
        for video_id, paths in rgb_groups.items()
        if len(paths) == 5
    )

    audio_by_id = {
        path.name.removesuffix("__vggish.npy"): path
        for path in audio_files
    }

    matched_ids = [
        video_id for video_id in valid_video_ids
        if video_id in audio_by_id
    ]

    rgb_list = []
    audio_list = []

    for video_id in matched_ids:
        crops = sorted(rgb_groups[video_id])

        for crop_path in crops:
            rgb_list.append(str(crop_path.resolve()))

        audio_list.append(str(audio_by_id[video_id].resolve()))

    rgb_list_path = REPO_LIST_DIR / "rgb_test.list"
    audio_list_path = REPO_LIST_DIR / "audio_test.list"

    rgb_list_path.write_text("\n".join(rgb_list) + "\n", encoding="utf-8")
    audio_list_path.write_text("\n".join(audio_list) + "\n", encoding="utf-8")

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matched_video_count": len(matched_ids),
        "rgb_list_count": len(rgb_list),
        "audio_list_count": len(audio_list),
        "expected_rgb_list_count": len(matched_ids) * 5,
        "rgb_list": str(rgb_list_path),
        "audio_list": str(audio_list_path),
        "checkpoint_mode": "MIX2",
        "status": (
            "PASS"
            if len(rgb_list) == len(matched_ids) * 5
            and len(audio_list) == len(matched_ids)
            else "REVIEW_REQUIRED"
        ),
    }

    report_path = OUTPUT_DIR / "xdviodel_test_lists.json"
    report_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"Status: {record['status']}")
    print(f"Matched videos: {len(matched_ids)}")
    print(f"RGB list entries: {len(rgb_list)}")
    print(f"Audio list entries: {len(audio_list)}")
    print(f"RGB list: {rgb_list_path}")
    print(f"Audio list: {audio_list_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()