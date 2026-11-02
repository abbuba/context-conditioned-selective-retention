import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "official_features"
    / "archives"
    / "videos.zip"
)

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "xd_raw_video_pipeline_subset_24.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_name(name):
    filename = Path(name).name
    filename = re.sub(r"\.(mp4|avi|mkv|mov|webm)$", "", filename, flags=re.I)
    return filename


def main():
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Archive not found: {ARCHIVE}"
        )

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Subset manifest not found: {MANIFEST}"
        )

    subset_df = pd.read_csv(MANIFEST)

    required_ids = set(
        subset_df["video_id"].astype(str)
    )

    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        files = [
            info
            for info in archive.infolist()
            if not info.is_dir()
        ]

        video_files = [
            info
            for info in files
            if Path(info.filename).suffix.lower()
            in {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        ]

        archive_map = {
            normalize_name(info.filename): info
            for info in video_files
        }

        match_rows = []

        for video_id in sorted(required_ids):
            info = archive_map.get(video_id)

            match_rows.append(
                {
                    "video_id": video_id,
                    "found_in_archive": info is not None,
                    "archive_path": (
                        info.filename if info else None
                    ),
                    "archive_size_bytes": (
                        info.file_size if info else None
                    ),
                    "archive_size_mb": (
                        round(
                            info.file_size / (1024 ** 2),
                            3,
                        )
                        if info
                        else None
                    ),
                }
            )

    match_df = pd.DataFrame(match_rows)

    match_path = (
        METADATA_DIR
        / "xd_raw_video_subset_archive_match.csv"
    )

    match_df.to_csv(
        match_path,
        index=False,
    )

    matched = int(match_df["found_in_archive"].sum())
    missing = int((~match_df["found_in_archive"]).sum())

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if missing == 0 else "REVIEW_REQUIRED",
        "archive": str(ARCHIVE),
        "archive_size_gb": round(
            ARCHIVE.stat().st_size / (1024 ** 3),
            3,
        ),
        "total_files_in_archive": len(files),
        "video_files_in_archive": len(video_files),
        "selected_subset_videos": len(required_ids),
        "matched_subset_videos": matched,
        "missing_subset_videos": missing,
        "match_file": str(match_path),
        "sample_archive_entries": [
            info.filename
            for info in video_files[:20]
        ],
    }

    report_path = (
        OUTPUT_DIR
        / "xd_raw_video_archive_audit.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Raw-video archive audit complete.")
    print(f"Archive size: {report['archive_size_gb']} GB")
    print(
        f"Video files in archive: "
        f"{report['video_files_in_archive']}"
    )
    print(
        f"Selected subset videos: "
        f"{report['selected_subset_videos']}"
    )
    print(
        f"Matched subset videos: "
        f"{report['matched_subset_videos']}"
    )
    print(
        f"Missing subset videos: "
        f"{report['missing_subset_videos']}"
    )
    print(f"Match file: {match_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()