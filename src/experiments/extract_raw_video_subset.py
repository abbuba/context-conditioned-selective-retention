import json
import shutil
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

MATCH_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "xd_raw_video_subset_archive_match.csv"
)

TARGET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "raw_subset"
)

LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

TARGET_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Archive missing: {ARCHIVE}"
        )

    if not MATCH_FILE.exists():
        raise FileNotFoundError(
            f"Match file missing: {MATCH_FILE}"
        )

    match_df = pd.read_csv(MATCH_FILE)

    selected = match_df[
        match_df["found_in_archive"] == True
    ].copy()

    if len(selected) != 24:
        raise RuntimeError(
            f"Expected 24 matched videos, found {len(selected)}."
        )

    extracted_rows = []

    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        for index, row in selected.iterrows():
            archive_path = row["archive_path"]
            video_id = row["video_id"]

            target_path = TARGET_DIR / Path(
                archive_path
            ).name

            if (
                target_path.exists()
                and target_path.stat().st_size
                == archive.getinfo(archive_path).file_size
            ):
                status = "already_exists"
            else:
                with archive.open(archive_path) as source:
                    with target_path.open("wb") as target:
                        shutil.copyfileobj(
                            source,
                            target,
                            length=1024 * 1024,
                        )

                status = "extracted"

            extracted_rows.append(
                {
                    "video_id": video_id,
                    "archive_path": archive_path,
                    "local_path": str(target_path),
                    "status": status,
                    "file_size_bytes": target_path.stat().st_size,
                    "file_size_mb": round(
                        target_path.stat().st_size
                        / (1024 ** 2),
                        3,
                    ),
                }
            )

            print(
                f"Processed {len(extracted_rows)}/"
                f"{len(selected)}: {target_path.name}"
            )

    total_bytes = sum(
        row["file_size_bytes"]
        for row in extracted_rows
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "archive": str(ARCHIVE),
        "selected_video_count": len(extracted_rows),
        "target_directory": str(TARGET_DIR),
        "total_subset_bytes": total_bytes,
        "total_subset_gb": round(
            total_bytes / (1024 ** 3),
            3,
        ),
        "videos": extracted_rows,
    }

    report_path = (
        LOG_DIR
        / "xd_raw_video_subset_extraction.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("Raw subset extraction: PASS")
    print(
        f"Videos extracted: {len(extracted_rows)}"
    )
    print(
        f"Subset size: "
        f"{report['total_subset_gb']} GB"
    )
    print(f"Directory: {TARGET_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()