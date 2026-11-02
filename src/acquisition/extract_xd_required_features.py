import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_DIR = (
    PROJECT_ROOT / "data" / "raw" / "xd_violence" / "official_features" / "archives"
)

I3D_ARCHIVE = ARCHIVE_DIR / "i3d-features.zip"
VGGISH_ARCHIVE = ARCHIVE_DIR / "vggish-features.zip"

RGB_TARGET = (
    PROJECT_ROOT / "data" / "raw" / "xd_violence" / "official_features" / "i3d_rgb_test"
)

AUDIO_TARGET = (
    PROJECT_ROOT / "data" / "raw" / "xd_violence" / "official_features" / "vggish_audio_test"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_selected_members(archive_path, target_dir, selector, expected_label):
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing archive: {archive_path}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        selected = [
            info for info in archive.infolist()
            if not info.is_dir() and selector(info.filename)
        ]

        if not selected:
            raise RuntimeError(
                f"No files selected from {archive_path.name} for {expected_label}"
            )

        extracted = 0
        skipped = 0
        total_bytes = 0

        for index, info in enumerate(selected, start=1):
            output_path = target_dir / Path(info.filename).name

            if output_path.exists() and output_path.stat().st_size == info.file_size:
                skipped += 1
                total_bytes += info.file_size
                continue

            with archive.open(info, "r") as source, output_path.open("wb") as target:
                shutil.copyfileobj(source, target)

            extracted += 1
            total_bytes += info.file_size

            if index % 500 == 0 or index == len(selected):
                print(
                    f"{expected_label}: {index}/{len(selected)} files processed "
                    f"({index / len(selected) * 100:.1f}%)"
                )

    return {
        "archive": archive_path.name,
        "target_directory": str(target_dir),
        "expected_content": expected_label,
        "selected_file_count": len(selected),
        "newly_extracted_file_count": extracted,
        "already_present_file_count": skipped,
        "selected_uncompressed_bytes": total_bytes,
        "selected_uncompressed_gb": round(total_bytes / (1024 ** 3), 3),
    }


def main():
    print("Extracting only required MIX2 test features.")
    print("I3D Flow and VGGish training features will be skipped.\n")

    rgb_result = extract_selected_members(
        archive_path=I3D_ARCHIVE,
        target_dir=RGB_TARGET,
        selector=lambda name: name.startswith("i3d-features/RGBTest/") and name.endswith(".npy"),
        expected_label="I3D RGB test features",
    )

    audio_result = extract_selected_members(
        archive_path=VGGISH_ARCHIVE,
        target_dir=AUDIO_TARGET,
        selector=lambda name: name.startswith("vggish-features/test/") and name.endswith(".npy"),
        expected_label="VGGish audio test features",
    )

    disk = shutil.disk_usage(PROJECT_ROOT)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_mode": "MIX2",
        "required_modalities": {
            "visual": "I3D RGB, 1024 dimensions",
            "audio": "VGGish, 128 dimensions",
        },
        "excluded_modalities": [
            "I3D optical flow",
            "VGGish training features",
        ],
        "rgb_extraction": rgb_result,
        "audio_extraction": audio_result,
        "disk_after_extraction": {
            "free_gb": round(disk.free / (1024 ** 3), 2),
            "used_gb": round(disk.used / (1024 ** 3), 2),
        },
    }

    output_path = OUTPUT_DIR / "xd_required_feature_extraction.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nExtraction complete.")
    print(f"RGB files available: {len(list(RGB_TARGET.glob('*.npy')))}")
    print(f"Audio files available: {len(list(AUDIO_TARGET.glob('*.npy')))}")
    print(f"Free disk after extraction: {report['disk_after_extraction']['free_gb']} GB")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()