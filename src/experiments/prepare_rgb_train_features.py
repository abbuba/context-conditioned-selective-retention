import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ZIP = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "official_features"
    / "archives"
    / "i3d-features.zip"
)
RGB_PREFIX = "i3d-features/RGB/"
TARGET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "official_features"
    / "i3d_rgb_train"
)
RGB_SOURCE_LIST = (
    PROJECT_ROOT
    / "external"
    / "XDVioDet"
    / "list"
    / "rgb.list"
)
RGB_LOCAL_LIST = (
    PROJECT_ROOT
    / "external"
    / "XDVioDet"
    / "list"
    / "rgb_train_local.list"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_rgb_train_feature_preparation.json"
)
MINIMUM_FREE_SPACE_RATIO = 1.25
PROGRESS_INTERVAL = 1000


def selected_rgb_members(archive):
    return [
        info
        for info in archive.infolist()
        if info.filename.startswith(RGB_PREFIX)
        and info.filename.endswith(".npy")
        and not info.is_dir()
    ]


def basename_counts(members):
    return Counter(
        PurePosixPath(info.filename).name
        for info in members
    )


def target_drive_free_bytes(path):
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise OSError(
                f"Could not find an existing parent for {path}"
            )
        probe = parent
    return shutil.disk_usage(probe).free


def write_manifest(report):
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    MANIFEST_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare local XD-Violence I3D RGB training features "
            "for the C1 RGB-only experiment."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing target feature files",
    )
    args = parser.parse_args()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_zip": str(SOURCE_ZIP),
        "target_folder": str(TARGET_DIR),
        "original_rgb_list": str(RGB_SOURCE_LIST),
        "local_rgb_list": str(RGB_LOCAL_LIST),
        "selected_member_count": 0,
        "selected_uncompressed_bytes": 0,
        "free_space_bytes": None,
        "required_free_space_bytes": 0,
        "original_list_count": 0,
        "extracted_file_count": 0,
        "local_list_count": 0,
        "missing_file_count": 0,
        "missing_feature_count": 0,
        "zero_byte_file_count": 0,
        "size_mismatch_count": 0,
        "duplicate_filename_count": 0,
        "duplicate_rgb_list_filename_count": 0,
        "extracted_bytes": 0,
        "newly_extracted_file_count": 0,
        "skipped_existing_file_count": 0,
        "force_overwrite": args.force,
        "status": "FAIL",
        "error": None,
    }

    try:
        if not SOURCE_ZIP.is_file():
            raise FileNotFoundError(
                f"Source feature archive not found: {SOURCE_ZIP}"
            )
        if not RGB_SOURCE_LIST.is_file():
            raise FileNotFoundError(
                f"RGB training list not found: {RGB_SOURCE_LIST}"
            )

        original_entries = [
            line.strip()
            for line in RGB_SOURCE_LIST.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        original_filenames = [
            PurePosixPath(entry.replace("\\", "/")).name
            for entry in original_entries
        ]
        report["original_list_count"] = len(original_entries)
        report["duplicate_rgb_list_filename_count"] = sum(
            count - 1
            for count in Counter(original_filenames).values()
            if count > 1
        )

        with zipfile.ZipFile(SOURCE_ZIP, "r") as archive:
            members = selected_rgb_members(archive)
            selected_names = basename_counts(members)
            report["selected_member_count"] = len(members)
            report["selected_uncompressed_bytes"] = sum(
                info.file_size
                for info in members
            )
            report["duplicate_filename_count"] = sum(
                count - 1
                for count in selected_names.values()
                if count > 1
            )

            if not members:
                raise RuntimeError(
                    f"No RGB .npy members found under {RGB_PREFIX}"
                )

            if report["duplicate_filename_count"]:
                raise RuntimeError(
                    "Flattening would collide on duplicate RGB member "
                    f"basenames ({report['duplicate_filename_count']} duplicates)."
                )

            required_space = math.ceil(
                report["selected_uncompressed_bytes"]
                * MINIMUM_FREE_SPACE_RATIO
            )
            free_space = target_drive_free_bytes(TARGET_DIR)
            report["required_free_space_bytes"] = required_space
            report["free_space_bytes"] = free_space

            print(
                "Selected RGB training members: "
                f"{report['selected_member_count']}"
            )
            print(
                "Selected uncompressed bytes: "
                f"{report['selected_uncompressed_bytes']}"
            )
            print(
                "Available free bytes on target drive: "
                f"{free_space}"
            )
            print(
                "Required free bytes (1.25x safety margin): "
                f"{required_space}"
            )

            if free_space < required_space:
                raise OSError(
                    "Insufficient free space: need at least "
                    f"{required_space} bytes, have {free_space}."
                )

            TARGET_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            newly_extracted = 0
            skipped_existing = 0

            for processed_count, info in enumerate(
                members,
                start=1,
            ):
                filename = PurePosixPath(info.filename).name
                output_path = TARGET_DIR / filename

                if output_path.exists() and not args.force:
                    skipped_existing += 1
                else:
                    temporary_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="wb",
                            dir=TARGET_DIR,
                            prefix=f".{filename}.",
                            suffix=".tmp",
                            delete=False,
                        ) as temporary_file:
                            temporary_path = Path(
                                temporary_file.name
                            )
                            with archive.open(info, "r") as source:
                                shutil.copyfileobj(
                                    source,
                                    temporary_file,
                                    length=1024 * 1024,
                                )

                        if temporary_path.stat().st_size != info.file_size:
                            raise OSError(
                                "Extracted byte count differs from ZIP "
                                f"metadata for {info.filename}"
                            )

                        if output_path.exists() and not args.force:
                            skipped_existing += 1
                        else:
                            os.replace(
                                temporary_path,
                                output_path,
                            )
                            newly_extracted += 1
                            temporary_path = None
                    finally:
                        if (
                            temporary_path is not None
                            and temporary_path.exists()
                        ):
                            temporary_path.unlink()

                if (
                    processed_count % PROGRESS_INTERVAL == 0
                    or processed_count == len(members)
                ):
                    print(
                        "Extraction progress: "
                        f"{processed_count}/{len(members)} files"
                    )

            report["newly_extracted_file_count"] = newly_extracted
            report["skipped_existing_file_count"] = skipped_existing

            missing_archive_files = 0
            zero_byte_files = 0
            size_mismatches = 0
            extracted_file_count = 0
            extracted_bytes = 0

            for info in members:
                output_path = TARGET_DIR / PurePosixPath(
                    info.filename
                ).name
                if not output_path.is_file():
                    missing_archive_files += 1
                    continue

                extracted_file_count += 1
                file_size = output_path.stat().st_size
                extracted_bytes += file_size

                if file_size == 0:
                    zero_byte_files += 1
                if file_size != info.file_size:
                    size_mismatches += 1

            member_sizes = {
                PurePosixPath(info.filename).name: info.file_size
                for info in members
            }
            missing_features = sum(
                1
                for filename in original_filenames
                if (
                    filename not in member_sizes
                    or not (TARGET_DIR / filename).is_file()
                    or (TARGET_DIR / filename).stat().st_size == 0
                    or (TARGET_DIR / filename).stat().st_size
                    != member_sizes.get(filename)
                )
            )

            local_entries = [
                str((TARGET_DIR / filename).resolve())
                for filename in original_filenames
            ]
            RGB_LOCAL_LIST.write_text(
                "\n".join(local_entries) + "\n",
                encoding="utf-8",
            )

            report.update(
                {
                    "extracted_file_count": extracted_file_count,
                    "local_list_count": len(local_entries),
                    "missing_file_count": missing_archive_files,
                    "missing_feature_count": missing_features,
                    "zero_byte_file_count": zero_byte_files,
                    "size_mismatch_count": size_mismatches,
                    "extracted_bytes": extracted_bytes,
                    "archive_rgb_member_not_in_original_list_count": len(
                        set(selected_names) - set(original_filenames)
                    ),
                }
            )

            report["status"] = (
                "PASS"
                if (
                    missing_archive_files == 0
                    and missing_features == 0
                    and zero_byte_files == 0
                    and size_mismatches == 0
                )
                else "FAIL"
            )

            if report["status"] == "FAIL":
                report["error"] = (
                    "Post-extraction verification found missing, zero-byte, "
                    "or size-mismatched RGB features."
                )

    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)

    write_manifest(report)

    print()
    print(f"RGB training feature preparation: {report['status']}")
    print(f"Extracted files: {report['extracted_file_count']}")
    print(f"Missing archive files: {report['missing_file_count']}")
    print(f"Missing list features: {report['missing_feature_count']}")
    print(f"Zero-byte files: {report['zero_byte_file_count']}")
    print(f"Total extracted bytes: {report['extracted_bytes']}")
    print(f"Local RGB list: {RGB_LOCAL_LIST}")
    print(f"Manifest: {MANIFEST_PATH}")

    if report["error"]:
        print(f"Error: {report['error']}")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())