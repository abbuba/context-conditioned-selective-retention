import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "official_features"
    / "archives"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_ARCHIVES = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )

    return {
        "command": " ".join(str(item) for item in command),
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def inspect_zip(path):
    with zipfile.ZipFile(path, "r") as archive:
        entries = archive.infolist()

    names = [entry.filename for entry in entries if not entry.is_dir()]

    return {
        "inspection_method": "python_zipfile",
        "file_count": len(names),
        "uncompressed_bytes": sum(entry.file_size for entry in entries),
        "sample_entries": names[:30],
        "all_entries": names,
    }


def inspect_with_7zip(path):
    seven_zip = shutil.which("7z") or shutil.which("7zz")

    if seven_zip is None:
        return {
            "inspection_method": "unavailable",
            "error": "7z or 7zz was not found in PATH.",
        }

    result = run_command([seven_zip, "l", "-slt", str(path)])

    if result["return_code"] != 0:
        return {
            "inspection_method": "7z_failed",
            "error": result["stderr"],
        }

    return {
        "inspection_method": "7z",
        "raw_listing": result["stdout"],
    }


def classify_entries(entries):
    lowered = [entry.lower() for entry in entries]

    return {
        "npy_file_count": sum(entry.endswith(".npy") for entry in lowered),
        "rgb_mentions": sum("rgb" in entry for entry in lowered),
        "flow_mentions": sum("flow" in entry for entry in lowered),
        "audio_mentions": sum(
            any(word in entry for word in ["audio", "vggish", "sound"])
            for entry in lowered
        ),
        "test_mentions": sum("test" in entry for entry in lowered),
        "train_mentions": sum("train" in entry for entry in lowered),
        "contains_rgb": any("rgb" in entry for entry in lowered),
        "contains_flow": any("flow" in entry for entry in lowered),
        "contains_audio": any(
            any(word in entry for word in ["audio", "vggish", "sound"])
            for entry in lowered
        ),
    }


def audit_archive(path):
    record = {
        "filename": path.name,
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "size_gb": round(path.stat().st_size / (1024 ** 3), 3),
        "sha256": sha256_file(path),
    }

    if path.suffix.lower() == ".zip":
        inspection = inspect_zip(path)
        record.update(inspection)

        classification = classify_entries(inspection["all_entries"])
        record["classification"] = classification

        record["uncompressed_gb"] = round(
            inspection["uncompressed_bytes"] / (1024 ** 3),
            3,
        )

        del record["all_entries"]

    else:
        record.update(inspect_with_7zip(path))

    return record


def create_markdown_report(report):
    archive_rows = []

    for archive in report["archives"]:
        classification = archive.get("classification", {})

        archive_rows.append(
            "| `{filename}` | {size_gb} | {uncompressed_gb} | {npy_count} | "
            "{rgb} | {flow} | {audio} | `{sha256}` |".format(
                filename=archive["filename"],
                size_gb=archive["size_gb"],
                uncompressed_gb=archive.get("uncompressed_gb", "Unknown"),
                npy_count=classification.get("npy_file_count", "Unknown"),
                rgb=classification.get("contains_rgb", "Unknown"),
                flow=classification.get("contains_flow", "Unknown"),
                audio=classification.get("contains_audio", "Unknown"),
                sha256=archive["sha256"],
            )
        )

    if not archive_rows:
        archive_rows = [
            "| No supported archives found | - | - | - | - | - | - | - |"
        ]

    samples = []

    for archive in report["archives"]:
        sample_entries = archive.get("sample_entries", [])

        if sample_entries:
            samples.append(f"### {archive['filename']}")
            samples.append("```text")
            samples.extend(sample_entries)
            samples.append("```")

    markdown = f"""# XD-Violence Feature Archive Audit

**Audit time (UTC):** {report["audit_timestamp_utc"]}

## Available Disk Space

| Field | Value |
|---|---:|
| Total disk capacity (GB) | {report["disk"]["total_gb"]} |
| Used disk capacity (GB) | {report["disk"]["used_gb"]} |
| Free disk capacity (GB) | {report["disk"]["free_gb"]} |

## Archives Detected

| Archive | Compressed GB | Uncompressed GB | `.npy` files | RGB | Flow | Audio | SHA-256 |
|---|---:|---:|---:|---|---|---|---|
{chr(10).join(archive_rows)}

## Extraction Decision

The selected XDVioDet checkpoint operates in MIX2 mode. Only I3D RGB and
VGGish audio features are required. Optical-flow features must not be extracted
unless a future documented ablation explicitly requires them.

The next extraction script will extract only:
1. I3D RGB feature files.
2. VGGish audio feature files.
3. Required test-list and annotation resources.

## Sample Archive Contents

{chr(10).join(samples) if samples else "No ZIP entry sample available. Install 7-Zip if archive contents could not be inspected."}
"""

    output_path = DOCS_DIR / "xd_feature_archive_audit.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def main():
    disk = shutil.disk_usage(PROJECT_ROOT)

    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "archive_directory": str(ARCHIVE_DIR),
        "disk": {
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "used_gb": round(disk.used / (1024 ** 3), 2),
            "free_gb": round(disk.free / (1024 ** 3), 2),
        },
        "archives": [],
    }

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in ARCHIVE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_ARCHIVES
    )

    for path in files:
        print(f"Auditing: {path.name}")
        report["archives"].append(audit_archive(path))

    json_path = OUTPUT_DIR / "xd_feature_archive_audit.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_path = create_markdown_report(report)

    print()
    print(f"Archives found: {len(report['archives'])}")
    print(f"Free disk space: {report['disk']['free_gb']} GB")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")

    for archive in report["archives"]:
        print()
        print(f"Archive: {archive['filename']}")
        print(f"Compressed size: {archive['size_gb']} GB")
        print(f"SHA-256: {archive['sha256']}")

        if "classification" in archive:
            info = archive["classification"]
            print(f"RGB detected: {info['contains_rgb']}")
            print(f"Flow detected: {info['contains_flow']}")
            print(f"Audio detected: {info['contains_audio']}")


if __name__ == "__main__":
    main()