import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTERNAL_DIR = PROJECT_ROOT / "external"
REPOSITORY_DIR = EXTERNAL_DIR / "XDVioDet"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

OFFICIAL_REPOSITORY_URL = "https://github.com/Roc-Ng/XDVioDet.git"

EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def run_command(command, cwd=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "command": " ".join(str(item) for item in command),
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError as error:
        return {
            "command": " ".join(str(item) for item in command),
            "return_code": -1,
            "stdout": "",
            "stderr": str(error),
        }


def sha256_file(file_path):
    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit(repo_dir):
    result = run_command(["git", "rev-parse", "HEAD"], cwd=repo_dir)

    if result["return_code"] == 0:
        return result["stdout"]

    return "Unavailable"


def get_git_remote(repo_dir):
    result = run_command(["git", "remote", "get-url", "origin"], cwd=repo_dir)

    if result["return_code"] == 0:
        return result["stdout"]

    return "Unavailable"


def find_files(repo_dir):
    code_extensions = {
        ".py",
        ".ipynb",
        ".yaml",
        ".yml",
        ".json",
        ".txt",
        ".md",
        ".sh",
        ".bat",
    }

    checkpoint_extensions = {
        ".pt",
        ".pth",
        ".ckpt",
        ".bin",
        ".pkl",
        ".tar",
    }

    important_files = []
    checkpoint_files = []
    feature_references = []
    label_references = []

    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        relative_path = str(path.relative_to(repo_dir))

        if suffix in checkpoint_extensions:
            checkpoint_files.append(
                {
                    "path": relative_path,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

        if suffix in code_extensions:
            important_files.append(
                {
                    "path": relative_path,
                    "size_bytes": path.stat().st_size,
                }
            )

            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                lowered = line.lower()

                if any(
                    phrase in lowered
                    for phrase in [
                        "vggish",
                        "i3d",
                        "rgb",
                        "flow",
                        ".npy",
                        ".h5",
                        "feature",
                    ]
                ):
                    feature_references.append(
                        {
                            "file": relative_path,
                            "line": line_number,
                            "text": line.strip()[:500],
                        }
                    )

                if any(
                    phrase in lowered
                    for phrase in [
                        "train.list",
                        "test.list",
                        "label",
                        "annotation",
                        "ground_truth",
                        "groundtruth",
                    ]
                ):
                    label_references.append(
                        {
                            "file": relative_path,
                            "line": line_number,
                            "text": line.strip()[:500],
                        }
                    )

    return {
        "important_files": important_files,
        "checkpoint_files": checkpoint_files,
        "feature_references": feature_references,
        "label_references": label_references,
    }


def extract_urls(repo_dir):
    urls = []

    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".md", ".txt", ".py", ".yaml", ".yml"}:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        found_urls = re.findall(r"https?://[^\s\]\)\}>\"']+", text)

        for url in found_urls:
            if url not in urls:
                urls.append(url)

    return urls


def write_markdown_report(report):
    checkpoint_rows = "\n".join(
        f'| `{item["path"]}` | {item["size_bytes"]:,} | `{item["sha256"]}` |'
        for item in report["repository_contents"]["checkpoint_files"]
    )

    if not checkpoint_rows:
        checkpoint_rows = "| No checkpoint file found in the cloned repository | - | - |"

    feature_rows = "\n".join(
        f'| `{item["file"]}` | {item["line"]} | `{item["text"]}` |'
        for item in report["repository_contents"]["feature_references"][:100]
    )

    if not feature_rows:
        feature_rows = "| No feature-path references identified automatically | - | - |"

    label_rows = "\n".join(
        f'| `{item["file"]}` | {item["line"]} | `{item["text"]}` |'
        for item in report["repository_contents"]["label_references"][:100]
    )

    if not label_rows:
        label_rows = "| No label-path references identified automatically | - | - |"

    url_rows = "\n".join(
        f"- {url}" for url in report["urls_found"]
    )

    if not url_rows:
        url_rows = "- No URLs found automatically."

    text = f"""# Official XDVioDet Repository Audit

**Audit time (UTC):** {report["audit_timestamp_utc"]}

## Repository Identity

| Field | Value |
|---|---|
| Intended official repository | {report["official_repository_url"]} |
| Local path | `{report["local_path"]}` |
| Remote URL | {report["git_remote"]} |
| Git commit | `{report["git_commit"]}` |
| Clone status | {report["clone_status"]} |

## Research Role

This repository is inspected as the detector baseline for the XD-Violence
feature-level experiments. It is not the proposed journal contribution.

The baseline is used to generate temporally indexed video-only and audio-video
event scores. Those scores will subsequently be calibrated and used by
selective evidence-preservation policies.

## Checkpoint Files Found Locally

| File | Size in bytes | SHA-256 |
|---|---:|---|
{checkpoint_rows}

**Checkpoint decision rule:** A checkpoint can be used only if it is present
locally or is linked from an official repository/documented author source. Its
path, source URL, Git commit, SHA-256 hash, expected input features, and
inference configuration must be recorded before it is used.

## Feature References Found in Code

| File | Line | Reference |
|---|---:|---|
{feature_rows}

## Label or Split References Found in Code

| File | Line | Reference |
|---|---:|---|
{label_rows}

## URLs Found in Repository Files

{url_rows}

## Required Manual Review

1. Read the repository README.
2. Confirm whether an official pretrained checkpoint is supplied or linked.
3. Identify the exact visual feature requirement: I3D RGB, I3D Flow, or both.
4. Identify the exact audio feature requirement: VGGish or another format.
5. Identify the expected feature-directory names and file naming convention.
6. Identify train, validation, and test list files.
7. Do not download any large feature archive until these requirements are clear.
"""

    report_path = DOCS_DIR / "xdviodel_repository_audit.md"
    report_path.write_text(text, encoding="utf-8")
    return report_path


def update_model_config(report):
    checkpoint_files = report["repository_contents"]["checkpoint_files"]

    config_lines = [
        "xdviodel_baseline:",
        '  repository_name: "Roc-Ng/XDVioDet"',
        f'  official_repository_url: "{OFFICIAL_REPOSITORY_URL}"',
        f'  local_repository_path: "{report["local_path"]}"',
        f'  git_commit: "{report["git_commit"]}"',
        '  role: "Existing baseline for event-score generation; not a novel detector"',
        '  status: "REPOSITORY_AUDITED_CHECKPOINT_NOT_YET_APPROVED"',
        "  selected_checkpoint:",
    ]

    if checkpoint_files:
        first_checkpoint = checkpoint_files[0]
        config_lines.extend(
            [
                f'    path: "{first_checkpoint["path"]}"',
                f'    sha256: "{first_checkpoint["sha256"]}"',
                '    status: "FOUND_LOCALLY_REQUIRES_MANUAL_COMPATIBILITY_CHECK"',
            ]
        )
    else:
        config_lines.extend(
            [
                '    path: null',
                "    sha256: null",
                '    status: "NO_LOCAL_CHECKPOINT_FOUND"',
            ]
        )

    config_lines.extend(
        [
            "  expected_modalities:",
            "    video: 'TO_BE_CONFIRMED_FROM_REPOSITORY_AUDIT'",
            "    audio: 'TO_BE_CONFIRMED_FROM_REPOSITORY_AUDIT'",
            "  next_action: >",
            "    Review docs/xdviodel_repository_audit.md and approve the exact",
            "    official checkpoint and feature archives before downloading them.",
        ]
    )

    config_path = CONFIGS_DIR / "xdviodel_baseline.yaml"
    config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    return config_path


def main():
    git_available = shutil.which("git") is not None

    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "official_repository_url": OFFICIAL_REPOSITORY_URL,
        "local_path": str(REPOSITORY_DIR),
        "git_available": git_available,
        "clone_status": "NOT_STARTED",
        "git_remote": "Unavailable",
        "git_commit": "Unavailable",
        "repository_contents": {
            "important_files": [],
            "checkpoint_files": [],
            "feature_references": [],
            "label_references": [],
        },
        "urls_found": [],
    }

    if not git_available:
        print("Git was not found in PATH.")
        print("Install Git for Windows, then rerun this script.")
        print("Download: https://git-scm.com/download/win")

    elif not REPOSITORY_DIR.exists():
        clone_result = run_command(
            ["git", "clone", OFFICIAL_REPOSITORY_URL, str(REPOSITORY_DIR)]
        )

        if clone_result["return_code"] != 0:
            report["clone_status"] = "FAILED"
            report["clone_error"] = clone_result
            print("Repository clone failed.")
            print(clone_result["stderr"])
        else:
            report["clone_status"] = "CLONED"

    else:
        report["clone_status"] = "ALREADY_PRESENT"

        pull_result = run_command(["git", "pull", "--ff-only"], cwd=REPOSITORY_DIR)
        report["git_pull_result"] = pull_result

        if pull_result["return_code"] != 0:
            print("Repository already exists. Git pull was not completed.")
            print("The local repository will still be audited.")

    if REPOSITORY_DIR.exists():
        report["git_remote"] = get_git_remote(REPOSITORY_DIR)
        report["git_commit"] = get_git_commit(REPOSITORY_DIR)
        report["repository_contents"] = find_files(REPOSITORY_DIR)
        report["urls_found"] = extract_urls(REPOSITORY_DIR)

    json_path = LOGS_DIR / "xdviodel_repository_audit.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_path = write_markdown_report(report)
    config_path = update_model_config(report)

    print()
    print("XDVioDet repository audit complete.")
    print(f"Repository status: {report['clone_status']}")
    print(f"Repository path: {REPOSITORY_DIR}")
    print(f"Git commit: {report['git_commit']}")
    print(
        "Checkpoint files found:",
        len(report["repository_contents"]["checkpoint_files"]),
    )
    print(
        "Feature references found:",
        len(report["repository_contents"]["feature_references"]),
    )
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")
    print(f"Baseline config: {config_path}")


if __name__ == "__main__":
    main()