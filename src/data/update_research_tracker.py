import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

AUDIT_FILE = LOGS_DIR / "environment_audit.json"
JSON_TRACKER = DOCS_DIR / "research_tracker.json"
MARKDOWN_TRACKER = DOCS_DIR / "research_tracker.md"


def load_json(path, default=None):
    if not path.exists():
        return {} if default is None else default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {} if default is None else default


def build_tracker(audit):
    gpu = audit.get("gpu", {})
    disk = audit.get("disk", {})
    ram = audit.get("ram", {})
    ffmpeg = audit.get("software", {}).get("ffmpeg", {})

    return {
        "project_name": "Context-Conditioned Selective Evidence Retention",
        "tracker_updated_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": (
            "Can calibrated multimodal event evidence and operational-policy "
            "context reduce retained surveillance footage while preserving "
            "relevant event evidence and enabling verifiable integrity checks?"
        ),
        "current_phase": {
            "id": "Phase 1A",
            "name": "XD-Violence acquisition and dataset audit",
            "status": "READY_TO_START",
            "next_action": (
                "Download XD-Violence from the official author-hosted source, "
                "place files under data/raw/xd_violence, and generate a manifest."
            ),
        },
        "completed_steps": [
            {
                "id": "Phase 0A",
                "name": "Project structure and reproducibility configuration",
                "status": "COMPLETE",
            },
            {
                "id": "Phase 0B",
                "name": "Environment audit",
                "status": "COMPLETE_WITH_RAM_PENDING",
            },
        ],
        "pending_steps": [
            {
                "id": "Phase 0B-Update",
                "name": "Install psutil and rerun environment audit",
                "status": "PENDING",
            },
            {
                "id": "Phase 1A",
                "name": "Download XD-Violence and inspect dataset contents",
                "status": "PENDING",
            },
            {
                "id": "Phase 1B",
                "name": "Generate reproducible XD-Violence manifest",
                "status": "PENDING",
            },
            {
                "id": "Phase 1C",
                "name": "Verify audio-video alignment and metadata availability",
                "status": "PENDING",
            },
            {
                "id": "Phase 2",
                "name": "Build unified temporal event and retention representation",
                "status": "PENDING",
            },
            {
                "id": "Phase 3",
                "name": "Run C1 video-only event and retention baseline",
                "status": "PENDING",
            },
            {
                "id": "Phase 4",
                "name": "Run C2 audio-video fusion baseline",
                "status": "PENDING",
            },
            {
                "id": "Phase 5",
                "name": "Calibration and temporal clip construction",
                "status": "PENDING",
            },
            {
                "id": "Phase 6",
                "name": "Run selective-preservation baselines C0-C7",
                "status": "PENDING",
            },
            {
                "id": "Phase 7",
                "name": "Context-conditioned preservation experiments",
                "status": "PENDING",
            },
            {
                "id": "Phase 8",
                "name": "IPFS and Hyperledger Fabric evidence experiments",
                "status": "PENDING",
            },
            {
                "id": "Phase 9",
                "name": "Ablations, failure analysis, and final tables",
                "status": "PENDING",
            },
        ],
        "environment": {
            "operating_system": (
                f'{audit.get("operating_system", {}).get("system", "Unknown")} '
                f'{audit.get("operating_system", {}).get("release", "")}'
            ).strip(),
            "python_version": audit.get("python", {}).get("version", "Unknown"),
            "python_executable": audit.get("python", {}).get("executable", "Unknown"),
            "torch_version": audit.get("software", {}).get("torch", "Unknown"),
            "cuda_available": gpu.get("cuda_available", False),
            "gpu_count": gpu.get("gpu_count", 0),
            "gpu_details": gpu.get("gpus", []),
            "ram_total_gb": ram.get("total_gb", "Pending psutil installation"),
            "ram_available_gb": ram.get("available_gb", "Pending psutil installation"),
            "disk_free_gb": disk.get("free_gb", "Unknown"),
            "ffmpeg_available": ffmpeg.get("available", False),
            "ffmpeg_version": ffmpeg.get("version", "Unknown"),
        },
        "dataset_status": {
            "xd_violence": {
                "status": "NOT_DOWNLOADED",
                "role": "Primary audio-video selective-retention dataset",
                "expected_location": "data/raw/xd_violence",
                "manifest": "data/metadata/xd_violence_manifest.csv",
            },
            "nwpu_campus": {
                "status": "NOT_DOWNLOADED",
                "role": "Secondary scene/context-policy validation dataset",
                "expected_location": "data/raw/nwpu_campus",
            },
            "cctv_fights": {
                "status": "NOT_DOWNLOADED",
                "role": "Temporal evidence-window validation dataset",
                "expected_location": "data/raw/cctv_fights",
            },
        },
        "conditions": {
            "C0": "Continuous retention",
            "C1": "Video-only selective retention",
            "C2": "Audio-video selective retention",
            "C3": "Audio-video with scene context",
            "C4": "Context-conditioned preservation policy",
            "C5": "Rule-only context policy",
            "C6": "Random duration-matched retention",
            "C7": "Periodic retention",
        },
        "frozen_initial_parameters": {
            "snippet_seconds": 1.0,
            "pre_event_seconds": 10.0,
            "post_event_seconds": 15.0,
            "merge_gap_seconds": 3.0,
            "minimum_trigger_seconds": 2.0,
            "minimum_release_seconds": 2.0,
            "event_coverage_threshold": 0.80,
            "initial_priority_threshold": 0.70,
            "critical_event_probability_threshold": 0.90,
            "random_seeds": [13, 27, 42, 77, 101],
        },
        "non_negotiable_rules": [
            "Do not use the final test set to select thresholds, weights, or models.",
            "Do not claim results before a matching CSV or JSON output exists.",
            "Do not claim blockchain makes video immutable.",
            "Do not call arbitrary site labels real operational context.",
            "Record all changes to configurations and experimental parameters.",
            "Store per-video and per-event metrics before computing aggregates.",
        ],
    }


def write_markdown(tracker):
    environment = tracker["environment"]
    phase = tracker["current_phase"]
    xd = tracker["dataset_status"]["xd_violence"]

    completed_rows = "\n".join(
        f'| {item["id"]} | {item["name"]} | {item["status"]} |'
        for item in tracker["completed_steps"]
    )

    pending_rows = "\n".join(
        f'| {item["id"]} | {item["name"]} | {item["status"]} |'
        for item in tracker["pending_steps"]
    )

    conditions_rows = "\n".join(
        f"| {key} | {value} |"
        for key, value in tracker["conditions"].items()
    )

    frozen_rows = "\n".join(
        f"| {key} | {value} |"
        for key, value in tracker["frozen_initial_parameters"].items()
    )

    rules = "\n".join(
        f"{index + 1}. {rule}"
        for index, rule in enumerate(tracker["non_negotiable_rules"])
    )

    markdown = f"""# Research Progress Tracker

**Project:** {tracker["project_name"]}

**Last updated (UTC):** {tracker["tracker_updated_utc"]}

## Research Question

{tracker["research_question"]}

## Current Position

| Field | Value |
|---|---|
| Current phase | {phase["id"]}: {phase["name"]} |
| Current status | {phase["status"]} |
| Next action | {phase["next_action"]} |

## Completed Steps

| ID | Step | Status |
|---|---|---|
{completed_rows}

## Pending Steps

| ID | Step | Status |
|---|---|---|
{pending_rows}

## Environment

| Field | Value |
|---|---|
| Operating system | {environment["operating_system"]} |
| Python executable | `{environment["python_executable"]}` |
| PyTorch | {environment["torch_version"]} |
| CUDA available | {environment["cuda_available"]} |
| GPU count | {environment["gpu_count"]} |
| Total RAM | {environment["ram_total_gb"]} GB |
| Available RAM | {environment["ram_available_gb"]} GB |
| Free disk space | {environment["disk_free_gb"]} GB |
| FFmpeg available | {environment["ffmpeg_available"]} |
| FFmpeg version | {environment["ffmpeg_version"]} |

## Dataset Status

| Dataset | Status | Role | Expected Location |
|---|---|---|---|
| XD-Violence | {xd["status"]} | {xd["role"]} | `{xd["expected_location"]}` |
| NWPU Campus | {tracker["dataset_status"]["nwpu_campus"]["status"]} | {tracker["dataset_status"]["nwpu_campus"]["role"]} | `{tracker["dataset_status"]["nwpu_campus"]["expected_location"]}` |
| CCTV-Fights | {tracker["dataset_status"]["cctv_fights"]["status"]} | {tracker["dataset_status"]["cctv_fights"]["role"]} | `{tracker["dataset_status"]["cctv_fights"]["expected_location"]}` |

## Planned Experimental Conditions

| ID | Condition |
|---|---|
{conditions_rows}

## Frozen Initial Parameters

These are initial values. Any later change must be logged and selected only using validation data.

| Parameter | Value |
|---|---|
{frozen_rows}

## Non-Negotiable Research Rules

{rules}

## Resume Prompt for a New Chat

Copy and paste the following into a new chat if needed:

```text
I am conducting the project “Context-Conditioned Selective Evidence Retention.”
Please read the attached files `docs/research_tracker.md`,
`docs/research_tracker.json`, `docs/experiment_log.md`,
`docs/paper_methodology_draft.md`, and `configs/study_protocol.yaml`.
Continue from the “Current Position” section of `research_tracker.md`.
Do not invent results. Give Python code for every implementation step and
paper-ready text for every completed experimental phase.
```
"""
    MARKDOWN_TRACKER.write_text(markdown, encoding="utf-8")


def main():
    audit = load_json(AUDIT_FILE)
    tracker = build_tracker(audit)

    JSON_TRACKER.write_text(json.dumps(tracker, indent=2), encoding="utf-8")
    write_markdown(tracker)

    print("Research tracker created successfully.")
    print(f"JSON tracker: {JSON_TRACKER}")
    print(f"Markdown tracker: {MARKDOWN_TRACKER}")
    print(f"Current phase: {tracker['current_phase']['id']}")
    print(f"Next action: {tracker['current_phase']['next_action']}")


if __name__ == "__main__":
    main()