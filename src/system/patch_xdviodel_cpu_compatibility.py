from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LAYERS_FILE = PROJECT_ROOT / "external" / "XDVioDet" / "layers.py"
BACKUP_FILE = PROJECT_ROOT / "external" / "XDVioDet" / "layers_original_cuda.py"

LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    if not LAYERS_FILE.exists():
        raise FileNotFoundError(f"Could not find: {LAYERS_FILE}")

    original_text = LAYERS_FILE.read_text(encoding="utf-8")

    if not BACKUP_FILE.exists():
        shutil.copy2(LAYERS_FILE, BACKUP_FILE)

    patched_text = original_text

    replacements = {
        ".to('cuda')": ".to(device)",
        '.to("cuda")': ".to(device)",
    }

    replacement_count = 0

    for old_text, new_text in replacements.items():
        count = patched_text.count(old_text)
        replacement_count += count
        patched_text = patched_text.replace(old_text, new_text)

    if replacement_count > 0:
        LAYERS_FILE.write_text(patched_text, encoding="utf-8")

    remaining_cuda_lines = [
        line.strip()
        for line in patched_text.splitlines()
        if ".to('cuda')" in line or '.to("cuda")' in line
    ]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_file": str(LAYERS_FILE),
        "backup_file": str(BACKUP_FILE),
        "replacement_count": replacement_count,
        "remaining_hardcoded_cuda_lines": remaining_cuda_lines,
        "patched_file_sha256": sha256_file(LAYERS_FILE),
        "backup_file_sha256": sha256_file(BACKUP_FILE),
        "status": "PASS" if not remaining_cuda_lines else "FAIL",
        "model_weights_modified": False,
        "feature_files_modified": False,
        "checkpoint_modified": False,
        "change_description": (
            "Replaced hard-coded CUDA tensor placement with device-aware "
            "placement. The model uses CPU when CUDA is unavailable."
        ),
    }

    report_path = LOG_DIR / "xdviodel_cpu_compatibility_patch.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    documentation = f"""# XDVioDet CPU Compatibility Patch

The official XDVioDet implementation contained hard-coded CUDA placement in
`external/XDVioDet/layers.py`.

All occurrences of:

```python
.to('cuda')
```

and:

```python
.to("cuda")
```

were replaced with:

```python
.to(device)
```

The inference runner defines the runtime device as CPU when CUDA is
unavailable.

| Field | Value |
|---|---|
| Replacements made | {replacement_count} |
| Remaining hard-coded CUDA lines | {len(remaining_cuda_lines)} |
| Model weights modified | No |
| Checkpoint modified | No |
| Feature files modified | No |
| Patched file SHA-256 | `{report["patched_file_sha256"]}` |
| Original backup SHA-256 | `{report["backup_file_sha256"]}` |
| Status | {report["status"]} |

Original file backup:

```text
{BACKUP_FILE}
```
"""

    documentation_path = DOCS_DIR / "xdviodel_cpu_compatibility_patch.md"
    documentation_path.write_text(documentation, encoding="utf-8")

    print("CPU compatibility patch completed.")
    print(f"Replacements made: {replacement_count}")
    print(f"Remaining hard-coded CUDA lines: {len(remaining_cuda_lines)}")
    print(f"Patch report: {report_path}")
    print(f"Documentation: {documentation_path}")

    if remaining_cuda_lines:
        raise SystemExit(
            "Patch failed: hard-coded CUDA calls still remain in layers.py."
        )


if __name__ == "__main__":
    main()