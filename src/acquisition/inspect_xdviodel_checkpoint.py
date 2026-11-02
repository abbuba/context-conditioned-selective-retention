import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "external" / "XDVioDet"
CHECKPOINT = REPO_ROOT / "ckpt" / "wsanodet_mix2.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = PROJECT_ROOT / "configs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_object(obj):
    if isinstance(obj, dict):
        return {
            "type": "dict",
            "key_count": len(obj),
            "keys": [str(key) for key in list(obj.keys())[:100]],
        }

    if isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "length": len(obj),
        }

    return {
        "type": type(obj).__name__,
        "representation": repr(obj)[:1000],
    }


def load_checkpoint(path):
    errors = []

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return obj, "torch.load(weights_only=False)", errors
    except Exception as error:
        errors.append(f"torch.load failed: {repr(error)}")

    try:
        with path.open("rb") as file:
            obj = pickle.load(file)
        return obj, "pickle.load", errors
    except Exception as error:
        errors.append(f"pickle.load failed: {repr(error)}")

    return None, None, errors


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint_hash = sha256_file(CHECKPOINT)
    checkpoint, loader, errors = load_checkpoint(CHECKPOINT)

    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_size_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "loader_used": loader,
        "load_errors_before_success": errors,
        "checkpoint_summary": summarize_object(checkpoint) if checkpoint is not None else None,
        "approved_input_contract": {
            "modality": "MIX2",
            "visual_feature": "I3D RGB",
            "visual_dimension": 1024,
            "audio_feature": "VGGish",
            "audio_dimension": 128,
            "combined_dimension": 1152,
            "optical_flow_required": False,
            "rgb_crops_per_video": 5,
            "audio_feature_mapping": "audio_list[index // 5]",
        },
        "status": "LOADABLE" if checkpoint is not None else "FAILED_TO_LOAD",
    }

    json_path = OUTPUT_DIR / "xdviodel_checkpoint_audit.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown = f"""# XDVioDet Checkpoint Contract

## Approved Baseline

| Field | Value |
|---|---|
| Repository commit | `f846d8cfc454943ec165bd2489b71e7dc0a1064d` |
| Checkpoint | `ckpt/wsanodet_mix2.pkl` |
| Checkpoint SHA-256 | `{checkpoint_hash}` |
| Checkpoint size | {CHECKPOINT.stat().st_size:,} bytes |
| Checkpoint load status | {report["status"]} |
| Loader used | `{loader}` |

## Input Contract

| Input | Representation | Dimension |
|---|---|---:|
| Visual stream | I3D RGB | 1024 |
| Audio stream | VGGish | 128 |
| Combined MIX2 input | I3D RGB + VGGish | 1152 |
| Optical flow | Not required | - |

The checkpoint requires MIX2 input. RGB features use five-crop augmentation,
whereas one audio feature file is associated with every five RGB crop files.

## Checkpoint Structure

```json
{json.dumps(report["checkpoint_summary"], indent=2)}
```

## Decision

This checkpoint is the fixed baseline event-score generator. It is not trained,
modified, or presented as the proposed contribution. Its output scores will be
calibrated and used by the selective evidence-preservation experiments.
"""

    markdown_path = DOCS_DIR / "xdviodel_checkpoint_contract.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    config = f"""xdviodel_baseline:
  repository_commit: "f846d8cfc454943ec165bd2489b71e7dc0a1064d"
  checkpoint:
    path: "external/XDVioDet/ckpt/wsanodet_mix2.pkl"
    sha256: "{checkpoint_hash}"
    size_bytes: {CHECKPOINT.stat().st_size}
    status: "{report["status"]}"
  modality: "MIX2"
  visual:
    representation: "I3D_RGB"
    dimension: 1024
    crops_per_video: 5
  audio:
    representation: "VGGish"
    dimension: 128
  combined_dimension: 1152
  optical_flow_required: false
  role: "Fixed pretrained baseline score generator"
"""

    config_path = CONFIG_DIR / "xdviodel_baseline.yaml"
    config_path.write_text(config, encoding="utf-8")

    print(f"Checkpoint status: {report['status']}")
    print(f"Loader used: {loader}")
    print(f"SHA-256: {checkpoint_hash}")
    print(f"JSON report: {json_path}")
    print(f"Contract report: {markdown_path}")
    print(f"Config file: {config_path}")


if __name__ == "__main__":
    main()