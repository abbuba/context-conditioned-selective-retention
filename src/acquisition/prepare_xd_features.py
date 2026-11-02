import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = PROJECT_ROOT / "data" / "raw" / "xd_violence" / "official_features"
LABEL_ROOT = PROJECT_ROOT / "data" / "raw" / "xd_violence" / "labels"
DOCS_ROOT = PROJECT_ROOT / "docs"

folders = [
    FEATURE_ROOT / "i3d_rgb_test",
    FEATURE_ROOT / "vggish_audio_test",
    FEATURE_ROOT / "archives",
    LABEL_ROOT / "official",
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

record = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint_mode": "MIX2",
    "required_downloads": [
        {
            "name": "I3D RGB test features",
            "target_directory": str(FEATURE_ROOT / "i3d_rgb_test"),
            "required": True,
        },
        {
            "name": "VGGish audio test features",
            "target_directory": str(FEATURE_ROOT / "vggish_audio_test"),
            "required": True,
        },
        {
            "name": "Official test annotations",
            "target_directory": str(LABEL_ROOT / "official"),
            "required": True,
        },
    ],
    "not_required_now": [
        "I3D Flow features",
        "Training features",
        "Full raw video dataset",
    ],
}

record_path = DOCS_ROOT / "xd_violence_feature_acquisition.json"
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

print("Feature directories created.")
for folder in folders:
    print(folder)

print("\nDownload only:")
print("1. I3D RGB test features")
print("2. VGGish audio test features")
print("3. Official test annotations")
print("\nDo not download Flow, training features, or full raw videos yet.")