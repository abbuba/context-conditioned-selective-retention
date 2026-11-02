import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "external" / "XDVioDet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FILES = [
    "README.md",
    "option.py",
    "test.py",
    "eval.py",
    "inference.py",
    "main.py",
]


def extract_relevant_lines(path):
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    results = []

    keywords = [
        "checkpoint",
        "ckpt",
        "test",
        "modality",
        "rgb-list",
        "audio-list",
        "test-rgb-list",
        "test-audio-list",
        "feature-size",
        "torch.load",
        "wsanodet",
    ]

    for number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()

        if any(keyword in lowered for keyword in keywords):
            results.append(
                {
                    "line": number,
                    "text": line.strip(),
                }
            )

    return results


def main():
    report = {
        "repository": str(REPO_ROOT),
        "files": {},
        "candidate_entrypoints": [],
        "recommended_fixed_values": {
            "modality": "MIX2",
            "feature_size": 1152,
            "checkpoint": "ckpt/wsanodet_mix2.pkl",
            "rgb_list": "list/rgb_test.list",
            "audio_list": "list/audio_test.list",
        },
    }

    for filename in TARGET_FILES:
        path = REPO_ROOT / filename

        if path.exists():
            report["files"][filename] = {
                "exists": True,
                "relevant_lines": extract_relevant_lines(path),
            }

            if filename in {"test.py", "eval.py", "inference.py", "main.py"}:
                report["candidate_entrypoints"].append(str(path))
        else:
            report["files"][filename] = {"exists": False}

    for path in REPO_ROOT.glob("*.py"):
        if path.name not in TARGET_FILES:
            lines = extract_relevant_lines(path)

            if lines:
                report["files"][path.name] = {
                    "exists": True,
                    "relevant_lines": lines,
                }

    output_path = OUTPUT_DIR / "xdviodel_inference_inspection.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Inference inspection complete.")
    print(f"Repository: {REPO_ROOT}")
    print("Candidate entrypoints:")

    for entrypoint in report["candidate_entrypoints"]:
        print(f"  {entrypoint}")

    print("\nRecommended fixed values:")
    for key, value in report["recommended_fixed_values"].items():
        print(f"  {key}: {value}")

    print(f"\nReport: {output_path}")


if __name__ == "__main__":
    main()