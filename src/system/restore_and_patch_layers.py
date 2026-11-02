from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
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
    if not BACKUP_FILE.exists():
        raise FileNotFoundError(
            f"Original backup is missing:\n{BACKUP_FILE}"
        )

    shutil.copy2(BACKUP_FILE, LAYERS_FILE)

    source_text = LAYERS_FILE.read_text(
        encoding="utf-8",
    )

    original_sha256 = sha256_file(BACKUP_FILE)

    class_start = source_text.find("class DistanceAdj")

    if class_start == -1:
        raise RuntimeError(
            "Could not find class DistanceAdj in layers.py."
        )

    prefix = source_text[:class_start]
    distance_adj_class = source_text[class_start:]

    cuda_single_count = distance_adj_class.count(".to('cuda')")
    cuda_double_count = distance_adj_class.count('.to("cuda")')

    distance_adj_class = distance_adj_class.replace(
        ".to('cuda')",
        ".to(device)",
    )

    distance_adj_class = distance_adj_class.replace(
        '.to("cuda")',
        ".to(device)",
    )

    forward_pattern = (
        r"(\n[ \t]*def forward\(\s*self\s*,\s*batch_size\s*,\s*[A-Za-z_]\w*\s*\)"
        r"\s*:\s*\n)"
    )

    match = re.search(
        forward_pattern,
        distance_adj_class,
    )

    if match is None:
        raise RuntimeError(
            "Could not find DistanceAdj.forward(batch_size, seq_len)."
        )

    forward_header = match.group(1)

    if "device = torch.device" not in distance_adj_class:
        function_match = re.match(
            r"\n([ \t]*)def forward",
            forward_header,
        )
        indentation = function_match.group(1)

        body_indentation = indentation + "    "

        device_lines = (
            f"{body_indentation}device = torch.device(\n"
            f"{body_indentation}    'cuda' if torch.cuda.is_available() else 'cpu'\n"
            f"{body_indentation})\n"
        )

        replacement_header = forward_header + device_lines

        distance_adj_class = (
            distance_adj_class[:match.start()]
            + replacement_header
            + distance_adj_class[match.end():]
        )

    patched_text = prefix + distance_adj_class

    remaining_cuda = [
        line.strip()
        for line in patched_text.splitlines()
        if ".to('cuda')" in line or '.to("cuda")' in line
    ]

    if remaining_cuda:
        raise RuntimeError(
            "Hard-coded CUDA calls remain:\n"
            + "\n".join(remaining_cuda)
        )

    if "device = torch.device" not in distance_adj_class:
        raise RuntimeError(
            "Device-selection logic was not inserted."
        )

    LAYERS_FILE.write_text(
        patched_text,
        encoding="utf-8",
    )

    patched_sha256 = sha256_file(LAYERS_FILE)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "original_backup": str(BACKUP_FILE),
        "patched_file": str(LAYERS_FILE),
        "original_sha256": original_sha256,
        "patched_sha256": patched_sha256,
        "cuda_single_quote_replacements": cuda_single_count,
        "cuda_double_quote_replacements": cuda_double_count,
        "total_cuda_replacements": (
            cuda_single_count + cuda_double_count
        ),
        "changes": [
            "Restored original author layers.py from backup.",
            "Patched only the DistanceAdj class.",
            "Inserted automatic CPU/CUDA device selection.",
            "Replaced hard-coded CUDA placement with .to(device).",
        ],
        "not_changed": [
            "Model checkpoint",
            "Model weights",
            "Model architecture",
            "Input features",
            "Score computation",
        ],
    }

    report_path = LOG_DIR / "xdviodel_final_cpu_patch.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("PATCH SUCCESSFUL")
    print(f"CUDA calls replaced: {report['total_cuda_replacements']}")
    print("Runtime device selection inserted: YES")
    print(f"Original SHA-256: {original_sha256}")
    print(f"Patched SHA-256: {patched_sha256}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()