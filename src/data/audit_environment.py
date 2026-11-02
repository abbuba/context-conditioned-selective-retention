import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import sklearn
except ImportError:
    sklearn = None

try:
    import yaml
except ImportError:
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_command_output(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=isinstance(command, str),
        )
        output = (result.stdout or result.stderr).strip()
        return output if output else "Not available"
    except Exception as error:
        return f"Unavailable: {error}"


def get_package_version(module):
    if module is None:
        return "Not installed"
    return getattr(module, "__version__", "Version unavailable")


def get_gpu_details():
    details = {
        "torch_installed": torch is not None,
        "cuda_available": False,
        "cuda_version": "Not available",
        "gpu_count": 0,
        "gpus": [],
    }

    if torch is None:
        return details

    details["cuda_available"] = bool(torch.cuda.is_available())
    details["cuda_version"] = torch.version.cuda or "Not available"

    if not details["cuda_available"]:
        return details

    details["gpu_count"] = torch.cuda.device_count()

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        details["gpus"].append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gb": round(properties.total_memory / (1024 ** 3), 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )

    return details


def get_disk_details():
    usage = shutil.disk_usage(PROJECT_ROOT)
    return {
        "project_drive": str(PROJECT_ROOT.drive),
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "used_gb": round(usage.used / (1024 ** 3), 2),
        "free_gb": round(usage.free / (1024 ** 3), 2),
    }


def get_ram_details():
    if psutil is None:
        return {
            "total_gb": "Install psutil to detect RAM automatically",
            "available_gb": "Install psutil to detect RAM automatically",
        }

    memory = psutil.virtual_memory()
    return {
        "total_gb": round(memory.total / (1024 ** 3), 2),
        "available_gb": round(memory.available / (1024 ** 3), 2),
    }


def get_ffmpeg_status():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return {
            "available": False,
            "path": "Not found in PATH",
            "version": "Not available",
        }

    return {
        "available": True,
        "path": ffmpeg_path,
        "version": get_command_output(["ffmpeg", "-version"]).splitlines()[0],
    }


def build_audit():
    gpu = get_gpu_details()

    return {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
        },
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
        },
        "processor": platform.processor() or "Not available",
        "ram": get_ram_details(),
        "disk": get_disk_details(),
        "gpu": gpu,
        "software": {
            "torch": get_package_version(torch),
            "opencv": get_package_version(cv2),
            "numpy": get_package_version(np),
            "pandas": get_package_version(pd),
            "scikit_learn": get_package_version(sklearn),
            "pyyaml": get_package_version(yaml),
            "ffmpeg": get_ffmpeg_status(),
            "pip_freeze": get_command_output([sys.executable, "-m", "pip", "freeze"]),
        },
    }


def write_hardware_log(audit):
    gpu_lines = ["No CUDA-compatible GPU detected by PyTorch."]
    if audit["gpu"]["cuda_available"]:
        gpu_lines = [
            f'- GPU {gpu["index"]}: {gpu["name"]} '
            f'({gpu["total_memory_gb"]} GB VRAM, '
            f'compute capability {gpu["compute_capability"]})'
            for gpu in audit["gpu"]["gpus"]
        ]

    ffmpeg = audit["software"]["ffmpeg"]

    lines = [
        "# Hardware and Environment Log",
        "",
        "This file was generated automatically before experimental execution.",
        "It records the computational environment used for reproducibility.",
        "",
        "## System",
        "",
        f'- Audit time (UTC): {audit["audit_timestamp_utc"]}',
        f'- Operating system: {audit["operating_system"]["system"]} '
        f'{audit["operating_system"]["release"]}',
        f'- Architecture: {audit["operating_system"]["architecture"]}',
        f'- Processor: {audit["processor"]}',
        f'- Total RAM: {audit["ram"]["total_gb"]} GB',
        f'- Available RAM at audit time: {audit["ram"]["available_gb"]} GB',
        f'- Project drive: {audit["disk"]["project_drive"]}',
        f'- Total disk capacity: {audit["disk"]["total_gb"]} GB',
        f'- Free disk space: {audit["disk"]["free_gb"]} GB',
        "",
        "## Python Environment",
        "",
        f'- Python executable: `{audit["python"]["executable"]}`',
        f'- Python version: {audit["python"]["version"]}',
        f'- PyTorch version: {audit["software"]["torch"]}',
        f'- CUDA available to PyTorch: {audit["gpu"]["cuda_available"]}',
        f'- CUDA version: {audit["gpu"]["cuda_version"]}',
        f'- OpenCV version: {audit["software"]["opencv"]}',
        f'- NumPy version: {audit["software"]["numpy"]}',
        f'- pandas version: {audit["software"]["pandas"]}',
        f'- scikit-learn version: {audit["software"]["scikit_learn"]}',
        f'- PyYAML version: {audit["software"]["pyyaml"]}',
        "",
        "## GPU",
        "",
        *gpu_lines,
        "",
        "## FFmpeg",
        "",
        f'- Available: {ffmpeg["available"]}',
        f'- Path: {ffmpeg["path"]}',
        f'- Version: {ffmpeg["version"]}',
        "",
        "## Notes",
        "",
        "- This audit was executed before dataset processing and model training.",
        "- Later experiments must record the same environment or explicitly document changes.",
        "- GPU and CUDA availability determine whether full feature extraction can be run locally.",
    ]

    hardware_log_path = DOCS_DIR / "hardware_log.md"
    hardware_log_path.write_text("\n".join(lines), encoding="utf-8")
    return hardware_log_path


def main():
    audit = build_audit()

    json_path = LOGS_DIR / "environment_audit.json"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    hardware_log_path = write_hardware_log(audit)

    print("Environment audit completed successfully.")
    print(f"JSON audit: {json_path}")
    print(f"Hardware log: {hardware_log_path}")
    print(f"CUDA available: {audit['gpu']['cuda_available']}")
    print(f"Free disk space: {audit['disk']['free_gb']} GB")
    print(f"FFmpeg available: {audit['software']['ffmpeg']['available']}")

    if not audit["gpu"]["cuda_available"]:
        print("\nWARNING: PyTorch did not detect a CUDA GPU.")
        print("We can still start dataset auditing and retention-policy code.")
        print("Before training/extracting large video features, we will decide whether")
        print("to install CUDA PyTorch, use pre-extracted features, or use Colab/Kaggle.")

    if not audit["software"]["ffmpeg"]["available"]:
        print("\nWARNING: FFmpeg was not found in PATH.")
        print("Install FFmpeg before extracting audio, video duration, FPS, or clips.")


if __name__ == "__main__":
    main()