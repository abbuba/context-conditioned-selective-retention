import hashlib
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

URL = "https://roc-ng.github.io/XD-Violence/images/annotations.txt"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "labels"
    / "official"
    / "annotations.txt"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    OUTPUT_FILE.write_bytes(response.content)

    print("Downloaded official XD-Violence test annotations.")
    print(f"Path: {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size:,} bytes")
    print(f"SHA-256: {sha256_file(OUTPUT_FILE)}")
    print(f"Lines: {len(OUTPUT_FILE.read_text(encoding='utf-8').splitlines())}")


if __name__ == "__main__":
    main()