from pathlib import Path
from getpass import getpass
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
import sys
import time
import uuid

import requests


# ============================================================
# PROJECT PATHS AND CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_DIR = PROJECT_ROOT / "evidence"

EVIDENCE_ZIP = (
    EVIDENCE_DIR
    / "cctv_selective_retention_evidence_package.zip"
)

EVIDENCE_SHA256_FILE = (
    EVIDENCE_DIR
    / "cctv_selective_retention_evidence_package.sha256"
)

IPFS_RECORDS_DIR = (
    EVIDENCE_DIR
    / "ipfs_records"
)

IPFS_RUNS_CSV = (
    IPFS_RECORDS_DIR
    / "ipfs_upload_runs.csv"
)

EXPECTED_SHA256 = (
    "0d6ff69cf2d7ae2c4b617d58217562a5eeeb0650906eb149401d451f2aecbda7"
)

PINATA_UPLOAD_URL = (
    "https://api.pinata.cloud/pinning/pinFileToIPFS"
)

ARTIFACT_COUNT = 24

# (connect, read/write) seconds; a stalled write previously hit the single 300 s limit.
UPLOAD_TIMEOUT_SECONDS = (30, 900)
UPLOAD_MAX_ATTEMPTS = 3
UPLOAD_RETRY_DELAY_SECONDS = 10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


class MultipartFileStream:
    # __len__ makes requests send Content-Length instead of chunked encoding.

    READ_BLOCK = 256 * 1024

    def __init__(self, file_path, fields, content_type):
        self.boundary = uuid.uuid4().hex
        self.file_path = file_path
        self.file_size = file_path.stat().st_size

        head = b""
        for name, value in fields.items():
            head += (
                f"--{self.boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")

        head += (
            f"--{self.boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")

        self._head = head
        self._tail = f"\r\n--{self.boundary}--\r\n".encode("utf-8")
        self._total = len(head) + self.file_size + len(self._tail)
        self._sent = 0
        self._next_report = 10
        self._file = open(file_path, "rb")
        self._stage = 0

    @property
    def content_type(self):
        return f"multipart/form-data; boundary={self.boundary}"

    def __len__(self):
        return self._total

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.READ_BLOCK

        size = min(size, self.READ_BLOCK)
        data = b""

        if self._stage == 0:
            data = self._head
            self._stage = 1
        elif self._stage == 1:
            data = self._file.read(size)
            if not data:
                data = self._tail
                self._stage = 2

        self._sent += len(data)
        percent = int(self._sent * 100 / self._total)

        while percent >= self._next_report and self._next_report <= 100:
            print(f"  sent {self._next_report}% ({self._sent} bytes)")
            self._next_report += 10

        return data

    def close(self):
        self._file.close()


def calculate_sha256(file_path, chunk_size=1024 * 1024):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def append_record_to_csv(record, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=record.keys(),
        )

        if not csv_exists:
            writer.writeheader()

        writer.writerow(record)


# ============================================================
# MAIN UPLOAD WORKFLOW
# ============================================================

def main():
    script_start_utc = utc_now()
    script_start_perf = time.perf_counter()

    run_id = (
        "IPFS-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8].upper()
    )

    IPFS_RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("CCTV SELECTIVE-RETENTION: PINATA IPFS EVIDENCE UPLOAD")
    print("=" * 90)

    print("\nRun ID:")
    print(run_id)

    print("\nScript start time (UTC):")
    print(script_start_utc)

    # --------------------------------------------------------
    # 1. Confirm required local files exist
    # --------------------------------------------------------

    if not EVIDENCE_ZIP.exists():
        raise FileNotFoundError(
            f"Evidence ZIP not found:\n{EVIDENCE_ZIP}"
        )

    if not EVIDENCE_SHA256_FILE.exists():
        raise FileNotFoundError(
            f"Evidence SHA-256 file not found:\n"
            f"{EVIDENCE_SHA256_FILE}"
        )

    # --------------------------------------------------------
    # 2. Read expected digest and verify ZIP integrity
    # --------------------------------------------------------

    hash_file_content = EVIDENCE_SHA256_FILE.read_text(
        encoding="utf-8"
    ).strip()

    hash_from_file = hash_file_content.split()[0].lower()

    sha256_start_perf = time.perf_counter()

    actual_sha256 = calculate_sha256(
        EVIDENCE_ZIP
    ).lower()

    sha256_verification_seconds = round(
        time.perf_counter() - sha256_start_perf,
        3,
    )

    evidence_zip_size_bytes = EVIDENCE_ZIP.stat().st_size
    evidence_zip_size_mb = round(
        evidence_zip_size_bytes / (1024 * 1024),
        3,
    )

    print("\nEvidence ZIP:")
    print(EVIDENCE_ZIP)

    print("\nEvidence ZIP size:")
    print(f"{evidence_zip_size_mb:.3f} MB")

    print("\nExpected project SHA-256:")
    print(EXPECTED_SHA256)

    print("\nSHA-256 read from .sha256 file:")
    print(hash_from_file)

    print("\nRecalculated local SHA-256:")
    print(actual_sha256)

    print("\nSHA-256 verification duration:")
    print(f"{sha256_verification_seconds:.3f} seconds")

    if hash_from_file != EXPECTED_SHA256:
        raise RuntimeError(
            "The SHA-256 value in the .sha256 file does not match "
            "the expected project SHA-256 digest."
        )

    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            "LOCAL INTEGRITY VERIFICATION FAILED.\n"
            "The ZIP differs from the verified evidence package.\n"
            "Do not upload this ZIP to Pinata."
        )

    print("\n✅ Local evidence-package integrity verification passed.")

    # --------------------------------------------------------
    # 3. Confirm public-IPFS upload
    # --------------------------------------------------------

    print("\nPublic-IPFS privacy reminder:")
    print(
        "This ZIP must not contain raw CCTV footage, personally "
        "identifiable information, private credentials, or secrets."
    )

    confirmation = input(
        "\nUpload this verified ZIP to Pinata Public IPFS?\n"
        "Type UPLOAD to continue: "
    ).strip()

    if confirmation != "UPLOAD":
        print("\nUpload cancelled. No file was sent to Pinata.")
        sys.exit(0)

    # --------------------------------------------------------
    # 4. Prompt securely for Pinata JWT
    # --------------------------------------------------------

    pinata_jwt = getpass(
        "\nPaste Pinata JWT (input remains hidden): "
    ).strip()

    if not pinata_jwt:
        raise RuntimeError(
            "No Pinata JWT was supplied."
        )

    # --------------------------------------------------------
    # 5. Prepare Pinata metadata
    # --------------------------------------------------------

    pinata_metadata = {
        "name": EVIDENCE_ZIP.name,
        "keyvalues": {
            "project": "cctv-selective-retention",
            "artifact_type": "verified-evidence-package",
            "sha256": actual_sha256,
            "artifact_count": str(ARTIFACT_COUNT),
            "c1_checkpoint": "c1_rgb.pkl",
            "c2_checkpoint": "c2_rgb_audio6.pkl",
            "raw_cctv_included": "false",
            "hyperledger_status": "pending",
            "run_id": run_id,
        },
    }

    pinata_options = {
        "cidVersion": 1,
    }

    headers = {
        "Authorization": f"Bearer {pinata_jwt}",
    }

    # --------------------------------------------------------
    # 6. Upload and measure Pinata API response duration
    # --------------------------------------------------------

    print("\nUploading to Pinata Public IPFS. Please wait...")

    upload_request_start_utc = utc_now()
    upload_request_start_perf = time.perf_counter()

    response = None
    last_error = None

    for attempt in range(1, UPLOAD_MAX_ATTEMPTS + 1):
        print(f"\nUpload attempt {attempt}/{UPLOAD_MAX_ATTEMPTS}")
        stream = MultipartFileStream(
            EVIDENCE_ZIP,
            {
                "pinataMetadata": json.dumps(pinata_metadata),
                "pinataOptions": json.dumps(pinata_options),
            },
            "application/zip",
        )

        try:
            response = requests.post(
                PINATA_UPLOAD_URL,
                headers={
                    **headers,
                    "Content-Type": stream.content_type,
                },
                data=stream,
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
            break
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as error:
            last_error = error
            print(
                f"Attempt {attempt}/{UPLOAD_MAX_ATTEMPTS} failed after "
                f"{stream._sent}/{len(stream)} bytes: "
                f"{type(error).__name__}"
            )

            if attempt < UPLOAD_MAX_ATTEMPTS:
                time.sleep(UPLOAD_RETRY_DELAY_SECONDS * attempt)
        finally:
            stream.close()

    if response is None:
        pinata_jwt = None
        raise RuntimeError(
            "Pinata upload failed after "
            f"{UPLOAD_MAX_ATTEMPTS} attempts. Check network, VPN, "
            "proxy, or antivirus HTTPS inspection, then rerun.\n"
            f"Last error: {last_error}"
        )

    pinata_upload_response_seconds = round(
        time.perf_counter() - upload_request_start_perf,
        3,
    )

    upload_response_received_utc = utc_now()

    # Remove credential reference after request.
    pinata_jwt = None

    try:
        response_data = response.json()
    except ValueError:
        response_data = {
            "raw_response": response.text
        }

    response_file = (
        IPFS_RECORDS_DIR
        / f"{run_id}_pinata_api_response.json"
    )

    response_file.write_text(
        json.dumps(response_data, indent=2),
        encoding="utf-8",
    )

    print("\nPinata HTTP status:")
    print(response.status_code)

    print("\nPinata upload-response duration:")
    print(f"{pinata_upload_response_seconds:.3f} seconds")

    if response.status_code not in [200, 201]:
        print("\nPinata response:")
        print(json.dumps(response_data, indent=2))

        raise RuntimeError(
            "\nPinata upload failed.\n"
            f"Saved API response: {response_file}"
        )

    cid = response_data.get("IpfsHash")
    pin_size_bytes = response_data.get("PinSize")
    pinata_timestamp = response_data.get("Timestamp")

    if not cid:
        print("\nUnexpected Pinata response:")
        print(json.dumps(response_data, indent=2))

        raise RuntimeError(
            "Pinata returned a success status but no IPFS CID was found."
        )

    # --------------------------------------------------------
    # 7. Create local provenance and timing record
    # --------------------------------------------------------

    script_end_utc = utc_now()

    wall_clock_script_seconds = round(
        time.perf_counter() - script_start_perf,
        3,
    )

    measured_automated_seconds = round(
        sha256_verification_seconds
        + pinata_upload_response_seconds,
        3,
    )

    upload_record = {
        "run_id": run_id,
        "record_type": "pinata_public_ipfs_upload",
        "upload_status": "SUCCESS",
        "project": "CCTV selective retention",
        "upload_service": "Pinata",
        "ipfs_network": "Public IPFS",
        "evidence_zip_file": EVIDENCE_ZIP.name,
        "evidence_artifact_count": ARTIFACT_COUNT,
        "evidence_zip_size_bytes": evidence_zip_size_bytes,
        "evidence_zip_size_mb": evidence_zip_size_mb,
        "expected_sha256": EXPECTED_SHA256,
        "sha256_from_hash_file": hash_from_file,
        "recalculated_sha256": actual_sha256,
        "local_integrity_verification": "PASSED",
        "sha256_verification_duration_seconds": sha256_verification_seconds,
        "pinata_upload_response_duration_seconds": (
            pinata_upload_response_seconds
        ),
        "automated_hash_plus_upload_seconds": (
            measured_automated_seconds
        ),
        "script_wall_clock_seconds_including_user_input": (
            wall_clock_script_seconds
        ),
        "upload_request_start_utc": upload_request_start_utc,
        "upload_response_received_utc": upload_response_received_utc,
        "script_start_utc": script_start_utc,
        "script_end_utc": script_end_utc,
        "ipfs_cid": cid,
        "cid_version": 1,
        "pinata_pin_size_bytes": pin_size_bytes,
        "pinata_timestamp": pinata_timestamp,
        "c1_checkpoint": "c1_rgb.pkl",
        "c2_checkpoint": "c2_rgb_audio6.pkl",
        "raw_cctv_included": False,
        "hyperledger_fabric_status": "PENDING",
        "python_version": sys.version.split()[0],
        "operating_system": platform.platform(),
        "requests_version": requests.__version__,
        "pinata_response_file": str(response_file),
    }

    record_file = (
        IPFS_RECORDS_DIR
        / f"{run_id}_ipfs_provenance_record.json"
    )

    record_file.write_text(
        json.dumps(upload_record, indent=2),
        encoding="utf-8",
    )

    append_record_to_csv(
        upload_record,
        IPFS_RUNS_CSV,
    )

    # --------------------------------------------------------
    # 8. Print paper-relevant final results
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("✅ PINATA IPFS UPLOAD SUCCESSFUL")
    print("=" * 90)

    print("\nEvidence ZIP SHA-256:")
    print(actual_sha256)

    print("\nIPFS CID:")
    print(cid)

    print("\nMeasured timing metrics:")

    print(
        "SHA-256 verification time:",
        f"{sha256_verification_seconds:.3f} seconds",
    )

    print(
        "Pinata upload-response time:",
        f"{pinata_upload_response_seconds:.3f} seconds",
    )

    print(
        "Automated SHA-256 + upload time:",
        f"{measured_automated_seconds:.3f} seconds",
    )

    print(
        "Total script wall-clock time:",
        f"{wall_clock_script_seconds:.3f} seconds",
    )

    print("\nPinata pin size:")
    print(f"{pin_size_bytes} bytes")

    print("\nPinata timestamp:")
    print(pinata_timestamp)

    print("\nIPFS provenance record:")
    print(record_file)

    print("\nCumulative IPFS run log:")
    print(IPFS_RUNS_CSV)

    print("\nRaw Pinata API response:")
    print(response_file)

    print("\nOptional public gateway format:")
    print(f"https://gateway.pinata.cloud/ipfs/{cid}")

    print(
        "\nNext stage: store the SHA-256 digest, CID, "
        "timestamps, and verification metadata in Hyperledger Fabric."
    )


if __name__ == "__main__":
    main()