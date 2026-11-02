import json
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_VIDEO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "raw_subset"
)

SUBSET_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "xd_raw_video_pipeline_subset_24.csv"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "official_features"
    / "i3d_rgb_test"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_PER_TEMPORAL_SCORE = 16
FEATURE_DERIVED_FPS = 24.0


def run_ffprobe(video_path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        (
            "format=duration,size,bit_rate:"
            "stream=codec_name,codec_type,width,height,"
            "avg_frame_rate,r_frame_rate,nb_frames"
        ),
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        return {
            "status": "ERROR",
            "error": result.stderr.strip(),
        }

    return {
        "status": "PASS",
        "data": json.loads(result.stdout),
    }


def fraction_to_float(value):
    try:
        if value in {None, "", "0/0"}:
            return None

        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def find_video_file(video_id):
    matches = list(
        RAW_VIDEO_DIR.glob(f"{video_id}.*")
    )

    video_extensions = {
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".webm",
    }

    matches = [
        path
        for path in matches
        if path.suffix.lower() in video_extensions
    ]

    if len(matches) != 1:
        return None

    return matches[0]


def main():
    if not SUBSET_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest missing: {SUBSET_MANIFEST}"
        )

    subset_df = pd.read_csv(SUBSET_MANIFEST)

    rows = []

    for _, item in subset_df.iterrows():
        video_id = item["video_id"]
        video_path = find_video_file(video_id)

        if video_path is None:
            rows.append(
                {
                    "video_id": video_id,
                    "status": "MISSING_OR_AMBIGUOUS",
                }
            )
            continue

        feature_path = FEATURE_DIR / f"{video_id}__0.npy"

        if not feature_path.exists():
            rows.append(
                {
                    "video_id": video_id,
                    "status": "MISSING_FEATURE",
                    "local_video_path": str(video_path),
                }
            )
            continue

        feature_array = np.load(
            feature_path,
            mmap_mode="r",
        )

        temporal_units = int(feature_array.shape[0])

        probe = run_ffprobe(video_path)

        if probe["status"] != "PASS":
            rows.append(
                {
                    "video_id": video_id,
                    "status": "FFPROBE_ERROR",
                    "local_video_path": str(video_path),
                    "ffprobe_error": probe["error"],
                }
            )
            continue

        format_data = probe["data"].get("format", {})
        streams = probe["data"].get("streams", [])

        video_stream = next(
            (
                stream
                for stream in streams
                if stream.get("codec_type") == "video"
            ),
            {},
        )

        audio_stream = next(
            (
                stream
                for stream in streams
                if stream.get("codec_type") == "audio"
            ),
            {},
        )

        actual_duration = float(
            format_data.get("duration", 0.0)
        )

        avg_fps = fraction_to_float(
            video_stream.get("avg_frame_rate")
        )

        real_fps = fraction_to_float(
            video_stream.get("r_frame_rate")
        )

        expected_duration = (
            temporal_units
            * FRAMES_PER_TEMPORAL_SCORE
            / FEATURE_DERIVED_FPS
        )

        duration_difference = (
            actual_duration - expected_duration
        )

        rows.append(
            {
                "video_id": video_id,
                "status": "PASS",
                "local_video_path": str(video_path),
                "file_size_bytes": video_path.stat().st_size,
                "file_size_mb": round(
                    video_path.stat().st_size
                    / (1024 ** 2),
                    3,
                ),
                "duration_seconds": round(
                    actual_duration,
                    6,
                ),
                "video_codec": video_stream.get(
                    "codec_name"
                ),
                "width": video_stream.get("width"),
                "height": video_stream.get("height"),
                "avg_fps": avg_fps,
                "real_fps": real_fps,
                "video_frame_count": video_stream.get(
                    "nb_frames"
                ),
                "audio_codec": audio_stream.get(
                    "codec_name"
                ),
                "feature_temporal_units": temporal_units,
                "feature_expected_duration_seconds": round(
                    expected_duration,
                    6,
                ),
                "duration_difference_seconds": round(
                    duration_difference,
                    6,
                ),
                "frames_per_temporal_score": (
                    FRAMES_PER_TEMPORAL_SCORE
                ),
            }
        )

        print(f"Audited: {video_id}")

    output_df = pd.DataFrame(rows)

    metadata_path = (
        OUTPUT_DIR
        / "xd_raw_video_metadata_24.csv"
    )

    output_df.to_csv(
        metadata_path,
        index=False,
    )

    successful = output_df[
        output_df["status"] == "PASS"
    ].copy()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS"
            if len(successful) == len(subset_df)
            else "REVIEW_REQUIRED"
        ),
        "selected_videos": len(subset_df),
        "successfully_audited_videos": len(successful),
        "feature_derived_fps": FEATURE_DERIVED_FPS,
        "frames_per_temporal_score": (
            FRAMES_PER_TEMPORAL_SCORE
        ),
        "mean_actual_fps": (
            float(successful["avg_fps"].mean())
            if not successful.empty
            else None
        ),
        "mean_duration_difference_seconds": (
            float(
                successful[
                    "duration_difference_seconds"
                ].mean()
            )
            if not successful.empty
            else None
        ),
        "max_absolute_duration_difference_seconds": (
            float(
                successful[
                    "duration_difference_seconds"
                ].abs().max()
            )
            if not successful.empty
            else None
        ),
        "metadata_file": str(metadata_path),
    }

    report_path = (
        LOG_DIR
        / "xd_raw_video_metadata_audit.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("Raw-video metadata audit complete.")
    print(
        f"Successfully audited: "
        f"{len(successful)}/{len(subset_df)}"
    )
    print(
        "Mean actual FPS: "
        f"{report['mean_actual_fps']}"
    )
    print(
        "Mean duration difference: "
        f"{report['mean_duration_difference_seconds']}"
    )
    print(
        "Maximum absolute duration difference: "
        f"{report['max_absolute_duration_difference_seconds']}"
    )
    print(f"Metadata: {metadata_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()