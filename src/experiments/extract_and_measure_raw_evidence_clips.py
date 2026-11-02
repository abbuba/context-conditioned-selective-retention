import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERVAL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "raw_video_pipeline"
    / "xd_raw_video_retention_intervals_24.csv"
)

VIDEO_SUMMARY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "raw_video_pipeline"
    / "xd_raw_video_retention_video_summary_24.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "raw_video_pipeline"
CLIP_DIR = OUTPUT_DIR / "retained_clips"
NORMALIZED_DIR = OUTPUT_DIR / "normalized_full_videos"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

CLIP_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_VIDEO_CODEC = "libx264"
FFMPEG_CRF = "23"
FFMPEG_PRESET = "medium"
FFMPEG_AUDIO_CODEC = "aac"
FFMPEG_AUDIO_BITRATE = "128k"


def safe_filename(name):
    return (
        name.replace("#", "_")
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def run_ffmpeg(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )

    return {
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def normalize_full_video(input_path, output_path):
    if output_path.exists() and output_path.stat().st_size > 0:
        return {
            "status": "already_exists",
            "return_code": 0,
        }

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
        "-c:v",
        FFMPEG_VIDEO_CODEC,
        "-preset",
        FFMPEG_PRESET,
        "-crf",
        FFMPEG_CRF,
        "-ac",
        "2",
        "-c:a",
        FFMPEG_AUDIO_CODEC,
        "-b:a",
        FFMPEG_AUDIO_BITRATE,
        str(output_path),
    ]

    result = run_ffmpeg(command)

    return {
        "status": (
            "success"
            if result["return_code"] == 0
            else "failed"
        ),
        **result,
    }


def extract_clip(
    input_path,
    output_path,
    start_seconds,
    duration_seconds,
):
    if output_path.exists() and output_path.stat().st_size > 0:
        return {
            "status": "already_exists",
            "return_code": 0,
        }

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
        "-c:v",
        FFMPEG_VIDEO_CODEC,
        "-preset",
        FFMPEG_PRESET,
        "-crf",
        FFMPEG_CRF,
        "-ac",
        "2",
        "-c:a",
        FFMPEG_AUDIO_CODEC,
        "-b:a",
        FFMPEG_AUDIO_BITRATE,
        str(output_path),
    ]

    result = run_ffmpeg(command)

    return {
        "status": (
            "success"
            if result["return_code"] == 0
            else "failed"
        ),
        **result,
    }


def main():
    if not INTERVAL_FILE.exists():
        raise FileNotFoundError(
            f"Interval file missing: {INTERVAL_FILE}"
        )

    if not VIDEO_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Video summary missing: {VIDEO_SUMMARY_FILE}"
        )

    intervals_df = pd.read_csv(INTERVAL_FILE)
    videos_df = pd.read_csv(VIDEO_SUMMARY_FILE)

    normalized_rows = []
    clip_rows = []

    print("Creating normalized full-video byte baseline...")

    for count, (_, row) in enumerate(
        videos_df.iterrows(),
        start=1,
    ):
        video_id = row["video_id"]
        input_path = Path(row["raw_video_path"])

        normalized_path = (
            NORMALIZED_DIR
            / f"{safe_filename(video_id)}_normalized.mp4"
        )

        result = normalize_full_video(
            input_path,
            normalized_path,
        )

        normalized_rows.append(
            {
                "video_id": video_id,
                "raw_video_path": str(input_path),
                "normalized_video_path": str(
                    normalized_path
                ),
                "status": result["status"],
                "raw_source_bytes": input_path.stat().st_size,
                "normalized_bytes": (
                    normalized_path.stat().st_size
                    if normalized_path.exists()
                    else None
                ),
                "raw_duration_seconds": row[
                    "raw_duration_seconds"
                ],
                "ffmpeg_error": (
                    result.get("stderr", "")[-1000:]
                    if result["status"] == "failed"
                    else None
                ),
            }
        )

        print(
            f"Normalized {count}/{len(videos_df)}: "
            f"{video_id}"
        )

    print()
    print("Extracting retained temporal evidence clips...")

    for count, (_, row) in enumerate(
        intervals_df.iterrows(),
        start=1,
    ):
        video_id = row["video_id"]
        interval_index = int(row["interval_index"])

        input_path = Path(row["raw_video_path"])

        output_path = (
            CLIP_DIR
            / (
                f"{safe_filename(video_id)}"
                f"_interval_{interval_index:03d}.mp4"
            )
        )

        result = extract_clip(
            input_path=input_path,
            output_path=output_path,
            start_seconds=float(row["start_seconds"]),
            duration_seconds=float(
                row["clip_duration_seconds"]
            ),
        )

        clip_rows.append(
            {
                "video_id": video_id,
                "subset_group": row["subset_group"],
                "interval_index": interval_index,
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "clip_duration_seconds": row[
                    "clip_duration_seconds"
                ],
                "source_video_path": str(input_path),
                "clip_path": str(output_path),
                "status": result["status"],
                "clip_bytes": (
                    output_path.stat().st_size
                    if output_path.exists()
                    else None
                ),
                "clip_size_mb": (
                    round(
                        output_path.stat().st_size
                        / (1024 ** 2),
                        3,
                    )
                    if output_path.exists()
                    else None
                ),
                "ffmpeg_error": (
                    result.get("stderr", "")[-1000:]
                    if result["status"] == "failed"
                    else None
                ),
            }
        )

        print(
            f"Extracted {count}/{len(intervals_df)}: "
            f"{output_path.name}"
        )

    normalized_df = pd.DataFrame(normalized_rows)
    clips_df = pd.DataFrame(clip_rows)

    normalized_path = (
        OUTPUT_DIR
        / "xd_normalized_full_video_baseline_24.csv"
    )

    clips_path = (
        OUTPUT_DIR
        / "xd_retained_clip_manifest_24.csv"
    )

    normalized_df.to_csv(
        normalized_path,
        index=False,
    )

    clips_df.to_csv(
        clips_path,
        index=False,
    )

    raw_source_bytes = int(
        normalized_df["raw_source_bytes"].sum()
    )

    normalized_bytes = int(
        normalized_df[
            "normalized_bytes"
        ].dropna().sum()
    )

    retained_clip_bytes = int(
        clips_df["clip_bytes"].dropna().sum()
    )

    raw_duration = float(
        normalized_df[
            "raw_duration_seconds"
        ].sum()
    )

    retained_duration = float(
        clips_df[
            "clip_duration_seconds"
        ].sum()
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS"
            if (
                (normalized_df["status"] != "failed").all()
                and (clips_df["status"] != "failed").all()
            )
            else "REVIEW_REQUIRED"
        ),
        "experiment_type": "RAW_VIDEO_PIPELINE_PROOF",
        "encoding": {
            "video_codec": FFMPEG_VIDEO_CODEC,
            "crf": FFMPEG_CRF,
            "preset": FFMPEG_PRESET,
            "audio_codec": FFMPEG_AUDIO_CODEC,
            "audio_bitrate": FFMPEG_AUDIO_BITRATE,
        },
        "raw_source_bytes": raw_source_bytes,
        "normalized_full_video_bytes": normalized_bytes,
        "retained_clip_bytes": retained_clip_bytes,
        "raw_duration_seconds": raw_duration,
        "retained_duration_seconds": retained_duration,
        "duration_retention_percent": (
            100.0 * retained_duration / raw_duration
        ),
        "duration_storage_reduction_percent": (
            100.0
            * (
                1.0
                - retained_duration / raw_duration
            )
        ),
        "source_byte_retention_percent": (
            100.0 * retained_clip_bytes / raw_source_bytes
        ),
        "source_byte_reduction_percent": (
            100.0
            * (
                1.0
                - retained_clip_bytes / raw_source_bytes
            )
        ),
        "normalized_byte_retention_percent": (
            100.0
            * retained_clip_bytes
            / normalized_bytes
        )
        if normalized_bytes > 0
        else None,
        "normalized_byte_reduction_percent": (
            100.0
            * (
                1.0
                - retained_clip_bytes / normalized_bytes
            )
        )
        if normalized_bytes > 0
        else None,
        "normalized_video_manifest": str(
            normalized_path
        ),
        "retained_clip_manifest": str(clips_path),
    }

    report_path = (
        LOG_DIR
        / "xd_raw_video_clip_extraction.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("Raw-video clip extraction: COMPLETE")
    print(
        f"Normalized full videos: "
        f"{len(normalized_df)}"
    )
    print(
        f"Retained clips: "
        f"{len(clips_df)}"
    )
    print(
        f"Original source bytes: "
        f"{raw_source_bytes}"
    )
    print(
        f"Normalized baseline bytes: "
        f"{normalized_bytes}"
    )
    print(
        f"Retained clip bytes: "
        f"{retained_clip_bytes}"
    )
    print(
        "Duration reduction: "
        f"{report['duration_storage_reduction_percent']:.2f}%"
    )
    print(
        "Normalized byte reduction: "
        + (
            f"{report['normalized_byte_reduction_percent']:.2f}%"
            if report["normalized_byte_reduction_percent"] is not None
            else "N/A (normalized byte baseline unavailable)"
        )
    )
    print(f"Clip manifest: {clips_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()