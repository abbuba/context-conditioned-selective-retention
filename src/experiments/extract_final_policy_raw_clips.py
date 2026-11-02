import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_INTERVALS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_final_retention_intervals_all_218.csv"
)

RAW_VIDEO_SUMMARY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "raw_video_pipeline"
    / "xd_raw_video_retention_video_summary_24.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "raw_video_pipeline"
CLIP_DIR = (
    OUTPUT_DIR
    / "retained_clips_final_official_policy_video_only"
)

NORMALIZED_DIR = (
    OUTPUT_DIR
    / "normalized_full_videos_video_only"
)

LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_VIDEO_CODEC = "libx264"
FFMPEG_CRF = "23"
FFMPEG_PRESET = "medium"
FFMPEG_AUDIO_CODEC = "aac"
FFMPEG_AUDIO_BITRATE = "128k"


def safe_filename(value):
    return (
        str(value)
        .replace("#", "_")
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


def normalize_video(input_path, output_path):
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
        "0:v:0",
        "-an",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-c:v",
        FFMPEG_VIDEO_CODEC,
        "-preset",
        FFMPEG_PRESET,
        "-crf",
        FFMPEG_CRF,
        "-movflags",
        "+faststart",
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
        "0:v:0",
        "-an",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-c:v",
        FFMPEG_VIDEO_CODEC,
        "-preset",
        FFMPEG_PRESET,
        "-crf",
        FFMPEG_CRF,
        "-movflags",
        "+faststart",
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
    for required_file in [
        FINAL_INTERVALS_FILE,
        RAW_VIDEO_SUMMARY_FILE,
    ]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Missing required file: {required_file}"
            )

    all_intervals_df = pd.read_csv(
        FINAL_INTERVALS_FILE
    )

    raw_videos_df = pd.read_csv(
        RAW_VIDEO_SUMMARY_FILE
    )

    required_raw_columns = {
        "video_id",
        "raw_video_path",
        "raw_duration_seconds",
    }

    missing_columns = required_raw_columns - set(
        raw_videos_df.columns
    )

    if missing_columns:
        raise RuntimeError(
            "Raw-video summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    raw_intervals_df = all_intervals_df.merge(
        raw_videos_df[
            [
                "video_id",
                "raw_video_path",
                "raw_duration_seconds",
            ]
        ],
        on="video_id",
        how="inner",
    ).copy()

    raw_intervals_df["end_seconds"] = np.minimum(
        raw_intervals_df["end_seconds"],
        raw_intervals_df["raw_duration_seconds"],
    )

    raw_intervals_df[
        "clip_duration_seconds"
    ] = (
        raw_intervals_df["end_seconds"]
        - raw_intervals_df["start_seconds"]
    ).clip(lower=0)

    raw_intervals_df = raw_intervals_df[
        raw_intervals_df[
            "clip_duration_seconds"
        ] > 0
    ].copy()

    raw_intervals_df = raw_intervals_df.sort_values(
        [
            "video_id",
            "start_seconds",
            "end_seconds",
        ]
    ).reset_index(drop=True)

    raw_intervals_df["interval_index"] = (
        raw_intervals_df.groupby("video_id").cumcount()
    )

    raw_interval_path = (
        OUTPUT_DIR
        / "xd_final_policy_raw_intervals_24.csv"
    )

    raw_intervals_df.to_csv(
        raw_interval_path,
        index=False,
    )

    normalized_rows = []
    clip_rows = []

    print("Preparing normalized full-video baseline...")

    for count, (_, row) in enumerate(
        raw_videos_df.iterrows(),
        start=1,
    ):
        video_id = row["video_id"]
        input_path = Path(row["raw_video_path"])

        output_path = (
            NORMALIZED_DIR
            / f"{safe_filename(video_id)}_normalized.mp4"
        )

        result = normalize_video(
            input_path,
            output_path,
        )

        normalized_rows.append(
            {
                "video_id": video_id,
                "raw_video_path": str(input_path),
                "normalized_video_path": str(output_path),
                "status": result["status"],
                "raw_source_bytes": (
                    input_path.stat().st_size
                    if input_path.exists()
                    else None
                ),
                "normalized_bytes": (
                    output_path.stat().st_size
                    if output_path.exists()
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
            f"Baseline {count}/{len(raw_videos_df)}: "
            f"{video_id}"
        )

    print()
    print(
        "Extracting clips with final official-event policy..."
    )

    for count, (_, row) in enumerate(
        raw_intervals_df.iterrows(),
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
                "split": row["split"],
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
            f"Clip {count}/{len(raw_intervals_df)}: "
            f"{output_path.name}"
        )

    normalized_df = pd.DataFrame(normalized_rows)
    clips_df = pd.DataFrame(clip_rows)

    normalized_manifest_path = (
        OUTPUT_DIR
        / "xd_final_policy_normalized_baseline_24.csv"
    )

    clip_manifest_path = (
        OUTPUT_DIR
        / "xd_final_policy_retained_clip_manifest_24.csv"
    )

    normalized_df.to_csv(
        normalized_manifest_path,
        index=False,
    )

    clips_df.to_csv(
        clip_manifest_path,
        index=False,
    )

    raw_source_bytes = int(
        normalized_df[
            "raw_source_bytes"
        ].dropna().sum()
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
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "PASS"
            if (
                (normalized_df["status"] != "failed").all()
                and (clips_df["status"] != "failed").all()
            )
            else "REVIEW_REQUIRED"
        ),
        "encoding_scope": (
            "Video-only. Audio streams, metadata, and chapters "
            "are intentionally excluded from both normalized "
            "full-video and retained-clip storage measurements."
        ),
        "policy": {
            "threshold": 0.12,
            "smoothing_window": 3,
            "minimum_trigger_units": 2,
            "pre_buffer_units": 0,
            "post_buffer_units": 0,
        },
        "raw_video_count": int(
            len(raw_videos_df)
        ),
        "retained_clip_count": int(
            len(clips_df)
        ),
        "raw_source_bytes": raw_source_bytes,
        "normalized_full_video_bytes": normalized_bytes,
        "retained_clip_bytes": retained_clip_bytes,
        "raw_duration_seconds": raw_duration,
        "retained_duration_seconds": retained_duration,
        "duration_retention_percent": (
            100.0 * retained_duration / raw_duration
        ),
        "duration_reduction_percent": (
            100.0
            * (
                1.0
                - retained_duration / raw_duration
            )
        ),
        "normalized_byte_retention_percent": (
            100.0
            * retained_clip_bytes
            / normalized_bytes
        ),
        "normalized_byte_reduction_percent": (
            100.0
            * (
                1.0
                - retained_clip_bytes / normalized_bytes
            )
        ),
        "raw_interval_file": str(raw_interval_path),
        "normalized_manifest_file": str(
            normalized_manifest_path
        ),
        "clip_manifest_file": str(
            clip_manifest_path
        ),
    }

    report_path = (
        LOG_DIR
        / "xd_final_policy_raw_clip_extraction.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print(
        "Final-policy video-only raw clip extraction: COMPLETE"
    )
    print(
        f"Raw videos: {report['raw_video_count']}"
    )
    print(
        f"Retained clips: {report['retained_clip_count']}"
    )
    print(
        "Retained duration: "
        f"{report['duration_retention_percent']:.2f}%"
    )
    print(
        "Duration reduction: "
        f"{report['duration_reduction_percent']:.2f}%"
    )
    print(
        "Normalized byte reduction: "
        f"{report['normalized_byte_reduction_percent']:.2f}%"
    )
    print(f"Intervals: {raw_interval_path}")
    print(f"Clip manifest: {clip_manifest_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()