import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBSET_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "xd_raw_video_pipeline_subset_24.csv"
)

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xd_violence"
    / "labels"
    / "official"
    / "annotations.txt"
)

RAW_METADATA = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_raw_video_metadata_24.csv"
)

CLIP_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "raw_video_pipeline"
    / "xd_final_policy_retained_clip_manifest_24.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "raw_video_pipeline"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

EVENT_COVERAGE_THRESHOLD = 0.80


def read_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_annotations():
    annotations = {}

    for line in read_lines(ANNOTATION_FILE):
        tokens = line.split()

        video_id = tokens[0].removeprefix("v=")

        values = [
            int(value)
            for value in tokens[1:]
        ]

        if len(values) % 2 != 0:
            continue

        intervals = []

        for index in range(0, len(values), 2):
            intervals.append(
                (
                    values[index],
                    values[index + 1],
                )
            )

        annotations[video_id] = intervals

    return annotations


def union_overlap_length(
    event_start,
    event_end,
    clip_intervals,
):
    overlaps = []

    for clip_start, clip_end in clip_intervals:
        start = max(event_start, clip_start)
        end = min(event_end, clip_end)

        if end > start:
            overlaps.append((start, end))

    if not overlaps:
        return 0.0

    overlaps.sort()

    merged = []
    current_start, current_end = overlaps[0]

    for start, end in overlaps[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append(
                (current_start, current_end)
            )
            current_start, current_end = start, end

    merged.append((current_start, current_end))

    return sum(
        end - start
        for start, end in merged
    )


def main():
    for path in [
        SUBSET_MANIFEST,
        ANNOTATION_FILE,
        RAW_METADATA,
        CLIP_MANIFEST,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

    subset_df = pd.read_csv(SUBSET_MANIFEST)
    metadata_df = pd.read_csv(RAW_METADATA)
    clips_df = pd.read_csv(CLIP_MANIFEST)

    annotations = parse_annotations()

    fps_map = {
        row["video_id"]: float(row["avg_fps"])
        for _, row in metadata_df.iterrows()
        if row["status"] == "PASS"
    }

    clip_map = {}

    successful_clips = clips_df[
        clips_df["status"].isin(
            ["success", "already_exists"]
        )
    ]

    for video_id, group in successful_clips.groupby(
        "video_id"
    ):
        clip_map[video_id] = [
            (
                float(row["start_seconds"]),
                float(row["end_seconds"]),
            )
            for _, row in group.iterrows()
        ]

    event_rows = []
    video_rows = []

    total_events = 0
    preserved_events = 0
    total_event_duration = 0.0
    covered_event_duration = 0.0

    for _, subset_row in subset_df.iterrows():
        video_id = subset_row["video_id"]

        if video_id not in annotations:
            video_rows.append(
                {
                    "video_id": video_id,
                    "subset_group": subset_row[
                        "subset_group"
                    ],
                    "annotated_event_count": 0,
                    "preserved_event_count": 0,
                    "event_recall": None,
                    "event_duration_seconds": 0.0,
                    "covered_event_duration_seconds": 0.0,
                    "event_coverage_ratio": None,
                    "retained_clip_count": len(
                        clip_map.get(video_id, [])
                    ),
                }
            )
            continue

        fps = fps_map.get(video_id)

        if fps is None or fps <= 0:
            raise RuntimeError(
                f"Invalid FPS for {video_id}"
            )

        clip_intervals = clip_map.get(video_id, [])

        video_event_count = 0
        video_preserved_count = 0
        video_event_duration = 0.0
        video_covered_duration = 0.0

        for event_index, (
            start_frame,
            end_frame,
        ) in enumerate(annotations[video_id]):
            event_start_seconds = start_frame / fps
            event_end_seconds = (
                (end_frame + 1) / fps
            )

            event_duration = (
                event_end_seconds
                - event_start_seconds
            )

            covered_duration = union_overlap_length(
                event_start_seconds,
                event_end_seconds,
                clip_intervals,
            )

            coverage_ratio = (
                covered_duration / event_duration
                if event_duration > 0
                else 0.0
            )

            preserved = int(
                coverage_ratio
                >= EVENT_COVERAGE_THRESHOLD
            )

            event_rows.append(
                {
                    "video_id": video_id,
                    "subset_group": subset_row[
                        "subset_group"
                    ],
                    "event_index": event_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_seconds": round(
                        event_start_seconds,
                        6,
                    ),
                    "end_seconds": round(
                        event_end_seconds,
                        6,
                    ),
                    "event_duration_seconds": round(
                        event_duration,
                        6,
                    ),
                    "covered_duration_seconds": round(
                        covered_duration,
                        6,
                    ),
                    "coverage_ratio": round(
                        coverage_ratio,
                        6,
                    ),
                    "preserved": preserved,
                    "retained_clip_count": len(
                        clip_intervals
                    ),
                }
            )

            video_event_count += 1
            video_preserved_count += preserved
            video_event_duration += event_duration
            video_covered_duration += covered_duration

        video_rows.append(
            {
                "video_id": video_id,
                "subset_group": subset_row[
                    "subset_group"
                ],
                "annotated_event_count": video_event_count,
                "preserved_event_count": (
                    video_preserved_count
                ),
                "event_recall": (
                    video_preserved_count
                    / video_event_count
                ),
                "event_duration_seconds": round(
                    video_event_duration,
                    6,
                ),
                "covered_event_duration_seconds": round(
                    video_covered_duration,
                    6,
                ),
                "event_coverage_ratio": round(
                    video_covered_duration
                    / video_event_duration,
                    6,
                )
                if video_event_duration > 0
                else None,
                "retained_clip_count": len(
                    clip_intervals
                ),
            }
        )

        total_events += video_event_count
        preserved_events += video_preserved_count
        total_event_duration += video_event_duration
        covered_event_duration += video_covered_duration

    event_df = pd.DataFrame(event_rows)
    video_df = pd.DataFrame(video_rows)

    event_path = (
        OUTPUT_DIR
        / "xd_final_policy_raw_event_coverage_24.csv"
    )

    video_path = (
        OUTPUT_DIR
        / "xd_final_policy_raw_evidence_recall_summary_24.csv"
    )

    event_df.to_csv(
        event_path,
        index=False,
    )

    video_df.to_csv(
        video_path,
        index=False,
    )

    event_recall = (
        preserved_events / total_events
        if total_events > 0
        else 0.0
    )

    duration_coverage = (
        covered_event_duration
        / total_event_duration
        if total_event_duration > 0
        else 0.0
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "experiment_type": "RAW_VIDEO_PIPELINE_PROOF",
        "event_coverage_threshold": (
            EVENT_COVERAGE_THRESHOLD
        ),
        "subset_video_count": len(subset_df),
        "annotated_event_count": total_events,
        "preserved_event_count": preserved_events,
        "raw_video_event_recall": event_recall,
        "total_event_duration_seconds": (
            total_event_duration
        ),
        "covered_event_duration_seconds": (
            covered_event_duration
        ),
        "event_duration_coverage_ratio": (
            duration_coverage
        ),
        "event_coverage_file": str(event_path),
        "video_summary_file": str(video_path),
        "scope_note": (
            "This result applies only to the selected 24-video "
            "raw-video pipeline proof subset."
        ),
    }

    report_path = (
        LOG_DIR
        / "xd_final_policy_raw_evidence_recall.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Final-policy raw-video evidence recall: PASS")
    print(
        f"Annotated events: "
        f"{total_events}"
    )
    print(
        f"Preserved events: "
        f"{preserved_events}"
    )
    print(
        f"Evidence recall: "
        f"{event_recall:.6f}"
    )
    print(
        f"Event-duration coverage: "
        f"{duration_coverage:.6f}"
    )
    print(f"Event coverage: {event_path}")
    print(f"Video summary: {video_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()