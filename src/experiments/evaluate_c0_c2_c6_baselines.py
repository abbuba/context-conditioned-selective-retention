import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DECISIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "xd_final_retention_decisions_all_218.csv"
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FPS = 24.0
FRAMES_PER_TEMPORAL_UNIT = 16
EVENT_COVERAGE_THRESHOLD = 0.80

RANDOM_SEEDS = 100
BASE_RANDOM_SEED = 20261002


def read_lines(path):
    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def parse_annotations():
    annotations = {}

    for line in read_lines(ANNOTATION_FILE):
        tokens = line.split()
        video_id = tokens[0].removeprefix("v=")

        frame_values = [
            int(value)
            for value in tokens[1:]
        ]

        if len(frame_values) % 2 != 0:
            continue

        intervals = []

        for index in range(
            0,
            len(frame_values),
            2,
        ):
            start_frame = frame_values[index]
            end_frame = frame_values[index + 1]

            intervals.append(
                (
                    start_frame / FPS,
                    (end_frame + 1) / FPS,
                )
            )

        annotations[video_id] = intervals

    return annotations


def retained_intervals(video_df, retain_column):
    retained = video_df[
        retain_column
    ].to_numpy(dtype=int)

    intervals = []
    run_start = None

    for index in range(len(retained) + 1):
        value = retained[index] if index < len(
            retained
        ) else 0

        if value == 1 and run_start is None:
            run_start = index

        if value == 0 and run_start is not None:
            start_temporal_index = int(
                video_df.iloc[run_start][
                    "temporal_index"
                ]
            )

            end_temporal_index = int(
                video_df.iloc[index - 1][
                    "temporal_index"
                ]
            )

            intervals.append(
                (
                    start_temporal_index
                    * FRAMES_PER_TEMPORAL_UNIT
                    / FPS,
                    (end_temporal_index + 1)
                    * FRAMES_PER_TEMPORAL_UNIT
                    / FPS,
                )
            )

            run_start = None

    return intervals


def union_overlap_duration(
    event_start,
    event_end,
    retained,
):
    overlaps = []

    for retain_start, retain_end in retained:
        start = max(event_start, retain_start)
        end = min(event_end, retain_end)

        if end > start:
            overlaps.append((start, end))

    if not overlaps:
        return 0.0

    overlaps.sort()

    merged = []
    current_start, current_end = overlaps[0]

    for start, end in overlaps[1:]:
        if start <= current_end:
            current_end = max(
                current_end,
                end,
            )
        else:
            merged.append(
                (
                    current_start,
                    current_end,
                )
            )

            current_start, current_end = start, end

    merged.append(
        (
            current_start,
            current_end,
        )
    )

    return sum(
        end - start
        for start, end in merged
    )


def evaluate_retention(
    evaluation_df,
    annotations,
    retain_column,
):
    total_units = 0
    retained_units = 0

    total_events = 0
    preserved_events = 0

    total_event_duration = 0.0
    covered_event_duration = 0.0

    for video_id, video_df in evaluation_df.groupby(
        "video_id",
        sort=False,
    ):
        video_df = video_df.sort_values(
            "temporal_index"
        )

        total_units += len(video_df)

        retained_units += int(
            video_df[retain_column].sum()
        )

        if video_id not in annotations:
            continue

        retained = retained_intervals(
            video_df,
            retain_column,
        )

        for event_start, event_end in annotations[
            video_id
        ]:
            event_duration = (
                event_end - event_start
            )

            overlap = union_overlap_duration(
                event_start,
                event_end,
                retained,
            )

            coverage = (
                overlap / event_duration
                if event_duration > 0
                else 0.0
            )

            total_events += 1
            total_event_duration += event_duration
            covered_event_duration += overlap

            if coverage >= EVENT_COVERAGE_THRESHOLD:
                preserved_events += 1

    retained_duration_percent = (
        100.0 * retained_units / total_units
        if total_units > 0
        else 0.0
    )

    event_recall = (
        preserved_events / total_events
        if total_events > 0
        else 0.0
    )

    event_duration_coverage = (
        covered_event_duration
        / total_event_duration
        if total_event_duration > 0
        else 0.0
    )

    return {
        "total_temporal_units": total_units,
        "retained_temporal_units": retained_units,
        "retained_duration_percent": (
            retained_duration_percent
        ),
        "planned_duration_reduction_percent": (
            100.0 - retained_duration_percent
        ),
        "total_official_events": total_events,
        "preserved_official_events": (
            preserved_events
        ),
        "missed_official_events": (
            total_events - preserved_events
        ),
        "official_event_recall": event_recall,
        "event_duration_coverage": (
            event_duration_coverage
        ),
    }


def create_random_retention(
    evaluation_df,
    target_retained_units,
    random_generator,
):
    result = evaluation_df.copy()

    result["retain_random"] = 0

    all_indices = result.index.to_numpy()

    selected_indices = random_generator.choice(
        all_indices,
        size=target_retained_units,
        replace=False,
    )

    result.loc[
        selected_indices,
        "retain_random",
    ] = 1

    return result


def main():
    if not DECISIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing decisions file: {DECISIONS_FILE}"
        )

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {ANNOTATION_FILE}"
        )

    decisions_df = pd.read_csv(
        DECISIONS_FILE
    )

    required_columns = {
        "video_id",
        "split",
        "temporal_index",
        "retain_final",
    }

    missing_columns = required_columns - set(
        decisions_df.columns
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    annotations = parse_annotations()

    evaluation_df = decisions_df[
        decisions_df["split"] == "evaluation"
    ].copy()

    if evaluation_df.empty:
        raise RuntimeError(
            "No rows found with split == 'evaluation'."
        )

    evaluation_df = evaluation_df.sort_values(
        [
            "video_id",
            "temporal_index",
        ]
    ).copy()

    evaluation_df["retain_c0"] = 1
    evaluation_df["retain_c2"] = (
        evaluation_df["retain_final"]
        .astype(int)
    )

    c0_result = evaluate_retention(
        evaluation_df,
        annotations,
        "retain_c0",
    )

    c2_result = evaluate_retention(
        evaluation_df,
        annotations,
        "retain_c2",
    )

    print("C0 and C2 evaluation: COMPLETE")
    print(
        "C0 event recall: "
        f"{c0_result['official_event_recall']:.6f}"
    )
    print(
        "C2 event recall: "
        f"{c2_result['official_event_recall']:.6f}"
    )
    print(
        "C2 planned reduction: "
        f"{c2_result['planned_duration_reduction_percent']:.2f}%"
    )
    print()

    target_retained_units = int(
        c2_result["retained_temporal_units"]
    )

    random_results = []

    for seed_offset in range(RANDOM_SEEDS):
        seed = BASE_RANDOM_SEED + seed_offset

        if (
            seed_offset == 0
            or (seed_offset + 1) % 10 == 0
            or seed_offset + 1 == RANDOM_SEEDS
        ):
            print(
                f"Running C6 random baseline: "
                f"{seed_offset + 1}/{RANDOM_SEEDS}"
            )

        random_generator = np.random.default_rng(
            seed
        )

        random_df = create_random_retention(
            evaluation_df,
            target_retained_units,
            random_generator,
        )

        result = evaluate_retention(
            random_df,
            annotations,
            "retain_random",
        )

        random_results.append(
            {
                "seed": seed,
                **result,
            }
        )

    random_df = pd.DataFrame(random_results)

    c6_result = {
        "total_temporal_units": int(
            random_df[
                "total_temporal_units"
            ].iloc[0]
        ),
        "retained_temporal_units": int(
            random_df[
                "retained_temporal_units"
            ].iloc[0]
        ),
        "retained_duration_percent_mean": float(
            random_df[
                "retained_duration_percent"
            ].mean()
        ),
        "retained_duration_percent_std": float(
            random_df[
                "retained_duration_percent"
            ].std(ddof=1)
        ),
        "planned_duration_reduction_percent_mean": (
            float(
                random_df[
                    "planned_duration_reduction_percent"
                ].mean()
            )
        ),
        "planned_duration_reduction_percent_std": (
            float(
                random_df[
                    "planned_duration_reduction_percent"
                ].std(ddof=1)
            )
        ),
        "total_official_events": int(
            random_df[
                "total_official_events"
            ].iloc[0]
        ),
        "preserved_official_events_mean": float(
            random_df[
                "preserved_official_events"
            ].mean()
        ),
        "preserved_official_events_std": float(
            random_df[
                "preserved_official_events"
            ].std(ddof=1)
        ),
        "missed_official_events_mean": float(
            random_df[
                "missed_official_events"
            ].mean()
        ),
        "missed_official_events_std": float(
            random_df[
                "missed_official_events"
            ].std(ddof=1)
        ),
        "official_event_recall_mean": float(
            random_df[
                "official_event_recall"
            ].mean()
        ),
        "official_event_recall_std": float(
            random_df[
                "official_event_recall"
            ].std(ddof=1)
        ),
        "event_duration_coverage_mean": float(
            random_df[
                "event_duration_coverage"
            ].mean()
        ),
        "event_duration_coverage_std": float(
            random_df[
                "event_duration_coverage"
            ].std(ddof=1)
        ),
    }

    summary_rows = [
        {
            "condition": "C0_continuous_retention",
            "description": (
                "Retain all evaluation temporal units"
            ),
            **c0_result,
        },
        {
            "condition": "C2_audio_video_conditioned",
            "description": (
                "Frozen calibrated MIX2 score policy"
            ),
            **c2_result,
        },
        {
            "condition": "C6_random_duration_matched",
            "description": (
                "Uniform random retention matched to "
                "C2 retained temporal-unit count"
            ),
            "total_temporal_units": c6_result[
                "total_temporal_units"
            ],
            "retained_temporal_units": c6_result[
                "retained_temporal_units"
            ],
            "retained_duration_percent": c6_result[
                "retained_duration_percent_mean"
            ],
            "retained_duration_percent_std": c6_result[
                "retained_duration_percent_std"
            ],
            "planned_duration_reduction_percent": (
                c6_result[
                    "planned_duration_reduction_percent_mean"
                ]
            ),
            "planned_duration_reduction_percent_std": (
                c6_result[
                    "planned_duration_reduction_percent_std"
                ]
            ),
            "total_official_events": c6_result[
                "total_official_events"
            ],
            "preserved_official_events": c6_result[
                "preserved_official_events_mean"
            ],
            "preserved_official_events_std": c6_result[
                "preserved_official_events_std"
            ],
            "missed_official_events": c6_result[
                "missed_official_events_mean"
            ],
            "missed_official_events_std": c6_result[
                "missed_official_events_std"
            ],
            "official_event_recall": c6_result[
                "official_event_recall_mean"
            ],
            "official_event_recall_std": c6_result[
                "official_event_recall_std"
            ],
            "event_duration_coverage": c6_result[
                "event_duration_coverage_mean"
            ],
            "event_duration_coverage_std": c6_result[
                "event_duration_coverage_std"
            ],
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    summary_path = (
        OUTPUT_DIR
        / "xd_c0_c2_c6_baseline_summary.csv"
    )

    random_path = (
        OUTPUT_DIR
        / "xd_c6_random_duration_matched_runs.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    random_df.to_csv(
        random_path,
        index=False,
    )

    report = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "scope": (
            "Evaluation split of the 218-video "
            "temporal retention experiment"
        ),
        "event_definition": (
            "Official annotation frame intervals"
        ),
        "event_coverage_threshold": (
            EVENT_COVERAGE_THRESHOLD
        ),
        "random_baseline_runs": RANDOM_SEEDS,
        "random_baseline_seed_start": (
            BASE_RANDOM_SEED
        ),
        "c0_continuous_retention": c0_result,
        "c2_audio_video_conditioned": c2_result,
        "c6_random_duration_matched": c6_result,
        "summary_file": str(summary_path),
        "random_runs_file": str(random_path),
    }

    report_path = (
        LOG_DIR
        / "xd_c0_c2_c6_baseline_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("C0-C2-C6 baseline evaluation: PASS")
    print()

    print("C0 continuous retention")
    print(
        "  Event recall: "
        f"{c0_result['official_event_recall']:.6f}"
    )
    print(
        "  Event-duration coverage: "
        f"{c0_result['event_duration_coverage']:.6f}"
    )
    print(
        "  Planned reduction: "
        f"{c0_result['planned_duration_reduction_percent']:.2f}%"
    )

    print()
    print("C2 audio-video-conditioned retention")
    print(
        "  Event recall: "
        f"{c2_result['official_event_recall']:.6f}"
    )
    print(
        "  Event-duration coverage: "
        f"{c2_result['event_duration_coverage']:.6f}"
    )
    print(
        "  Planned reduction: "
        f"{c2_result['planned_duration_reduction_percent']:.2f}%"
    )

    print()
    print(
        "C6 random duration-matched retention "
        f"({RANDOM_SEEDS} seeds)"
    )
    print(
        "  Event recall: "
        f"{c6_result['official_event_recall_mean']:.6f} "
        f"± {c6_result['official_event_recall_std']:.6f}"
    )
    print(
        "  Event-duration coverage: "
        f"{c6_result['event_duration_coverage_mean']:.6f} "
        f"± {c6_result['event_duration_coverage_std']:.6f}"
    )
    print(
        "  Planned reduction: "
        f"{c6_result['planned_duration_reduction_percent_mean']:.2f}%"
    )

    print()
    print(f"Summary: {summary_path}")
    print(f"Random runs: {random_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()