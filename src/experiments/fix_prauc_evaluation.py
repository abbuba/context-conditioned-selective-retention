from pathlib import Path
import csv
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "evidence" / "extracted" / "results"

CONFIGURATIONS = (
    ("C1 RGB", "c1_rgb_xdviolence_video_scores.csv", "offline_max_score"),
    ("C2 RGB + Audio", "c2_rgb_audio_xdviolence_video_scores.csv", "offline_max_score"),
    (
        "C3 RGB + Audio + Context",
        "c3_rgb_audio_context_video_scores.csv",
        "c3_offline_max_score",
    ),
)


def evaluate_video_scores(configuration, filename, score_column):
    video_scores_file = RESULTS_DIR / filename

    with video_scores_file.open("r", newline="", encoding="utf-8") as file_handle:
        video_rows = list(csv.DictReader(file_handle))

    if not video_rows:
        raise RuntimeError(f"No video-score rows found in {video_scores_file}")

    required_columns = {"ground_truth_video_label", score_column}
    missing_columns = required_columns - set(video_rows[0])

    if missing_columns:
        raise RuntimeError(
            f"{video_scores_file.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    labels = np.array([
        1 if row["ground_truth_video_label"] == "anomalous" else 0
        for row in video_rows
    ])
    scores = np.array([float(row[score_column]) for row in video_rows])

    if labels.min() == labels.max():
        raise RuntimeError(f"{video_scores_file.name} contains only one label class")

    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    prevalence = float(labels.mean())

    return {
        "configuration": configuration,
        "score_column": score_column,
        "pr_auc": pr_auc,
        "random_baseline_prevalence": prevalence,
        "above_random": pr_auc > prevalence,
        "total_videos": len(video_rows),
        "anomalous_videos": int(labels.sum()),
        "normal_videos": int(len(labels) - labels.sum()),
    }


def main():
    print("=" * 80)
    print("CORRECTED VIDEO-LEVEL PR-AUC EVALUATION")
    print("=" * 80)

    results = [
        evaluate_video_scores(*configuration)
        for configuration in CONFIGURATIONS
    ]

    for result in results:
        print(f"\n{result['configuration']}")
        print(f"Score column: {result['score_column']}")
        print(f"Videos: {result['total_videos']}")
        print(f"Anomalous / normal: {result['anomalous_videos']} / "
              f"{result['normal_videos']}")
        print(f"PR-AUC: {result['pr_auc']:.4f}")
        print(f"Random baseline: {result['random_baseline_prevalence']:.4f}")
        print(f"Above random: {result['above_random']}")

    output_file = RESULTS_DIR / "corrected_prauc_results.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()