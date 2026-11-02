import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "scores"
    / "xdviodel_mix2_raw_video_scores.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VALIDATION_RATIO = 0.20


def summarize_scores(labels, scores):
    precision, recall, thresholds = precision_recall_curve(
        labels,
        scores,
    )

    return {
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "roc_auc": float(
            roc_auc_score(labels, scores)
        ),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "pr_curve_points": int(len(precision)),
        "threshold_count": int(len(thresholds)),
    }


def main():
    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            f"Raw score file not found: {SCORES_FILE}"
        )

    df = pd.read_csv(SCORES_FILE)

    required_columns = {
        "video_index",
        "video_id",
        "weak_label",
        "offline_raw_score",
        "online_raw_score",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(df) != 800:
        raise RuntimeError(
            f"Expected 800 rows, found {len(df)}."
        )

    if df.isna().any().any():
        raise RuntimeError(
            "Score file contains missing values."
        )

    labels = df["weak_label"].to_numpy(dtype=int)

    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError(
            "Weak labels must contain both 0 and 1."
        )

    offline_scores = df["offline_raw_score"].to_numpy(dtype=float)
    online_scores = df["online_raw_score"].to_numpy(dtype=float)

    offline_metrics = summarize_scores(
        labels,
        offline_scores,
    )

    online_metrics = summarize_scores(
        labels,
        online_scores,
    )

    validation_indices, evaluation_indices = train_test_split(
        np.arange(len(df)),
        test_size=1.0 - VALIDATION_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    df["split"] = "evaluation"
    df.loc[validation_indices, "split"] = "validation"

    split_path = (
        METADATA_DIR
        / "xd_violence_policy_calibration_evaluation_split.csv"
    )

    df.to_csv(
        split_path,
        index=False,
    )

    validation_df = df[df["split"] == "validation"]
    evaluation_df = df[df["split"] == "evaluation"]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "input_score_file": str(SCORES_FILE),
        "random_seed": RANDOM_SEED,
        "validation_ratio": VALIDATION_RATIO,
        "evaluation_ratio": 1.0 - VALIDATION_RATIO,
        "total_videos": int(len(df)),
        "validation_videos": int(len(validation_df)),
        "evaluation_videos": int(len(evaluation_df)),
        "validation_positive_videos": int(
            validation_df["weak_label"].sum()
        ),
        "validation_negative_videos": int(
            len(validation_df) - validation_df["weak_label"].sum()
        ),
        "evaluation_positive_videos": int(
            evaluation_df["weak_label"].sum()
        ),
        "evaluation_negative_videos": int(
            len(evaluation_df) - evaluation_df["weak_label"].sum()
        ),
        "offline_raw_metrics_all_800": offline_metrics,
        "online_raw_metrics_all_800": online_metrics,
        "split_file": str(split_path),
        "important_scope_note": (
            "This is a fixed policy calibration/evaluation split created "
            "from the locally available XD-Violence test-feature partition. "
            "It must not be reported as the official XD-Violence benchmark "
            "test protocol. It is used for selective-retention policy "
            "experiments because no training is performed in this study."
        ),
    }

    report_path = LOG_DIR / "xdviodel_raw_score_audit.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    metrics_path = OUTPUT_DIR / "xdviodel_raw_score_metrics.csv"

    metrics_df = pd.DataFrame(
        [
            {
                "score_type": "offline_raw",
                **offline_metrics,
            },
            {
                "score_type": "online_raw",
                **online_metrics,
            },
        ]
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print("Raw-score audit: PASS")
    print(f"Total videos: {len(df)}")
    print(
        f"Normal videos: {int((df['weak_label'] == 0).sum())}"
    )
    print(
        f"Anomalous videos: {int((df['weak_label'] == 1).sum())}"
    )
    print()
    print(
        "Offline AP: "
        f"{offline_metrics['average_precision']:.6f}"
    )
    print(
        "Offline ROC-AUC: "
        f"{offline_metrics['roc_auc']:.6f}"
    )
    print(
        "Online AP: "
        f"{online_metrics['average_precision']:.6f}"
    )
    print(
        "Online ROC-AUC: "
        f"{online_metrics['roc_auc']:.6f}"
    )
    print()
    print(
        f"Validation videos: {len(validation_df)}"
    )
    print(
        f"Evaluation videos: {len(evaluation_df)}"
    )
    print(f"Split file: {split_path}")
    print(f"Metrics file: {metrics_path}")
    print(f"Audit report: {report_path}")


if __name__ == "__main__":
    main()