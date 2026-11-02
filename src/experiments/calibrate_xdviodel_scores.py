import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "xd_violence_policy_calibration_evaluation_split.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scores"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

RAW_SCORE_COLUMN = "offline_raw_score"
LABEL_COLUMN = "weak_label"
SPLIT_COLUMN = "split"


def calculate_metrics(labels, probabilities):
    return {
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(labels, probabilities)
        ),
        "probability_min": float(np.min(probabilities)),
        "probability_max": float(np.max(probabilities)),
        "probability_mean": float(np.mean(probabilities)),
        "probability_std": float(np.std(probabilities)),
    }


def main():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}"
        )

    df = pd.read_csv(SPLIT_FILE)

    required_columns = {
        RAW_SCORE_COLUMN,
        LABEL_COLUMN,
        SPLIT_COLUMN,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    validation_df = df[df[SPLIT_COLUMN] == "validation"].copy()
    evaluation_df = df[df[SPLIT_COLUMN] == "evaluation"].copy()

    if len(validation_df) != 160 or len(evaluation_df) != 640:
        raise RuntimeError(
            "Expected 160 validation and 640 evaluation videos."
        )

    x_validation = validation_df[[RAW_SCORE_COLUMN]].to_numpy()
    y_validation = validation_df[LABEL_COLUMN].to_numpy(dtype=int)

    x_evaluation = evaluation_df[[RAW_SCORE_COLUMN]].to_numpy()
    y_evaluation = evaluation_df[LABEL_COLUMN].to_numpy(dtype=int)

    calibrator = LogisticRegression(
        random_state=42,
        solver="lbfgs",
    )

    calibrator.fit(
        x_validation,
        y_validation,
    )

    validation_probability = calibrator.predict_proba(
        x_validation
    )[:, 1]

    evaluation_probability = calibrator.predict_proba(
        x_evaluation
    )[:, 1]

    validation_df["event_probability"] = validation_probability
    evaluation_df["event_probability"] = evaluation_probability

    combined_df = pd.concat(
        [validation_df, evaluation_df],
        ignore_index=True,
    )

    output_path = (
        OUTPUT_DIR
        / "xdviodel_offline_calibrated_event_probabilities.csv"
    )

    combined_df.to_csv(
        output_path,
        index=False,
    )

    validation_raw_metrics = calculate_metrics(
        y_validation,
        validation_df[RAW_SCORE_COLUMN].to_numpy(),
    )

    validation_calibrated_metrics = calculate_metrics(
        y_validation,
        validation_probability,
    )

    evaluation_raw_metrics = calculate_metrics(
        y_evaluation,
        evaluation_df[RAW_SCORE_COLUMN].to_numpy(),
    )

    evaluation_calibrated_metrics = calculate_metrics(
        y_evaluation,
        evaluation_probability,
    )

    fraction_positive, mean_predicted = calibration_curve(
        y_validation,
        validation_probability,
        n_bins=10,
        strategy="quantile",
    )

    reliability_rows = []

    for observed, predicted in zip(
        fraction_positive,
        mean_predicted,
    ):
        reliability_rows.append(
            {
                "mean_predicted_probability": float(predicted),
                "observed_positive_fraction": float(observed),
            }
        )

    reliability_path = (
        METRICS_DIR
        / "xdviodel_validation_calibration_curve.csv"
    )

    pd.DataFrame(reliability_rows).to_csv(
        reliability_path,
        index=False,
    )

    metrics_rows = [
        {
            "subset": "validation",
            "score_version": "raw_offline",
            **validation_raw_metrics,
        },
        {
            "subset": "validation",
            "score_version": "calibrated_probability",
            **validation_calibrated_metrics,
        },
        {
            "subset": "evaluation",
            "score_version": "raw_offline",
            **evaluation_raw_metrics,
        },
        {
            "subset": "evaluation",
            "score_version": "calibrated_probability",
            **evaluation_calibrated_metrics,
        },
    ]

    metrics_path = (
        METRICS_DIR
        / "xdviodel_calibration_metrics.csv"
    )

    pd.DataFrame(metrics_rows).to_csv(
        metrics_path,
        index=False,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "calibration_method": "Platt scaling using logistic regression",
        "input_score": RAW_SCORE_COLUMN,
        "validation_videos": int(len(validation_df)),
        "evaluation_videos": int(len(evaluation_df)),
        "random_seed": 42,
        "model_coefficient": float(calibrator.coef_[0][0]),
        "model_intercept": float(calibrator.intercept_[0]),
        "validation_raw_metrics": validation_raw_metrics,
        "validation_calibrated_metrics": validation_calibrated_metrics,
        "evaluation_raw_metrics": evaluation_raw_metrics,
        "evaluation_calibrated_metrics": evaluation_calibrated_metrics,
        "calibrated_score_file": str(output_path),
        "reliability_curve_file": str(reliability_path),
        "metrics_file": str(metrics_path),
        "scope_note": (
            "Calibration is fitted only on the fixed validation subset. "
            "Evaluation probabilities are generated without refitting."
        ),
    }

    report_path = LOG_DIR / "xdviodel_score_calibration.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Calibration: PASS")
    print(f"Validation videos: {len(validation_df)}")
    print(f"Evaluation videos: {len(evaluation_df)}")
    print()
    print(
        "Validation Brier score, raw: "
        f"{validation_raw_metrics['brier_score']:.6f}"
    )
    print(
        "Validation Brier score, calibrated: "
        f"{validation_calibrated_metrics['brier_score']:.6f}"
    )
    print(
        "Evaluation Brier score, raw: "
        f"{evaluation_raw_metrics['brier_score']:.6f}"
    )
    print(
        "Evaluation Brier score, calibrated: "
        f"{evaluation_calibrated_metrics['brier_score']:.6f}"
    )
    print()
    print(f"Probabilities: {output_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()