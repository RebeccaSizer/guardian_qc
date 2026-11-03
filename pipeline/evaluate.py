"""Evaluate Guardian-QC Isolation Forest predictions and explanation methods."""

from pathlib import Path
import os 
from utils.logger import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import config
import argparse

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def clean_sample_name(series: pd.Series) -> pd.Series:
    """Standardise sample identifiers before matching."""
    return series.astype("string").str.strip()


def normalise_label(value):
    """Convert anomaly labels to -1 (anomaly) or 1 (normal)."""
    if pd.isna(value):
        return pd.NA

    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Unexpected anomaly label {value!r}; expected -1 or 1."
        ) from exc

    if number not in (-1, 1):
        raise ValueError(f"Unexpected anomaly label {value!r}; expected -1 or 1.")

    return int(number)


def clean_feature(value):
    """Standardise feature names for exact matching."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    return value or None


def get_predicted_features(row: pd.Series, columns: list[str]) -> set[str]:
    """Return unique, non-empty feature names from the specified columns."""
    return {
        feature
        for column in columns
        if column in row.index
        if (feature := clean_feature(row[column])) is not None
    }


def evaluate_reason_method(row: pd.Series, columns: list[str]) -> dict:
    """Compare predicted explanation features against known truth features."""

    # truth_features is already a cleaned (lowercased) list built in evaluate_model.
    truth_features = row.get("truth_features")
    true_features = (
        set(truth_features)
        if isinstance(truth_features, (list, set, tuple))
        else set()
    )

    predicted_features = get_predicted_features(row, columns)

    # Explanation quality is only assessed for known anomalies.
    is_anomaly = pd.notna(row["truth_anomaly"]) and row["truth_anomaly"] == -1

    if not is_anomaly or not true_features:
        return {
            "predicted_features": sorted(predicted_features),
            "matching_features": [],
            "reason_hit": pd.NA,
            "reason_precision": pd.NA,
            "reason_recall": pd.NA,
        }

    matches = true_features & predicted_features

    return {
        "predicted_features": sorted(predicted_features),
        "matching_features": sorted(matches),
        "reason_hit": bool(matches),
        "reason_precision": (
            len(matches) / len(predicted_features)
            if predicted_features else 0.0
        ),
        "reason_recall": len(matches) / len(true_features),
    }

def quality_metric_flag_success_all(
    scored: pd.DataFrame,
    truth_set: pd.DataFrame,
    assay: str,
    version: str,
    split: str,
):
    """
    Evaluate Isolation Forest predictions against a labelled truth set.

    Both DataFrames must contain:
        sample_name: sample identifier
        anomaly_label: -1 = anomaly, 1 = normal

    Evaluation is performed at sample level, using only samples
    present in both the scored dataset and the truth set.
    """

    scored = scored.copy()
    truth_set = truth_set.copy()

    required_columns = {"sample_name", "anomaly_label"}

    for name, df in [("scored", scored), ("truth_set", truth_set)]:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    # Keep only the sample ID and true label from the truth set.
    truth = truth_set[["sample_name", "truth_label"]].rename(
        columns={"truth_label": "true_label"}
    )

    # Aggregate model predictions to one prediction per sample.
    # A sample is an anomaly if any of its rows was flagged.
    predictions = (
        scored
        .agg(predicted_label=(
                "anomaly_label",
                lambda x: -1 if (x == -1).any() else 1
            )
        )
    )

    invalid_labels = set(truth["true_label"].dropna().unique()) - {-1, 1}
    if invalid_labels:
        raise ValueError(
            f"Truth labels must be -1 or 1. Found: {invalid_labels}"
        )

    # Inner join: evaluate only samples present in both datasets.
    evaluated = predictions.merge(
        truth,
        on="sample_name",
        how="inner",
        validate="one_to_one",
    )

    # Remove missing truth labels, if any.
    evaluated = evaluated.dropna(subset=["true_label"])

    if evaluated.empty:
        raise ValueError(
            "No samples with known truth labels overlap with scored data."
        )

    # Confusion matrix counts at sample level.
    true_anomaly = evaluated["true_label"] == -1
    predicted_anomaly = evaluated["predicted_label"] == -1

    tp = int((true_anomaly & predicted_anomaly).sum())
    fn = int((true_anomaly & ~predicted_anomaly).sum())
    fp = int((~true_anomaly & predicted_anomaly).sum())
    tn = int((~true_anomaly & ~predicted_anomaly).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    denominator = precision + recall
    f1_score = (
        2 * precision * recall / denominator
        if denominator else 0
    )

    total_scored = len(predictions)
    total_truth = truth["sample_name"].nunique()
    evaluated_samples = len(evaluated)
    model_outlier_samples = int(predicted_anomaly.sum())
    truth_anomaly_samples = int(true_anomaly.sum())

    flag_rate = (
        model_outlier_samples / evaluated_samples
        if evaluated_samples else 0
    )

    logging.info(
        f"\n{'=' * 50}\n"
        f"Evaluation: {assay.upper()} | {version} | {split}\n"
        f"{'=' * 50}\n"
        f"Samples scored by model:       {total_scored}\n"
        f"Samples in truth set:          {total_truth}\n"
        f"Samples evaluated (overlap):   {evaluated_samples}\n"
        f"True anomalies:                {truth_anomaly_samples}\n"
        f"Model outliers in overlap:     {model_outlier_samples}\n"
        f"True positives:                {tp}\n"
        f"False positives:               {fp}\n"
        f"False negatives:               {fn}\n"
        f"True negatives:                {tn}\n"
        f"Precision:                     {precision:.3f}\n"
        f"Recall:                        {recall:.3f}\n"
        f"F1-score:                      {f1_score:.3f}\n"
        f"Flag rate (overlap):           {flag_rate:.1%}\n"
        f"{'=' * 50}"
    )

    return {
        "assay": assay,
        "version": version,
        "split": split,
        "total_scored_samples": total_scored,
        "total_truth_samples": total_truth,
        "evaluated_samples": evaluated_samples,
        "true_anomaly_samples": truth_anomaly_samples,
        "model_outlier_samples": model_outlier_samples,
        "detected_known_failures": tp,
        "missed_known_failures": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "flag_rate": flag_rate,
    }


# ---------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------

def evaluate_model(
    scored_data: str | Path,
    truth_file: str | Path,
    assay: str,
    version: str,
    out_dir: str
) -> dict:
    """
    Evaluate predictions and explanations for one assay/version.

    The truth file must contain:
        - sample
        - anomaly
        - one or more truth_feature_* columns

    Returns a dictionary of DataFrames/Series needed by graphs_for_evaluation().

    Writes the detailed evaluation CSVs to:
        out_dir / assay / version
    """
    result_dir = Path(out_dir) / assay / version
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load model predictions and truth data.
    model = pd.read_csv(scored_data)
    truth = pd.read_csv(truth_file)

    # Identify all truth feature columns automatically.
    truth_feature_columns = [
        col for col in truth.columns
        if col.startswith("truth_feature_")
    ]

    # Check that all required columns are present.
    required_model = {
        "sample_name",
        "anomaly_label",
        *config.SHAP_COLUMNS,
        *config.ROBUST_Z_COLUMNS,
    }

    required_truth = {"sample", "truth_label"}

    missing_model = required_model - set(model.columns)
    missing_truth = required_truth - set(truth.columns)

    if missing_model:
        raise ValueError(
            f"Missing model columns: {sorted(missing_model)}"
        )

    if missing_truth:
        raise ValueError(
            f"Missing truth-set columns: {sorted(missing_truth)}"
        )

    if not truth_feature_columns:
        raise ValueError(
            "No truth_feature_* columns found in truth set."
        )

    model = model.copy()
    truth = truth.copy()

    # Clean sample identifiers.
    model["sample_name"] = clean_sample_name(model["sample_name"])
    truth["sample"] = clean_sample_name(truth["sample"])

    model = model.dropna(subset=["sample_name"])
    truth = truth.dropna(subset=["sample"])

    # Standardise anomaly labels.
    model["anomaly_label"] = model["anomaly_label"].map(normalise_label)
    truth["truth_label"] = truth["truth_label"].map(normalise_label)

    # Check that model output contains one record per sample.
    # If samples can recur across runs, use a shared unique identifier
    # (e.g. sample_name + worklist) instead.
    duplicate_names = model.loc[
        model["sample_name"].duplicated(keep=False), "sample_name"
    ].unique()

    if len(duplicate_names):
        raise ValueError(
            "Duplicate sample_name values in model output. Match using a shared "
            "unique identifier (for example sample_name + worklist) before "
            f"evaluating. Examples: {duplicate_names[:10].tolist()}"
        )

    # Check that each sample has one consistent truth anomaly label.
    truth_label_counts = truth.groupby("sample")["truth_label"].nunique()
    inconsistent = truth_label_counts[truth_label_counts > 1].index.tolist()

    if inconsistent:
        raise ValueError(
            "Truth-set anomaly labels conflict within these samples: "
            f"{inconsistent[:10]}"
        )

    # Group truth data to create one record per sample.
    # Combine all truth_feature_* columns into a single list of features.
    truth_by_sample = (
        truth.groupby("sample")
        .apply(
            lambda group: pd.Series({
                "truth_anomaly": group["truth_label"].iloc[0],
                "truth_features": sorted({
                    feature
                    for col in truth_feature_columns
                    for value in group[col]
                    if (feature := clean_feature(value)) is not None
                }),
            }),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"sample": "sample_name"})
    )

    # Match truth samples to model predictions.
    # All truth samples are retained, even if no model prediction exists.
    evaluation = truth_by_sample.merge(
        model,
        on="sample_name",
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    evaluation["sample_present"] = evaluation["_merge"].eq("both")
    evaluation.drop(columns="_merge", inplace=True)

    # Identify whether each model classification agrees with the truth.
    evaluation["classification_correct"] = (
        evaluation["sample_present"]
        & evaluation["truth_anomaly"].eq(evaluation["anomaly_label"])
    )

    evaluation.to_csv(
        result_dir / "sample_level_evaluation.csv",
        index=False,
    )

    # Save truth samples that were not present in the model output.
    missing_samples = evaluation.loc[
        ~evaluation["sample_present"],
        ["sample_name", "truth_anomaly"],
    ]

    missing_samples.to_csv(
        result_dir / "missing_truth_samples.csv",
        index=False,
    )

    # Save model samples that were not present in the truth set.
    extra_samples = model.loc[
        ~model["sample_name"].isin(truth_by_sample["sample_name"])
    ]

    extra_samples.to_csv(
        result_dir / "model_samples_not_in_truth.csv",
        index=False,
    )

    # Only matched samples can be used for classification evaluation.
    matched = evaluation.loc[evaluation["sample_present"]].copy()

    if matched.empty:
        raise ValueError(
            "No sample names matched between model output and truth set."
        )

    # Check for missing model predictions before calculating metrics.
    if matched["anomaly_label"].isna().any():
        raise ValueError(
            "Some matched model records have missing anomaly_label values."
        )

    y_true = matched["truth_anomaly"].astype(int)
    y_pred = matched["anomaly_label"].astype(int)

    # Identify false-positive and false-negative samples.
    # False positive: model flags a normal sample as anomalous.
    false_positives = matched.loc[
        (y_true == 1) & (y_pred == -1),
        "sample_name",
    ]

    # False negative: model predicts normal, but truth says anomalous.
    false_negatives = matched.loc[
        (y_true == -1) & (y_pred == 1),
        "sample_name",
    ]

    print("\nFALSE POSITIVE SAMPLES")
    print(f"Count: {len(false_positives)}")
    print(false_positives.to_string(index=False))

    print("\nFALSE NEGATIVE SAMPLES")
    print(f"Count: {len(false_negatives)}")
    print(false_negatives.to_string(index=False))

    # Calculate the confusion matrix.
    # Labels are ordered [normal, anomaly]:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    tn, fp, fn, tp = cm.ravel()

    # Calculate classification performance metrics.
    classification_metrics = {
        "truth_samples": len(truth_by_sample),
        "matched_samples": len(matched),
        "missing_truth_samples": len(missing_samples),
        "model_samples_not_in_truth": len(extra_samples),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "anomaly_precision": precision_score(
            y_true, y_pred, pos_label=-1, zero_division=0
        ),
        "anomaly_recall_sensitivity": recall_score(
            y_true, y_pred, pos_label=-1, zero_division=0
        ),
        "anomaly_f1": f1_score(
            y_true, y_pred, pos_label=-1, zero_division=0
        ),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "false_positive_rate": (
            fp / (fp + tn) if (fp + tn) else float("nan")
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    pd.DataFrame(
        classification_metrics.items(),
        columns=["metric", "value"],
    ).to_csv(
        result_dir / "classification_metrics.csv",
        index=False,
    )

    print("\nCLASSIFICATION PERFORMANCE")
    print(pd.Series(classification_metrics).to_string())

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[-1, 1],
            target_names=["Anomaly (-1)", "Normal (1)"],
            zero_division=0,
        )
    )

    # Evaluate explanation features for matched samples.
    reason_results = matched.copy()

    for method_name, columns in [
        ("shap", config.SHAP_COLUMNS),
        ("robust_z", config.ROBUST_Z_COLUMNS),
    ]:
        details = reason_results.apply(
            lambda row: pd.Series(
                evaluate_reason_method(row, columns)
            ),
            axis=1,
        )

        for column in details.columns:
            reason_results[f"{method_name}_{column}"] = details[column]

    reason_results.to_csv(
        result_dir / "reason_evaluation_by_sample.csv",
        index=False,
    )

    # Only evaluate explanation quality for known anomalies
    # that have at least one reference truth feature.
    valid = reason_results.loc[
        reason_results["truth_anomaly"].eq(-1)
        & reason_results["truth_features"].map(bool)
    ].copy()

    summaries = []

    for method_name in ["shap", "robust_z"]:
        hit_col = f"{method_name}_reason_hit"
        precision_col = f"{method_name}_reason_precision"
        recall_col = f"{method_name}_reason_recall"

        # Columns mix pd.NA with numbers (object dtype), so force numeric.
        hits = pd.to_numeric(valid[hit_col], errors="coerce").dropna()
        precisions = pd.to_numeric(valid[precision_col], errors="coerce")
        recalls = pd.to_numeric(valid[recall_col], errors="coerce")

        summaries.append({
            "method": method_name,
            "anomalous_samples_with_reference_reasons": len(valid),
            "samples_with_any_correct_reason": int(hits.sum()),
            "reason_hit_rate": hits.mean() if len(hits) else float("nan"),
            "mean_reason_precision": (
                precisions.mean() if precisions.notna().any() else float("nan")
            ),
            "mean_reason_recall": (
                recalls.mean() if recalls.notna().any() else float("nan")
            ),
        })

    reason_summary = pd.DataFrame(summaries)

    reason_summary.to_csv(
        result_dir / "reason_method_comparison.csv",
        index=False,
    )

    print("\nREASON METHOD COMPARISON")
    print(reason_summary.to_string(index=False))

    print(f"\nEvaluation files saved to: {result_dir.resolve()}")

    return {
        "evaluation": evaluation,
        "matched": matched,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_metrics": classification_metrics,
        "reason_results": reason_results,
        "reason_summary": reason_summary,
        "result_dir": result_dir,
    }


# ---------------------------------------------------------------------
# Graphing function
# ---------------------------------------------------------------------

def graphs_for_evaluation(results: dict, assay: str, version: str, out_dir) -> None:
    """Create and save evaluation plots using the output from evaluate_model()."""
    plot_dir = Path(out_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    cm = results["confusion_matrix"]
    metrics = results["classification_metrics"]
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    matched = results["matched"]
    reason_results = results["reason_results"]
    reason_summary = results["reason_summary"]

    # 1. Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Number of samples")
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Normal (1)", "Anomaly (-1)"],
        yticklabels=["Normal (1)", "Anomaly (-1)"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{assay} {version}: Confusion Matrix",
    )
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=13,
            )
    fig.tight_layout()
    fig.savefig(plot_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    # 2. Classification metrics
    metric_names = [
        "anomaly_precision", "anomaly_recall_sensitivity",
        "specificity", "anomaly_f1", "balanced_accuracy",
    ]
    metric_labels = ["Precision", "Recall", "Specificity", "F1-score", "Balanced accuracy"]
    values = [metrics[name] for name in metric_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(metric_labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"{assay} {version}: Classification Performance")
    ax.axhline(0.8, linestyle="--", linewidth=1, label="0.80 reference")
    ax.legend()
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.02,
                f"{value:.2f}", ha="center", va="bottom",
            )
    fig.tight_layout()
    fig.savefig(plot_dir / "classification_metrics.png", dpi=300)
    plt.close(fig)

    # 3. Explanation reason hit rate: SHAP vs robust z-score
    hit_rates = reason_summary.set_index("method")["reason_hit_rate"].reindex(["shap", "robust_z"])
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["SHAP", "Robust z-score (MAD)"], hit_rates.values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Reason hit rate")
    ax.set_title("How Often Does Each Method Find a Known Reason?")
    for bar, value in zip(bars, hit_rates.values):
        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.02,
                f"{value:.1%}", ha="center", va="bottom",
            )
    fig.tight_layout()
    fig.savefig(plot_dir / "reason_hit_rate_comparison.png", dpi=300)
    plt.close(fig)

    # 4. Explanation precision and recall
    reason_metrics = reason_summary.set_index("method").reindex(["shap", "robust_z"])
    x = np.arange(2)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    precision_bars = ax.bar(
        x - width / 2, reason_metrics["mean_reason_precision"], width,
        label="Reason precision",
    )
    recall_bars = ax.bar(
        x + width / 2, reason_metrics["mean_reason_recall"], width,
        label="Reason recall",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["SHAP", "Robust z-score (MAD)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Explanation Quality: SHAP vs Robust z-score")
    ax.legend()
    for bars in [precision_bars, recall_bars]:
        for bar in bars:
            value = bar.get_height()
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, value + 0.02,
                    f"{value:.2f}", ha="center", va="bottom", fontsize=9,
                )
    fig.tight_layout()
    fig.savefig(plot_dir / "reason_precision_recall.png", dpi=300)
    plt.close(fig)

    # 5. Anomaly-score distributions by true class
    if "anomaly_score" in matched.columns:
        score_data = matched.copy()
        score_data["anomaly_score"] = pd.to_numeric(score_data["anomaly_score"], errors="coerce")
        normal_scores = score_data.loc[score_data["truth_anomaly"].eq(1), "anomaly_score"].dropna()
        anomaly_scores = score_data.loc[score_data["truth_anomaly"].eq(-1), "anomaly_score"].dropna()

        groups, labels = [], []
        if len(normal_scores):
            groups.append(normal_scores)
            labels.append("True normal")
        if len(anomaly_scores):
            groups.append(anomaly_scores)
            labels.append("True anomaly")

        if groups:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(groups, tick_labels=labels, showmeans=True)
            ax.set_ylabel("Model anomaly score")
            ax.set_title("Anomaly Scores by True Class")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(plot_dir / "anomaly_score_distributions.png", dpi=300)
            plt.close(fig)
    else:
        print("No 'anomaly_score' column; skipping score distribution plot.")

    # 6. Most frequent known reasons vs predicted reasons
    reason_samples = reason_results.loc[
        reason_results["truth_anomaly"].eq(-1)
        & reason_results["truth_features"].map(bool)
    ].copy()

    truth_counts = {}
    for features in reason_samples["truth_features"]:
        for feature in features:
            truth_counts[feature] = truth_counts.get(feature, 0) + 1

    def count_predicted_reasons(data: pd.DataFrame, columns: list[str]) -> pd.Series:
        counts = {}
        for _, row in data.iterrows():
            for feature in get_predicted_features(row, columns):
                counts[feature] = counts.get(feature, 0) + 1
        return pd.Series(counts, dtype=int)

    shap_counts = count_predicted_reasons(reason_samples, config.SHAP_COLUMNS)
    robust_counts = count_predicted_reasons(reason_samples, config.ROBUST_Z_COLUMNS)
    all_counts = pd.DataFrame({
        "Truth set": pd.Series(truth_counts, dtype=int),
        "SHAP": shap_counts,
        "Robust z-score": robust_counts,
    }).fillna(0)

    if not all_counts.empty:
        top_features = all_counts.sum(axis=1).sort_values(ascending=False).head(15).index
        plot_data = all_counts.loc[top_features].sort_values("Truth set", ascending=True)
        ax = plot_data.plot.barh(figsize=(11, 7))
        ax.set_xlabel("Number of anomalous samples")
        ax.set_ylabel("Feature")
        ax.set_title("Reference Reasons vs Model-Identified Reasons")
        ax.legend(title="Source")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(plot_dir / "reason_feature_frequencies.png", dpi=300)
        plt.close(fig)
    else:
        print("No reference reasons available for the feature-frequency plot.")

    print(f"All plots saved to: {plot_dir.resolve()}")


# ---------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Change assay/version to match the files being evaluated.
    parser = argparse.ArgumentParser(
            description='Settings for training the Isolation Model'
        )
    
    parser.add_argument(
        "--assay",
        choices=["st", "haem"],
        required=True,
        help='Assay to train model: "st" or "haem"'
    )

    parser.add_argument(
        "--version",
        choices=["v2", "v3", "v2_v3"],
        help="Capture version to train the model: 'v2', 'v3', or 'v2_v3'"
    )

    args = parser.parse_args()

    OUTPUT_DIR = os.path.join("data/evaluate")


    results = evaluate_model(
        'data/trained/st/v2_v3/st_v2_v3_c0.15, n100_ms256_mf1.0_test_scored.csv',
        truth_file=os.path.join(config.TRUTH_SET_DIR, f'truth_set_{args.assay}_500.csv'),
        assay=args.assay,
        version=args.version,
        out_dir=OUTPUT_DIR,
    )
    graphs_for_evaluation(results, assay=args.assay, version=args.assay, out_dir="outputs/graphs/evaluate/")
