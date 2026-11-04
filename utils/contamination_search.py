from pipeline.preprocess import run_preprocessing
from pipeline.train_composite import run_model_train
from pipeline.evaluate import clean_sample_name, normalise_label, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from utils.logger import logging
import config
import os
import pandas as pd
import argparse

# Graph functions for contamination metric tunring. 
import matplotlib.pyplot as plt
import os


def quality_metric_flag_success_all(
    scored: pd.DataFrame,
    truth_set: pd.DataFrame,
    assay: str,
    version: str,
    split: str,
):
    """
    Evaluate Isolation Forest predictions against a labelled truth set.

    Uses the same sample matching, label normalisation and
    classification metrics as evaluate_model().

    Both datasets must contain:
        sample_name
        anomaly_label

    Labels:
        -1 = anomaly
         1 = normal
    """

    scored = scored.copy()
    truth_set = truth_set.copy()

    required_columns = {"sample_name", "anomaly_label"}

    # ---------------------------------------------------------
    # 1. Check required columns
    # ---------------------------------------------------------

    for name, df in [
        ("scored", scored),
        ("truth_set", truth_set)
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    # ---------------------------------------------------------
    # 2. Clean sample identifiers
    # ---------------------------------------------------------

    scored["sample_name"] = clean_sample_name(
        scored["sample_name"]
    )

    truth_set["sample_name"] = clean_sample_name(
        truth_set["sample_name"]
    )

    scored = scored.dropna(subset=["sample_name"])
    truth_set = truth_set.dropna(subset=["sample_name"])

    # ---------------------------------------------------------
    # 3. Normalise anomaly labels
    # ---------------------------------------------------------

    scored["anomaly_label"] = scored["anomaly_label"].map(
        normalise_label
    )

    truth_set["anomaly_label"] = truth_set["anomaly_label"].map(
        normalise_label
    )

    # Remove missing truth labels
    truth_set = truth_set.dropna(subset=["anomaly_label"])

    # Check valid labels
    for name, df in [
        ("scored", scored),
        ("truth_set", truth_set)
    ]:

        invalid_labels = (
            set(df["anomaly_label"].dropna().unique()) - {-1, 1}
        )

        if invalid_labels:
            raise ValueError(
                f"{name} contains invalid labels: {invalid_labels}"
            )

    # ---------------------------------------------------------
    # 4. Check duplicate model sample names
    # ---------------------------------------------------------

    duplicate_names = scored.loc[
        scored["sample_name"].duplicated(keep=False),
        "sample_name"
    ].unique()

    if len(duplicate_names):

        raise ValueError(
            "Duplicate sample names in model output. "
            "Use a shared unique identifier before evaluation. "
            f"Examples: {duplicate_names[:10].tolist()}"
        )

    # ---------------------------------------------------------
    # 5. Check consistency of truth labels
    # ---------------------------------------------------------

    truth_label_counts = (
        truth_set.groupby("sample_name")["anomaly_label"].nunique()
    )

    inconsistent = truth_label_counts[
        truth_label_counts > 1
    ].index.tolist()

    if inconsistent:
        raise ValueError(
            "Conflicting truth labels for samples: "
            f"{inconsistent[:10]}"
        )

    # ---------------------------------------------------------
    # 6. Aggregate truth set to sample level
    # ---------------------------------------------------------

    truth = (
        truth_set.groupby("sample_name", as_index=False)
        ["anomaly_label"]
        .first()
        .rename(columns={"anomaly_label": "true_label"})
    )

    # ---------------------------------------------------------
    # 7. Match truth samples to model predictions
    # ---------------------------------------------------------

    evaluation = truth.merge(
        scored[["sample_name", "anomaly_label"]],
        on="sample_name",
        how="left",
        indicator=True,
        validate="one_to_one"
    )

    evaluation["sample_present"] = (
        evaluation["_merge"] == "both"
    )

    evaluation.drop(columns="_merge", inplace=True)

    # Retain matched samples only
    matched = evaluation.loc[
        evaluation["sample_present"]
    ].copy()

    if matched.empty:
        raise ValueError(
            "No samples with known truth labels overlap with scored data."
        )

    # Check missing predictions
    if matched["anomaly_label"].isna().any():
        raise ValueError(
            "Some matched samples have missing model predictions."
        )

    # ---------------------------------------------------------
    # 8. Calculate confusion matrix
    # ---------------------------------------------------------

    y_true = matched["true_label"].astype(int)
    y_pred = matched["anomaly_label"].astype(int)

    # Order: normal (1), anomaly (-1)
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[1, -1]
    )

    tn, fp, fn, tp = cm.ravel()

    # ---------------------------------------------------------
    # 9. Calculate classification metrics
    # ---------------------------------------------------------

    precision = precision_score(
        y_true, y_pred, pos_label=-1, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, pos_label=-1, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, pos_label=-1, zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_true, y_pred
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) else float("nan")
    )

    false_positive_rate = (
        fp / (fp + tn) if (fp + tn) else float("nan")
    )

    # ---------------------------------------------------------
    # 10. Sample counts
    # ---------------------------------------------------------

    total_scored = scored["sample_name"].nunique()

    total_truth = truth["sample_name"].nunique()

    evaluated_samples = len(matched)

    missing_samples = int(
        (~evaluation["sample_present"]).sum()
    )

    model_outlier_samples = int(
        (y_pred == -1).sum()
    )

    truth_anomaly_samples = int(
        (y_true == -1).sum()
    )

    flag_rate = (
        model_outlier_samples / evaluated_samples
        if evaluated_samples else 0
    )

    # ---------------------------------------------------------
    # 11. Log results
    # ---------------------------------------------------------

    logging.info(
        f"\n{'=' * 50}\n"
        f"Evaluation: {assay.upper()} | {version} | {split}\n"
        f"{'=' * 50}\n"
        f"Samples scored:                {total_scored}\n"
        f"Samples in truth set:          {total_truth}\n"
        f"Matched samples:               {evaluated_samples}\n"
        f"Missing truth samples:         {missing_samples}\n"
        f"True anomalies:                {truth_anomaly_samples}\n"
        f"True positives:                {tp}\n"
        f"False positives:               {fp}\n"
        f"False negatives:               {fn}\n"
        f"True negatives:                {tn}\n"
        f"Precision:                     {precision:.3f}\n"
        f"Recall:                        {recall:.3f}\n"
        f"F1-score:                      {f1:.3f}\n"
        f"Specificity:                   {specificity:.3f}\n"
        f"False positive rate:            {false_positive_rate:.3f}\n"
        f"Flag rate:                     {flag_rate:.1%}\n"
        f"{'=' * 50}"
    )

    # ---------------------------------------------------------
    # 12. Return results
    # ---------------------------------------------------------

    return {
        "assay": assay,
        "version": version,
        "split": split,

        "total_scored_samples": total_scored,
        "total_truth_samples": total_truth,
        "evaluated_samples": evaluated_samples,
        "missing_truth_samples": missing_samples,

        "true_anomaly_samples": truth_anomaly_samples,
        "model_outlier_samples": model_outlier_samples,

        "detected_known_failures": int(tp),
        "missed_known_failures": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),

        "precision": precision,
        "recall": recall,
        "f1_score": f1,

        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,

        "flag_rate": flag_rate,
    }


def plot_parameter_results(metric, results_df, assay, version, split):
    """Plot precision, recall and flag rate across parameter values."""

    os.makedirs(f"outputs/graphs/parameter_tuning/{assay}/{metric}/", exist_ok=True)

    plot_df = results_df.copy()

    # Plot precision, recall and F1-score
    metrics = ["precision", "recall", "f1_score"]

    ax = plot_df.set_index(
        plot_df.index.astype(float)
    )[metrics].plot(
        kind="line",
        marker="o",
        figsize=(9, 6),
        linewidth=2
    )

    ax.set_title(
        f"Precision, Recall and F1-score Across {metric} Values: "
        f"{assay} {version} ({split})"
    )
    ax.set_xlabel(f"{metric.title()}")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    x_values = results_df[metric].values

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(
        x_values.astype(str),
        rotation=45,
        ha="right"
    )
    ax.legend(title="Metric")
    ax.grid(axis="both", linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_path = os.path.join(
        f"outputs/graphs/parameter_tuning/{assay}/{metric}/",
        f"{assay}_{version}_{metric}.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Parameter tuning plot saved to {output_path}")


def parameter_tuning(metric, summary_qc_metrics: str, assay: str, version: str, out_dir):

    # run the preprocessing step for the model 
    X_train, X_test, train_meta, test_meta = run_preprocessing(summary_qc_metrics, assay, version, out_dir)
    
    # set default values for contamination, n_estimators, max_samples, and max_features
    contamination = 0.15 if assay == "st" else 0.08
    n_estimators = config.N_ESTIMATORS
    max_samples = config.MAX_SAMPLES
    max_features = config.MAX_FEATURES

    # Baseline parameters
    baseline = {
        "contamination": 0.15 if assay == "st" else 0.08,
        "n_estimators": config.N_ESTIMATORS,
        "max_samples": config.MAX_SAMPLES,
        "max_features": config.MAX_FEATURES
    }
    
    # Candidate values
    parameter_values = {
        "contamination": [0.01, 0.03, 0.05, 0.08,
                          0.10, 0.15, 0.20, 0.25],

        "n_estimators": [50, 100, 150, 200, 250, 300],

        "max_samples": [0.5, 0.75, 1.0],

        "max_features": [0.5, 0.75, 1.0]
    }

    if metric not in parameter_values:
        raise ValueError(f"Invalid metric: {metric}. Must be one of {list(parameter_values.keys())}")

    all_results = []

    # load the truth set for the assay
    truth_df = pd.read_csv(f'data/raw/truth_set_{assay}_500.csv', sep=',')

    for value in parameter_values[metric]:

        # Start with baseline parameters
        params = baseline.copy()

        # Change only the parameter being investigated
        params[metric] = value

        logging.info(f"Testing {metric} = {value}")

        explained_model_train, explained_model_test = run_model_train(X_train, X_test, train_meta, test_meta, assay, version, params['contamination'], params['n_estimators'], params['max_samples'], params['max_features'], os.path.join('models/trained/composite', assay))

        scored_explained = explained_model_test.copy()

        returned_analysis = quality_metric_flag_success_all(
            scored=scored_explained,
            truth_set=truth_df,
            assay=assay,
            version=version,
            split='test',
        )

        returned_analysis[metric] = value

        all_results.append(returned_analysis)

    results_df = pd.DataFrame(all_results)
    
    results_df.to_csv(os.path.join("outputs/evaluation/", f"{assay}_{version}.csv"))

    plot_parameter_results(
        metric,
        results_df,
        assay,
        version,
        'test'
    )



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description = "Evaluate a model against known flagged fails"
    )

    parser.add_argument(
        "--assay",
        choices=["st", "haem"],
        help="Assay type: st or haem"
    )

    parser.add_argument(
        "--version",
        choices=["v2", "v3", "v2_v3"],
        help="Cature version of pipeline: v2, v3 or v2_v3"
    )

    parser.add_argument(
            "--split",
            choices=["test", "train" ],
            required=False,
            help="Cature version of pipeline: v2, v3 or v2_v3"
        )

    parser.add_argument(
        "--metric",
        choices=["contamination", "n_estimators", "max_samples", "max_features"],
        required=True,
        help="Metric to tune: contamination, n_estimators, max_samples, or max_features"
    )

    args= parser.parse_args()

    parameter_tuning(args.metric, config.SUMMARY_QC_METRICS, assay=args.assay, version=args.version, out_dir=None)