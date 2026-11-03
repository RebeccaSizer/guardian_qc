from pipeline.preprocess import run_preprocessing
from pipeline.train_composite import run_model_train
from pipeline.evaluate import quality_metric_flag_success_all
from utils.logger import logging
import config
import os
import pandas as pd
import argparse

# Graph functions for contamination metric tunring. 
import matplotlib.pyplot as plt
import os

def plot_parameter_results(metric, results_df, assay, version, split):
    """Plot precision, recall and flag rate across parameter values."""

    os.makedirs(f"outputs/graph/parameter_tuning/{assay}/{metric}/", exist_ok=True)

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
        f"Precision, Recall and F1-score Across Contamination Values: "
        f"{assay} {version} ({split})"
    )
    ax.set_xlabel(f"{metric.title()}")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(plot_df.index)
    ax.set_xticklabels(
        [f"{float(value):.0%}" for value in plot_df.index],
        rotation=45,
        ha="right"
    )
    ax.legend(title="Metric")
    ax.grid(axis="both", linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_path = os.path.join(
        f"outputs/graphs/parameter_tuning/{assay}/{metric}/",
        f"{assay}_{version}_{split}_{metric}.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Parameter tuning plot saved to {output_path}")


def parameter_tuning(metric, summary_qc_metrics: str, assay: str, version: str, out_dir, split):

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
            split=split,
        )

        all_results.append(returned_analysis)

        results_df = pd.DataFrame(all_results)
    
        results_df.to_csv(os.path.join("outputs/evaluation/", f"{assay}_{version}_{split}.csv"))

    plot_parameter_results(
        metric,
        results_df,
        assay,
        version,
        split
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

    args= parser.parse_args()

    parameter_tuning("contamination", config.SUMMARY_QC_METRICS, assay=args.assay, version=args.version, out_dir=None, split=args.split)