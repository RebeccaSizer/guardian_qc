from pipeline.preprocess import run_preprocessing
from pipeline.train_composite import run_model_train
from pipeline.evaluate import get_truth_set, quality_metric_flag_success_all
from utils.logger import logging
import config
import os
import pandas as pd
import argparse
# Graph functions for contamination metric tunring. 
import matplotlib.pyplot as plt
import os



def plot_contamination_results(results_df, assay, version, split):
    """Plot precision, recall and flag rate across contamination values."""

    os.makedirs("outputs/evaluation/", exist_ok=True)

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
    ax.set_xlabel("Contamination")
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
        "outputs/evaluation/",
        f"{assay}_{version}_{split}_contamination.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Contamination plot saved to {output_path}")


def contamination_search(summary_qc_metrics: str, assay: str, version: str, out_dir, split):

    X_train, X_test, train_meta, test_meta = run_preprocessing(summary_qc_metrics, assay, version, out_dir)
    
    contamination = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]
    all_results = {}
    truth_df = pd.read_csv('data/raw/ST_test.csv')

    for cont_value in contamination:
        explained_model_train, explained_model_test = run_model_train(X_train, X_test, train_meta, test_meta, assay, version, split, cont_value, os.path.join('models/trained/composite', assay))

        truth_set = get_truth_set('data/raw/', assay)

        if split == 'train':
            scored_explained = explained_model_train.copy()
        elif split == 'test':
            scored_explained = explained_model_test.copy()

        returned_analysis = quality_metric_flag_success_all(
            scored=scored_explained,
            truth_set=truth_df,
            assay=assay,
            version=version,
            split=split,
        )

        all_results[cont_value] = returned_analysis
    
        results_df = pd.DataFrame.from_dict(
        all_results,
        orient="index"
        )
    
        results_df.to_csv(os.path.join("outputs/evaluation/", f"{assay}_{version}_{split}.csv"))

    plot_contamination_results(
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

    contamination_search(config.SUMMARY_QC_METRICS, assay=args.assay, version=args.version, out_dir=None, split=args.split)