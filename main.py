import argparse
import os

import config

from pipeline.ingest import (
    get_run_file_paths,
    get_qc_summary_file_path,
    get_run_sample_file_paths,
    filter_df,
    filter_run_qc,
    sample_level_qc,
)
from pipeline.preprocess import run_preprocessing
from pipeline.train import run_model_train
from pipeline.evaluate import quality_metirc_flag_success
from utils.logger import logging


def run_ingest():
    """Run the data ingestion and QC filtering steps."""

    logging.info("Starting ingestion pipeline...")

    df_sample = get_qc_summary_file_path()
    df_run = get_run_file_paths()

    df = get_run_sample_file_paths(df_run, df_sample)

    df.to_csv(
        config.QC_FILE_PATHS,
        sep="\t",
        index=False,
    )
    logging.info(f"File paths written to {config.QC_FILE_PATHS}")

    df = filter_df(df)
    df = filter_run_qc(df)

    df.to_csv(
        config.FILTERED_RUN_QC_DATA,
        sep="\t",
        index=False,
    )
    logging.info(
        f"Filtered data written to {config.FILTERED_RUN_QC_DATA}"
    )

    summary_qc_metrics_df = sample_level_qc(df)

    summary_qc_metrics_df.to_csv(
        config.SUMMARY_QC_METRICS,
        sep="\t",
        index=False,
    )

    logging.info(
        f"Summary QC metrics written to "
        f"{config.SUMMARY_QC_METRICS} "
        f"({len(summary_qc_metrics_df)} rows)"
    )


def run_preprocess(assay: str, version: str):
    """Run preprocessing for a given assay and capture version."""

    logging.info(
        f"Starting preprocessing for assay: {assay}, version: {version}"
    )

    output_dir = os.path.join(
        config.PREPROCESSING_OUTDIR,
        assay,
        version,
    )

    X_train, X_test, train_meta, test_meta = run_preprocessing(
        file_path=config.SUMMARY_QC_METRICS,
        out_dir=output_dir,
        assay_type=assay,
        version=version,
    )

    logging.info(f"{assay} {version} preprocessing complete.")
    logging.info(f"Training samples: {len(X_train)}")
    logging.info(f"Test samples: {len(X_test)}")

    return X_train, X_test, train_meta, test_meta


def get_truth_set(assay: str):
    """Load the known failed sample names for the selected assay."""

    if assay == "haem":
        truth_file = "data/raw/qc_fail_haem.tsv"

    elif assay == "st":
        truth_file = "data/raw/qc_fail_ST.tsv"

    else:
        raise ValueError(f"Unknown assay: {assay}")

    truth_set = []

    with open(truth_file, "r") as qc_fail:
        for line in qc_fail:
            line = line.strip()

            if line:
                truth_set.append(line)

    logging.info(
        f"Loaded {len(truth_set)} known failed samples "
        f"from {truth_file}"
    )

    return truth_set


def run_train_and_evaluate(
    assay: str,
    version: str,
    contamination: float,
):
    """Run preprocessing, model training and evaluation."""

    X_train, X_test, train_meta, test_meta = run_preprocess(
        assay,
        version,
    )

    model_output_dir = os.path.join(
        "models",
        "trained",
        assay,
        version,
    )

    explained_model_train, explained_model_test = run_model_train(
        X_train,
        X_test,
        train_meta,
        test_meta,
        assay,
        version,
        contamination,
        model_output_dir,
    )

    truth_set = get_truth_set(assay)

    results = quality_metirc_flag_success(
        explained_model_test,
        truth_set,
        assay,
        version,
        "test",
    )

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the Guardian-QC Isolation Forest pipeline."
    )

    parser.add_argument(
        "--step",
        choices=[
            "ingest",
            "preprocess",
            "train",
            "evaluate",
            "all",
        ],
        required=True,
        help=(
            "Pipeline step to run: "
            "ingest, preprocess, train, evaluate or all"
        ),
    )

    parser.add_argument(
        "--assay",
        choices=["st", "haem"],
        required=True,
        help='Assay to analyse: "st" or "haem"',
    )

    parser.add_argument(
        "--version",
        choices=["v2", "v3", "v2_v3"],
        required=True,
        help="Capture version: v2, v3 or v2_v3",
    )

    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help=(
            "Expected proportion of outliers for the Isolation Forest. "
            "Default: 0.05"
        ),
    )

    args = parser.parse_args()

    # Validate contamination
    if not 0 < args.contamination <= 0.5:
        parser.error(
            "--contamination must be greater than 0 and no greater than 0.5"
        )

    if args.step == "ingest":

        run_ingest()

    elif args.step == "preprocess":

        run_preprocess(
            args.assay,
            args.version,
        )

    elif args.step == "train":

        X_train, X_test, train_meta, test_meta = run_preprocess(
            args.assay,
            args.version,
        )

        model_output_dir = os.path.join(
            "models",
            "trained",
            args.assay,
            args.version,
        )

        run_model_train(
            X_train,
            X_test,
            train_meta,
            test_meta,
            args.assay,
            args.version,
            args.contamination,
            model_output_dir,
        )

    elif args.step == "evaluate":

        # You need to load your previously trained model/results here.
        # This depends on how run_model_train saves its output.
        raise NotImplementedError(
            "Add model loading here for the evaluate-only step."
        )

    elif args.step == "all":

        run_ingest()

        results = run_train_and_evaluate(
            args.assay,
            args.version,
            args.contamination,
        )

        logging.info(
            f"Pipeline complete: "
            f"{args.assay} {args.version} "
            f"contamination={args.contamination}"
        )

        logging.info(f"Evaluation results: {results}")