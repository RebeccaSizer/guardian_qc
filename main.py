import argparse
import os
import pandas as pd
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
from pipeline.train import run_model_train, load_model
from pipeline.evaluate import quality_metric_flag_success, get_truth_set
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

    results = quality_metric_flag_success(
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
        "--split",
        choices=["train", "test"],
        required=False,
        help='Split used for evaluation: Train or Test"',
    )

    args = parser.parse_args()

    # Validate contamination
    if args.assay == 'haem':
        contamination = 0.08
    
    elif args.assay == 'ST':
        contamination = 0.15

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
            contamination,
            model_output_dir,
        )

    elif args.step == "evaluate":
        if args.split == False:
            split = 'test'
        else:
            split = args.split

        scored_file = os.path.join('models/trained', args.assay, f"{args.assay}_{args.version}_test_scored.csv")
        scored_df = pd.read_csv(scored_file)
        truth_set = get_truth_set('data/raw/', args.assay)
        quality_metric_flag_success(scored_df, truth_set, args.assay, args.version, split)


    elif args.step == "all":

        run_ingest()

        results = run_train_and_evaluate(
            args.assay,
            args.version,
            contamination,
        )

        logging.info(
            f"Pipeline complete: "
            f"{args.assay} {args.version} "
            f"contamination={contamination}"
        )

        logging.info(f"Evaluation results: {results}")