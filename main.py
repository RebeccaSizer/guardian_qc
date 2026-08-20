import argparse
import os
import config

from pipeline.ingest import (
    get_run_file_paths, get_qc_summary_file_path,
    get_run_sample_file_paths, filter_df, filter_run_qc, sample_level_qc
)
from pipeline.preprocess import run_preprocessing
from utils.logger import logging


def run_ingest():
    logging.info("Starting ingestion pipeline...")

    df_sample = get_qc_summary_file_path()
    df_run    = get_run_file_paths()

    df = get_run_sample_file_paths(df_run, df_sample)
    df.to_csv(config.QC_FILE_PATHS, sep="\t", index=False)
    logging.info(f"File paths written to {config.QC_FILE_PATHS}")

    df = filter_df(df)
    df = filter_run_qc(df)
    df.to_csv(config.FILTERED_RUN_QC_DATA, sep="\t", index=False)
    logging.info(f"Filtered data written to {config.FILTERED_RUN_QC_DATA}")

    summary_qc_metrics_df = sample_level_qc(df)
    summary_qc_metrics_df.to_csv(config.SUMMARY_QC_METRICS, sep="\t", index=False)
    logging.info(f"Summary QC metrics written to {config.SUMMARY_QC_METRICS} ({len(summary_qc_metrics_df)} rows)")


def run_preprocess(assay: str):
    logging.info(f"Starting preprocessing pipeline for assay: {assay}")

    output_dir = os.path.join(config.PREPROCESSING_OUTDIR, assay)

    X_train, X_test, train_meta, test_meta = run_preprocessing(
        file_path=config.SUMMARY_QC_METRICS,
        out_dir=output_dir,
        assay_type=assay,
    )

    logging.info(f"{assay} preprocessing complete.")
    logging.info(f"Training samples: {len(X_train)}")
    logging.info(f"Test samples:     {len(X_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Guardian-QC pipeline runner."
    )
    parser.add_argument(
        "--step",
        choices=["ingest", "preprocess", "all"],
        required=True,
        help=(
            "Pipeline step to run. "
            "'ingest' collects and filters raw QC data. "
            "'preprocess' runs feature engineering and scaling on existing summary metrics. "
            "'all' runs both in sequence."
        )
    )
    parser.add_argument(
        "--assay",
        choices=["ST", "haem"],
        required=False,
        help="Required when --step is 'preprocess' or 'all'."
    )

    args = parser.parse_args()

    # Validate: assay is required if preprocessing
    if args.step in ("preprocess", "all") and not args.assay:
        parser.error("--assay is required when --step is 'preprocess' or 'all'")

    if args.step == "ingest":
        run_ingest()

    elif args.step == "preprocess":
        run_preprocess(args.assay)

    elif args.step == "all":
        run_ingest()
        run_preprocess(args.assay)