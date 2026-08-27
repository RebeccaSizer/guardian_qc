""" 
evaluate the IsolationForest model I have built """
import pandas as pd
import config
from pipeline.train import run_model_train
from pipeline.preprocess import run_preprocessing
from sklearn.metrics import (confusion_matrix,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score
                             )
import argparse
import os
from utils.logger import logging 

def quality_metric_flag_success(scored: pd.DataFrame, truth_set: list, assay:str, version:str, split:str):
    """
    Evaluate Isolation Forest performance against a subset of known failed samples.

    Because truth_set contains only a subset of all failed samples, samples not in
    truth_set are treated as unknown rather than as confirmed passes.

    Parameters:
        scored:
            DataFrame containing metadata, QC metrics and model predictions.
        truth_set:
            List of sample_names previously flagged as failed.
        assay:
            Assay the dataframe relates to, e.g. 'st' or 'haem'.
        version:
            Capture version, e.g. 'v2', 'v3' or 'v2_v3'.
        split:
            Dataset split, e.g. 'train' or 'test'.

    Returns:
        Dictionary containing evaluation metrics.
    """

    logging.info("Starting evaluation of the model")

    # Identify samples in the known failure subset
    scored["known_failed"] = scored["sample_name"].isin(truth_set)

    # Identify samples flagged as outliers by the model
    scored["model_outlier"] = scored["anomaly_label"] == -1

    # Number of known failures
    known_failed_rows = scored["known_failed"].sum()
    known_failed_samples = (
        scored.loc[scored["known_failed"], "sample_name"].nunique()
    )

    # Model outliers
    model_outlier_rows = scored["model_outlier"].sum()
    model_outlier_samples = (
        scored.loc[scored["model_outlier"], "sample_name"].nunique()
    )

    # Known failures detected by the model
    detected_known_failures = (
        scored["known_failed"] &
        scored["model_outlier"]
    ).sum()

    # Known failures missed by the model
    missed_known_failures = (
        scored["known_failed"] &
        ~scored["model_outlier"]
    ).sum()

    # Model outliers which are not in the known failure subset.
    # These cannot be called false positives because they may be genuine
    # failures that have not been included in truth_set.
    unconfirmed_outliers = (
        ~scored["known_failed"] &
        scored["model_outlier"]
    ).sum()

    # Recall/sensitivity against the known failure subset
    recall = (
        detected_known_failures / known_failed_rows
        if known_failed_rows > 0
        else 0
    )

    logging.info(
        f"Evaluation: {assay} {version} {split}\n"
        f"Known failed samples: {known_failed_samples}\n"
        f"Known failed rows: {known_failed_rows}\n"
        f"Model predicted outlier rows: {model_outlier_rows}\n"
        f"Model predicted outlier samples: {model_outlier_samples}\n"
        f"Known failures detected by model: {detected_known_failures}\n"
        f"Known failures missed by model: {missed_known_failures}\n"
        f"Unconfirmed model outliers: {unconfirmed_outliers}\n"
        f"Recall against known failures: {recall:.3f}"
    )

    return {
        "assay": assay,
        "version": version,
        "split": split,
        "known_failed_samples": known_failed_samples,
        "known_failed_rows": known_failed_rows,
        "model_outlier_rows": model_outlier_rows,
        "model_outlier_samples": model_outlier_samples,
        "detected_known_failures": detected_known_failures,
        "missed_known_failures": missed_known_failures,
        "unconfirmed_outliers": unconfirmed_outliers,
        "recall": recall,
    }
   


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

    args= parser.parse_args()

    X_train, X_test, train_meta, test_meta = run_preprocessing(config.SUMMARY_QC_METRICS, args.assay, out_dir=None, version=args.version)
    
    explained_model_train, explained_model_test = run_model_train(X_train, X_test, train_meta, test_meta, args.assay, args.version, 0.05, os.path.join('models/trained', args.assay ))


    truth_set_haem = []
    truth_set_st = []
    with open("data/raw/qc_fail_haem.tsv", "r") as qc_fail:
        for line in qc_fail:
            line = line.strip()
            truth_set_haem.append(line)

    with open("data/raw/qc_fail_ST.tsv", "r") as qc_fail:
        for line in qc_fail:
            line = line.strip()
            truth_set_st.append(line)

    quality_metric_flag_success(explained_model_train, truth_set_haem, args.assay, args.version, "train")