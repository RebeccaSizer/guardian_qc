""" 
evaluate the IsolationForest model I have built """
import pandas as pd
import config
from pipeline.train import run_model_train
from pipeline.preprocess import run_preprocessing
import argparse
import os
from utils.logger import logging 
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def quality_metric_flag_success(scored: pd.DataFrame, truth_set: list, assay: str, version: str, split: str):
    """
    Evaluate Isolation Forest performance against a subset of known failed samples.
    Returns metrics dict and prints a summary. Saves a bar chart to outputs/.
    """

    scored = scored.copy()
    scored["known_failed"] = scored["sample_name"].isin(truth_set)
    scored["model_outlier"] = scored["anomaly_label"] == -1

    known_failed_rows         = scored["known_failed"].sum()
    known_failed_samples      = scored.loc[scored["known_failed"], "sample_name"].nunique()
    model_outlier_rows        = scored["model_outlier"].sum()
    model_outlier_samples     = scored.loc[scored["model_outlier"], "sample_name"].nunique()
    detected_known_failures   = (scored["known_failed"] & scored["model_outlier"]).sum()
    missed_known_failures     = (scored["known_failed"] & ~scored["model_outlier"]).sum()
    unconfirmed_outliers      = (~scored["known_failed"] & scored["model_outlier"]).sum()
    unconfirmed_passes          = (~scored["known_failed"] & ~scored["model_outlier"]).sum()

    recall = detected_known_failures / known_failed_rows if known_failed_rows > 0 else 0

    # Flag rate — what fraction of all samples did the model flag?
    total_rows = len(scored)
    flag_rate  = model_outlier_rows / total_rows if total_rows > 0 else 0

    # Of known failures, what fraction did we catch?
    # (same as recall — shown separately for clarity in logs)
    detection_rate = recall

    logging.info(
        f"\n{'='*50}\n"
        f"Evaluation: {assay.upper()} | {version} | {split}\n"
        f"{'='*50}\n"
        f"  Total rows:                  {total_rows}\n"
        f"  Known failed rows:           {known_failed_rows} ({known_failed_samples} samples)\n"
        f"\n  -- Model output --\n"
        f"  Flagged as outlier:          {model_outlier_rows} rows ({model_outlier_samples} samples)\n"
        f"  Flag rate:                   {flag_rate:.1%}\n"
        f"\n  -- Against truth set --\n"
        f"  Detected (TP):               {detected_known_failures}\n"
        f"  Missed   (FN):               {missed_known_failures}\n"
        f"  Unconfirmed outliers*:       {unconfirmed_outliers}\n"
        f"  Unconfirmed passes:            {unconfirmed_passes}\n"
        f"\n  Recall (sensitivity):        {recall:.3f}\n"
        f"\n  * Cannot be called FP — truth set is partial.\n"
        f"{'='*50}"
    )

    return {
        "assay":                    assay,
        "version":                  version,
        "split":                    split,
        "total_rows":               total_rows,
        "known_failed_samples":     known_failed_samples,
        "known_failed_rows":        known_failed_rows,
        "model_outlier_rows":       model_outlier_rows,
        "model_outlier_samples":    model_outlier_samples,
        "detected_known_failures":  detected_known_failures,
        "missed_known_failures":    missed_known_failures,
        "unconfirmed_outliers":     unconfirmed_outliers,
        "unconfirmed_passes":       unconfirmed_passes,
        "recall":                   recall,
        "flag_rate":                flag_rate,
    }

def get_truth_set(out_dir, assay):

    file_path = os.path.join(out_dir, f"qc_fail_{assay}.tsv")

    truth_set = []

    with open(file_path, "r") as qc_fail:
            for line in qc_fail:
                line = line.strip()
                truth_set.append(line)

    return truth_set

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description = "Evaluate a model against known flagged fails"
    )

    parser.add_argument(
        "--assay",
        choices=["ST", "haem"],
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

    X_train, X_test, train_meta, test_meta = run_preprocessing(config.SUMMARY_QC_METRICS, args.assay, out_dir=None, version=args.version)

    contamination = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]
    all_results = {}
    for i in contamination:
        explained_model_train, explained_model_test = run_model_train(X_train, X_test, train_meta, test_meta, args.assay, args.version, i, os.path.join('models/trained', args.assay ))

        truth_set = get_truth_set('data/raw/', args.assay)

        if args.split == 'train':
            scored_explained = explained_model_train.copy()
        elif args.split == 'test':
            scored_explained = explained_model_test.copy()

        returned_analysis = quality_metric_flag_success(scored_explained, truth_set, args.assay, args.version, args.split)

        
        all_results[i] = returned_analysis

    results_df = pd.DataFrame.from_dict(
    all_results,
    orient="index"
    )

    results_df.to_csv(os.path.join("outputs/evaluation/", f"{args.assay}_{args.version}_{args.split}.csv"))


    results_df.index.name = "contamination"

    print(results_df)