"""
guardian_qc model.py
 
Model training, scoring, and evaluation for Guardian-QC.
 
Fits one Isolation Forest per assay on preprocessed QC data,
scores new and held-out samples, and produces evaluation outputs
including anomaly score distributions, per-feature explainability,
and (where labelled data is available) precision-recall curves.
 
Workflow:
    Preprocessed X_train / X_test (per assay: ST or haem)
        |
        v
    Fit Isolation Forest per assay
        |
        v
    Score X_train and X_test
        |
        v
    Explain flagged samples (MAD-based feature ranking)
        |
        v
    Evaluate: score distributions, flagging rates, PR curve if labels available
        |
        v
    Save models and evaluation outputs
 
Functions
---------
fit_isolation_forest()
    Fit an Isolation Forest on a single assay's training data.
 
save_model()
    Persist a fitted model to disk with joblib.
 
load_model()
    Load a saved model from disk.
 
score_samples()
    Score a feature matrix and return anomaly scores and binary labels.
 
explain_outlier()
    Rank features by deviation from training median for a flagged sample.
 
explain_all_outliers()
    Apply explain_outlier to all flagged samples in a scored DataFrame.
 
run_model_pipeline()
    Orchestrator: fit, score, explain, evaluate, save for all assays.
 
Date: 2026-08-19
Author: Rebecca Sizer
"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
 
from sklearn.ensemble import IsolationForest
from pipeline.preprocess import run_preprocessing
from outputs.graphs.train.train_graphs import plot_score_distribution, plot_flagging_rates, plot_top_deviant_features, evaluate_with_labels
from utils.logger import logging
import config
import argparse
 
N_ESTIMATORS = 100 # The number of base estimators in the ensemble (Number of decision trees) - can up to 300 if needed
RANDOM_STATE = 42 # Controls the pseudo-randomness of the selection of the features and the split values - needed to build a reproducible random sequence
 
# Top features to report
TOP_N_FEATURES = 3
 
# Unsupervised model training
# Isolation forest
def fit_isolation_forest(X_train_feature, assay: str, version: str, contamination: float):
    """
    Fit an Isolation Forest on a single assay's preprocessed training data.
   
    params:
        X_train: DataFrame
             Scaled, imputed, feature selected training data for one assay
        assay: str
             Assay label
        Contamination: float
             Expected proportion of anomilies in the training data.
             Set based on historical QC failure rate
 
    Return:
        IsolationForest: Fitted model
    """
    logging.info(f"[{assay}_{version}] Fitting isolation forest |"
                 f"n_samples={len(X_train)} | n_features={X_train.shape[1]}"
                 f"contamination={contamination}")
   
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1 #run jobs in parallel
    )

    model.fit(X_train_feature)
    logging.info(f"Isolation forest fitted for assay: {assay}_{version} | n_samples: {len(X_train)}")
    return model
 
 
# Save the models to a set location
 
def save_model(feature, model: IsolationForest, assay: str, version: str, outdir: str) -> None:
    """ Save a fitted IsolationForest to disk"""
 
    os.makedirs(outdir, exist_ok = True)
    path = os.path.join(outdir, f"{feature}_{assay}_{version}_isolation_forest.pkl")
    joblib.dump(model, path)
 
    logging.info(f"{feature}_{assay}_{version} | Model saved to {path}")
 
# Load the models
def load_model(feature, assay: str, version: str, model_dir: str) -> IsolationForest:
    """ Load a fitted IsolationForest model"""

    path = os.join.path(model_dir, f"{feature}_{assay}_{version}_isolation_forest.pkl")
    model = joblib.load(path)
    logging.info(f"[{feature}_{assay}_{version}] Model loaded from path {path}")
    return model


# This is a function that will add a score to the data 
# to indicate if it or isn't an outlier
def score_samples(feature, model: IsolationForest, X: pd.DataFrame, assay: str, version: str, split: str) -> pd.DataFrame:
    """
    Score a feature matrix using a fitted Isolation Forest.
 
    Adds two columns to a copy of X:
        anomaly_score : continuous score from decision_function().
                        More negative = more anomalous.
                        Threshold is at 0 (negative = flagged).
        anomaly_label : binary prediction. -1 = anomaly, 1 = normal.
 
    params:
        model : IsolationForest
            Fitted model for this assay.
        X : pd.DataFrame
            Feature matrix to score (must have same columns as training data).
        assay : str
            Assay label (used for logging).
        split : str
            'train' or 'test' — used in log messages only.
 
    output:
        pd.DataFrame
            X with anomaly_score for each metric and anomaly_label column appended to end.
    """

    X_scored = X.copy()
    X_scored[f'{feature}_anomaly_score'] = model.decision_function(X[[feature]]) # Apply the model - this gives a metric of how far away the value is from the median?
    X_scored[f'{feature}_anomaly_label'] = model.predict(X[[feature]]) # Apple the model to the data so it predicts if it is an outlier (-1 anomaly)

    n_flagged = (X_scored[f'{feature}_anomaly_label'] == -1).sum()
    n_total = len(X_scored)
    flag_rate = 100 * n_flagged / n_total

    logging.info(
        f"[{feature}_{assay}_{version}] Scored {split} set | "
        f"n={n_total} | flagged={n_flagged} ({flag_rate:.1f}%)"
    )

    return X_scored

# ── 7. Summary logging ────────────────────────────────────────────────────────
 
def log_model_summary(
    assay: str,
    version: str,
    n_train: int,
    n_test: int,
    n_flagged_train: int,
    n_flagged_test: int,
    contamination: float,
    auprc: float = None,
) -> None:
    sep = "=" * 55
    logging.info(sep)
    logging.info(f"MODEL SUMMARY — {assay}_{version}")
    logging.info(sep)
    logging.info(f"  {'Contamination parameter:':<35} {contamination}")
    logging.info(f"  {'Training samples:':<35} {n_train}")
    logging.info(f"  {'Test samples:':<35} {n_test}")
    logging.info(
        f"  {'Flagged in training:':<35} "
        f"{n_flagged_train} ({100*n_flagged_train/n_train:.1f}%)"
    )
    logging.info(
        f"  {'Flagged in test:':<35} "
        f"{n_flagged_test} ({100*n_flagged_test/n_test:.1f}%)"
    )
    if auprc is not None:
        logging.info(f"  {'AUPRC (labelled eval):':<35} {auprc:.4f}")
    logging.info(sep)


def attach_metadata(X_scored: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Join sample metadata back onto a scored feature matrix by index.
    Metadata columns are prepended so they appear first in the output.
    """
    merged = meta.join(X_scored, how="inner")
    return merged

 
def run_model_train(X_train, X_test, train_meta, test_meta, assay, version, split, contamination, out_dir):

    for col in X_train.columns:
        model = fit_isolation_forest(X_train[[col]], assay, version, contamination)
        save_model(col, model, assay, version, out_dir)

        X_train = score_samples(col, model, X_train, assay, version, 'train')
        X_test = score_samples(col, model, X_test, assay, version, 'test')

    # Attach metadata so you can identify samples by name
    train_results = attach_metadata(X_train, train_meta)
    test_results  = attach_metadata(X_test,  test_meta)

    # For model summary 
    n_train = len(X_train)
    n_test = len(X_test)

    test_flagged = 0
    train_flagged = 0

    for col in train_results:
        if col.endswith('anomaly_label'):
            n_flagged_train = (train_results[col] == -1).sum()
            logging.info(f"For feature {col}, number of anomolies detected: {n_flagged_train}")
            train_flagged += n_flagged_train

    for col in test_results:
        if col.endswith('anomaly_label'):
            n_flagged_test = (test_results[col] == -1).sum()
            logging.info(f"For feature {col}, number of anomolies detected: {n_flagged_test}")
            test_flagged += n_flagged_test



    log_model_summary(
        assay,
        version,
        n_train,
        n_test,
        n_flagged_train,
        n_flagged_test,
        contamination
        )

    # Save the full scored + metadata output for manual review
    os.makedirs(out_dir, exist_ok=True)
    train_results.to_csv(os.path.join(out_dir, f"{assay}_{version}_{split}_scored.csv"), index=True)
    test_results.to_csv( os.path.join(out_dir, f"{assay}_{version}_{split}_scored.csv"),  index=True)
    logging.info(f"[{assay}] Scored outputs with metadata saved to {out_dir}")

    return train_results, test_results
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Settings for training the Isolation Model'
    )

    parser.add_argument(
        "--assay",
        choices=["ST", "haem"],
        required=True,
        help='Assay to train model: "st" or "haem"'
    )

    parser.add_argument(
        "--version",
        choices=["v2", "v3", "v2_v3"],
        help="Capture version to train the model: 'v2', 'v3', or 'v2_v3'"
    )

    parser.add_argument(
            "--split",
            choices=["train", "test"],
            help="Capture split of data: train or test"
    )

    args = parser.parse_args()
    out_dir = os.path.join(config.PREPROCESSING_OUTDIR,
                           args.assay
                           )

    if args.assay == 'haem':
        contamination = 0.08

    elif args.assay == 'ST':
        contamination = 0.15

    X_train, X_test, train_meta, test_meta = run_preprocessing(config.SUMMARY_QC_METRICS, args.assay, out_dir=None, version=args.version)

    explained_model_train, explained_model_test = run_model_train(X_train, X_test, train_meta, test_meta, args.assay, args.version, args.split, contamination, os.path.join('models/trained', args.assay ))

    #plot_score_distribution(X_train, X_test_scored, args.assay, os.path.join('outputs/graphs/train', args.assay, args.version))

    #plot_flagging_rates(flagging_rate, 'outputs/graphs/train/')

    #plot_top_deviant_features(X_train_scored_explained, 'haem_train', 'outputs/graphs/train/haem/', 10)
    #plot_top_deviant_features(X_test_scored_explained, 'haem_test', 'outputs/graphs/train/haem/', 10)

    #plot_top_deviant_features(X_train_scored_explained, 'st_train', 'outputs/graphs/train/st/', 10)
    #plot_top_deviant_features(X_test_scored_explained, 'st_test', 'outputs/graphs/train/st/', 10)