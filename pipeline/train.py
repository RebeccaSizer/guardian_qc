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
 
# Set constants that will be used in the script
CONTAMINATION = 0.05
N_ESTIMATORS = 100 # The number of base estimators in the ensemble (Number of decision trees) - can up to 300 if needed
RANDOM_STATE = 42 # Controls the pseudo-randomness of the selection of the features and the split values - needed to build a reproducible random sequence
 
# Top features to report
TOP_N_FEATURES = 5
 
# Unsupervised model training
# Isolation forest
def fit_isolation_forest(X_train: pd.DataFrame, assay: str, contamination: float):
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
    logging.info(f"[{assay}] Fitting isolation forest |"
                 f"n_samples={len(X_train)} | n_features={X_train.shape[1]}"
                 f"contamination={contamination}")
   
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1 #run jobs in parallel
    )
    model.fit(X_train)
    logging.info(f"Isolation forest fitted for assay: {assay} | n_samples: {len(X_train)}")
    return model
 
 
# Save the models to a set location
 
def save_model(model: IsolationForest, assay: str, outdir: str) -> None:
    """ Save a fitted IsolationForest to disk"""
 
    os.makedirs(outdir, exist_ok = True)
    path = os.path.join(outdir, f"{assay}_isolation_forest.pkl")
    joblib.dump(model, path)
 
    logging.info(f"{assay} | Model saved to {path}")
 
# Load the models
def load_model(assay: str, model_dir: str) -> IsolationForest:
    """ Load a fitted IsolationForest model"""

    path = os.join.path(model_dir, f"{assay}_isolation_forest.pkl")
    model = joblib.load(path)
    logging.info(f"[{assay}] Model loaded from path {path}")
    return model


# This is a function that will add a score to the data 
# to indicate if it or isn't an outlier
def score_samples(model: IsolationForest, X: pd.DataFrame, assay: str, split: str) -> pd.DataFrame:
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
    X_scored['anomaly_score'] = model.decision_function(X) # Apply the model - this gives a metric of how far away the value is from the median?
    X_scored['anomaly_label'] = model.predict(X) # Apple the model to the data so it predicts if it is an outlier (-1 anomaly)

    n_flagged = (X_scored['anomaly_label'] == -1).sum()
    n_total = len(X_scored)
    flag_rate = 100 * n_flagged / n_total

    logging.info(
        f"[{assay}] Scored {split} set | "
        f"n={n_total} | flagged={n_flagged} ({flag_rate:.1f}%)"
    )

    return X_scored

def explain_outlier(
    sample: pd.Series,
    X_train: pd.DataFrame,
    top_n: int = TOP_N_FEATURES,
    ) -> pd.DataFrame:
    """
    For a single flagged sample, rank features by how far they deviate
    from the training median, using Median Absolute Deviation (MAD)
    as a robust spread estimate.

    Returns a DataFrame with columns:
        feature         : feature name
        sample_value    : the sample's raw (scaled) value
        train_median    : training set median for that feature
        mad             : training set MAD for that feature
        robust_z_score  : |sample - median| / MAD — higher = more deviant

    params
        sample : pd.Series
            One row from a scored feature matrix (excluding anomaly_score/label).
        X_train : pd.DataFrame
            Training feature matrix (same columns, pre-scoring).
        top_n : int
            Number of top deviant features to return.
    """
    feature_cols  = [c for c in sample.index if c not in ("anomaly_score", "anomaly_label")] # gives a list of feature columns
    sample_feats  = sample[feature_cols]
    train_feats   = X_train[feature_cols]

    train_median  = train_feats.median() # calculate median from the training data
    train_mad     = train_feats.apply(lambda col: (col - col.median()).abs().median()) # calculate the median absolute devisation

    robust_z      = (sample_feats - train_median) / (train_mad + 1e-9) # hwo many SD the value is from the mean 

    result = pd.DataFrame({
        "feature":        feature_cols,
        "sample_value":   sample_feats.values,
        "train_median":   train_median.values,
        "mad":            train_mad.values,
        "robust_z_score": robust_z.abs().values,
    }).sort_values("robust_z_score", ascending=False).head(top_n).reset_index(drop=True) # sort the z score and only keep n number (most affecting the score)

    #print(f"explain outlier result: {result}")
    return result

def explain_all_outliers(
        X_scored: pd.DataFrame,
        X_train: pd.DataFrame,
        assay: str,
        out_dir: str,
        split: str,
        top_n: int = TOP_N_FEATURES,
    ) -> pd.DataFrame:
    """
    Apply explain_outlier to every flagged sample in a scored DataFrame.
    Saves a summary CSV and logs the top deviant feature per flagged sample.
 
    Returns a long-format DataFrame with one row per (sample, feature).
    """
    flagged = X_scored[X_scored["anomaly_label"] == -1]
    logging.info(f"[{assay}] Explaining {len(flagged)} flagged samples...")
 
    explanation_rows = []
 
    for idx, row in flagged.iterrows():
        explanation = explain_outlier(row, X_train, top_n=top_n)
        explanation.insert(0, "sample_index", idx)
        explanation.insert(1, "anomaly_score", row["anomaly_score"])
        explanation_rows.append(explanation)
 
        top_feature = explanation.iloc[0]
        logging.info(
            f"  Sample {idx} | score={row['anomaly_score']:.4f} | "
            f"top deviant feature: {top_feature['feature']} "
            f"(z={top_feature['robust_z_score']:.2f}, "
            f"value={top_feature['sample_value']:.3f}, "
            f"median={top_feature['train_median']:.3f})"
        )
 
    if not explanation_rows:
        logging.info(f"[{assay}] No flagged samples to explain.")
        return pd.DataFrame()
 
    all_explanations = pd.concat(explanation_rows, ignore_index=True)
 
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{assay}_{split}_outlier_explanations.csv")
    all_explanations.to_csv(out_path, index=False)
    logging.info(f"[{assay}] Explanations saved to {out_path}")
 
    return all_explanations
 
 
# ── 7. Summary logging ────────────────────────────────────────────────────────
 
def log_model_summary(
    assay: str,
    n_train: int,
    n_test: int,
    n_flagged_train: int,
    n_flagged_test: int,
    contamination: float,
    auprc: float = None,
) -> None:
    sep = "=" * 55
    logging.info(sep)
    logging.info(f"MODEL SUMMARY — {assay}")
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
    merged = meta.join(X_scored[["anomaly_score", "anomaly_label"]], how="inner")
    return merged

 
def run_model_train(X_train, X_test, train_meta, test_meta, assay, contamination, out_dir):

    model = fit_isolation_forest(X_train, assay, contamination)
    save_model(model, assay, out_dir)

    X_train_scored = score_samples(model, X_train, assay, 'train')
    X_test_scored = score_samples(model, X_test, assay, 'test')

    # Attach metadata so you can identify samples by name
    train_results = attach_metadata(X_train_scored, train_meta)
    test_results  = attach_metadata(X_test_scored,  test_meta)

    # Save the full scored + metadata output for manual review
    os.makedirs(out_dir, exist_ok=True)
    train_results.to_csv(os.path.join(out_dir, f"{assay}_train_scored.csv"), index=True)
    test_results.to_csv( os.path.join(out_dir, f"{assay}_test_scored.csv"),  index=True)
    logging.info(f"[{assay}] Scored outputs with metadata saved to {out_dir}")

    X_train_scored_explained = explain_all_outliers(X_train_scored, X_train, assay, out_dir, 'train', TOP_N_FEATURES)
    X_test_scored_explained = explain_all_outliers(X_test_scored, X_train, assay, out_dir, 'test', TOP_N_FEATURES)

    # For model summary 
    n_train = len(X_train)
    n_test = len(X_test)
    n_flagged_train = (X_train_scored['anomaly_label'] == -1).sum()
    n_flagged_test = (X_test_scored['anomaly_label'] == -1).sum()

    log_model_summary(
        assay,
        n_train,
        n_test,
        n_flagged_train,
        n_flagged_test,
        contamination
        )
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    X_train_haem, X_test_haem, train_meta_haem, test_meta_haem = run_preprocessing(config.SUMMARY_QC_METRICS, 'haem', out_dir=None)
    X_train_ST, X_test_ST, train_meta_ST, test_meta_ST = run_preprocessing(config.SUMMARY_QC_METRICS, 'ST', out_dir=None)

    run_model_train(X_train_haem, X_test_haem, train_meta_haem, test_meta_haem, 'haem', 0.05, os.path.join('models/trained', 'haem'))
    run_model_train(X_train_ST, X_test_ST, train_meta_ST, test_meta_ST, 'ST', 0.20, os.path.join('models/trained', 'ST'))