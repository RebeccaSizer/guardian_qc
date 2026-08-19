"""
guardian_qc model.py
 
Model training, scoring, and evaluation for Guardian-QC.
 
Fits one Isolation Forest per assay on preprocessed QC data,
scores new and held-out samples, and produces evaluation outputs
including anomaly score distributions, per-feature explainability,
and (where labelled data is available) precision-recall curves.
 
Workflow:
    Preprocessed X_train / X_test (per assay)
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
 
plot_score_distribution()
    Plot the distribution of anomaly scores for an assay.
 
plot_flagging_rate()
    Bar chart of flagging rates across assays.
 
evaluate_with_labels()
    If labelled known-bad runs are available, compute AUPRC and plot PR curve.
 
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
 
 
# Load X_train and X_test data
X_train, X_test = run_preprocessing(file_path=config.SUMMARY_QC_METRICS,
        out_dir=os.path.join(config.PREPROCESSING_OUTDIR, 'feature_selection_correlation')
        )
 
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
        n_estimatorts=N_ESTIMATORS,
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
 
    os.makedir(outdir, exist_ok = True)
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
            X with anomaly_score and anomaly_label columns appended.
    """

    X_scored = X.copy()
    X_scored['anomaly_score'] = model.decision_function(X) # Apply the model - this gives a metric of how far away the value is from the median?
    X_scored['anomaly_label'] = model.predict(X) # Apple the model to the data so it predicts if it is an outlier

    n_flagged = (X_scored['anomaly_lable'] == -1).sum()
    n_total = len(X_scored)
    flag_rate = 100 * n_flagged / n_total

    logging.info(
        f"[{assay}] Scored {split} set | "
        f"n={n_total} | flagged={n_flagged} ({flag_rate:.1f}%)"
    )
    print(X_scored)
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
    feature_cols  = [c for c in sample.index if c not in ("anomaly_score", "anomaly_label")]
    sample_feats  = sample[feature_cols]
    train_feats   = X_train[feature_cols]

    train_median  = train_feats.median()
    train_mad     = train_feats.apply(lambda col: (col - col.median()).abs().median())

    robust_z      = (sample_feats - train_median) / (train_mad + 1e-9)

    result = pd.DataFrame({
        "feature":        feature_cols,
        "sample_value":   sample_feats.values,
        "train_median":   train_median.values,
        "mad":            train_mad.values,
        "robust_z_score": robust_z.abs().values,
    }).sort_values("robust_z_score", ascending=False).head(top_n).reset_index(drop=True)

    return result

def explain_all_outliers(
        X_scored: pd.DataFrame,
        X_train: pd.DataFrame,
        assay: str,
        out_dir: str,
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
    out_path = os.path.join(out_dir, f"{assay}_outlier_explanations.csv")
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
 
 
# ── 8. Orchestrator ───────────────────────────────────────────────────────────
 
def run_model_pipeline(
    assay_data: dict,
    model_dir: str,
    plot_dir: str,
    explanation_dir: str,
    contamination_map: dict = None,
    labelled_data: dict = None,
) -> dict:
    """
    Fit, score, explain, and evaluate Isolation Forest models for all assays.
 
    Parameters
    ----------
    assay_data : dict
        Keys: assay name (str).
        Values: dict with keys:
            'X_train' : pd.DataFrame — preprocessed training features
            'X_test'  : pd.DataFrame — preprocessed test features
        e.g. {
            "NovaSeq_lung": {"X_train": df_train, "X_test": df_test},
            ...
        }
 
    model_dir : str
        Directory to save fitted models.
 
    plot_dir : str
        Directory to save evaluation plots.
 
    explanation_dir : str
        Directory to save outlier explanation CSVs.
 
    contamination_map : dict, optional
        Per-assay contamination rates.
        e.g. {"NovaSeq_lung": 0.03, "MiSeq_colorectal": 0.07}
        Falls back to DEFAULT_CONTAMINATION if assay not in map.
 
    labelled_data : dict, optional
        Per-assay ground truth labels for PR curve evaluation.
        Keys: assay name. Values: pd.Series (1=bad, 0=good), indexed
        to match X_test rows.
        Only provide if you have manually verified known-bad runs.
 
    Returns
    -------
    dict
        Per-assay summary: flagging rates, AUPRC if available.
    """
    if contamination_map is None:
        contamination_map = {}
    if labelled_data is None:
        labelled_data = {}
 
    summary       = {}
    flagging_summary = {}
 
    for assay, data in assay_data.items():
        logging.info(f"\n{'='*55}")
        logging.info(f"Processing assay: {assay}")
        logging.info(f"{'='*55}")
 
        X_train = data["X_train"]
        X_test  = data["X_test"]
        contamination = contamination_map.get(assay, CONTAMINATION)
 
        # Fit
        model = fit_isolation_forest(X_train, assay, contamination)
        save_model(model, assay, model_dir)
 
        # Score
        X_train_scored = score_samples(model, X_train, assay, split="train")
        X_test_scored  = score_samples(model, X_test,  assay, split="test")
 
        n_flagged_train = (X_train_scored["anomaly_label"] == -1).sum()
        n_flagged_test  = (X_test_scored["anomaly_label"]  == -1).sum()
 
        flagging_summary[assay] = {
            "n_total":   len(X_test_scored),
            "n_flagged": n_flagged_test,
        }
 
        # Explain flagged test samples
        explanations = explain_all_outliers(
            X_test_scored, X_train, assay, explanation_dir
        )
 
        # Plots
        plot_score_distribution(X_train_scored, X_test_scored, assay, plot_dir)
        if not explanations.empty:
            plot_top_deviant_features(explanations, assay, plot_dir)
 
        # Evaluate with labels if provided
        auprc = None
        if assay in labelled_data:
            eval_result = evaluate_with_labels(
                X_test_scored, labelled_data[assay], assay, plot_dir
            )
            auprc = eval_result["auprc"]
 
        # Log summary
        log_model_summary(
            assay=assay,
            n_train=len(X_train),
            n_test=len(X_test),
            n_flagged_train=n_flagged_train,
            n_flagged_test=n_flagged_test,
            contamination=contamination,
            auprc=auprc,
        )
 
        summary[assay] = {
            "n_train":        len(X_train),
            "n_test":         len(X_test),
            "n_flagged_train": n_flagged_train,
            "n_flagged_test":  n_flagged_test,
            "flag_rate_test":  100 * n_flagged_test / len(X_test),
            "contamination":   contamination,
            "auprc":           auprc,
        }
 
    # Cross-assay flagging rate plot
    plot_flagging_rates(flagging_summary, plot_dir)
 
    return summary
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print(X_test)
    assay_data = {
        "novaseqx_solid_tumour": {
            "X_train": X_train_novaseqx_solid_tumour,
            "X_test": X_test_novaseqx_solid_tumour,
        },

        "novaseqx_haem": {
            "X_train": X_train_novaseqx_haem,
            "X_test": X_test_novaseqx_haem,
        },

        "novaseq6000_solid_tumour": {
            "X_train": X_train_novaseq6000_solid_tumour,
            "X_test": X_test_novaseq6000_solid_tumour,
        },

        "novaseq6000_haem": {
            "X_train": X_train_novaseq6000_haem,
            "X_test": X_test_novaseq6000_haem,
        },
    }

    # ------------------------------------------------------------------
    # Isolation Forest contamination
    # ------------------------------------------------------------------
    #
    # contamination represents the expected proportion of anomalous
    # samples in each assay.
    #
    # These are example values only. Ideally, determine these from
    # historical QC failure rates or labelled data.

    contamination_map = {
        "novaseqx_solid_tumour": 0.05,
        "novaseqx_haem": 0.05,
        "novaseq6000_solid_tumour": 0.05,
        "novaseq6000_haem": 0.05,
    }

    # ------------------------------------------------------------------
    # Optional labelled data
    # ------------------------------------------------------------------
    #
    # If you have manually reviewed QC runs, you can provide their
    # labels here to evaluate model performance.
    #
    # Example:
    #
    # labelled_data = {
    #     "novaseqx_solid_tumour": pd.Series({
    #         10: 1,   # known bad run
    #         25: 0,   # known good run
    #         31: 1,
    #     })
    # }

    summary = run_model_pipeline(
        assay_data=assay_data,
        model_dir=config.MODEL_DIR,
        plot_dir=config.MODEL_PLOT_DIR,
        explanation_dir=config.EXPLANATION_DIR,
        contamination_map=contamination_map,
        # labelled_data=labelled_data,
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    logging.info("Model pipeline complete.")

    summary_df = pd.DataFrame(summary).T

    logging.info(
        "\n%s",
        summary_df.to_string()
    )
