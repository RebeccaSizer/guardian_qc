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

####################
# install packages
#####################

import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from pipeline.preprocess import run_preprocessing
#from outputs.graphs.train.train_graphs import plot_score_distribution, plot_flagging_rates, plot_top_deviant_features, evaluate_with_labels
from utils.logger import logging
import config
import argparse
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import shap 
from pathlib import Path


#####################
# Train model 
####################
# Unsupervised model training
# Isolation forest
def fit_isolation_forest(X_train: pd.DataFrame, assay: str, version: str, contamination: float):
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
        n_estimators=config.N_ESTIMATORS,
        contamination=contamination,
        random_state=config.RANDOM_STATE,
        n_jobs=-1 #run jobs in parallel
    )

    model.fit(X_train)
    logging.info(f"Isolation forest fitted for assay: {assay}_{version} | n_samples: {len(X_train)}")
    return model
 
 
# Save the models to a set location
 
def save_model(model: IsolationForest, assay: str, version: str, outdir: str) -> None:
    """ Save a fitted IsolationForest to disk"""
 
    os.makedirs(outdir, exist_ok = True)
    path = os.path.join(outdir, f"{assay}_{version}_isolation_forest.pkl")
    joblib.dump(model, path)
 
    logging.info(f"{assay}_{version} | Model saved to {path}")
 
# Load the models
def load_model(assay: str, version: str, model_dir: str) -> IsolationForest:
    """ Load a fitted IsolationForest model"""

    path = os.path.join(model_dir, f"{assay}_{version}_isolation_forest.pkl")
    model = joblib.load(path)
    logging.info(f"[{assay}_{version}] Model loaded from path {path}")
    return model


# This is a function that will add a score to the data 
# to indicate if it or isn't an outlier
def score_samples(model: IsolationForest, X: pd.DataFrame, assay: str, version: str, split: str) -> pd.DataFrame:
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
        f"[{assay}_{version}] Scored {split} set | "
        f"n={n_total} | flagged={n_flagged} ({flag_rate:.1f}%)"
    )

    return X_scored

def explain_outlier(
    sample: pd.Series,
    X_train: pd.DataFrame,
    top_n: int = config.TOP_N_FEATURES,
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
            One row from a scored feature matrix (excluding anomaly_score/label/sample_name/sequencer/cancer_type/assay_type).
        X_train : pd.DataFrame
            Training feature matrix (same columns, pre-scoring).
        top_n : int
            Number of top deviant features to return.
    """
    feature_cols = [
        c for c in sample.index
        if c in X_train.columns
    ]
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
    version: str,
    out_dir: str,
    top_n: int = config.TOP_N_FEATURES,
) -> pd.DataFrame:
    """
    Explain every flagged sample.

    Returns one row per anomalous record, with each top deviant
    feature stored in separate columns.
    """
    os.makedirs(out_dir, exist_ok=True)

    anomalies = X_scored[X_scored["anomaly_label"] == -1]

    logging.info(
        f"[{assay}_{version}] Explaining {len(anomalies)} flagged records..."
    )

    explanation_rows = []

    for idx, row in anomalies.iterrows():
        explanation = explain_outlier(row, X_train, top_n=top_n)

        sample_explanation = {
            "sample_index": idx,
            "sample_name": row["sample_name"],
            "anomaly_score": row["anomaly_score"],
        }

        for rank, (_, feature_row) in enumerate(
            explanation.iterrows(), start=1
        ):
            sample_explanation[f"top_{rank}_feature"] = feature_row["feature"]
            sample_explanation[f"top_{rank}_sample_value"] = (
                feature_row["sample_value"]
            )
            sample_explanation[f"top_{rank}_train_median"] = (
                feature_row["train_median"]
            )
            sample_explanation[f"top_{rank}_mad"] = feature_row["mad"]
            sample_explanation[f"top_{rank}_robust_z_score"] = (
                feature_row["robust_z_score"]
            )

        explanation_rows.append(sample_explanation)

    if not explanation_rows:
        logging.info(f"[{assay}_{version}] No flagged samples to explain.")
        return pd.DataFrame()

    return pd.DataFrame(explanation_rows)


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

def explain_outliers_shap(X_scored: pd.DataFrame,
    model, features, out_dir, top_n
    ) -> pd.DataFrame:
    
    os.makedirs(out_dir, exist_ok=True)
    
    explainer = shap.TreeExplainer(model)

    outlier_mask = (X_scored["anomaly_label"] == -1).to_numpy()
    X_outliers = X_scored.loc[outlier_mask, features].copy()

    shap_values = explainer(X_outliers)
    print("Number of features:", len(features))
    print("SHAP shape:", shap_values.values.shape)
    print("Feature names:", features)

    def get_top_shap_values(features, top_n=3):

        shap_array = shap_values.values
        results = []
        
        for i in range(shap_array.shape[0]):

            sample_shap = shap_array[i] # list of metrics for that sample
            top_indices = np.argsort( # return the sorted values indexes
                np.abs(sample_shap)
            )[::-1][:top_n] #reverse the order bc i want largest magnitude, and filter for top 3 only 

            sample_result = {}

            for rank, feature_idx in enumerate(top_indices, start=1):

                feature = features[int(feature_idx)]
                shap_value = float(sample_shap[int(feature_idx)])

                sample_result[f"top_{rank}_metric"] = feature
                sample_result[f"top_{rank}_shap"] = shap_value
                sample_result[f"top_{rank}_abs_shap"] = abs(shap_value)

            results.append(sample_result)

        return pd.DataFrame(results)

    top_shap = get_top_shap_values(features, top_n)
    print(top_shap.head())
    print(top_shap.dtypes)
    print(top_shap.shape)

    shap_columns = top_shap.columns

    metric_columns = [
    col for col in top_shap.columns
    if col.endswith("_metric")
    ]

    numeric_columns = [
        col for col in top_shap.columns
        if col.endswith("_shap") or col.endswith("_abs_shap")
    ]

    # Create metric columns as object dtype
    for col in metric_columns:
        X_scored[col] = pd.Series(
            pd.NA,
            index=X_scored.index,
            dtype="object"
        )

    # Create numeric SHAP columns
    for col in numeric_columns:
        X_scored[col] = np.nan

    # Add metric names for the outlier rows
    X_scored.loc[
        outlier_mask,
        metric_columns
    ] = top_shap[metric_columns].to_numpy()

    # Add SHAP values for the outlier rows
    X_scored.loc[
        outlier_mask,
        numeric_columns
    ] = top_shap[numeric_columns].to_numpy()

    print(X_scored)

    return X_scored


############################
# Graphs
############################

def plot_score_distribution(
    X_train_scored: pd.DataFrame,
    X_test_scored: pd.DataFrame,
    assay: str,
    out_dir: str,
) -> None:
    """
    Plot the distribution of anomaly scores for train and test sets.
    The vertical dashed line at x=0 is the decision boundary — left = flagged.
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(f"Anomaly score distribution — {assay}", fontsize=13)
 
    for ax, scored, label in zip(
        axes,
        [X_train_scored, X_test_scored],
        ["Training set", "Test set"],
    ):
        colours = scored["anomaly_label"].map({1: "#4C72B0", -1: "#DD4949"})
        ax.scatter(
            scored["anomaly_score"],
            np.zeros(len(scored)) + np.random.uniform(-0.1, 0.1, len(scored)),
            c=colours,
            alpha=0.5,
            s=15,
            edgecolors="none",
        )
        sns.kdeplot(
            data=scored, x="anomaly_score",
            ax=ax, color="black", linewidth=1.5,
        )
        ax.axvline(0, linestyle="--", color="red", linewidth=1, label="Decision boundary")
        ax.set_xlabel("Anomaly score (lower = more anomalous)")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_title(label)
        ax.legend(fontsize=9)
 
        n_flagged = (scored["anomaly_label"] == -1).sum()
        ax.text(
            0.02, 0.95,
            f"Flagged: {n_flagged} / {len(scored)} ({100*n_flagged/len(scored):.1f}%)",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", color="#DD4949",
        )
 
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{assay}_score_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"[{assay}] Score distribution plot saved to {out_path}")
 
 
def plot_top_deviant_features(
    explanations: pd.DataFrame,
    assay: str,
    out_dir: str,
    top_n: int = 10,
) -> None:
    """
    Bar chart showing which features appear most frequently among
    the deviant features across all flagged samples for an assay.
    Helps identify which QC metrics are most frequently associated
    with flags.
    """
    os.makedirs(out_dir, exist_ok=True)

    if explanations.empty:
        return

    # Count how often each feature appears in the explanations.
    feature_counts = (
        explanations["top_1_feature"]
        .value_counts()
        .head(top_n)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    feature_counts.plot(
        kind="barh",
        ax=ax,
        color="#4C72B0",
        edgecolor="white",
    )

    ax.set_xlabel("Number of occurrences")
    ax.set_ylabel("Feature")
    ax.set_title(f"Most frequently deviant features — {assay}")

    plt.tight_layout()

    out_path = os.path.join(
        out_dir, f"{assay}_top_deviant_features.png"
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(
        f"[{assay}] Top deviant features plot saved to {out_path}"
    )
    

def plot_anomaly_score(X_scored, assay, version):

    plt.figure(figsize=(10, 5))

    X_scored_copy = X_scored.reset_index(drop=True)
    X_scored_copy["instance"] = X_scored_copy.index

    normal = X_scored_copy[X_scored_copy["anomaly_label"] == 1]
    anomalies = X_scored_copy[X_scored_copy["anomaly_label"] == -1]

    plt.scatter(
        normal["instance"],
        normal["anomaly_score"],
        label="Normal",
        s=10
    )

    plt.scatter(
        anomalies["instance"],
        anomalies["anomaly_score"],
        label="Outlier",
        s=10
    )

    plt.title(f"Anomaly scores: {assay}_{version}")
    plt.xlabel("Instance")
    plt.ylabel("Anomaly score")
    plt.legend()

    output_path = os.path.join(config.TRAINING_PLOT_DIR, assay, version,
        f"anomaly_score_scatter_{assay}_{version}.png")
    

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def plot_shap(X_scored, model, features, assay, version, out_dir):

    explainer = shap.TreeExplainer(model)
    X_explain_global = X_scored[features]
    shap_values = explainer(X_explain_global)

    # Isolate Outliers and Norms
    outliers = (X_scored["anomaly_label"] == -1).to_numpy()
    normal = (X_scored["anomaly_label"] == 1).to_numpy()

    # calculate shap values
    shap_outliers = shap_values[outliers]
    shap_normal = shap_values[normal]   

    # Work out the path length for comparison
    path_length = shap_values.base_values + shap_values.values.sum(axis=1)

    path_length_anomalies = path_length[outliers]
    path_length_normal = path_length[normal]

    # Check for interactions. 
    shap_interaction_values = explainer.shap_interaction_values(X_explain_global)
    
    # Interaction values
    mean_shap = np.abs(shap_interaction_values).mean(0)
    mean_shap = np.round(mean_shap, 1)
    df = pd.DataFrame(mean_shap, index=X_explain_global.columns, columns=X_explain_global.columns)
    df.where(df.values == np.diagonal(df), df.values * 2, inplace = True)

    # plots a heatmap of the average shap interaction values
    sns.set(font_scale=1)
    sns.heatmap(df, cmap="coolwarm", annot=True)
    plt.yticks(rotation=0)
    plt.savefig(
            os.path.join(out_dir, assay, version, "shap_interactions.png"),
            bbox_inches="tight",
            dpi=300
        )
    plt.close()

    # Plot graphs
    shap.plots.waterfall(shap_outliers[1]) # f(x) = average path length across all branches
    plt.savefig(
        os.path.join(out_dir, assay, version, "shap_waterfall_sample_outlier.png"),
        bbox_inches="tight",
        dpi=300
    )
    plt.close()
    shap.plots.waterfall(shap_normal[1])
    plt.savefig(
        os.path.join(out_dir, assay, version, "shap_waterfall_sample_normal.png"),
        bbox_inches="tight",
        dpi=300
    )
    plt.close()
    # Outliers
    shap.plots.bar(shap_outliers, show=False)
    plt.savefig(
        os.path.join(out_dir, assay, version, "shap_bar_outliers.png"),
        bbox_inches="tight",
        dpi=300
    )
    plt.close()

    # Normal
    shap.plots.bar(shap_normal, show=False)
    plt.savefig(
        os.path.join(out_dir, assay, version, "shap_bar_normal.png"),
        bbox_inches="tight",
        dpi=300
    )
    plt.close()
  
    shap.plots.beeswarm(shap_outliers, show=False)
    plt.savefig(
        os.path.join(out_dir, assay, version, "shap_beeswarm_outliers.png"),
        bbox_inches="tight",
        dpi=300
    )
    plt.close()

    shap.plots.beeswarm(shap_normal, show=False)
    plt.savefig(
        os.path.join(out_dir, assay, version, "shap_beeswarm_normal.png"),
        bbox_inches="tight",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.boxplot([path_length_anomalies, path_length_normal], tick_labels=['Outlier', 'Normal'])
    plt.ylabel('Average Path Length f(x)')
    plt.savefig(
            os.path.join(out_dir, assay, version, "average_path_length_comparison.png"),
            bbox_inches="tight",
            dpi=300
        )
    plt.close()
    
    

########################
# Run all
#########################
 
def run_model_train(X_train, X_test, train_meta, test_meta, assay, version, split, contamination, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    model = fit_isolation_forest(X_train, assay, version, contamination)
    save_model(model, assay, version, os.path.join(config.TRAINED_COMPOSITE_MODEL_OUTDIR, assay))

    X_train_scored = score_samples(model, X_train, assay, version, 'train')
    X_test_scored = score_samples(model, X_test, assay, version, 'test')

    print("Duplicates in train_results in X_test_scored:")
    print(X_test_scored.shape)

    # Attach metadata so you can identify samples by name
    train_results = attach_metadata(X_train_scored, train_meta)
    test_results  = attach_metadata(X_test_scored,  test_meta)

    print("Duplicates in train_results in X_test_scored:")
    print(train_results.shape)

    if assay == 'haem':
        features = ['bcftools_tv', 'bcftools_tstv', 'bcftools_snvs', 'bcftools_indels', 
                    'picard_mode_insert', 'picard_median_insert', 'picard_mad_insert', 
                    'picard_total_reads', 'picard_pf_q30_bases', 'picard_read_length', 
                    'picard_at_dropout', 'picard_fold_enrichment', 'picard_fold80', 
                    'picard_median_target_coverage', 'picard_target_bases_100x', 
                    'fastqc_duplication_rate', 'fastp_duplication_rate']
    if assay == 'st':
        features = ['bcftools_ts', 'bcftools_tv', 'bcftools_tstv', 'bcftools_snvs', 
                    'bcftools_indels', 'picard_mode_insert', 'picard_median_insert', 
                    'picard_mad_insert', 'picard_pf_q30_bases', 'picard_read_length', 
                    'picard_at_dropout', 'picard_gc_dropout', 'picard_fold_enrichment', 
                    'picard_fold80', 'picard_target_bases_100x', 'fastqc_duplication_rate', 
                    'fastp_duplication_rate']

    # Plot distribution
    #plot_score_distribution(X_train_scored, X_test_scored, assay, os.path.join(config.TRAINING_PLOT_DIR, assay, version))
    #plot_anomaly_score(test_results, assay, version)
    
    explain_outliers_shap(test_results, model, features, config.TRAINING_PLOT_DIR, 3)
    explain_outliers_shap(train_results, model, features, config.TRAINING_PLOT_DIR, 3)
    
    #plot_shap(test_results, model, features, assay, version, config.TRAINING_PLOT_DIR)

    X_train_scored_explained = explain_all_outliers(train_results, X_train, assay, version, out_dir, config.TOP_N_FEATURES)
    X_test_scored_explained = explain_all_outliers(test_results, X_train, assay, version, out_dir, config.TOP_N_FEATURES)


    #plot_top_deviant_features(X_test_scored_explained, assay, os.path.join(config.TRAINING_PLOT_DIR, assay, version), 10)
    
    # For model summary 
    n_train = len(X_train)
    n_test = len(X_test)
    n_flagged_train = (X_train_scored['anomaly_label'] == -1).sum()
    n_flagged_test = (X_test_scored['anomaly_label'] == -1).sum()

    log_model_summary(
        assay,
        version,
        n_train,
        n_test,
        n_flagged_train,
        n_flagged_test,
        contamination
        )

    #merge explanations with the scored daraframe
    model_output_train = (
        train_results
        .assign(sample_index=train_results.index)
        .merge(
            X_train_scored_explained.drop(columns=["sample_name", "anomaly_score"]),
            on="sample_index",
            how="left",
            validate="one_to_one",
        )
        .drop(columns=["sample_index"])
    )

    model_output_test = (
        test_results
        .assign(sample_index=test_results.index)
        .merge(
            X_test_scored_explained.drop(columns=["sample_name", "anomaly_score"]),
            on="sample_index",
            how="left",
            validate="one_to_one",
        )
        .drop(columns=["sample_index"])
    )

    # Save the full scored + metadata output for manual review
    os.makedirs(out_dir, exist_ok=True)
    model_output_train.to_csv(os.path.join(out_dir, f"{assay}_{version}_train_scored.csv"), index=True)
    model_output_test.to_csv( os.path.join(out_dir, f"{assay}_{version}_test_scored.csv"),  index=True)
    logging.info(f"[{assay}] Scored outputs with metadata saved to {out_dir}")

    logging.info(f"Training complete. model_output_train: {model_output_train.shape}")
    logging.info(f"Training complete. model_output_test: {model_output_test.shape}")

    return model_output_train, model_output_test
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Settings for training the Isolation Model'
    )

    parser.add_argument(
        "--assay",
        choices=["st", "haem"],
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

    elif args.assay == 'st':
        contamination = 0.15

    X_train, X_test, train_meta, test_meta = run_preprocessing(config.SUMMARY_QC_METRICS, args.assay, out_dir=None, version=args.version)

    explained_model_train, explained_model_test = run_model_train(X_train, X_test, train_meta, test_meta, args.assay, args.version, args.split, contamination, os.path.join('data/processed/from_model/', args.assay ))
