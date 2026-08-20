import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.logger import logging
import os
from pathlib import Path
import config

def load_data(file_path):
    """
    Loads the raw summary qc file containing all of the qc metrics.
    
    params:
        DataFrame containing qc metrics
        
    output:"""

    summary_qc_df = pd.read_csv(file_path, header=0, sep="\t")
    logging.info(f"Loaded {summary_qc_df.shape[0]} rows x {summary_qc_df.shape[1]} columns from {file_path}")
    return summary_qc_df

def plot_correlation_matrix(X_train: pd.DataFrame, assay: str, out_dir: str) -> None:
    corr = X_train.corr()

    fig, ax = plt.subplots(figsize=(16, 14))

    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    
    sns.heatmap(
        corr,
        annot=True,           # show values in cells
        fmt=".2f",            # 2 decimal places
        cmap="coolwarm",      # blue = negative, red = positive
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(f"Feature correlation matrix — {assay}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_matrix_{assay}.png", dpi=150)
    plt.close()
    logging.info(f"Correlation matrix saved for assay: {assay}")

def plot_feature_distributions(
    df,
    features,
    assay_col="assay",
    out_dir=None
):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for feature in features:

        # Make a copy and convert the feature to numeric
        plot_df = df.copy()
        plot_df[feature] = pd.to_numeric(
            plot_df[feature],
            errors="coerce"
        )

        # Remove missing values for this feature
        plot_df = plot_df.dropna(
            subset=[feature, assay_col]
        )

        if plot_df.empty:
            logging.warning(
                f"No numeric data available for {feature}; skipping plot."
            )
            continue

        # Create one panel per assay
        g = sns.FacetGrid(
            plot_df,
            col=assay_col,
            height=3,
            sharex=False,
            sharey=False
        )

        g.map(
            sns.histplot,
            feature,
            kde=True
        )

        g.set_titles("{col_name}")
        g.figure.suptitle(feature, y=1.02)

        # Set a separate x-axis range for EACH assay
        for ax, assay in zip(g.axes.flat, g.col_names):

            assay_values = plot_df.loc[
                plot_df[assay_col] == assay,
                feature
            ].dropna()

            if assay_values.empty:
                continue

            min_val = assay_values.min()
            max_val = assay_values.max()

            data_range = max_val - min_val

            # Handle constant features
            if data_range == 0:
                padding = max(abs(min_val) * 0.05, 1)
            else:
                padding = data_range * 0.05

            ax.set_xlim(
                min_val - padding,
                max_val + padding
            )

        plt.tight_layout()

        g.figure.savefig(
            out_dir / f"dist_{feature}.png",
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(g.figure)

        logging.info(
            f"Feature distribution plot saved: {feature}"
        )

if __name__=="__main__":

    df = load_data(
        file_path=config.SUMMARY_QC_METRICS_CLEANED
    )

    # Feature distributions
    plot_feature_distributions(
        df,
        config.MODEL_FEATURES,
        assay_col="assay_type",
        out_dir=os.path.join(
            config.PREPROCESSING_PLOT_DIR,
            "feature_distribution"
        )
    )

    # Correlation matrices
    for assay in df["assay_type"].dropna().unique():

        assay_features = df.loc[
            df["assay_type"] == assay,
            config.MODEL_FEATURES
        ].copy()

        plot_correlation_matrix(
            assay_features,
            assay=assay,
            out_dir=os.path.join(
                config.PREPROCESSING_PLOT_DIR,
                "metric_correlation_matrix_1"
            )
        )