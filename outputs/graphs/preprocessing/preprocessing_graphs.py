import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.logger import logging
import os

def plot_correlation_matrix(X_train: pd.DataFrame, assay: str, out_dir: str) -> None:
    corr = X_train.corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    
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

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for feature in features:

        # Create one panel per assay
        g = sns.FacetGrid(
            df,
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

            assay_values = df.loc[
                df[assay_col] == assay,
                feature
            ].dropna()

            if assay_values.empty:
                continue

            min_val = assay_values.min()
            max_val = assay_values.max()

            # Add 5% padding around the data
            data_range = max_val - min_val
            padding = data_range * 0.05

            # Prevent zero-width axis for constant features
            if padding == 0:
                padding = max(abs(min_val) * 0.05, 1)

            ax.set_xlim(
                min_val - padding,
                max_val + padding
            )

        plt.tight_layout()

        if out_dir:
            g.figure.savefig(
                f"{out_dir}/dist_{feature}.png",
                dpi=120,
                bbox_inches="tight"
            )

        plt.close(g.figure)