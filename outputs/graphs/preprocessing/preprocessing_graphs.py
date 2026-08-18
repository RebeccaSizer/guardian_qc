import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.logger import logging

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