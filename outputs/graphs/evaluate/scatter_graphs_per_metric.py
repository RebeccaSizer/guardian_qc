import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
from utils.logger import logging


def plot_metric_scatter_by_label_composite(X_scored: pd.DataFrame, feature_cols: list[str], 
                                 assay: str, version:str, split:str, out_dir: str, 
                                 worklist_col: str = "worklist"):

    """
    For each feature, plot a scatter of sample index vs metric value,
    coloured by anomaly_label (-1 = outlier, 1 = normal).

    params:
        X_scored     : scored DataFrame — must contain anomaly_label + feature_cols
        feature_cols : list of QC metric column names to plot
        assay        : assay label (for title/filename)
        version      : version label
        split        : 'train' or 'test'
        out_dir      : output directory
        sample_col   : column to use as x-axis label (optional)
    """
    worklists = X_scored[worklist_col].astype(str) # Creates a panda series of worklists
    unique_wl = worklists.unique().tolist() # Produces a list of worklists deduplicated
    wl_to_x = {} 

    for i, wl in enumerate(unique_wl):
        wl_to_x[wl] = i

    x_vals = worklists.map(wl_to_x).values

    n_features = len(feature_cols)
    n_cols = 4 # 4 plots per row 
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    axes = axes.flatten() # turns grid coordinates into one long array so you can easily loop through them 

    colours = {1: "#4C72B0", -1: "#DD4444"}
    labels  = {1: "Normal", -1: "Outlier"}

    for i, feature in enumerate(feature_cols):
        ax = axes[i]

        for label_val in [1, -1]:
            mask = (X_scored["anomaly_label"] == label_val).values # Returns true of false if outlier or not 
            ax.scatter(
                x=x_vals[mask],
                y=X_scored.loc[mask, feature].values,
                c=colours[label_val],
                label=labels[label_val],
                alpha=0.5,
                s=12,
                linewidths=0,
            )

        # Show a subset of worklist tick labels so they don't overlap
        step = max(1, len(unique_wl) // 8) # work out a good value to use as a step increase for 
        ax.set_xticks(range(0, len(unique_wl), step))
        ax.set_xticklabels(unique_wl[::step], rotation=45, ha="right", fontsize=6)

        ax.set_title(feature, fontsize=9, pad=4)
        ax.set_xlabel("Worklist", fontsize=7)
        ax.set_ylabel("Scaled value", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)): # Hide any unused subplots 
        axes[j].set_visible(False)

    handles = [
        mpatches.Patch(color=colours[1],  label="Normal"),
        mpatches.Patch(color=colours[-1], label="Outlier"),
    ]

    fig.legend(handles=handles, loc="upper right", fontsize=9)

    fig.suptitle(f"{assay}_{version} | {split} | metric scatter by worklist", fontsize=11, y=1.01)
    plt.tight_layout() # Automatically adjust the spacing between the subplots 

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{assay}_{version}_{split}_metric_scatter_composite.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"[{assay}_{version}] Metric scatter plot saved to {out_path}")
    

if __name__=="__main__":

    X_scored_test_st = pd.read_csv('models/trained/ST/ST_v2_v3_test_scored.csv', sep=',')
    X_scored_train_st = pd.read_csv('models/trained/ST/ST_v2_v3_train_scored.csv', sep=',')

    X_scored_test_haem = pd.read_csv('models/trained/haem/haem_v2_v3_test_scored.csv', sep=',')
    X_scored_train_haem = pd.read_csv('models/trained/haem/haem_v2_v3_train_scored.csv', sep=',')


    feature_cols = [
    c for c in X_scored_train_st.columns
        if c in [
            'bcftools_ts',
            'bcftools_tv',
            'bcftools_tstv',
            'bcftools_snvs',
            'bcftools_indels',
            'picard_mode_insert',
            'picard_median_insert',
            'picard_pf_q30_bases',
            'picard_read_length',
            'picard_at_dropout',
            'picard_gc_dropout',
            'picard_fold_enrichment',
            'picard_fold80',
            'picard_target_bases_100x',
            'fastqc_duplication_rate',
            'fastp_duplication_rate'
        ]
    ]

    print(feature_cols)
    
    plot_metric_scatter_by_label_composite(X_scored_train_st, feature_cols, 'st', 'v2_v3', "train", 'outputs/graphs/evaluate')
    plot_metric_scatter_by_label_composite(X_scored_train_st, feature_cols, 'st', 'v2_v3', "test", 'outputs/graphs/evaluate')
    plot_metric_scatter_by_label_composite(X_scored_train_haem, feature_cols, 'haem', 'v2_v3', "train", 'outputs/graphs/evaluate')
    plot_metric_scatter_by_label_composite(X_scored_train_haem, feature_cols, 'haem', 'v2_v3', "test", 'outputs/graphs/evaluate')