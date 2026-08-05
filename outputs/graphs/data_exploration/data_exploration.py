""" 
This script generates a pie chart showing the proportion of runs that pass and fail run level QC metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from itertools import combinations

metric_cols = [
        "bcftools_ts",
        "bcftools_tv",
        "bcftools_tstv",
        "bcftools_variants",
        "bcftools_snvs",
        "bcftools_indels",
        "picard_mode_insert",
        "picard_mean_insert",
        "picard_median_insert",
        "picard_mad_insert",
        "picard_total_reads",
        "picard_pf_reads",
        "picard_pf_q30_bases",
        "picard_read_length",
        "picard_at_dropout",
        "picards_gc_dropout",
        "picard_fold_enrichment",
        #"picard_fold80",
        "picard_mean_target_coverage",
        "picard_median_target_coverage",
        "picard_target_bases_20x",
        "picard_target_bases_30x",
        "picard_target_bases_50x",
        "picard_target_bases_100x",
        "fastqc_duplication_rate",
        #"fastqc_basic_status",
        "fastp_duplication_rate",
    ]

def pie_chart_run_metric_pass_rate (df):
    counts = df['pass_run_qc'].value_counts()

    labels = [
        "Pass" if label == "Yes" else label
        for label in counts.index
    ]

    colours = {
        'Yes': '#4ECFF7',  # Blue
        'Percent Q30 < 80 AND Error rate > 2': '#AF4C82',  # Pink
        'Error rate > 2': '#FFC107'  # Yellow
    }

    pie_colours = [colours.get(label, '#D3D3D3') for label in counts.index]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    wedges, _, autotexts = ax.pie(
        counts,
        labels=None,
        colors=pie_colours,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
        pctdistance=1.1,
    )

    # Move the two smallest percentages
    autotexts[1].set_position((0.3, 1.06))
    autotexts[2].set_position((-0.03, 1.1))

    # Equal aspect ratio keeps the pie circular
    ax.axis("equal")

    # Title
    ax.set_title(
        "Proportion of Runs Passing Run-Level QC",
        fontsize=14,
        fontweight="bold",
    )

    # Legend
    ax.legend(
        wedges,
        labels,
        title="Run QC Status",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.tight_layout()

    # Save figure
    plt.show()
    plt.savefig('outputs/graphs/data_exploration/pie_chart_pass_run_qc.png')

def bar_chart_sample_count_by_sequencer_and_cancer_type(df):

    counts = df[['sequencer', 'cancer_type']].value_counts().reset_index(name="count")

    # Create a combined label for the x-axis
    counts["label"] = counts["sequencer"] + ": " + counts["cancer_type"]

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.figure(figsize=(8, 6))
    plt.bar(counts["label"], counts["count"])

    plt.xlabel("Sequencer / Cancer Type")
    plt.ylabel("Number of Samples")
    plt.title("Samples by Sequencer and Cancer Type")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig('outputs/graphs/data_exploration/sample_count_by_cancer_type_and_sequencer.png')

def comparing_sequencer_cancer_type (df):

    # Scale
    X = df[metric_cols].fillna(df[metric_cols].median())
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=metric_cols,
        columns=["PC1", "PC2"]
    )
    print(loadings["PC1"].abs().sort_values(ascending=False).head(10))
    print(loadings["PC2"].abs().sort_values(ascending=False).head(10))

    print(pca.explained_variance_ratio_)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # By sequencer
    for seq in df["sequencer"].unique():
        mask = df["sequencer"] == seq
        axes[0].scatter(df.loc[mask,"PC1"], df.loc[mask,"PC2"],
                        label=seq, alpha=0.3, s=6)

    axes[0].set_title("By sequencer")
    axes[0].set_xlabel(f"PC1 ({var1:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({var2:.1f}%)")
    axes[0].legend(markerscale=4, fontsize=8)

    # By cancer type
    for ct in df["cancer_type"].unique():
        mask = df["cancer_type"] == ct
        axes[1].scatter(df.loc[mask,"PC1"], df.loc[mask,"PC2"],
                        label=ct, alpha=0.3, s=6)
    axes[1].set_title("By cancer type")
    axes[1].set_xlabel(f"PC1 ({var1:.1f}%)")
    axes[1].legend(markerscale=4, fontsize=8)

    # By both — combine labels
    df["group"] = df["sequencer"] + " | " + df["cancer_type"]
    for grp in df["group"].unique():
        mask = df["group"] == grp
        axes[2].scatter(df.loc[mask,"PC1"], df.loc[mask,"PC2"],
                        label=grp, alpha=0.3, s=6)
    axes[2].set_title("By sequencer + cancer type")
    axes[2].set_xlabel(f"PC1 ({var1:.1f}%)")
    axes[2].legend(markerscale=4, fontsize=8, 
                bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("outputs/graphs/data_exploration/pca_by_group.png", dpi=150, bbox_inches="tight")

def test_group_differences(df, cols, group_col):
    """
    For each metric, run Kruskal-Wallis across groups.
    Returns a summary DataFrame sorted by p-value.
    """
    results = []
    groups = df[group_col].unique()
    
    for col in cols:
        # One list of values per group, dropping NaNs
        group_data = [
            df.loc[df[group_col] == g, col].dropna().values
            for g in groups
        ]
        # Only test if all groups have data
        if all(len(g) > 1 for g in group_data):
            stat, p = stats.kruskal(*group_data)
            results.append({
                "metric":   col,
                "group_by": group_col,
                "H_stat":   round(stat, 3),
                "p_value":  p,
                "significant": p < 0.05
            })
    
    return pd.DataFrame(results).sort_values("p_value")


# test in main 
if __name__ == "__main__":
    df_run_metrics = pd.read_csv("outputs/filtered_run_metrics.csv", sep="\t")
    df_qc_metrics = pd.read_csv("outputs/summary_qc_metrics.csv", sep = "\t")

    #pie_chart_run_metric_pass_rate(df_run_metrics)
    #bar_chart_sample_count_by_sequencer_and_cancer_type(df_qc_metrics)
    #comparing_sequencer_cancer_type(df_qc_metrics)
    # Run for each grouping
    #results_seq    = test_group_differences(df_qc_metrics, metric_cols, "sequencer")
    #results_cancer = test_group_differences(df_qc_metrics, metric_cols, "cancer_type")

    #df_qc_metrics_extra = df_qc_metrics.copy()
    #df_qc_metrics_extra["cancer_sequencer"] = (df_qc_metrics_extra["cancer_type"] + "_" + df_qc_metrics_extra["sequencer"])
    #results_both = test_group_differences(df_qc_metrics_extra, metric_cols, "cancer_sequencer")

    #print("=== By sequencer ===")
    #print(results_seq[["metric","H_stat","p_value","significant"]].to_string())

    #print("\n=== By cancer type ===")
    #print(results_cancer[["metric","H_stat","p_value","significant"]].to_string())

    #print("\n=== By sequencer and cancer type ===")
    #print(results_both[["metric","H_stat","p_value","significant"]].to_string())



