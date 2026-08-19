"""
guardian_qc assess_grouping.py
 
Assesses whether QC metrics differ significantly between sequencers
and pipeline versions within each cancer type group (ST and Haem).
 
If metrics are statistically equivalent across sequencers/pipeline versions
within a cancer type, the data can be safely pooled and only split by
cancer type (ST vs Haem), simplifying the per-assay model structure.
 
Approach
--------
For each QC metric and each cancer type group:
 
1. Statistical tests
   - Two or more groups: Kruskal-Wallis (non-parametric, no normality assumption)
   - Pairwise follow-up: Mann-Whitney U with Bonferroni correction
   - Effect size: Cohen's d and rank-biserial correlation
 
2. Visual inspection
   - Box plots per metric per group, faceted by sequencer/pipeline version
   - Overlaid KDE plots to compare distributions
 
3. Summary table
   - One row per metric per cancer type
   - Reports p-value, effect size, and a recommendation (pool / do not pool)
 
Outputs
-------
- assess_grouping_summary.csv   : full statistical summary
- boxplots/                     : one plot per metric
- kde_plots/                    : one plot per metric
- assess_grouping.log           : full log
 
Date: 2026-08-19
Author: Rebecca Sizer
"""
 
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
 
from itertools import combinations
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
 
import config
from utils.logger import logging
from pipeline.preprocess import load_data, fix_data_types
 
# ── Constants ─────────────────────────────────────────────────────────────────
 
# Significance threshold after Bonferroni correction
ALPHA = 0.05
 
# Effect size threshold below which a difference is considered negligible
# Uses rank-biserial correlation (0 = no effect, 1 = complete separation)
NEGLIGIBLE_EFFECT_SIZE = 0.1
 
# The grouping variables to test
CANCER_TYPE_COL  = "cancer_type"
SEQUENCER_COL    = "sequencer"
PIPELINE_COL     = "pipeline_version"   # rename to match your actual column name
 
# Which column to split assays on — adjust if you don't have pipeline_version
GROUP_COLS = [CANCER_TYPE_COL]            # add PIPELINE_COL if you have it
 
 
# ── Statistical tests ─────────────────────────────────────────────────────────
 
def rank_biserial_correlation(u_stat: float, n1: int, n2: int) -> float:
    """
    Compute rank-biserial correlation as effect size for Mann-Whitney U.
    Ranges from -1 to 1. Values near 0 = negligible effect.
    """
    return 1 - (2 * u_stat) / (n1 * n2)
 
 
def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Cohen's d effect size between two groups."""
    mean_diff = group1.mean() - group2.mean()
    pooled_std = np.sqrt(
        ((len(group1) - 1) * group1.std()**2 +
         (len(group2) - 1) * group2.std()**2) /
        (len(group1) + len(group2) - 2)
    )
    return mean_diff / (pooled_std + 1e-9)
 
 
def test_metric_across_groups(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
) -> dict:
    """
    Test whether a QC metric differs significantly across groups
    (e.g. sequencers or pipeline versions).
 
    Uses Kruskal-Wallis for overall test, then pairwise Mann-Whitney U
    with Bonferroni correction if more than two groups.
 
    Returns a dict with test results.
    """
    groups      = df[group_col].dropna().unique()
    group_data  = [df.loc[df[group_col] == g, metric].dropna() for g in groups]
 
    # Drop groups with fewer than 5 samples
    valid = [(g, d) for g, d in zip(groups, group_data) if len(d) >= 5]
 
    if len(valid) < 2:
        return {
            "metric":           metric,
            "group_col":        group_col,
            "n_groups":         len(valid),
            "kruskal_p":        np.nan,
            "significant":      False,
            "max_effect_size":  np.nan,
            "recommendation":   "insufficient data",
            "notes":            f"Only {len(valid)} group(s) with n>=5",
        }
 
    groups, group_data = zip(*valid)
 
    # Overall Kruskal-Wallis
    if len(groups) == 2:
        kw_stat, kw_p = kruskal(*group_data)
    else:
        kw_stat, kw_p = kruskal(*group_data)
 
    # Pairwise Mann-Whitney U with Bonferroni correction
    pairs        = list(combinations(range(len(groups)), 2))
    n_pairs      = len(pairs)
    pairwise_results = []
 
    for i, j in pairs:
        g1, g2 = group_data[i], group_data[j]
        u_stat, mw_p = mannwhitneyu(g1, g2, alternative="two-sided")
        corrected_p  = min(mw_p * n_pairs, 1.0)   # Bonferroni
        rbc          = abs(rank_biserial_correlation(u_stat, len(g1), len(g2)))
        d            = abs(cohens_d(g1, g2))
 
        pairwise_results.append({
            "group_a":      groups[i],
            "group_b":      groups[j],
            "mw_p":         mw_p,
            "corrected_p":  corrected_p,
            "rbc":          rbc,
            "cohens_d":     d,
            "significant":  corrected_p < ALPHA,
        })
 
    pairwise_df  = pd.DataFrame(pairwise_results)
    max_effect   = pairwise_df["rbc"].max()
    any_sig      = pairwise_df["significant"].any()
 
    # Recommendation
    if not any_sig:
        recommendation = "pool"
    elif any_sig and max_effect < NEGLIGIBLE_EFFECT_SIZE:
        recommendation = "pool (significant but negligible effect)"
    else:
        recommendation = "do not pool"
 
    return {
        "metric":          metric,
        "group_col":       group_col,
        "n_groups":        len(groups),
        "group_sizes":     {g: len(d) for g, d in zip(groups, group_data)},
        "kruskal_stat":    round(kw_stat, 4),
        "kruskal_p":       round(kw_p, 6),
        "significant":     any_sig,
        "max_effect_size": round(max_effect, 4),
        "recommendation":  recommendation,
        "pairwise":        pairwise_df,
    }
 
 
# ── Plots ─────────────────────────────────────────────────────────────────────
 
def plot_metric_boxplot(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    cancer_type: str,
    out_dir: str,
) -> None:
    """Box plot of a metric split by group_col, for one cancer type."""
    fig, ax = plt.subplots(figsize=(8, 5))
 
    groups  = sorted(df[group_col].dropna().unique())
    colours = sns.color_palette("Set2", len(groups))
 
    data_to_plot = [df.loc[df[group_col] == g, metric].dropna() for g in groups]
 
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
 
    ax.set_xticklabels(
        [f"{g}\n(n={len(df[df[group_col]==g][metric].dropna())})" for g in groups]
    )
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by {group_col} — {cancer_type}")
    plt.tight_layout()
 
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        os.path.join(out_dir, f"{cancer_type}_{metric}_{group_col}_boxplot.png"),
        dpi=120, bbox_inches="tight"
    )
    plt.close()
 
 
def plot_metric_kde(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    cancer_type: str,
    out_dir: str,
) -> None:
    """Overlaid KDE plot of a metric split by group_col, for one cancer type."""
    fig, ax = plt.subplots(figsize=(8, 5))
 
    for group, colour in zip(
        sorted(df[group_col].dropna().unique()),
        sns.color_palette("Set2"),
    ):
        subset = df.loc[df[group_col] == group, metric].dropna()
        if len(subset) >= 5:
            sns.kdeplot(subset, ax=ax, label=f"{group} (n={len(subset)})",
                        fill=True, alpha=0.3, color=colour)
 
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    ax.set_title(f"{metric} distribution by {group_col} — {cancer_type}")
    ax.legend()
    plt.tight_layout()
 
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        os.path.join(out_dir, f"{cancer_type}_{metric}_{group_col}_kde.png"),
        dpi=120, bbox_inches="tight"
    )
    plt.close()
 
 
def plot_summary_heatmap(
    summary_df: pd.DataFrame,
    group_col: str,
    cancer_type: str,
    out_dir: str,
) -> None:
    """
    Heatmap of effect sizes across all metrics for one cancer type and group_col.
    Green = negligible difference (safe to pool), red = meaningful difference.
    """
    subset = summary_df[
        (summary_df["group_col"]    == group_col) &
        (summary_df["cancer_type"]  == cancer_type) &
        (summary_df["max_effect_size"].notna())
    ].set_index("metric")[["max_effect_size", "significant"]].copy()
 
    if subset.empty:
        return
 
    fig, ax = plt.subplots(figsize=(4, max(5, len(subset) * 0.35)))
 
    cmap = sns.diverging_palette(130, 10, as_cmap=True)
    sns.heatmap(
        subset[["max_effect_size"]],
        annot=True, fmt=".3f",
        cmap=cmap,
        vmin=0, vmax=0.5,
        center=NEGLIGIBLE_EFFECT_SIZE,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Rank-biserial correlation (effect size)"},
    )
 
    # Mark significant metrics with an asterisk
    for i, (metric, row) in enumerate(subset.iterrows()):
        if row["significant"]:
            ax.text(1.02, i + 0.5, "*", transform=ax.transAxes,
                    fontsize=12, color="red", va="center")
 
    ax.set_title(f"Effect sizes — {group_col} — {cancer_type}\n* = significant after Bonferroni")
    ax.set_ylabel("")
    plt.tight_layout()
 
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        os.path.join(out_dir, f"{cancer_type}_{group_col}_effect_size_heatmap.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    logging.info(f"Effect size heatmap saved: {cancer_type} | {group_col}")
 
 
# ── Orchestrator ──────────────────────────────────────────────────────────────
 
def run_grouping_assessment(
    file_path: str,
    out_dir: str,
    group_cols: list = GROUP_COLS,
    features: list = None,
) -> pd.DataFrame:
    """
    Full grouping assessment pipeline.
 
    For each cancer type, tests every QC metric for significant differences
    across each grouping variable (sequencer, pipeline version, etc.).
 
    Produces:
    - Statistical summary CSV
    - Box plots and KDE plots per metric
    - Effect size heatmaps per cancer type per group variable
 
    Parameters
    ----------
    file_path : str
        Path to QC summary TSV.
    out_dir : str
        Root output directory.
    group_cols : list
        Columns to test grouping by (e.g. ['sequencer', 'pipeline_version']).
    features : list
        QC metrics to test. Defaults to MODEL_FEATURES (numeric only).
 
    Returns
    -------
    pd.DataFrame
        Summary table with one row per metric / cancer type / group column.
    """
    if features is None:
        # Use only numeric model features — exclude OHE columns
        features = [f for f in config.MODEL_FEATURES if not f.startswith("fastqc_basic_status_")]
 
    # Load and fix types
    df = load_data(file_path)
    df = fix_data_types(df)
 
    cancer_types = df[CANCER_TYPE_COL].unique()
    logging.info(f"Cancer types found: {cancer_types.tolist()}")
    logging.info(f"Group columns to test: {group_cols}")
    logging.info(f"Metrics to test: {len(features)}")
 
    boxplot_dir = os.path.join(out_dir, "boxplots")
    kde_dir     = os.path.join(out_dir, "kde_plots")
    heatmap_dir = os.path.join(out_dir, "heatmaps")
 
    all_results = []
 
    for cancer_type in cancer_types:
        ct_df = df[df[CANCER_TYPE_COL] == cancer_type].copy()
        logging.info(f"\n{'='*55}")
        logging.info(f"Cancer type: {cancer_type} | n={len(ct_df)}")
        logging.info(f"{'='*55}")
 
        for group_col in group_cols:
            if group_col not in ct_df.columns:
                logging.warning(f"Column '{group_col}' not found — skipping.")
                continue
 
            group_counts = ct_df[group_col].value_counts()
            logging.info(f"\nGroup col: {group_col}")
            logging.info(f"Groups:\n{group_counts.to_string()}")
 
            for metric in features:
                if metric not in ct_df.columns:
                    continue
 
                result = test_metric_across_groups(ct_df, metric, group_col)
                result["cancer_type"] = cancer_type
                all_results.append(result)
 
                logging.info(
                    f"  {metric:<45} "
                    f"p={result['kruskal_p']:.4f}  "
                    f"effect={result['max_effect_size']:.3f}  "
                    f"→ {result['recommendation']}"
                )
 
                # Plots
                plot_metric_boxplot(ct_df, metric, group_col, cancer_type, boxplot_dir)
                plot_metric_kde(ct_df, metric, group_col, cancer_type, kde_dir)
 
    # Build summary DataFrame
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "cancer_type":     r.get("cancer_type"),
            "group_col":       r.get("group_col"),
            "metric":          r.get("metric"),
            "n_groups":        r.get("n_groups"),
            "kruskal_p":       r.get("kruskal_p"),
            "significant":     r.get("significant"),
            "max_effect_size": r.get("max_effect_size"),
            "recommendation":  r.get("recommendation"),
            "notes":           r.get("notes", ""),
        })
 
    summary_df = pd.DataFrame(summary_rows)
 
    # Heatmaps
    for cancer_type in cancer_types:
        for group_col in group_cols:
            plot_summary_heatmap(summary_df, group_col, cancer_type, heatmap_dir)
 
    # Overall recommendation per cancer type / group col
    sep = "=" * 55
    logging.info(f"\n{sep}")
    logging.info("GROUPING ASSESSMENT — OVERALL RECOMMENDATIONS")
    logging.info(sep)
 
    for cancer_type in cancer_types:
        for group_col in group_cols:
            subset = summary_df[
                (summary_df["cancer_type"] == cancer_type) &
                (summary_df["group_col"]   == group_col)
            ]
            if subset.empty:
                continue
 
            n_do_not_pool = (subset["recommendation"] == "do not pool").sum()
            n_pool        = (subset["recommendation"].str.startswith("pool")).sum()
            n_total       = len(subset)
 
            logging.info(f"\n  {cancer_type} | {group_col}")
            logging.info(f"  Metrics where pooling is safe:    {n_pool} / {n_total}")
            logging.info(f"  Metrics where pooling is NOT safe: {n_do_not_pool} / {n_total}")
 
            if n_do_not_pool == 0:
                logging.info(f"  ✓ RECOMMENDATION: pool across {group_col} for {cancer_type}")
            elif n_do_not_pool <= 2:
                logging.info(
                    f"  ~ RECOMMENDATION: pooling likely acceptable — "
                    f"review {n_do_not_pool} metric(s) manually"
                )
                do_not_pool_metrics = subset[
                    subset["recommendation"] == "do not pool"
                ]["metric"].tolist()
                logging.info(f"    Metrics to review: {do_not_pool_metrics}")
            else:
                logging.info(
                    f"  ✗ RECOMMENDATION: do not pool across {group_col} for {cancer_type} — "
                    f"{n_do_not_pool} metrics differ meaningfully"
                )
 
    # Save summary
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "assess_grouping_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"\nSummary saved to {summary_path}")
 
    return summary_df
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
 
    summary = run_grouping_assessment(
        file_path=config.SUMMARY_QC_METRICS,
        out_dir=os.path.join(config.PREPROCESSING_OUTDIR, "grouping_assessment"),
        group_cols=[SEQUENCER_COL],   # add PIPELINE_COL here if you have it
    )
 
    # Print a quick readable table of do-not-pool metrics
    do_not_pool = summary[summary["recommendation"] == "do not pool"]
    if do_not_pool.empty:
        logging.info("\n✓ All metrics safe to pool across tested grouping variables.")
    else:
        logging.info(f"\nMetrics flagged as do-not-pool:\n{do_not_pool[['cancer_type','group_col','metric','kruskal_p','max_effect_size']].to_string(index=False)}")