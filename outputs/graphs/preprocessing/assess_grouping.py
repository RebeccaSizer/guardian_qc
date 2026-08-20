"""
guardian_qc assess_grouping.py
 
Assesses whether QC metrics differ significantly between:
    1. Pipeline versions (solid_tumour vs solid_tumour_v3,
                          haem_v2 vs haem_v3)
    2. Sequencers (within each cancer type group)
 
This determines whether data can be pooled across pipeline versions
and/or sequencers before training per-assay Isolation Forest models,
or whether separate models are needed for each combination.
 
cancer_type column contains: solid_tumour, solid_tumour_v3, haem_v2, haem_v3
sequencer column contains:   two sequencer identifiers
 
Outputs
-------
    assess_grouping_summary.csv     Full statistical results
    heatmaps/                       Effect size heatmaps per comparison
    boxplots/                       Box plots per metric per comparison
    kde_plots/                      KDE plots per metric per comparison
 
Date: 2026-08-19
Author: Rebecca Sizer
"""
 
import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu
 
warnings.filterwarnings("ignore", category=UserWarning)
 
# ── Column names ──────────────────────────────────────────────────────────────
 
CANCER_TYPE_COL = "cancer_type"
SEQUENCER_COL   = "sequencer"
 
# Map cancer_type values to their parent group and version
# Adjust these if your values differ
PIPELINE_GROUPS = {
    "solid_tumour":    {"parent": "solid_tumour", "version": "v1"},
    "solid_tumour_v3": {"parent": "solid_tumour", "version": "v3"},
    "haem_v2":         {"parent": "haem",         "version": "v2"},
    "haem_v3":         {"parent": "haem",         "version": "v3"},
}
 
# ── QC metrics to test ────────────────────────────────────────────────────────
 
NUMERIC_FEATURES = [
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
    "picard_gc_dropout",
    "picard_fold_enrichment",
    "picard_fold80",
    "picard_mean_target_coverage",
    "picard_median_target_coverage",
    "picard_target_bases_20x",
    "picard_target_bases_30x",
    "picard_target_bases_50x",
    "picard_target_bases_100x",
    "fastqc_duplication_rate",
    "fastp_duplication_rate",
]
 
# ── Thresholds ────────────────────────────────────────────────────────────────
 
ALPHA                   = 0.05    # significance threshold
NEGLIGIBLE_EFFECT       = 0.1     # rank-biserial r below this → negligible
SMALL_EFFECT            = 0.3     # r below this → small effect
 
 
# ── Statistics ────────────────────────────────────────────────────────────────
 
def rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """Effect size for Mann-Whitney U. Range: -1 to 1."""
    return 1 - (2 * u_stat) / (n1 * n2)
 
 
def interpret_effect(r: float) -> str:
    r = abs(r)
    if r < NEGLIGIBLE_EFFECT:
        return "negligible"
    elif r < SMALL_EFFECT:
        return "small"
    else:
        return "meaningful"
 
 
def recommend(significant: bool, effect: str) -> str:
    if not significant:
        return "pool"
    if effect == "negligible":
        return "pool (sig but negligible effect)"
    if effect == "small":
        return "review"
    return "do not pool"
 
 
def test_two_groups(
    group_a: pd.Series,
    group_b: pd.Series,
    label_a: str,
    label_b: str,
    metric: str,
    n_comparisons: int = 1,
) -> dict:
    """
    Mann-Whitney U test between two groups with Bonferroni correction.
    Returns a result dict.
    """
    a = group_a.dropna()
    b = group_b.dropna()
 
    if len(a) < 5 or len(b) < 5:
        return {
            "metric": metric, "group_a": label_a, "group_b": label_b,
            "n_a": len(a), "n_b": len(b),
            "u_stat": np.nan, "raw_p": np.nan, "corrected_p": np.nan,
            "effect_r": np.nan, "effect_interp": "insufficient data",
            "significant": False, "recommendation": "insufficient data",
        }
 
    u_stat, raw_p       = mannwhitneyu(a, b, alternative="two-sided")
    corrected_p         = min(raw_p * n_comparisons, 1.0)
    r                   = rank_biserial_r(u_stat, len(a), len(b))
    effect_interp       = interpret_effect(r)
    significant         = corrected_p < ALPHA
    rec                 = recommend(significant, effect_interp)
 
    return {
        "metric":       metric,
        "group_a":      label_a,
        "group_b":      label_b,
        "n_a":          len(a),
        "n_b":          len(b),
        "u_stat":       round(u_stat, 2),
        "raw_p":        round(raw_p, 6),
        "corrected_p":  round(corrected_p, 6),
        "effect_r":     round(abs(r), 4),
        "effect_interp": effect_interp,
        "significant":  significant,
        "recommendation": rec,
    }
 
 
# ── Plots ─────────────────────────────────────────────────────────────────────
 
def plot_boxplot(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    title: str,
    out_path: str,
) -> None:
    groups     = sorted(df[group_col].dropna().unique())
    group_data = [df.loc[df[group_col] == g, metric].dropna() for g in groups]
    colours    = sns.color_palette("Set2", len(groups))
 
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 2), 5))
    bp = ax.boxplot(group_data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
 
    ax.set_xticklabels(
        [f"{g}\n(n={len(df[df[group_col]==g][metric].dropna())})"
         for g in groups],
        rotation=15, ha="right"
    )
    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
 
 
def plot_kde(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    title: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = sns.color_palette("Set2")
 
    for i, group in enumerate(sorted(df[group_col].dropna().unique())):
        subset = df.loc[df[group_col] == group, metric].dropna()
        if len(subset) >= 5:
            sns.kdeplot(subset, ax=ax, label=f"{group} (n={len(subset)})",
                        fill=True, alpha=0.3, color=colours[i % len(colours)])
 
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
 
 
def plot_effect_heatmap(
    results_df: pd.DataFrame,
    comparison_label: str,
    out_path: str,
) -> None:
    """
    Heatmap of effect sizes per metric for one comparison.
    Green = negligible (safe to pool), red = meaningful difference.
    """
    pivot = results_df.set_index("metric")[["effect_r", "recommendation"]].copy()
    pivot = pivot[pivot["effect_r"].notna()].sort_values("effect_r", ascending=False)
 
    if pivot.empty:
        return
 
    fig, ax = plt.subplots(figsize=(4, max(6, len(pivot) * 0.38)))
    cmap = sns.diverging_palette(130, 10, as_cmap=True)
 
    sns.heatmap(
        pivot[["effect_r"]],
        annot=True, fmt=".3f",
        cmap=cmap,
        vmin=0, vmax=0.5,
        center=NEGLIGIBLE_EFFECT,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Effect size (rank-biserial r)"},
    )
 
    # Flag significant meaningful differences
    for i, (metric, row) in enumerate(pivot.iterrows()):
        if row["recommendation"] == "do not pool":
            ax.text(1.03, i + 0.5, "✗", transform=ax.transAxes,
                    fontsize=11, color="red", va="center")
        elif row["recommendation"].startswith("pool"):
            ax.text(1.03, i + 0.5, "✓", transform=ax.transAxes,
                    fontsize=11, color="green", va="center")
 
    ax.set_title(
        f"{comparison_label}\n"
        f"✓ = pool  ✗ = do not pool",
        fontsize=10
    )
    ax.set_ylabel("")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Heatmap saved: {out_path}")
 
 
# ── Orchestrator ──────────────────────────────────────────────────────────────
 
def run_grouping_assessment(
    df: pd.DataFrame,
    out_dir: str,
    features: list = None,
) -> pd.DataFrame:
    """
    Runs two sets of comparisons:
 
    1. Pipeline version comparisons (within each parent group):
           solid_tumour vs solid_tumour_v3
           haem_v2 vs haem_v3
 
    2. Sequencer comparisons (within each cancer_type):
           for each of solid_tumour, solid_tumour_v3, haem_v2, haem_v3:
               sequencer A vs sequencer B
 
    Returns a summary DataFrame with one row per metric per comparison.
    """
    if features is None:
        features = [f for f in NUMERIC_FEATURES if f in df.columns]
 
    # Fix picard_fold80 type issue
    df = df.copy()
    df["picard_fold80"] = pd.to_numeric(
        df["picard_fold80"].replace("?", np.nan), errors="coerce"
    )
 
    # Add parent group and version columns
    df["parent_group"] = df[CANCER_TYPE_COL].map(
        lambda x: PIPELINE_GROUPS.get(x, {}).get("parent", x)
    )
    df["pipeline_version"] = df[CANCER_TYPE_COL].map(
        lambda x: PIPELINE_GROUPS.get(x, {}).get("version", x)
    )
 
    all_results = []
    sep = "=" * 60
 
    # ── 1. Pipeline version comparisons ──────────────────────────────────────
    logging.info(f"\n{sep}")
    logging.info("COMPARISON 1: Pipeline versions")
    logging.info("Can solid_tumour and solid_tumour_v3 be pooled?")
    logging.info("Can haem_v2 and haem_v3 be pooled?")
    logging.info(sep)
 
    for parent in df["parent_group"].unique():
        parent_df = df[df["parent_group"] == parent]
        versions  = parent_df[CANCER_TYPE_COL].unique()
 
        if len(versions) < 2:
            logging.warning(f"  {parent}: only one version found, skipping.")
            continue
 
        # For exactly two versions, one pairwise comparison per metric
        v_a, v_b     = versions[0], versions[1]
        comparison   = f"pipeline_version_{parent}"
        n_metrics    = len(features)
 
        logging.info(f"\n  {parent}: {v_a} vs {v_b}")
        logging.info(f"  n_{v_a} = {(parent_df[CANCER_TYPE_COL]==v_a).sum()}")
        logging.info(f"  n_{v_b} = {(parent_df[CANCER_TYPE_COL]==v_b).sum()}")
 
        comp_results = []
        for metric in features:
            result = test_two_groups(
                group_a      = parent_df.loc[parent_df[CANCER_TYPE_COL]==v_a, metric],
                group_b      = parent_df.loc[parent_df[CANCER_TYPE_COL]==v_b, metric],
                label_a      = v_a,
                label_b      = v_b,
                metric       = metric,
                n_comparisons = n_metrics,   # Bonferroni across all metrics
            )
            result["comparison"] = comparison
            result["parent"]     = parent
            comp_results.append(result)
            all_results.append(result)
 
            logging.info(
                f"    {metric:<45} "
                f"p={result['corrected_p']:.4f}  "
                f"r={result['effect_r']:.3f} ({result['effect_interp']})  "
                f"→ {result['recommendation']}"
            )
 
            # Plots
            plot_boxplot(
                parent_df, metric, CANCER_TYPE_COL,
                title=f"{metric} — {parent}: pipeline version comparison",
                out_path=os.path.join(out_dir, "boxplots", f"{comparison}_{metric}.png"),
            )
            plot_kde(
                parent_df, metric, CANCER_TYPE_COL,
                title=f"{metric} — {parent}: pipeline version comparison",
                out_path=os.path.join(out_dir, "kde_plots", f"{comparison}_{metric}.png"),
            )
 
        # Heatmap for this comparison
        comp_df = pd.DataFrame(comp_results)
        plot_effect_heatmap(
            comp_df,
            comparison_label=f"{parent}: {v_a} vs {v_b}",
            out_path=os.path.join(out_dir, "heatmaps", f"{comparison}_heatmap.png"),
        )
 
        # Overall recommendation for this pipeline comparison
        n_do_not_pool = (comp_df["recommendation"] == "do not pool").sum()
        n_review      = (comp_df["recommendation"] == "review").sum()
        logging.info(f"\n  Summary for {parent} pipeline version comparison:")
        logging.info(f"    Safe to pool:   {(comp_df['recommendation'].str.startswith('pool')).sum()} / {n_metrics}")
        logging.info(f"    Review:         {n_review} / {n_metrics}")
        logging.info(f"    Do not pool:    {n_do_not_pool} / {n_metrics}")
        if n_do_not_pool == 0 and n_review == 0:
            logging.info(f"    ✓ RECOMMENDATION: pool {v_a} and {v_b}")
        elif n_do_not_pool == 0:
            metrics_to_review = comp_df[comp_df["recommendation"]=="review"]["metric"].tolist()
            logging.info(f"    ~ RECOMMENDATION: likely poolable — review: {metrics_to_review}")
        else:
            do_not_pool_metrics = comp_df[comp_df["recommendation"]=="do not pool"]["metric"].tolist()
            logging.info(f"    ✗ RECOMMENDATION: do NOT pool — differing metrics: {do_not_pool_metrics}")
 
    # ── 2. Sequencer comparisons ──────────────────────────────────────────────
    logging.info(f"\n{sep}")
    logging.info("COMPARISON 2: Sequencers")
    logging.info("Within each cancer_type, can sequencer A and B be pooled?")
    logging.info(sep)
 
    for cancer_type in df[CANCER_TYPE_COL].unique():
        ct_df      = df[df[CANCER_TYPE_COL] == cancer_type]
        sequencers = ct_df[SEQUENCER_COL].dropna().unique()
 
        if len(sequencers) < 2:
            logging.warning(f"  {cancer_type}: only one sequencer found, skipping.")
            continue
 
        seq_a, seq_b = sequencers[0], sequencers[1]
        comparison   = f"sequencer_{cancer_type}"
        n_metrics    = len(features)
 
        logging.info(f"\n  {cancer_type}: {seq_a} vs {seq_b}")
        logging.info(f"  n_{seq_a} = {(ct_df[SEQUENCER_COL]==seq_a).sum()}")
        logging.info(f"  n_{seq_b} = {(ct_df[SEQUENCER_COL]==seq_b).sum()}")
 
        comp_results = []
        for metric in features:
            result = test_two_groups(
                group_a       = ct_df.loc[ct_df[SEQUENCER_COL]==seq_a, metric],
                group_b       = ct_df.loc[ct_df[SEQUENCER_COL]==seq_b, metric],
                label_a       = seq_a,
                label_b       = seq_b,
                metric        = metric,
                n_comparisons = n_metrics,
            )
            result["comparison"] = comparison
            result["cancer_type_tested"] = cancer_type
            comp_results.append(result)
            all_results.append(result)
 
            logging.info(
                f"    {metric:<45} "
                f"p={result['corrected_p']:.4f}  "
                f"r={result['effect_r']:.3f} ({result['effect_interp']})  "
                f"→ {result['recommendation']}"
            )
 
            # Plots
            plot_boxplot(
                ct_df, metric, SEQUENCER_COL,
                title=f"{metric} — {cancer_type}: sequencer comparison",
                out_path=os.path.join(out_dir, "boxplots", f"{comparison}_{metric}.png"),
            )
            plot_kde(
                ct_df, metric, SEQUENCER_COL,
                title=f"{metric} — {cancer_type}: sequencer comparison",
                out_path=os.path.join(out_dir, "kde_plots", f"{comparison}_{metric}.png"),
            )
 
        # Heatmap
        comp_df = pd.DataFrame(comp_results)
        plot_effect_heatmap(
            comp_df,
            comparison_label=f"{cancer_type}: {seq_a} vs {seq_b}",
            out_path=os.path.join(out_dir, "heatmaps", f"{comparison}_heatmap.png"),
        )
 
        # Overall recommendation
        n_do_not_pool = (comp_df["recommendation"] == "do not pool").sum()
        n_review      = (comp_df["recommendation"] == "review").sum()
        logging.info(f"\n  Summary for {cancer_type} sequencer comparison:")
        logging.info(f"    Safe to pool:   {(comp_df['recommendation'].str.startswith('pool')).sum()} / {n_metrics}")
        logging.info(f"    Review:         {n_review} / {n_metrics}")
        logging.info(f"    Do not pool:    {n_do_not_pool} / {n_metrics}")
        if n_do_not_pool == 0 and n_review == 0:
            logging.info(f"    ✓ RECOMMENDATION: pool {seq_a} and {seq_b} for {cancer_type}")
        elif n_do_not_pool == 0:
            metrics_to_review = comp_df[comp_df["recommendation"]=="review"]["metric"].tolist()
            logging.info(f"    ~ RECOMMENDATION: likely poolable — review: {metrics_to_review}")
        else:
            do_not_pool_metrics = comp_df[comp_df["recommendation"]=="do not pool"]["metric"].tolist()
            logging.info(f"    ✗ RECOMMENDATION: do NOT pool — differing metrics: {do_not_pool_metrics}")
 
    # ── Final summary ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_results)
    out_path   = os.path.join(out_dir, "assess_grouping_summary.csv")
    os.makedirs(out_dir, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    logging.info(f"\nFull summary saved to {out_path}")
 
    return summary_df
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import config
    from pipeline.preprocess import load_data
 
    df = load_data(config.SUMMARY_QC_METRICS)
 
    summary = run_grouping_assessment(
        df=df,
        out_dir=os.path.join(config.PREPROCESSING_PLOT_DIR, "grouping_assessment"),
    )
 
    # Quick readable table of anything flagged as do-not-pool or review
    flagged = summary[summary["recommendation"].isin(["do not pool", "review"])]
    if flagged.empty:
        logging.info("\n✓ All metrics safe to pool across all tested comparisons.")
    else:
        logging.info(
            f"\nMetrics requiring attention:\n"
            f"{flagged[['comparison','metric','corrected_p','effect_r','recommendation']].to_string(index=False)}"
        )