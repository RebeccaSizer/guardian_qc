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
    effect_size_heatmap_all_comparisons.png
                                    One heatmap summarising effect sizes across
                                    all tested comparisons
 
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
 
 
# ── Combined plotting ─────────────────────────────────────────────────────────

def plot_combined_comparison(
    df: pd.DataFrame,
    features: list,
    group_col: str,
    comparison_label: str,
    results_df: pd.DataFrame,
    out_path: str,
) -> None:
    """Save all metric plots for one comparison in a single multi-panel figure.

    Each row is a QC metric. Columns show its group-wise boxplot, KDE
    distributions, and rank-biserial effect size from the statistical test.
    """
    groups = sorted(df[group_col].dropna().unique())
    if not groups:
        logging.warning("No groups available for %s; skipping plot.", comparison_label)
        return

    colours = sns.color_palette("Set2", max(len(groups), 2))
    result_by_metric = results_df.set_index("metric").to_dict("index")
    n_metrics = len(features)
    fig_height = max(12, n_metrics * 1.65)

    fig, axes = plt.subplots(
        n_metrics, 3,
        figsize=(17, fig_height),
        gridspec_kw={"width_ratios": [1.0, 1.35, 0.9]},
        squeeze=False,
    )
    fig.suptitle(
        f"{comparison_label}: QC metric distributions and effect sizes",
        fontsize=16, fontweight="bold", y=0.999,
    )

    for row_idx, metric in enumerate(features):
        ax_box, ax_kde, ax_effect = axes[row_idx]

        # Boxplot: compare group distributions for this metric.
        group_data = [
            pd.to_numeric(df.loc[df[group_col] == group, metric], errors="coerce").dropna()
            for group in groups
        ]
        nonempty_data = [values for values in group_data if len(values) > 0]
        nonempty_positions = [i + 1 for i, values in enumerate(group_data) if len(values) > 0]
        if nonempty_data:
            bp = ax_box.boxplot(
                nonempty_data, positions=nonempty_positions, patch_artist=True,
                widths=0.6, medianprops={"color": "black", "linewidth": 1.2},
                showfliers=False,
            )
            for patch, pos in zip(bp["boxes"], nonempty_positions):
                patch.set_facecolor(colours[(pos - 1) % len(colours)])
                patch.set_alpha(0.7)
        ax_box.set_xticks(range(1, len(groups) + 1))
        ax_box.set_xticklabels(
            [f"{g}\n(n={len(values)})" for g, values in zip(groups, group_data)],
            fontsize=7, rotation=20, ha="right",
        )
        ax_box.set_ylabel(metric, fontsize=8)
        ax_box.tick_params(axis="y", labelsize=7)
        ax_box.grid(axis="y", alpha=0.2)

        # KDE: overlay group distributions where there are enough observations.
        plotted_kde = False
        for group_idx, (group, values) in enumerate(zip(groups, group_data)):
            if len(values) >= 5 and values.nunique() > 1:
                try:
                    sns.kdeplot(
                        x=values, ax=ax_kde, label=f"{group} (n={len(values)})",
                        fill=False, linewidth=1.4,
                        color=colours[group_idx % len(colours)],
                    )
                    plotted_kde = True
                except (ValueError, np.linalg.LinAlgError):
                    logging.debug("KDE could not be estimated for %s / %s", metric, group)
        if row_idx == 0 and plotted_kde:
            ax_kde.legend(fontsize=7, loc="best")
        ax_kde.set_ylabel("Density", fontsize=7)
        ax_kde.set_xlabel("")
        ax_kde.tick_params(axis="both", labelsize=7)
        ax_kde.grid(alpha=0.2)

        # Effect size panel: rank-biserial r and corrected p-value.
        result = result_by_metric.get(metric, {})
        effect = result.get("effect_r", np.nan)
        corrected_p = result.get("corrected_p", np.nan)
        recommendation = result.get("recommendation", "not tested")
        if pd.notna(effect):
            bar_colour = (
                "#2b8cbe" if recommendation.startswith("pool")
                else "#fdae61" if recommendation == "review"
                else "#d7191c" if recommendation == "do not pool"
                else "#999999"
            )
            ax_effect.barh([0], [effect], color=bar_colour, height=0.5)
            ax_effect.text(
                min(float(effect) + 0.02, 0.82), 0,
                f"r={effect:.2f}", va="center", fontsize=7,
            )
            p_text = f"adjusted p={corrected_p:.3g}" if pd.notna(corrected_p) else "adjusted p=NA"
            ax_effect.text(0.02, 0.78, p_text, transform=ax_effect.transAxes, fontsize=7)
            ax_effect.text(
                0.02, 0.57, recommendation, transform=ax_effect.transAxes,
                fontsize=7, wrap=True,
            )
        else:
            ax_effect.text(0.05, 0.5, "Insufficient data", transform=ax_effect.transAxes, fontsize=7)
        ax_effect.set_xlim(0, 1)
        ax_effect.set_yticks([])
        ax_effect.set_xlabel("Effect size (|r|)", fontsize=7)
        ax_effect.tick_params(axis="x", labelsize=7)
        ax_effect.grid(axis="x", alpha=0.2)

        if row_idx == 0:
            ax_box.set_title("Boxplots by group", fontsize=10)
            ax_kde.set_title("Overlaid KDE distributions", fontsize=10)
            ax_effect.set_title("Statistical effect size", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.997], h_pad=0.65, w_pad=1.0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logging.info("Combined comparison figure saved: %s", out_path)



def plot_all_comparisons_heatmap(
    results_df: pd.DataFrame,
    out_path: str,
) -> None:
    """Create one heatmap of absolute rank-biserial effect sizes across comparisons.

    Rows are QC metrics; columns are pipeline-version and sequencer comparisons.
    A '*' marks a Bonferroni-adjusted p-value below ALPHA.
    """
    required = {"metric", "comparison", "effect_r", "corrected_p"}
    missing = required.difference(results_df.columns)
    if missing:
        logging.warning(
            "Cannot create overall effect-size heatmap; missing columns: %s",
            sorted(missing),
        )
        return

    data = results_df.copy()
    data["effect_r"] = pd.to_numeric(data["effect_r"], errors="coerce")
    data["corrected_p"] = pd.to_numeric(data["corrected_p"], errors="coerce")

    # Keep metrics in the same order as NUMERIC_FEATURES where possible.
    effect_matrix = data.pivot_table(
        index="metric", columns="comparison", values="effect_r", aggfunc="first"
    )
    p_matrix = data.pivot_table(
        index="metric", columns="comparison", values="corrected_p", aggfunc="first"
    )

    ordered_metrics = [m for m in NUMERIC_FEATURES if m in effect_matrix.index]
    ordered_metrics += [m for m in effect_matrix.index if m not in ordered_metrics]
    effect_matrix = effect_matrix.reindex(index=ordered_metrics)
    p_matrix = p_matrix.reindex(index=ordered_metrics, columns=effect_matrix.columns)

    # More readable comparison labels for the x-axis.
    def readable_label(value):
        return (
            value.replace("pipeline_version_", "Pipeline versions:\n")
                 .replace("sequencer_", "Sequencers:\n")
                 .replace("_", " ")
        )

    effect_matrix.columns = [readable_label(str(c)) for c in effect_matrix.columns]
    p_matrix.columns = effect_matrix.columns

    n_rows, n_cols = effect_matrix.shape
    if n_rows == 0 or n_cols == 0:
        logging.warning("No valid results available for the overall heatmap.")
        return

    fig_width = max(10, 2.0 * n_cols + 5)
    fig_height = max(8, 0.34 * n_rows + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Effect sizes are absolute rank-biserial r values (0 to 1).
    sns.heatmap(
        effect_matrix,
        ax=ax,
        cmap=sns.diverging_palette(130, 10, as_cmap=True),
        vmin=0,
        vmax=1,
        annot=False,
        linewidths=0.45,
        linecolor="white",
        mask=effect_matrix.isna(),
        cbar_kws={"label": "Absolute rank-biserial effect size (|r|)"},
    )

    # Annotate effect sizes; * indicates adjusted p < ALPHA.
    for row_idx, metric in enumerate(effect_matrix.index):
        for col_idx, comparison in enumerate(effect_matrix.columns):
            value = effect_matrix.loc[metric, comparison]
            if pd.isna(value):
                continue
            p_value = p_matrix.loc[metric, comparison]
            star = "*" if pd.notna(p_value) and p_value < ALPHA else ""
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                f"{value:.2f}{star}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    ax.set_title(
        "QC metric effect sizes across pipeline-version and sequencer comparisons",
        fontsize=13,
        pad=14,
    )
    ax.set_xlabel("")
    ax.set_ylabel("QC metric")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.text(
        0.01, 0.01,
        f"* Bonferroni-adjusted p < {ALPHA}. "
        "Colour and cell values show effect size; significance should be interpreted alongside effect size.",
        ha="left", va="bottom", fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Overall effect-size heatmap saved: %s", out_path)


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
 
        comp_df = pd.DataFrame(comp_results)
 
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
 
        comp_df = pd.DataFrame(comp_results)
 
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

    # One summary heatmap for all tested comparisons.
    plot_all_comparisons_heatmap(
        results_df=summary_df,
        out_path=os.path.join(out_dir, "effect_size_heatmap_all_comparisons.png"),
    )

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