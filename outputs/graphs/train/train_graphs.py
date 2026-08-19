from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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
 
 
def plot_flagging_rates(
    flagging_summary: dict,
    out_dir: str,
) -> None:
    """
    Bar chart of flagging rates across all assays.
 
    Parameters
    ----------
    flagging_summary : dict
        Keys: assay names. Values: dict with keys 'n_total', 'n_flagged'.
        e.g. {"NovaSeq_lung": {"n_total": 500, "n_flagged": 25}, ...}
    """
    assays    = list(flagging_summary.keys())
    rates     = [
        100 * v["n_flagged"] / v["n_total"]
        for v in flagging_summary.values()
    ]
 
    fig, ax = plt.subplots(figsize=(max(6, len(assays) * 1.5), 5))
    bars = ax.bar(assays, rates, color="#4C72B0", edgecolor="white", width=0.6)
 
    for bar, rate, assay in zip(bars, rates, assays):
        n_flagged = flagging_summary[assay]["n_flagged"]
        n_total   = flagging_summary[assay]["n_total"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{n_flagged}/{n_total}",
            ha="center", va="bottom", fontsize=9,
        )
 
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Flagging rate")
    ax.set_xlabel("Assay")
    ax.set_title("Isolation Forest flagging rates by assay (test set)")
    ax.set_ylim(0, max(rates) * 1.25 + 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
 
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "flagging_rates_by_assay.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Flagging rate plot saved to {out_path}")
 
 
def plot_top_deviant_features(
    explanations: pd.DataFrame,
    assay: str,
    out_dir: str,
    top_n: int = 10,
) -> None:
    """
    Bar chart showing which features most frequently appear in the top
    deviant features across all flagged samples for an assay.
    Helps identify which QC metrics are driving the most flags.
    """
    if explanations.empty:
        return
 
    feature_counts = (
        explanations.groupby("feature")["robust_z_score"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
 
    fig, ax = plt.subplots(figsize=(10, 5))
    feature_counts.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Mean robust z-score across flagged samples")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top deviant features in flagged samples — {assay}")
    ax.invert_yaxis()
    plt.tight_layout()
 
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{assay}_top_deviant_features.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"[{assay}] Top deviant features plot saved to {out_path}")
 
 
# ── 6. Evaluate — with labels (optional) ─────────────────────────────────────
 
def evaluate_with_labels(
    X_scored: pd.DataFrame,
    labels: pd.Series,
    assay: str,
    out_dir: str,
) -> dict:
    """
    If you have manually labelled known-bad runs, compute AUPRC and plot
    a precision-recall curve. This is optional — only call if you have labels.
 
    Parameters
    ----------
    X_scored : pd.DataFrame
        Output of score_samples() — must contain 'anomaly_score'.
    labels : pd.Series
        Ground truth. 1 = known bad run, 0 = known good run.
        Must be aligned to X_scored by index.
        Even a small set (20-30 samples) is useful.
    assay : str
        Assay label.
    out_dir : str
        Directory to save the PR curve plot.
 
    Returns
    -------
    dict
        Contains 'auprc' and 'n_labelled'.
    """
    # Isolation Forest scores: more negative = more anomalous
    # Invert so that higher score = more likely to be a bad run,
    # which is what precision-recall expects
    scores = -X_scored.loc[labels.index, "anomaly_score"]
 
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = average_precision_score(labels, scores)
 
    logging.info(
        f"[{assay}] AUPRC = {auprc:.4f} | "
        f"n_labelled = {len(labels)} | "
        f"n_known_bad = {labels.sum()} | "
        f"n_known_good = {(labels == 0).sum()}"
    )
 
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay(precision=precision, recall=recall).plot(
        ax=ax,
        name=f"Isolation Forest (AUPRC = {auprc:.3f})",
        color="#4C72B0",
    )
    # Baseline: random classifier at the positive rate
    baseline = labels.mean()
    ax.axhline(baseline, linestyle="--", color="grey", label=f"Baseline ({baseline:.2f})")
    ax.set_title(f"Precision-Recall curve — {assay}")
    ax.legend()
    plt.tight_layout()
 
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{assay}_precision_recall.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"[{assay}] PR curve saved to {out_path}")
 
    return {"auprc": auprc, "n_labelled": len(labels)}
 