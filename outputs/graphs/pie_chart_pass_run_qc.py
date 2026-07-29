""" 
This script generates a pie chart showing the proportion of runs that pass and fail run level QC metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/filtered_run_metrics.csv", sep="\t")

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
plt.savefig("outputs/pie_chart_pass_run_qc.png", dpi=300)
plt.show()
plt.savefig('outputs/pie_chart_pass_run_qc.png')