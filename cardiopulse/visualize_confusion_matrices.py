"""Confusion matrices for the representative uniform-overlap model and the
deployed class-specific-overlap model, stacked vertically.
Values are from PCG_Report_Visualizations_FINAL.ipynb (cell 24)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ["Normal", "Murmur", "Extrasystole", "Artifact"]
cm_uniform = np.array([[187, 19, 28, 8], [39, 45, 0, 0], [11, 0, 5, 0], [1, 0, 0, 35]])
cm_class_specific = np.array([[151, 11, 61, 19], [16, 56, 9, 3], [15, 0, 32, 0], [1, 1, 0, 34]])
vmax = max(cm_uniform.max(), cm_class_specific.max())

fig, axes = plt.subplots(2, 1, figsize=(8.2, 12.6))
panels = [(cm_uniform, "Representative Uniform-Overlap Model (YAMNet weight = 0.50)"),
          (cm_class_specific, "Deployed Class-Specific-Overlap Model (YAMNet weight = 0.95)")]
for ax, (cm, title) in zip(axes, panels):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", vmin=0, vmax=vmax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                annot_kws={"size": 18}, cbar_kws={"shrink": 0.9})
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel("Predicted class", fontsize=16)
    ax.set_ylabel("True class", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_yticklabels(CLASS_NAMES, rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=13)
fig.tight_layout(h_pad=3.0)
fig.savefig("images/fig_confusion_matrices_overlap_comparison_stacked.png", dpi=200, bbox_inches="tight")
print("saved")
