#!/usr/bin/env python3
"""
best_model_report.py  -  confusion matrix + top-10 features for one model.

Usage:  python3 best_model_report.py dataset_v2.csv "XGBoost" [--z]
        --z  also use per-student normalised features (stage 5)
Writes: confusion_matrix.png, feature_importance.png, feature_importance.csv
"""
import sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, f1_score
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from evaluate_step2 import models, per_student_z, META, STATES
warnings.filterwarnings("ignore")

path, model_name = sys.argv[1], sys.argv[2]
use_z = "--z" in sys.argv
d = pd.read_csv(path)
y = d.label3.map({s: i for i, s in enumerate(STATES)}).values
g = d.participant
cols = [c for c in d.columns if c not in META and c != "gaze_samples"]  # gaze_samples = duration proxy
X = d[cols]
if use_z:
    X = pd.concat([X, per_student_z(d, cols, g)], axis=1)
names, Xv = list(X.columns), X.astype(float).values

pred = np.empty(len(y), dtype=int)
imps = []
for tr, te in LeaveOneGroupOut().split(Xv, y, g):
    m = models()[model_name]
    if model_name == "XGBoost":
        m.fit(Xv[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
    else:
        m.fit(Xv[tr], y[tr])
    pred[te] = m.predict(Xv[te])
    if hasattr(m, "feature_importances_"):
        imps.append(m.feature_importances_)

print(f"{model_name}{' + per-student normalisation' if use_z else ''}: "
      f"macro-F1 {f1_score(y, pred, average='macro'):.3f}")
INK, MUTED, GRID = "#1A1A1A", "#595959", "#D9D9D9"
LABELS = ["Locating", "Understanding", "Reasoning"]

# ---- confusion matrix (row-normalised, counts shown) ----
cm = confusion_matrix(y, pred, labels=[0, 1, 2])
pct = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center", fontsize=12,
                color="white" if pct[i, j] > 0.55 else INK)
ax.set_xticks(range(3), LABELS, fontsize=11, color=INK)
ax.set_yticks(range(3), LABELS, fontsize=11, color=INK)
ax.set_xlabel("Predicted state", fontsize=12, color=MUTED)
ax.set_ylabel("Self-reported state", fontsize=12, color=MUTED)
ax.set_title(f"Confusion matrix: {model_name}", fontsize=13, color=INK, pad=12)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig("confusion_matrix.png")
print(cm)

# ---- feature importance (mean over the 47 folds) ----
if imps:
    imp = pd.Series(np.mean(imps, axis=0), index=names).sort_values(ascending=False)
    imp.to_csv("feature_importance.csv", header=["importance"])
    NICE = {"total_time_s": "Total time", "stroke_count": "Pen strokes", "pen_down_time_s": "Pen-down time",
            "fix_count": "Number of fixations", "ink_length_px": "Ink length", "writing_phase_s": "Writing-phase length",
            "lb_passage_per_min": "Look-backs to passage while writing (/min)", "pause_total_s": "Total pause time (>1 s)",
            "pr_ink_per_stroke": "Ink per stroke", "pressure_mean": "Mean pen pressure",
            "time_to_first_stroke_s": "Time to first stroke", "pause_count_gt1s": "Pauses > 1 s",
            "fix_rate_per_s": "Fixations per second", "pupil_mean_mm": "Pupil size"}
    top = imp.head(10)[::-1]
    top.index = [NICE.get(i, i) for i in top.index]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    ax.barh(range(len(top)), top.values, color="#2F6DB5", height=0.6)
    ax.set_yticks(range(len(top)), top.index, fontsize=10, color=INK)
    ax.set_xlabel("Importance (mean over 47 test rounds)", fontsize=11, color=MUTED)
    ax.set_title(f"Top 10 features: {model_name}", fontsize=13, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig("feature_importance.png")
    print(imp.head(10).round(4).to_string())
