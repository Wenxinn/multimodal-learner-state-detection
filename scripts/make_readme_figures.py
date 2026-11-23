"""Regenerates the README charts in docs/figures/ from the aggregate result CSVs.
Usage: python3 scripts/make_readme_figures.py   (run from the repo root)"""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R = sys.argv[1] if len(sys.argv) > 1 else "."; OUT = R + "/docs/figures/"
SURF="#ffffff"; T1="#0b0b0b"; T2="#52514e"; GRID="#e6e5e0"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; GRAY="#a3a29b"
plt.rcParams.update({"font.family":"Helvetica Neue","font.size":11,"axes.edgecolor":GRID,
  "axes.labelcolor":T2,"xtick.color":T2,"ytick.color":T2,"text.color":T1,
  "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF})
def style(ax, grid_axis="y"):
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)
def title(fig, t, sub):
    H=fig.get_figheight()
    fig.text(0.02, 1-0.18/H, t, fontsize=15, weight="bold", ha="left", va="top")
    fig.text(0.02, 1-0.5/H, sub, fontsize=10.5, color=T2, ha="left", va="top")
def rbar(ax, x, h, w, color, horiz=False):
    # bar with rounded data-end (approx via FancyBbox), square at baseline
    r=0.02
    if not horiz:
        ax.add_patch(FancyBboxPatch((x-w/2, 0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=color, ec=SURF, lw=1.5, mutation_aspect=1))
        ax.add_patch(plt.Rectangle((x-w/2,0), w, min(h, r*1.2), fc=color, ec="none"))
s = pd.read_csv(R+"/analysis_v3/results_v3_summary.csv")

# ---------- 1. model comparison ----------
setups=[("flat-v2","Flat\n49 features"),("flat-v3","Flat\n94 features"),("hier-v3","Two-stage\n94 features"),
        ("hier-v3-gaze","Two-stage\ngaze only"),("hier-v3-pen","Two-stage\npen only")]
models=[("Random Forest",BLUE),("XGBoost",ORANGE),("LightGBM",AQUA)]
fig, ax = plt.subplots(figsize=(10,5.4)); fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.98)
w=0.25
for i,(m,c) in enumerate(models):
    for j,(k,_) in enumerate(setups):
        v=s[(s.setup==k)&(s.model==m)].macro_F1.iloc[0]
        sd=s[(s.setup==k)&(s.model==m)].macro_F1_sd.iloc[0]
        ax.bar(j+(i-1)*w, v, w-0.03, color=c, label=m if j==0 else None)
        ax.errorbar(j+(i-1)*w, v, yerr=sd, color=T1, lw=1, capsize=0)
        if (k,m)==("hier-v3","LightGBM"):
            ax.annotate(f"best  {v:.2f}", (j+(i-1)*w, v+0.012), ha="center", va="bottom", fontsize=10, weight="bold")
ax.axhline(0.505, color=T2, ls=(0,(4,3)), lw=1.3)
ax.text(4.55, 0.512, "Rule-based baseline 0.51", ha="right", va="bottom", color=T2, fontsize=10)
ax.set_xticks(range(len(setups))); ax.set_xticklabels([l for _,l in setups])
ax.set_ylim(0,0.6); ax.set_ylabel("Macro-F1 (3 classes)"); style(ax)
ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0,1.08), handlelength=1, handleheight=1)
title(fig,"Model comparison","Leave-one-participant-out, mean of 5 seeds (error bars: SD across seeds)")
fig.savefig(OUT+"model_comparison.png", dpi=200); plt.close(fig)

# ---------- 2. per-state F1: baseline vs best ----------
states=["Locating","Understanding","Reasoning"]
base=[0.836,0.0,0.678]
b=s[(s.setup=="hier-v3")&(s.model=="LightGBM")].iloc[0]
best=[b.F1_locating,b.F1_understanding,b.F1_reasoning]
fig, ax = plt.subplots(figsize=(10,4.6)); fig.subplots_adjust(top=0.74, bottom=0.12, left=0.15, right=0.97)
y=np.arange(3); h=0.36
ax.barh(y-h/2, base, h-0.04, color=GRAY, label="Rule-based baseline (most common state per question position)")
ax.barh(y+h/2, best, h-0.04, color=BLUE, label="Two-stage LightGBM (gaze + pen)")
for yi,v in zip(y-h/2,base): ax.text(v+0.01, yi, f"{v:.2f}", va="center", color=T2, fontsize=10)
for yi,v in zip(y+h/2,best): ax.text(v+0.01, yi, f"{v:.2f}", va="center", color=T1, fontsize=10, weight="bold")
ax.set_yticks(y); ax.set_yticklabels(states); ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel("F1 score")
style(ax,"x"); ax.spines["bottom"].set_visible(False)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.13,1.2), ncol=1, handlelength=1, fontsize=10)
title(fig,"Where the sensor model helps","The baseline never predicts 'understanding'; gaze and pen features recover part of it")
fig.savefig(OUT+"per_state_f1.png", dpi=200); plt.close(fig)

# ---------- 3. feature importance, two stages ----------
names={"total_time_s":"Time on sub-question","m_gaze_pos_sd":"Gaze position spread","lb_passage_per_min":"Look-backs to passage",
"fix_count":"Fixation count","stroke_dur_mean_ms":"Mean stroke duration","pen_down_time_s":"Pen-down time","m_blink_dur_max":"Longest blink",
"pressure_mean":"Mean pen pressure","m_pen_pos_sd":"Writing position spread","pr_ink_per_stroke":"Ink per stroke","stroke_count":"Stroke count",
"m_pupil_max":"Max pupil diameter","dwell_question_pct":"Gaze dwell on question (%)","pre_fix_rate_per_s":"Fixation rate before writing",
"m_fixation_dur_sd":"Fixation duration SD","rd_coverage_pct":"Passage reading coverage"}
gaze_keys={"m_gaze_pos_sd","lb_passage_per_min","fix_count","m_blink_dur_max","m_pupil_max","dwell_question_pct","pre_fix_rate_per_s","m_fixation_dur_sd","rd_coverage_pct"}
pen_keys={"stroke_dur_mean_ms","pen_down_time_s","pressure_mean","m_pen_pos_sd","pr_ink_per_stroke","stroke_count"}
def col(k): return BLUE if k in gaze_keys else (ORANGE if k in pen_keys else GRAY)
fig, axes = plt.subplots(1,2, figsize=(11,5)); fig.subplots_adjust(top=0.72, bottom=0.1, left=0.2, right=0.97, wspace=0.85)
for ax,(f,t) in zip(axes,[("importance_stage1_lightgbm.csv","Stage 1: locating vs. other"),("importance_stage2_lightgbm.csv","Stage 2: understanding vs. reasoning")]):
    d=pd.read_csv(R+"/analysis_v3/"+f, index_col=0).gain_share.head(8)
    ax.barh(range(8), d.values*100, 0.7, color=[col(k) for k in d.index])
    ax.set_yticks(range(8)); ax.set_yticklabels([names.get(k,k) for k in d.index]); ax.invert_yaxis()
    for i,v in enumerate(d.values*100): ax.text(v+0.15, i, f"{v:.1f}%", va="center", fontsize=9.5, color=T2)
    ax.set_title(t, loc="left", fontsize=11.5, weight="bold", pad=10); ax.set_xlim(0, d.values.max()*100*1.25)
    style(ax,"x"); ax.spines["bottom"].set_visible(False); ax.set_xticks([])
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BLUE,label="Gaze"),Patch(color=ORANGE,label="Pen"),Patch(color=GRAY,label="Timing")],
           ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0.012,0.87), handlelength=1, handleheight=1)
title(fig,"What the model looks at","Top 8 features by LightGBM gain share in each stage of the two-stage classifier")
fig.savefig(OUT+"feature_importance_stages.png", dpi=200); plt.close(fig)

# ---------- 4. dataset composition ----------
fig, ax = plt.subplots(figsize=(10,2.4)); fig.subplots_adjust(top=0.6, bottom=0.08, left=0.02, right=0.98)
vals=[("Locating",539,BLUE),("Understanding",175,ORANGE),("Reasoning",180,AQUA)]; left=0
for n,v,c in vals:
    ax.barh(0, v, 0.6, left=left, color=c, edgecolor=SURF, lw=2)
    ax.text(left+v/2, 0, f"{n}\n{v}  ({v/894:.0%})", ha="center", va="center", color="white", fontsize=10.5, weight="bold")
    left+=v
ax.set_xlim(0,894); ax.axis("off")
title(fig,"Dataset: 894 sub-questions from 47 participants","Self-reported dominant state per sub-question; the classes are imbalanced")
fig.savefig(OUT+"dataset_composition.png", dpi=200); plt.close(fig)
print("ok")
