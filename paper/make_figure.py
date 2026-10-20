"""Figure 1: paired differences (decision-trained minus same untrained base) in macro-F1 and ECE, zero-shot.

    python paper/make_figure.py   -> paper/fig1_paired.png (+ .pdf)
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
df = pd.read_csv(HERE.parent / "runs" / "paired.csv")
df = df[df.kind == "primary"]

PAIRS = [("jeff-0.8b", "Jeff 0.8B - Qwen3.5 0.8B"), ("jeff-2b", "Jeff 2B - Qwen3.5 2B"),
         ("openjev-fp8", "OpenJev* - Qwen3.8 27B")]  # *: own prompt format and calibration (caption)
TASKS = [("medabs", "Medical Abstracts"), ("hoc", "HoC"), ("cxr", "MIMIC-CXR")]
K = {0: dict(color="#2a78d6", marker="o", mfc="#2a78d6", label="zero-shot (k=0)", dy=0.0)}  # one series: no legend
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"axes.unicode_minus": False, "font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.size": 0})
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True, gridspec_kw={"wspace": 0.08})

rows, labels, y = [], [], 0
for task, tname in TASKS:
    for model, pname in PAIRS:
        rows.append((task, model, y))
        labels.append((y, pname))
        y += 1
    y += 0.6  # gap between task groups

for ax, metric, title in ((axes[0], "f1", "Δ F1-score\n> 0: System One model better"),
                          (axes[1], "ece", "Δ ECE\n< 0: System One model better")):
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    for task, model, yy in rows:
        for k, st in K.items():
            r = df[(df.task == task) & (df.a == model) & (df.k == k)].iloc[0]
            ax.plot([r[f"{metric}_lo"], r[f"{metric}_hi"]], [yy + st["dy"]] * 2, color=st["color"], lw=1.6,
                    solid_capstyle="round", zorder=2)
            ax.plot(r[metric], yy + st["dy"], marker=st["marker"], ms=4.2, mfc=st["mfc"], mec=st["color"], mew=1.2,
                    ls="none", zorder=3, label=st["label"] if (task, model) == ("medabs", "jeff-0.8b") else None)
    ax.set_title(title, fontsize=7.5, color=INK, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

axes[0].set_yticks([t for t, _ in labels], [n for _, n in labels])
axes[0].invert_yaxis()
for (task, tname), first in zip(TASKS, range(0, 9, 3)):
    yy = rows[first][2] - 0.75
    axes[0].text(-0.02, yy, tname, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                 fontweight="bold", color=INK)
fig.subplots_adjust(left=0.27, right=0.98, top=0.84, bottom=0.1)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig1_paired.{ext}", dpi=300)
print("wrote", HERE / "fig1_paired.png")
