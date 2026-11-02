"""Draw the revised streaming-ESM timing architecture."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


OUT = Path("results/architecture")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#1674B8", "blue_fill": "#E6F3FC",
    "green": "#2D9B47", "green_fill": "#E9F7E9",
    "orange": "#E58A00", "orange_fill": "#FFF4DF",
    "purple": "#7652B5", "purple_fill": "#F1ECFB",
    "teal": "#008F95", "teal_fill": "#E3F7F6",
    "magenta": "#B33A83", "magenta_fill": "#FBEAF5",
    "red": "#D73B3E", "red_fill": "#FDEBEC",
    "navy": "#0C2748", "grid": "#AAB4C0", "text": "#10243E",
}


def rounded(ax, x, y, w, h, face, edge, text, *, size=7.4, weight="normal",
            linestyle="-", z=3):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.012", facecolor=face,
        edgecolor=edge, linewidth=1, linestyle=linestyle, zorder=z)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=COLORS["text"], weight=weight, zorder=z + 1)
    return patch


def vertical_arrow(ax, x, y0, y1, color):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle="-|>", lw=1.15, color=color), zorder=5)


fig, ax = plt.subplots(figsize=(18, 10.2), facecolor="white")
ax.set_xlim(-.37, 1.54)
ax.set_ylim(-.18, 8.25)
ax.axis("off")

lanes = [
    (7, "1. PDW Stream", "continuous input from\nES receiver", "blue"),
    (6, "2. HDBSCAN Discovery", "batch clustering / recovery", "green"),
    (5, "3. Incremental Groups", "fast PDW-to-group\nassociation", "orange"),
    (4, "4. Group Events", "new · update · inactive\n· reappeared", "purple"),
    (3, "5. Pulse-Train Association", "PRI · phase · jitter\n· missed pulses", "teal"),
    (2, "6. Cross-RF / Mode Association", "train families +\ncompeting hypotheses", "magenta"),
    (1, "7. Mode Prediction / Filtering", "per-emitter model bank", "red"),
    (0, "8. Emitter Track Manager", "state · continuity · confidence", "red"),
]

for y, title, subtitle, color in lanes:
    fill, edge = COLORS[f"{color}_fill"], COLORS[color]
    ax.add_patch(Rectangle((-.36, y + .05), 1.89, .83,
                           facecolor=fill, edgecolor="none", alpha=.72, zorder=0))
    ax.add_patch(FancyBboxPatch((-.355, y + .12), .335, .68,
        boxstyle="round,pad=0.012,rounding_size=0.012", facecolor=fill,
        edgecolor=edge, linewidth=1, zorder=3))
    ax.text(-.1875, y + .57, title, ha="center", va="center",
            fontsize=8.3, color=COLORS["text"], weight="bold", zorder=4)
    ax.text(-.1875, y + .34, subtitle, ha="center", va="center",
            fontsize=7.1, color=COLORS["text"], zorder=4)

# Per-emitter state boundary around the two tracker-owned lanes.
ax.add_patch(FancyBboxPatch((-.012, .03), 1.27, 1.88,
    boxstyle="round,pad=0.015,rounding_size=0.02", facecolor="none",
    edgecolor=COLORS["red"], linewidth=1.2, linestyle="--", zorder=1))
ax.text(.008, 1.84, "PER-EMITTER TRACKING STATE", fontsize=7.5,
        weight="bold", color=COLORS["red"], zorder=5)

# Shared time grid.
for t in np.arange(0, 1.21, .1):
    ax.plot([t, t], [.02, 7.89], color=COLORS["grid"], lw=.65,
            alpha=.5, linestyle="--", zorder=0)
    ax.text(t, -.08, f"{t:.1f}", ha="center", va="top", fontsize=7.5,
            color=COLORS["text"])
ax.annotate("", xy=(1.27, -.05), xytext=(-.01, -.05),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["text"], lw=1.1))
ax.text(-.355, -.05, "Time (seconds)", ha="left", va="center", fontsize=8.5)

# 1. Continuous PDW stream.
rng = np.random.default_rng(46)
pdw_x = np.sort(rng.uniform(0, 1.2, 285))
gaps = ((pdw_x > .48) & (pdw_x < .535)) | ((pdw_x > .97) & (pdw_x < 1.015))
pdw_x = pdw_x[~gaps]
heights = rng.uniform(.18, .38, len(pdw_x))
ax.vlines(pdw_x, 7.28, 7.28 + heights, color=COLORS["blue"], lw=.9, zorder=3)
ax.text(.60, 7.73, "Individual PDWs arrive continuously (each tick = one PDW)",
        ha="center", va="center", fontsize=7.7, color=COLORS["navy"])

# 2. HDBSCAN windows.
for i, start in enumerate([.02, .32, .62, .92], 1):
    rounded(ax, start, 6.18, .275, .48, COLORS["green_fill"], COLORS["green"],
            f"Discovery window {i}\n10,000 accumulated PDWs", size=7.3, weight="bold")
    ax.annotate("", xy=(start + .275, 6.72), xytext=(start, 6.72),
                arrowprops=dict(arrowstyle="<->", color=COLORS["green"], lw=1))

# 3. Incremental group update blocks every 100 ms.
for t in np.arange(.02, 1.2, .1):
    rounded(ax, t, 5.31, .062, .18, COLORS["orange_fill"], COLORS["orange"], "",
            size=6)
    ax.annotate("", xy=(t + .031, 5.72), xytext=(t + .031, 5.51),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["orange"], lw=.9))
ax.text(.60, 5.18, "New PDWs gated continuously; group statistics published every 100 ms",
        ha="center", fontsize=7.3, color=COLORS["text"])

# 4. Significant group events.
events = [
    (.08, "New group\nG7"), (.29, "Update\nG7"), (.51, "G7\ninactive"),
    (.62, "New group\nG12"), (.86, "Update\nG12"), (1.07, "G7\nreappeared")]
for t, label in events:
    vertical_arrow(ax, t, 4.58, 4.78, COLORS["purple"])
    rounded(ax, t - .045, 4.18, .09, .36, COLORS["purple_fill"],
            COLORS["purple"], label, size=6.7)

# 5. Pulse-train association outputs.
train_boxes = [
    (.07, .17, "T7\nPRI 85.1 µs\nRMS 2.1 µs"),
    (.27, .17, "T7 update\n57 pulses\nscore .84"),
    (.58, .18, "T12\nPRI 85.0 µs\nscore .79"),
    (.82, .18, "T12 update\njitter model\nscore .86"),
    (1.04, .16, "T7 reacquired\nphase test\nscore .81"),
]
for x, w, label in train_boxes:
    rounded(ax, x, 3.17, w, .50, COLORS["teal_fill"], COLORS["teal"],
            label, size=6.8)
for x in [.12, .35, .67, .91, 1.11]:
    vertical_arrow(ax, x, 3.72, 3.96, COLORS["teal"])

# 6. Cross-RF and mode association.
family_boxes = [
    (.12, .22, "Candidate F1\nG7 + RF modes\nmerge vs separate"),
    (.52, .24, "F1 updated\nT7 + T12\nRF-hop evidence"),
    (.89, .24, "F1 maintained\n8 RF modes\nassociation .82"),
]
for x, w, label in family_boxes:
    rounded(ax, x, 2.18, w, .50, COLORS["magenta_fill"], COLORS["magenta"],
            label, size=6.9)
for x in [.20, .65, 1.00]:
    vertical_arrow(ax, x, 2.72, 2.96, COLORS["magenta"])

# 7. Per-track mode predictor / filter.
mode_boxes = [
    (.06, .18, "Predict E1\nnext TOA + RF"),
    (.28, .23, "Update model bank\nhopping .70\njittered .17"),
    (.57, .18, "Predict through\nsignal gap"),
    (.79, .23, "Update residuals\nRF + timing + AOA"),
    (1.06, .17, "Reacquisition\ngate passes"),
]
for x, w, label in mode_boxes:
    rounded(ax, x, 1.18, w, .50, COLORS["red_fill"], COLORS["red"],
            label, size=6.7)
for x in [.13, .39, .64, .91, 1.14]:
    vertical_arrow(ax, x, 1.72, 1.96, COLORS["red"])

# Prediction feedback to pulse association and group gating.
feedback = FancyArrowPatch((.74, 1.67), (.49, 3.18),
    connectionstyle="arc3,rad=-.34", arrowstyle="-|>", mutation_scale=10,
    linewidth=1.25, linestyle="--", color=COLORS["red"], zorder=6)
ax.add_patch(feedback)
ax.text(.47, 2.42, "prediction feedback\nTOA · RF · uncertainty", fontsize=7,
        color=COLORS["red"], ha="center", va="center", rotation=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.5), zorder=7)

# 8. Persistent emitter-track state.
track_boxes = [
    (.08, .17, "Create E1\nscore 89%"),
    (.31, .17, "Update E1\nscore 84%"),
    (.55, .18, "Coast E1\nscore 61%"),
    (.79, .19, "Maintain E1\nmulti-mode 82%"),
    (1.05, .18, "Reacquire E1\nscore 86%"),
]
for x, w, label in track_boxes:
    rounded(ax, x, .18, w, .48, "#FFF7F7", COLORS["red"], label, size=7)
for x in [.15, .39, .64, .89, 1.13]:
    vertical_arrow(ax, x, .70, .96, COLORS["red"])

# Explanatory notes at right.
notes = [
    (7.43, "Continuous sensor input"),
    (6.43, "Find new groups and recover\nassociations missed by predictions"),
    (5.43, "Fast measurement updates\nbetween discovery windows"),
    (4.43, "Event-driven lifecycle messages"),
    (3.43, "PDW → pulse-train confidence\nTiming residuals stay explicit"),
    (2.43, "Train → emitter-family confidence\nRetain competing hypotheses"),
    (1.43, "Predict and update fixed, jittered,\nstaggered and hopping models"),
    (.43, "Persistent identity, coasting,\nreacquisition and track confidence"),
]
for y, text in notes:
    ax.text(1.29, y, text, ha="left", va="center", fontsize=7.2,
            color=COLORS["text"])

fig.suptitle("Processing, Association and Tracking in a Streaming ESM",
             x=.5, y=.985, fontsize=18, weight="bold", color=COLORS["navy"])
ax.text(.635, 8.03,
        "Discovery is periodic; pulse association and emitter tracking update asynchronously as evidence arrives",
        ha="center", va="center", fontsize=9.5, color=COLORS["text"])

png = OUT / "streaming_esm_architecture_v2.png"
svg = OUT / "streaming_esm_architecture_v2.svg"
fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(svg, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(png)
print(svg)
