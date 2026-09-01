# ============================================================
#   Fig 1: All methods in lit review compared
# ============================================================
import matplotlib
matplotlib.rcParams["pdf.use14corefonts"] = True 
matplotlib.rcParams["font.family"] = "Helvetica"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(9.5, 6.5))

# ── Data ──────────────────────────────────────────────────────────────────────
# (prior_knowledge, guarantee_strength)
# x: 0=none, 0.25=source/target, 0.5=anchors, 0.75=env labels, 1.0=labels+anchors
# y: 0=none, 0.25=heuristic, 0.5=DRO bound, 0.75=coverage, 1.0=exact

methods = {
    "EIIL":        (0.00, 0.00, "#ff7744", "#c04020"),
    "HRM":         (0.04, 0.04, "#ff7744", "#c04020"),
    "ACR ★":       (0.00, 0.75, "#66bb66", "#2a7a40"),
    "DANN":        (0.25, 0.50, "#ffbb55", "#a05000"),
    "IRM":         (0.75, 0.25, "#ffbb55", "#a05000"),
    "REx / Fishr": (0.75, 0.20, "#ffbb55", "#a05000"),
    "GroupDRO":    (0.75, 0.50, "#ffbb55", "#a05000"),
    "CR":          (0.75, 0.72, "#bb99dd", "#5b4a8a"),
    "Anchor Reg.": (0.50, 0.75, "#bb99dd", "#5b4a8a"),
    "ICP":         (0.75, 1.00, "#bb99dd", "#5b4a8a"),
}

# label offsets (dx, dy) to avoid overlap
offsets = {
    "EIIL":        ( 0.035, -0.055),
    "HRM":         ( 0.035,  0.045),
    "ACR":         ( 0.045,  0.035),
    "DANN":        ( 0.035,  0.035),
    "IRM":         ( 0.035,  0.035),
    "REx / Fishr": ( 0.035, -0.055),
    "GroupDRO":    (-0.035,  0.045),
    "CR":          (-0.065,  0.045),
    "Anchor Reg.": ( 0.035,  0.035),
    "ICP":         ( 0.035,  0.035),
}

for name, (x, y, face, edge) in methods.items():
    is_acr = name == "ACR"
    if is_acr:                    
        ax.scatter(x, y, s=420, marker="*", color=face, edgecolors=edge, linewidths=1.6, zorder=4)
    else:
        ax.scatter(x, y, s=95, color=face, edgecolors=edge, linewidths=1.2, zorder=3)
    dx, dy = offsets[name]
    ax.text(x + dx, y + dy, name, fontsize=10, color=edge, fontweight="bold" if is_acr else "normal", va="center", zorder=5)


# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.12, 1.05)
ax.set_ylim(-0.15, 1.12)

ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
ax.set_xticklabels(["None", "Source /\ntarget", "Anchor\nvariables",
                    "Env labels", "Labels +\nanchors"], fontsize=9.5)
ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax.set_yticklabels(["None", "Heuristic", "DRO bound",
                    "Coverage", "Exact"], fontsize=9.5)
ax.text(0.455, -0.20, "Prior knowledge required", transform=ax.transAxes, ha="right", va="center", fontsize=12)
ax.annotate("", xy=(0.575, -0.20), xytext=(0.475, -0.20), xycoords="axes fraction", 
            textcoords="axes fraction", arrowprops=arrow, annotation_clip=False)

ax.text(-0.155, 0.40, "Theoretical guarantee", transform=ax.transAxes, ha="center", va="center", rotation=90, fontsize=12)
ax.annotate("", xy=(-0.155, 0.80), xytext=(-0.155, 0.68), xycoords="axes fraction", 
            textcoords="axes fraction", arrowprops=arrow, annotation_clip=False)

ax.grid(True, color="#e8e4dc", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_color("#cccccc")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor="#bb99dd", edgecolor="#5b4a8a", label="Statistics"),
    mpatches.Patch(facecolor="#ffbb55", edgecolor="#a05000", label="ML"),
    mpatches.Patch(facecolor="#ff7744", edgecolor="#c04020", label="Env discovery"),
    Line2D([], [], marker="*", markersize=13, linestyle="none", markerfacecolor="#66bb66", markeredgecolor="#2a7a40", label="ACR (this thesis)"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9.5, framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("fig_position_map.pdf", bbox_inches="tight")
plt.show()
