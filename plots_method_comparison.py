# ============================================================
#   Fig 1: All methods in lit review compared
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 5.5))

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
    "EIIL":        ( 0.03, -0.06),
    "HRM":         ( 0.03,  0.04),
    "ACR ★":       ( 0.03,  0.03),
    "DANN":        ( 0.03,  0.03),
    "IRM":         ( 0.03,  0.03),
    "REx / Fishr": ( 0.03, -0.06),
    "GroupDRO":    (-0.03,  0.04),
    "CR":          (-0.06,  0.04),
    "Anchor Reg.": ( 0.03,  0.03),
    "ICP":         ( 0.03,  0.03),
}

for name, (x, y, face, edge) in methods.items():
    s = 120 if name == "ACR ★" else 80
    lw = 2.0 if name == "ACR ★" else 1.2
    ax.scatter(x, y, s=s, color=face, edgecolors=edge, linewidths=lw, zorder=3)
    dx, dy = offsets[name]
    weight = "bold" if name == "ACR ★" else "normal"
    ax.text(x + dx, y + dy, name, fontsize=9, color=edge,
            fontweight=weight, va="center")


# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.12, 1.05)
ax.set_ylim(-0.15, 1.12)

ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
ax.set_xticklabels(["None", "Source /\ntarget", "Anchor\nvariables",
                    "Env labels", "Labels +\nanchors"], fontsize=8.5)
ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax.set_yticklabels(["None", "Heuristic", "DRO bound",
                    "Coverage", "Exact"], fontsize=8.5)

ax.set_xlabel("Prior knowledge required →", fontsize=10, labelpad=8)
ax.set_ylabel("Theoretical guarantee →", fontsize=10, labelpad=8)

ax.grid(True, color="#e8e4dc", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_color("#cccccc")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor="#bb99dd", edgecolor="#5b4a8a", label="Statistics"),
    mpatches.Patch(facecolor="#ffbb55", edgecolor="#a05000", label="ML"),
    mpatches.Patch(facecolor="#ff7744", edgecolor="#c04020", label="Env discovery"),
    mpatches.Patch(facecolor="#66bb66", edgecolor="#2a7a40", label="ACR"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8.5,
          framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("fig_position_map.pdf", dpi=200, bbox_inches="tight")
plt.show()
