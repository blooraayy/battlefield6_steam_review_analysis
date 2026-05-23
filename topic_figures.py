"""Regenera figuras modificadas para la presentación final."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

FIGS = Path(__file__).parent / "outputs" / "figures"
DATA = Path(__file__).parent / "data" / "battlefield6_reviews_topics.csv"

TOPIC_COLORS = {
    "Comparación con entregas anteriores": "#006EA6",
    "Mecánicas de combate y anti-cheat":   "#EF5350",
    "Mapas, armas y vehículos":             "#FFA726",
    "Problemas técnicos y servidores":      "#26A69A",
    "Monetización y Battlepass":            "#66BB6A",
}
SHORT_LABELS = {
    "Comparación con entregas anteriores": "Comparación\nBF3/BF4",
    "Mecánicas de combate y anti-cheat":   "Mecánicas\ny anti-cheat",
    "Mapas, armas y vehículos":             "Mapas, armas\ny vehículos",
    "Problemas técnicos y servidores":      "Bugs\ny servidores",
    "Monetización y Battlepass":            "Monetización\ny Battlepass",
}

plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False})

df = pd.read_csv(DATA)

# ── FIGURA 1: Distribución de temas — colores correctos, sin nombres largos ──
counts = df["topic_label"].value_counts()
ordered = list(TOPIC_COLORS.keys())
vals    = [counts.get(t, 0) for t in ordered]
colors  = [TOPIC_COLORS[t] for t in ordered]
labels  = [SHORT_LABELS[t] for t in ordered]
pcts    = [v / sum(vals) * 100 for v in vals]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9FA")

bars = ax.barh(labels, vals, color=colors, alpha=0.88, height=0.55)

for bar, pct, val in zip(bars, pcts, vals):
    ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%  ({val:,})",
            va="center", ha="left", fontsize=10.5, fontweight="bold",
            color="#333333")

ax.set_xlabel("Número de reseñas", fontsize=10, color="#555")
ax.set_xlim(0, max(vals) * 1.28)
ax.tick_params(axis="y", labelsize=11)
ax.tick_params(axis="x", labelsize=9, colors="#888")
ax.set_title("Distribución por tema dominante  —  91.197 reseñas",
             fontsize=13, fontweight="bold", color="#003A5C", pad=12)
ax.grid(axis="x", alpha=0.25, linestyle="--")
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#CCCCCC")

plt.tight_layout()
plt.savefig(FIGS / "26_topic_dist_colors.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close()
print("OK 26_topic_dist_colors.png")

# ── FIGURA 2: Temas positivos vs negativos — verde/rojo graduado ─────────────
pos_df = df[df["voted_up"] == True]["topic_label"].value_counts(normalize=True) * 100
neg_df = df[df["voted_up"] == False]["topic_label"].value_counts(normalize=True) * 100

pos_vals = [pos_df.get(t, 0) for t in ordered]
neg_vals = [neg_df.get(t, 0) for t in ordered]

def graduated(base_hex, values):
    """Escala la luminosidad: mayor valor → color más oscuro/saturado."""
    rgb = np.array(mcolors.to_rgb(base_hex))
    min_v, max_v = min(values), max(values)
    result = []
    for v in values:
        if max_v == min_v:
            alpha = 0.85
        else:
            alpha = 0.45 + 0.55 * (v - min_v) / (max_v - min_v)
        white = np.array([1, 1, 1])
        blended = alpha * rgb + (1 - alpha) * white
        result.append(blended)
    return result

green_colors = graduated("#006EA6", pos_vals)  # azul para positivos
red_colors   = graduated("#EF5350", neg_vals)  # rojo para negativos

x     = np.arange(len(ordered))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9FA")

bars_pos = ax.bar(x - width/2, pos_vals, width, color=green_colors,
                  edgecolor="white", linewidth=0.8, label="Reseñas positivas (voted_up)")
bars_neg = ax.bar(x + width/2, neg_vals, width, color=red_colors,
                  edgecolor="white", linewidth=0.8, label="Reseñas negativas")

for bar, val in zip(bars_pos, pos_vals):
    if val >= 5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color="#003A5C")

for bar, val in zip(bars_neg, neg_vals):
    if val >= 5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color="#7B0000")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("% de reseñas del grupo", fontsize=10, color="#555")
ax.set_ylim(0, max(max(pos_vals), max(neg_vals)) * 1.18)
ax.set_title("¿De qué hablan los que recomiendan vs los que no?",
             fontsize=13, fontweight="bold", color="#003A5C", pad=12)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.tick_params(axis="x", labelsize=10)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#CCCCCC")

patch_pos = mpatches.Patch(color="#006EA6", alpha=0.85, label="Reseñas positivas ✓")
patch_neg = mpatches.Patch(color="#EF5350", alpha=0.85, label="Reseñas negativas ✗")
ax.legend(handles=[patch_pos, patch_neg], fontsize=10, loc="upper right",
          framealpha=0.9)

plt.tight_layout()
plt.savefig(FIGS / "27_topics_voted_up_v2.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close()
print("OK 27_topics_voted_up_v2.png")
