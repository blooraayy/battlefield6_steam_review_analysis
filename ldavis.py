"""
Genera una imagen estática del mapa pyLDAvis a partir del HTML ya existente.
Datos extraídos directamente del JSON embebido en lda_visualization.html.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "outputs" / "figures" / "25_ldavis_static.png"

# ── Datos extraídos del HTML (mdsDat + top terms por tema) ────────────────────
# Coordenadas MDS de cada tema (PC1, PC2) y frecuencia global del tema
topics = [
    {"id": 1, "label": "Comparación con\nentregas anteriores",
     "x": -0.346, "y": -0.096, "freq": 29.15,
     "terms": ["bf4", "cod", "bf3", "campaign", "multiplayer", "fps", "2042", "love", "shooter", "recommend"],
     "color": "#006EA6"},
    {"id": 2, "label": "Mecánicas de combate\ny anti-cheat",
     "x":  0.236, "y": -0.338, "freq": 10.99,
     "terms": ["kill", "hit", "tank", "die", "shoot", "spawn", "cheater", "cheat", "sniper", "enemy"],
     "color": "#EF5350"},
    {"id": 3, "label": "Mapas, armas\ny vehículos",
     "x": -0.140, "y":  0.019, "freq": 21.10,
     "terms": ["map", "weapon", "gun", "vehicle", "mode", "big", "small", "class", "balance", "movement"],
     "color": "#FFA726"},
    {"id": 4, "label": "Problemas técnicos\ny servidores",
     "x":  0.105, "y":  0.276, "freq": 20.92,
     "terms": ["fix", "issue", "bug", "server", "crash", "update", "launch", "run", "unplayable", "refund"],
     "color": "#26A69A"},
    {"id": 5, "label": "Monetización\ny Battlepass",
     "x":  0.144, "y":  0.139, "freq": 17.84,
     "terms": ["buy", "skin", "money", "pass", "pay", "season", "battle", "battlepass", "price", "free"],
     "color": "#66BB6A"},
]

# ── Figura con dos paneles ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6.5))
fig.patch.set_facecolor("white")

ax_map  = fig.add_axes([0.03, 0.08, 0.44, 0.84])   # panel izquierdo
ax_bars = fig.add_axes([0.54, 0.08, 0.44, 0.84])   # panel derecho

# ── PANEL IZQUIERDO: mapa de distancias entre temas ──────────────────────────
ax_map.set_facecolor("#F8F9FA")
ax_map.axhline(0, color="#CCCCCC", lw=0.8, zorder=0)
ax_map.axvline(0, color="#CCCCCC", lw=0.8, zorder=0)

scale = 900  # factor de escala para tamaño de los círculos
for t in topics:
    circle = plt.Circle((t["x"], t["y"]), radius=np.sqrt(t["freq"]) * 0.04,
                         color=t["color"], alpha=0.55, zorder=2)
    ax_map.add_patch(circle)
    ax_map.scatter(t["x"], t["y"], s=t["freq"] * scale,
                   color=t["color"], alpha=0.30, edgecolors=t["color"],
                   linewidths=2, zorder=3)
    ax_map.text(t["x"], t["y"],
                f"{t['id']}\n({t['freq']:.1f}%)",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4)
    # Etiqueta exterior
    offset_x = -0.07 if t["x"] < 0 else 0.07
    offset_y = -0.07 if t["y"] < 0 else 0.07
    ax_map.text(t["x"] + offset_x, t["y"] + offset_y,
                t["label"], ha="center", va="center",
                fontsize=7.5, color="#333333", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=t["color"],
                          alpha=0.85, lw=1.2))

ax_map.set_xlim(-0.55, 0.55)
ax_map.set_ylim(-0.55, 0.55)
ax_map.set_xlabel("PC1  (primera componente MDS)", fontsize=9, color="#555")
ax_map.set_ylabel("PC2  (segunda componente MDS)", fontsize=9, color="#555")
ax_map.set_title("Mapa de distancias entre temas", fontsize=11, fontweight="bold",
                 color="#003A5C", pad=10)
ax_map.tick_params(labelsize=8, colors="#888")
for spine in ax_map.spines.values():
    spine.set_edgecolor("#DDDDDD")

# Leyenda de tamaño
for freq, label in [(10, "10%"), (20, "20%"), (30, "30%")]:
    ax_map.scatter([], [], s=freq * scale, color="#AAAAAA", alpha=0.4,
                   label=f"Peso: {label}")
ax_map.legend(title="Tamaño = % del corpus", title_fontsize=7.5,
              fontsize=7.5, loc="lower right", framealpha=0.8)

# ── PANEL DERECHO: top términos del tema más grande ───────────────────────────
# Mostrar términos del Tema 1 (Comparación) como ejemplo, con barras azul/rojo
# al estilo pyLDAvis
ax_bars.set_facecolor("#F8F9FA")

# Frecuencias aproximadas del tema y del corpus global (normalizadas)
# Extraídas del JSON para Topic1 (primeras 10 entradas)
terms_t1 = ["bf4", "cod", "bf3", "campaign", "multiplayer", "fps", "2042", "love", "shooter", "recommend"]
freq_topic  = [3981, 9908, 3305, 3578, 3556, 3283, 3213, 4828, 2609, 2813]  # en tema
freq_corpus = [3981, 9908, 3305, 3578, 3556, 3283, 3213, 4828, 2609, 2813]  # global (similares para top1)
# Normalizar a proporción
total_topic  = sum(freq_topic)
total_corpus = 91197 * 5  # aprox tokens totales

ft = [f / total_topic * 100 for f in freq_topic]
fc = [f / total_corpus * 100 for f in freq_corpus]

y_pos = np.arange(len(terms_t1))
bar_h = 0.38

# Barras del corpus (rojas)
bars_corpus = ax_bars.barh(y_pos + bar_h/2, fc, bar_h,
                            color="#E74C3C", alpha=0.55, label="Frecuencia global en corpus")
# Barras del tema (azules)
bars_topic  = ax_bars.barh(y_pos - bar_h/2, ft, bar_h,
                            color="#006EA6", alpha=0.80, label="Frecuencia en Tema 1")

ax_bars.set_yticks(y_pos)
ax_bars.set_yticklabels(terms_t1, fontsize=10)
ax_bars.set_xlabel("Frecuencia (%)", fontsize=9, color="#555")
ax_bars.set_title("Términos relevantes — Tema 1:\nComparación con entregas anteriores",
                  fontsize=11, fontweight="bold", color="#003A5C", pad=10)
ax_bars.tick_params(labelsize=8)
for spine in ax_bars.spines.values():
    spine.set_edgecolor("#DDDDDD")
ax_bars.set_facecolor("#F8F9FA")

patch_blue = mpatches.Patch(color="#006EA6", alpha=0.80, label="Frecuencia en el tema")
patch_red  = mpatches.Patch(color="#E74C3C", alpha=0.55, label="Frecuencia global")
ax_bars.legend(handles=[patch_blue, patch_red], fontsize=8, loc="lower right")

# Nota al pie de la figura
fig.text(0.5, 0.01,
         "Visualización estática del mapa interactivo pyLDAvis  "
         "·  Interactive version: outputs/figures/lda_visualization.html",
         ha="center", fontsize=8, color="#888888", style="italic")

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Guardado: {OUT}")
