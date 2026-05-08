"""
Análisis complementario: temporal y de engagement.
Carga el CSV enriquecido con temas (battlefield6_reviews_topics.csv) y genera
ocho gráficos sobre la evolución temporal del sentimiento y la relación
entre el tiempo de juego / puntuación útil y el sentimiento.

Nota sobre 'votes_helpful': el CSV no incluye esa columna directamente;
se usa weighted_vote_score (puntuación ponderada de Steam) como proxy,
ya que combina votos útiles y total de votos.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
INPUT_CSV  = ROOT / "data" / "battlefield6_reviews_topics.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Carga de datos ─────────────────────────────────────────────────────────────
print("[0] Cargando datos...")
df = pd.read_csv(INPUT_CSV)
df["date"] = pd.to_datetime(df["date"], utc=True)
# Semana ISO (lunes) para agrupar: periodos comparables de 7 días
df["week"] = df["date"].dt.to_period("W").dt.start_time
print(f"    {len(df)} reseñas | rango: {df['date'].min().date()} – {df['date'].max().date()}")


# ════════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 – ANÁLISIS TEMPORAL
# ════════════════════════════════════════════════════════════════════════════════

# ── Gráfico 1: Sentimiento medio semanal + volumen de reseñas (doble eje) ─────
# Combinar ambas métricas en un único gráfico permite detectar si los picos
# negativos coinciden con lanzamientos masivos (muchos usuarios = mayor ruido)
# o son señal fiable de un problema real.
print("[1] Sentimiento medio semanal + volumen (doble eje)...")

weekly = (
    df.groupby("week")
    .agg(sentiment_mean=("vader_compound", "mean"), volume=("vader_compound", "count"))
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.bar(weekly["week"], weekly["volume"], width=5,
        color="#90CAF9", alpha=0.6, label="Volumen de reseñas")
ax2.plot(weekly["week"], weekly["sentiment_mean"], color="#1565C0",
         marker="o", linewidth=2, markersize=5, label="Sentimiento medio")
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)

ax1.set_xlabel("Semana")
ax1.set_ylabel("Número de reseñas", color="#90CAF9")
ax2.set_ylabel("Vader compound medio", color="#1565C0")
ax1.tick_params(axis="x", rotation=30)
ax2.set_ylim(-1, 1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.set_title("Sentimiento medio semanal y volumen de reseñas", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "09_weekly_sentiment_volume.png", dpi=150)
plt.close(fig)
print("    Guardado: 09_weekly_sentiment_volume.png")


# ── Gráfico 2: Evolución semanal por tema dominante ───────────────────────────
# Ver qué temas "dominan" en cada semana revela si ciertos eventos (parche,
# DLC, cobertura mediática) desplazan las preocupaciones de la comunidad.
print("[2] Evolución semanal por tema...")

weekly_topic = (
    df.groupby(["week", "topic_label"])
    .size()
    .reset_index(name="count")
)

# Normalizar a proporción dentro de cada semana para comparar semanas con
# distinto volumen sin que los picos de lanzamiento distorsionen la lectura.
total_per_week = weekly_topic.groupby("week")["count"].transform("sum")
weekly_topic["proportion"] = weekly_topic["count"] / total_per_week

topic_colors = {
    "Rendimiento":                   "#EF5350",
    "Mecánicas de combate":          "#5C6BC0",
    "Valoración general positiva":   "#66BB6A",
    "Ambientación y universo del juego": "#FFA726",
    "Historia y personajes":         "#26A69A",
}

fig, ax = plt.subplots(figsize=(13, 5))
for label, grp in weekly_topic.groupby("topic_label"):
    color = topic_colors.get(label, "gray")
    ax.plot(grp["week"], grp["proportion"], marker="o", linewidth=2,
            markersize=4, label=label, color=color)

ax.set_title("Proporción semanal de reseñas por tema dominante", fontsize=13)
ax.set_xlabel("Semana")
ax.set_ylabel("Proporción de reseñas")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "10_weekly_topic_evolution.png", dpi=150)
plt.close(fig)
print("    Guardado: 10_weekly_topic_evolution.png")


# ════════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 – ANÁLISIS DE ENGAGEMENT
# ════════════════════════════════════════════════════════════════════════════════

# ── Gráfico 3: playtime_hours vs vader_compound (dispersión + tendencia) ───────
# Un usuario que juega muchas horas y deja una reseña negativa es una señal
# más fuerte que alguien con <1 h: detectar esa correlación es clave.
# Usamos regresión lineal simple; el p-valor indica si la pendiente es real.
print("[3] Dispersión playtime_hours vs vader_compound...")

x_pt = df["playtime_hours"].values
y_pt = df["vader_compound"].values
slope, intercept, r, p, _ = stats.linregress(x_pt, y_pt)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(x_pt, y_pt, alpha=0.25, s=18, color="#5C6BC0", edgecolors="none")
x_line = np.linspace(x_pt.min(), x_pt.max(), 200)
ax.plot(x_line, slope * x_line + intercept, color="#EF5350", linewidth=2,
        label=f"Tendencia  r={r:.3f}  p={p:.3f}")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Horas jugadas")
ax.set_ylabel("Vader compound")
ax.set_title("Horas jugadas vs sentimiento (vader_compound)", fontsize=13)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "11_playtime_vs_sentiment.png", dpi=150)
plt.close(fig)
print(f"    r={r:.3f}, p={p:.4f} → Guardado: 11_playtime_vs_sentiment.png")


# ── Gráfico 4: weighted_vote_score vs vader_compound (dispersión + tendencia) ──
# weighted_vote_score es el proxy de 'votos útiles' de Steam. Si correlaciona
# con vader_compound, las reseñas positivas reciben más validación de la comunidad,
# lo que puede amplificar o suavizar la percepción pública del juego.
print("[4] Dispersión weighted_vote_score vs vader_compound...")

x_ws = df["weighted_vote_score"].values
y_ws = df["vader_compound"].values
slope2, intercept2, r2, p2, _ = stats.linregress(x_ws, y_ws)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(x_ws, y_ws, alpha=0.25, s=18, color="#26A69A", edgecolors="none")
x_line2 = np.linspace(x_ws.min(), x_ws.max(), 200)
ax.plot(x_line2, slope2 * x_line2 + intercept2, color="#EF5350", linewidth=2,
        label=f"Tendencia  r={r2:.3f}  p={p2:.3f}")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("weighted_vote_score (proxy votos útiles)")
ax.set_ylabel("Vader compound")
ax.set_title("Puntuación ponderada de Steam vs sentimiento (vader_compound)", fontsize=13)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "12_vote_score_vs_sentiment.png", dpi=150)
plt.close(fig)
print(f"    r={r2:.3f}, p={p2:.4f} → Guardado: 12_vote_score_vs_sentiment.png")


# ── Gráfico 5: Medias de playtime y weighted_vote_score por vader_sentiment ────
# Agrupar por etiqueta de sentimiento (positive/neutral/negative) permite ver
# si los usuarios más comprometidos (más horas) tienden a opinar de forma distinta
# a los que dejan reseñas rápidas. También revela si la comunidad recompensa
# más (vote_score) las reseñas positivas o las negativas.
print("[5] Medias de playtime y vote_score por sentiment...")

engagement = (
    df.groupby("vader_sentiment")[["playtime_hours", "weighted_vote_score"]]
    .mean()
    .reindex(["negative", "neutral", "positive"])
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subgráfico A: playtime medio por sentimiento
colors_sent = ["#EF5350", "#FFC107", "#4CAF50"]
axes[0].bar(engagement.index, engagement["playtime_hours"],
            color=colors_sent, edgecolor="white")
for i, (idx, val) in enumerate(engagement["playtime_hours"].items()):
    axes[0].text(i, val + 0.3, f"{val:.1f}h", ha="center", fontsize=10)
axes[0].set_title("Horas medias de juego por sentimiento", fontsize=12)
axes[0].set_xlabel("Sentimiento")
axes[0].set_ylabel("Horas jugadas (media)")
axes[0].set_ylim(0, engagement["playtime_hours"].max() * 1.2)

# Subgráfico B: weighted_vote_score medio por sentimiento
axes[1].bar(engagement.index, engagement["weighted_vote_score"],
            color=colors_sent, edgecolor="white")
for i, (idx, val) in enumerate(engagement["weighted_vote_score"].items()):
    axes[1].text(i, val + 0.003, f"{val:.3f}", ha="center", fontsize=10)
axes[1].set_title("Puntuación ponderada media por sentimiento", fontsize=12)
axes[1].set_xlabel("Sentimiento")
axes[1].set_ylabel("weighted_vote_score (media)")
axes[1].set_ylim(0, engagement["weighted_vote_score"].max() * 1.2)

fig.suptitle("Engagement medio por tipo de sentimiento", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "13_engagement_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("    Guardado: 13_engagement_by_sentiment.png")


# ── Correlaciones numéricas ────────────────────────────────────────────────────
# Imprimimos los coeficientes de Pearson ya calculados y añadimos Spearman
# (más robusto ante outliers y distribuciones no normales) para validar.
print("\n── Correlaciones ───────────────────────────────────────────────────")

r_pt_sp, p_pt_sp = stats.spearmanr(df["playtime_hours"], df["vader_compound"])
r_ws_sp, p_ws_sp = stats.spearmanr(df["weighted_vote_score"], df["vader_compound"])

print(f"  vader_compound ~ playtime_hours")
print(f"    Pearson  r={r:.4f}   p={p:.4f}")
print(f"    Spearman r={r_pt_sp:.4f}   p={p_pt_sp:.4f}")
print()
print(f"  vader_compound ~ weighted_vote_score")
print(f"    Pearson  r={r2:.4f}   p={p2:.4f}")
print(f"    Spearman r={r_ws_sp:.4f}   p={p_ws_sp:.4f}")
print("────────────────────────────────────────────────────────────────────")

print(f"\nTodos los gráficos guardados en: {FIGURES_DIR}")
print("── Análisis complementario completado ──────────────────────────────")
