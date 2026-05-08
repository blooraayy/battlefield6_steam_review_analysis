"""
Análisis de sentimiento sobre reseñas de Steam usando VADER (NLTK).
Genera gráficos, métricas y CSV de resultados.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Rutas del proyecto ────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
CLEAN_CSV = ROOT / "data" / "battlefield6_reviews_clean.csv"
OUTPUT_CSV = ROOT / "data" / "battlefield6_reviews_sentiment.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ── Descarga del lexicon VADER si no está disponible ─────────────────────────
nltk.download("vader_lexicon", quiet=True)


# ── 1. Carga de datos ─────────────────────────────────────────────────────────
print("[1/7] Cargando CSV limpio...")
df = pd.read_csv(CLEAN_CSV, parse_dates=["date"])

# Convertir voted_up a booleano por si viene como string
df["voted_up"] = df["voted_up"].map(
    lambda x: True if str(x).strip().lower() == "true" else False
)

df["playtime_hours"] = pd.to_numeric(df["playtime_hours"], errors="coerce")
df["weighted_vote_score"] = pd.to_numeric(df["weighted_vote_score"], errors="coerce")

print(f"    Filas cargadas: {len(df)}")


# ── 2. Análisis de sentimiento VADER ─────────────────────────────────────────
print("[2/7] Aplicando VADER sobre text_cleaned...")
sia = SentimentIntensityAnalyzer()

# Puntuación compuesta: va de -1 (muy negativo) a +1 (muy positivo)
df["vader_compound"] = df["text_cleaned"].apply(
    lambda text: sia.polarity_scores(str(text))["compound"]
)

# Etiqueta de sentimiento según umbrales estándar de VADER
def _label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"

df["vader_sentiment"] = df["vader_compound"].apply(_label)

print("    Distribución de etiquetas VADER:")
print(df["vader_sentiment"].value_counts().to_string())


# ── 3. Guardar CSV con sentimiento ───────────────────────────────────────────
print(f"[3/7] Guardando CSV enriquecido en: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)


# ── 4. Gráficos ───────────────────────────────────────────────────────────────
print("[4/7] Generando gráficos...")

# Paleta de colores consistente
SENTIMENT_COLORS = {"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#F44336"}

# ── 4.1 Distribución de sentimientos (barras) ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["vader_sentiment"].value_counts().reindex(["positive", "neutral", "negative"])
bars = ax.bar(
    counts.index,
    counts.values,
    color=[SENTIMENT_COLORS[s] for s in counts.index],
    edgecolor="white",
)
# Añadir etiquetas de valor encima de cada barra
for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(int(bar.get_height())),
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.set_title("Distribución de sentimientos VADER", fontsize=13)
ax.set_xlabel("Sentimiento")
ax.set_ylabel("Número de reseñas")
ax.set_ylim(0, counts.max() * 1.15)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "01_sentiment_distribution.png", dpi=150)
plt.close(fig)

# ── 4.2 Comparación vader_sentiment vs voted_up ──────────────────────────────
# Tabla de contingencia: eje X = vader_sentiment, grupos = voted_up
fig, ax = plt.subplots(figsize=(8, 5))
cross = (
    df.groupby(["vader_sentiment", "voted_up"])
    .size()
    .unstack(fill_value=0)
    .reindex(["positive", "neutral", "negative"])
)
cross.columns = ["Reseña negativa (voted_up=False)", "Reseña positiva (voted_up=True)"]
cross.plot(kind="bar", ax=ax, color=["#F44336", "#4CAF50"], edgecolor="white", rot=0)
ax.set_title("vader_sentiment vs voted_up", fontsize=13)
ax.set_xlabel("Sentimiento VADER")
ax.set_ylabel("Número de reseñas")
ax.legend(title="voted_up")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "02_vader_vs_voted_up.png", dpi=150)
plt.close(fig)

# ── 4.3 Sentimiento medio semanal ────────────────────────────────────────────
# Agrupamos por semana y calculamos la media del compound
weekly = (
    df.set_index("date")["vader_compound"]
    .resample("W")
    .mean()
    .dropna()
)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(weekly.index, weekly.values, marker="o", linewidth=1.5, color="#2196F3")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)  # línea de referencia neutral
ax.set_title("Sentimiento medio (vader_compound) por semana", fontsize=13)
ax.set_xlabel("Semana")
ax.set_ylabel("vader_compound medio")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "03_weekly_sentiment.png", dpi=150)
plt.close(fig)

# ── 4.4 Dispersión playtime_hours vs vader_compound ─────────────────────────
plot_df = df.dropna(subset=["playtime_hours", "vader_compound"])
fig, ax = plt.subplots(figsize=(8, 5))
for label, color in SENTIMENT_COLORS.items():
    mask = plot_df["vader_sentiment"] == label
    ax.scatter(
        plot_df.loc[mask, "playtime_hours"],
        plot_df.loc[mask, "vader_compound"],
        alpha=0.4,
        s=20,
        color=color,
        label=label,
    )
ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.7)
ax.axhline(-0.05, color="gray", linestyle="--", linewidth=0.7)
ax.set_title("playtime_hours vs vader_compound", fontsize=13)
ax.set_xlabel("Horas jugadas")
ax.set_ylabel("vader_compound")
ax.legend(title="Sentimiento")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "04_playtime_vs_compound.png", dpi=150)
plt.close(fig)

# ── 4.5 Dispersión weighted_vote_score vs vader_compound ────────────────────
plot_df2 = df.dropna(subset=["weighted_vote_score", "vader_compound"])
fig, ax = plt.subplots(figsize=(8, 5))
for label, color in SENTIMENT_COLORS.items():
    mask = plot_df2["vader_sentiment"] == label
    ax.scatter(
        plot_df2.loc[mask, "weighted_vote_score"],
        plot_df2.loc[mask, "vader_compound"],
        alpha=0.4,
        s=20,
        color=color,
        label=label,
    )
ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.7)
ax.axhline(-0.05, color="gray", linestyle="--", linewidth=0.7)
ax.set_title("weighted_vote_score vs vader_compound", fontsize=13)
ax.set_xlabel("weighted_vote_score")
ax.set_ylabel("vader_compound")
ax.legend(title="Sentimiento")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "05_weighted_score_vs_compound.png", dpi=150)
plt.close(fig)

print(f"    Gráficos guardados en: {FIGURES_DIR}")


# ── 5. Métricas: vader_sentiment vs voted_up ─────────────────────────────────
# Tratamos voted_up=True como "positive" para comparar con vader_sentiment
print("[5/7] Calculando métricas...")

# Mapeamos vader_sentiment a binario: positive → True, resto → False
vader_binary = df["vader_sentiment"].map({"positive": True}).fillna(False)
voted_binary = df["voted_up"]

correct = int((vader_binary == voted_binary).sum())
total = len(df)
wrong = total - correct
accuracy = correct / total

# Tabla de fallos por categoría
fp = int(((vader_binary == True) & (voted_binary == False)).sum())   # VADER dice + pero Steam -
fn = int(((vader_binary == False) & (voted_binary == True)).sum())   # VADER dice - pero Steam +
tp = int(((vader_binary == True) & (voted_binary == True)).sum())
tn = int(((vader_binary == False) & (voted_binary == False)).sum())

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

metrics = {
    "total_reviews": total,
    "correct_predictions": correct,
    "wrong_predictions": wrong,
    "accuracy": round(accuracy, 4),
    "true_positives": tp,
    "true_negatives": tn,
    "false_positives_vader": fp,
    "false_negatives_vader": fn,
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
}

print("\n── Métricas VADER vs voted_up ──────────────────────────────")
print(f"  Total reseñas      : {total}")
print(f"  Aciertos           : {correct}")
print(f"  Fallos             : {wrong}")
print(f"  Accuracy           : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Verdaderos positivos (TP): {tp}")
print(f"  Verdaderos negativos (TN): {tn}")
print(f"  Falsos positivos VADER (FP): {fp}  ← VADER + pero Steam -")
print(f"  Falsos negativos VADER (FN): {fn}  ← VADER - pero Steam +")
print(f"  Precision          : {precision:.4f}")
print(f"  Recall             : {recall:.4f}")
print(f"  F1-score           : {f1:.4f}")
print("────────────────────────────────────────────────────────────\n")


# ── 6. Guardar métricas en JSON ───────────────────────────────────────────────
print("[6/7] Guardando métricas...")
metrics_path = METRICS_DIR / "vader_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"    Métricas guardadas en: {metrics_path}")


print("[7/7] Análisis completado.")
print(f"    CSV de sentimiento : {OUTPUT_CSV}")
print(f"    Gráficos           : {FIGURES_DIR}")
print(f"    Métricas           : {metrics_path}")
