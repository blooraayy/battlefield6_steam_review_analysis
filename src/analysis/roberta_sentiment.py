"""
Análisis de sentimiento con RoBERTa (cardiffnlp/twitter-roberta-base-sentiment).

Lee el CSV limpio y añade:
  - roberta_sentiment : etiqueta (positive / neutral / negative)
  - roberta_score     : confianza del modelo (0–1)
  - roberta_compound  : puntuación compuesta (-1 a +1, análoga a vader_compound)
                        positive → +score | negative → −score | neutral → 0

Ejecutar directamente:
    python src/analysis/roberta_sentiment.py
"""

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
INPUT_CSV   = ROOT / "data" / "battlefield6_reviews_clean.csv"
OUTPUT_CSV  = ROOT / "data" / "battlefield6_reviews_sentiment.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
BATCH_SIZE = 32

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

SENTIMENT_COLORS = {"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#F44336"}


# ── 1. Carga de datos ─────────────────────────────────────────────────────────
print("[1/6] Cargando CSV limpio...")
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
df["voted_up"] = df["voted_up"].map(
    lambda x: True if str(x).strip().lower() == "true" else False
)
df["playtime_hours"] = pd.to_numeric(df["playtime_hours"], errors="coerce")
df["weighted_vote_score"] = pd.to_numeric(df["weighted_vote_score"], errors="coerce")
print(f"    Filas cargadas: {len(df)}")


# ── 2. Cargar modelo RoBERTa ──────────────────────────────────────────────────
print("[2/6] Cargando modelo RoBERTa...")

# device=0 usa la primera GPU; device=-1 fuerza CPU
device = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
print(f"    Dispositivo: {device_name}")

# truncation=True es obligatorio: RoBERTa tiene límite de 512 tokens
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=device,
    truncation=True,
    max_length=512,
)


# ── 3. Inferencia en batches con barra de progreso ────────────────────────────
print(f"[3/6] Ejecutando inferencia en batches de {BATCH_SIZE}...")
texts = df["text_cleaned"].fillna("").tolist()
results = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="RoBERTa inference", unit="batch"):
    batch = texts[i : i + BATCH_SIZE]
    results.extend(classifier(batch))

df["roberta_sentiment"] = [LABEL_MAP[r["label"]] for r in results]
df["roberta_score"] = [round(r["score"], 4) for r in results]

# Puntuación compuesta: positive → +score, negative → −score, neutral → 0
def _signed_score(row) -> float:
    if row["roberta_sentiment"] == "positive":
        return round(row["roberta_score"], 4)
    if row["roberta_sentiment"] == "negative":
        return round(-row["roberta_score"], 4)
    return 0.0

df["roberta_compound"] = df.apply(_signed_score, axis=1)

print("    Distribución RoBERTa:")
print(df["roberta_sentiment"].value_counts().to_string())


# ── 4. Guardar CSV enriquecido ────────────────────────────────────────────────
print(f"[4/6] Guardando CSV en: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)


# ── 5. Gráficos ───────────────────────────────────────────────────────────────
print("[5/6] Generando gráficos...")

# ── 5.1 Distribución de sentimientos ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["roberta_sentiment"].value_counts().reindex(["positive", "neutral", "negative"])
bars = ax.bar(
    counts.index, counts.values,
    color=[SENTIMENT_COLORS[s] for s in counts.index],
    edgecolor="white",
)
for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(int(bar.get_height())),
        ha="center", va="bottom", fontsize=10,
    )
ax.set_title("Distribución de sentimientos RoBERTa", fontsize=13)
ax.set_xlabel("Sentimiento")
ax.set_ylabel("Número de reseñas")
ax.set_ylim(0, counts.max() * 1.15)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "01_sentiment_distribution.png", dpi=150)
plt.close(fig)

# ── 5.2 roberta_sentiment vs voted_up ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
cross = (
    df.groupby(["roberta_sentiment", "voted_up"])
    .size()
    .unstack(fill_value=0)
    .reindex(["positive", "neutral", "negative"])
)
cross.columns = ["Reseña negativa (voted_up=False)", "Reseña positiva (voted_up=True)"]
cross.plot(kind="bar", ax=ax, color=["#F44336", "#4CAF50"], edgecolor="white", rot=0)
ax.set_title("roberta_sentiment vs voted_up", fontsize=13)
ax.set_xlabel("Sentimiento RoBERTa")
ax.set_ylabel("Número de reseñas")
ax.legend(title="voted_up")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "02_roberta_vs_voted_up.png", dpi=150)
plt.close(fig)

# ── 5.3 Sentimiento medio semanal (roberta_compound) ──────────────────────────
weekly = df.set_index("date")["roberta_compound"].resample("W").mean().dropna()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(weekly.index, weekly.values, marker="o", linewidth=1.5, color="#2196F3")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Sentimiento medio (RoBERTa) por semana", fontsize=13)
ax.set_xlabel("Semana")
ax.set_ylabel("Puntuación de sentimiento media")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "03_weekly_sentiment.png", dpi=150)
plt.close(fig)

# ── 5.4 playtime_hours vs roberta_compound ────────────────────────────────────
plot_df = df.dropna(subset=["playtime_hours", "roberta_compound"])
fig, ax = plt.subplots(figsize=(8, 5))
for label, color in SENTIMENT_COLORS.items():
    mask = plot_df["roberta_sentiment"] == label
    ax.scatter(
        plot_df.loc[mask, "playtime_hours"],
        plot_df.loc[mask, "roberta_compound"],
        alpha=0.4, s=20, color=color, label=label,
    )
ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax.set_title("playtime_hours vs roberta_compound", fontsize=13)
ax.set_xlabel("Horas jugadas")
ax.set_ylabel("roberta_compound")
ax.legend(title="Sentimiento")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "04_playtime_vs_compound.png", dpi=150)
plt.close(fig)

# ── 5.5 weighted_vote_score vs roberta_compound ───────────────────────────────
plot_df2 = df.dropna(subset=["weighted_vote_score", "roberta_compound"])
fig, ax = plt.subplots(figsize=(8, 5))
for label, color in SENTIMENT_COLORS.items():
    mask = plot_df2["roberta_sentiment"] == label
    ax.scatter(
        plot_df2.loc[mask, "weighted_vote_score"],
        plot_df2.loc[mask, "roberta_compound"],
        alpha=0.4, s=20, color=color, label=label,
    )
ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax.set_title("weighted_vote_score vs roberta_compound", fontsize=13)
ax.set_xlabel("weighted_vote_score")
ax.set_ylabel("roberta_compound")
ax.legend(title="Sentimiento")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "05_weighted_score_vs_compound.png", dpi=150)
plt.close(fig)

print(f"    Gráficos guardados en: {FIGURES_DIR}")


# ── 6. Métricas: roberta_sentiment vs voted_up ────────────────────────────────
print("[6/6] Calculando métricas...")

roberta_binary = df["roberta_sentiment"].map({"positive": True}).fillna(False)
voted_binary = df["voted_up"]

tp = int(((roberta_binary == True)  & (voted_binary == True)).sum())
tn = int(((roberta_binary == False) & (voted_binary == False)).sum())
fp = int(((roberta_binary == True)  & (voted_binary == False)).sum())
fn = int(((roberta_binary == False) & (voted_binary == True)).sum())

total     = len(df)
correct   = tp + tn
accuracy  = correct / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

metrics = {
    "total_reviews":       total,
    "correct_predictions": correct,
    "wrong_predictions":   total - correct,
    "accuracy":            round(accuracy, 4),
    "true_positives":      tp,
    "true_negatives":      tn,
    "false_positives":     fp,
    "false_negatives":     fn,
    "precision":           round(precision, 4),
    "recall":              round(recall, 4),
    "f1_score":            round(f1, 4),
}

print("\n── Métricas RoBERTa vs voted_up ────────────────────────────")
print(f"  Total reseñas : {total}")
print(f"  Aciertos      : {correct}")
print(f"  Fallos        : {total - correct}")
print(f"  Accuracy      : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-score  : {f1:.4f}")
print("────────────────────────────────────────────────────────────\n")

metrics_path = METRICS_DIR / "roberta_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"    Métricas guardadas en: {metrics_path}")

print("  Análisis RoBERTa completado.")
print(f"    CSV      : {OUTPUT_CSV}")
print(f"    Gráficos : {FIGURES_DIR}")
print(f"    Métricas : {metrics_path}")
