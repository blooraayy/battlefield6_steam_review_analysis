"""
Clasificación ensemble VADER + RoBERTa.

Regla:
  - Si ambos modelos coinciden → se usa esa etiqueta directamente.
  - Si difieren → se elige el modelo con mayor score de confianza.
      · VADER:   abs(vader_compound)  — cuánto se aleja del umbral neutral
      · RoBERTa: roberta_score        — probabilidad de la etiqueta ganadora

Ambas métricas están en [0, 1], por lo que son comparables directamente.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
INPUT_CSV   = ROOT / "data" / "battlefield6_reviews_roberta.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Carga ───────────────────────────────────────────────────────────────────
print("[1/4] Cargando CSV...")
df = pd.read_csv(INPUT_CSV)
df["voted_up"] = df["voted_up"].map(
    lambda x: True if str(x).strip().lower() == "true" else False
)
print(f"    Filas: {len(df)}")

# ── 2. Ensemble ────────────────────────────────────────────────────────────────
print("[2/4] Aplicando lógica ensemble...")

# Puntuación de confianza de cada modelo normalizada a [0, 1]
# VADER usa el valor absoluto del compound como proxy de confianza
df["vader_score"] = df["vader_compound"].abs()

def _ensemble(row) -> tuple[str, str]:
    if row["vader_sentiment"] == row["roberta_sentiment"]:
        # Ambos de acuerdo: la etiqueta es segura, fuente = ambos
        return row["vader_sentiment"], "both"
    # Desacuerdo: gana el modelo más seguro
    if row["roberta_score"] >= row["vader_score"]:
        return row["roberta_sentiment"], "roberta"
    return row["vader_sentiment"], "vader"

results = df.apply(_ensemble, axis=1, result_type="expand")
df["ensemble_sentiment"] = results[0]
df["ensemble_source"]    = results[1]

# ── 3. Métricas ────────────────────────────────────────────────────────────────
print("[3/4] Calculando métricas...")

def _metrics(pred_col: str) -> dict:
    binary = df[pred_col].map({"positive": True}).fillna(False)
    voted  = df["voted_up"]
    tp = int(((binary == True)  & (voted == True)).sum())
    tn = int(((binary == False) & (voted == False)).sum())
    fp = int(((binary == True)  & (voted == False)).sum())
    fn = int(((binary == False) & (voted == True)).sum())
    acc  = (tp + tn) / len(df)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

vader_m    = _metrics("vader_sentiment")
roberta_m  = _metrics("roberta_sentiment")
ensemble_m = _metrics("ensemble_sentiment")

# Cuántas veces cada fuente resolvió el desacuerdo
source_counts = df["ensemble_source"].value_counts()
agree_n    = source_counts.get("both",    0)
roberta_n  = source_counts.get("roberta", 0)
vader_n    = source_counts.get("vader",   0)

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║          Resultados Ensemble VADER + RoBERTa                 ║")
print(f"╠{'═'*22}╦{'═'*12}╦{'═'*12}╦{'═'*12}╣")
print(f"║{'Métrica':22s}║{'VADER':^12}║{'RoBERTa':^12}║{'Ensemble':^12}║")
print(f"╠{'═'*22}╬{'═'*12}╬{'═'*12}╬{'═'*12}╣")
for key, label in [("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),("f1","F1-score")]:
    print(f"║{label:22s}║{vader_m[key]:^12.4f}║{roberta_m[key]:^12.4f}║{ensemble_m[key]:^12.4f}║")
print(f"╠{'═'*22}╩{'═'*12}╩{'═'*12}╩{'═'*12}╣")
print(f"║  Coinciden (ambos)          : {agree_n:>5} ({agree_n/len(df)*100:.1f}%){'':>18}║")
print(f"║  Desacuerdo → ganó VADER    : {vader_n:>5} ({vader_n/len(df)*100:.1f}%){'':>18}║")
print(f"║  Desacuerdo → ganó RoBERTa  : {roberta_n:>5} ({roberta_n/len(df)*100:.1f}%){'':>18}║")
print("╚══════════════════════════════════════════════════════════════╝\n")

# ── 4. Gráfico ────────────────────────────────────────────────────────────────
print("[4/4] Generando gráfico...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── 4.1 Métricas comparadas (accuracy, precision, recall, f1) ────────────────
ax = axes[0]
metric_keys   = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
x = range(len(metric_keys))
width = 0.25
model_colors = {"VADER": "#2196F3", "RoBERTa": "#FF5722", "Ensemble": "#9C27B0"}

for i, (name, m, color) in enumerate(zip(
    ["VADER", "RoBERTa", "Ensemble"],
    [vader_m, roberta_m, ensemble_m],
    model_colors.values(),
)):
    positions = [xi + i * width for xi in x]
    bars = ax.bar(positions, [m[k] for k in metric_keys], width=width,
                  label=name, color=color, edgecolor="white")
    for bar, val in zip(bars, [m[k] for k in metric_keys]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=7,
        )

ax.set_title("Métricas vs voted_up de Steam", fontsize=12)
ax.set_ylabel("Valor")
ax.set_ylim(0, 1.2)
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(metric_labels)
ax.legend()

# ── 4.2 Origen de la decisión ensemble ───────────────────────────────────────
ax = axes[1]
src_labels = ["Coinciden\n(ambos)", "Ganó\nRoBERTa", "Ganó\nVADER"]
src_values = [agree_n, roberta_n, vader_n]
src_colors = ["#4CAF50", "#FF5722", "#2196F3"]
bars = ax.bar(src_labels, src_values, color=src_colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, src_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + len(df) * 0.005,
        f"{val}\n({val/len(df)*100:.1f}%)",
        ha="center", va="bottom", fontsize=10,
    )
ax.set_title("Origen de la clasificación ensemble", fontsize=12)
ax.set_ylabel("Número de reseñas")
ax.set_ylim(0, max(src_values) * 1.2)

fig.suptitle("Ensemble VADER + RoBERTa", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "19_ensemble_results.png", dpi=150)
plt.close(fig)

# ── 4.3 Distribución de clasificaciones ensemble ──────────────────────────────
# Muestra cuántas reseñas quedaron clasificadas como positive/neutral/negative
SENTIMENT_COLORS = {"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#F44336"}

counts = (
    df["ensemble_sentiment"]
    .value_counts()
    .reindex(["positive", "neutral", "negative"])
    .fillna(0)
    .astype(int)
)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    counts.index,
    counts.values,
    color=[SENTIMENT_COLORS[s] for s in counts.index],
    edgecolor="white",
)
for bar, val in zip(bars, counts.values):
    pct = val / len(df) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + len(df) * 0.005,
        f"{val}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=11,
    )
ax.set_title("Distribución de reseñas — clasificación Ensemble\n(VADER + RoBERTa)", fontsize=12)
ax.set_xlabel("Sentimiento")
ax.set_ylabel("Número de reseñas")
ax.set_ylim(0, counts.max() * 1.2)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "20_ensemble_sentiment_distribution.png", dpi=150)
plt.close(fig)

print(f"    Gráfico guardado en: {FIGURES_DIR / '19_ensemble_results.png'}")
print(f"    Gráfico guardado en: {FIGURES_DIR / '20_ensemble_sentiment_distribution.png'}")
print("\n  Test ensemble completado.")
