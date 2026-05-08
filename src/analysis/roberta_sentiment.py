"""
Análisis de sentimiento con RoBERTa (cardiffnlp/twitter-roberta-base-sentiment).

Script independiente: lee el CSV con resultados VADER ya calculados y añade
dos columnas nuevas (roberta_sentiment, roberta_score). Luego compara ambos
modelos contra la etiqueta real voted_up de Steam.

Ejecutar directamente:
    python src/analysis/roberta_sentiment.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
# Leemos el CSV que ya tiene VADER para poder comparar ambos modelos directamente
INPUT_CSV  = ROOT / "data" / "battlefield6_reviews_sentiment.csv"
# Guardamos en un archivo separado para no tocar los resultados de VADER
OUTPUT_CSV = ROOT / "data" / "battlefield6_reviews_roberta.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
BATCH_SIZE = 32  # equilibrio entre velocidad y uso de VRAM

# El modelo devuelve etiquetas genéricas; las mapeamos a nombres legibles
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


# ── 1. Carga de datos ─────────────────────────────────────────────────────────
print("[1/6] Cargando CSV con resultados VADER...")
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])

# voted_up puede llegar como string "True"/"False" según cómo se guardó el CSV
df["voted_up"] = df["voted_up"].map(
    lambda x: True if str(x).strip().lower() == "true" else False
)
print(f"    Filas cargadas: {len(df)}")


# ── 2. Cargar modelo RoBERTa ──────────────────────────────────────────────────
print("[2/6] Cargando modelo RoBERTa...")

# device=0 usa la primera GPU; device=-1 fuerza CPU
# Preferimos GPU porque la inferencia sobre miles de reseñas es muy lenta en CPU
device = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
print(f"    Dispositivo: {device_name}")

# truncation=True es obligatorio: RoBERTa tiene límite de 512 tokens
# y algunas reseñas largas lo superan
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=device,
    truncation=True,
    max_length=512,
)


# ── 3. Inferencia en batches con barra de progreso ────────────────────────────
print(f"[3/6] Ejecutando inferencia en batches de {BATCH_SIZE}...")

# fillna("") para evitar que el modelo falle con valores nulos
texts = df["text_cleaned"].fillna("").tolist()
results = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="RoBERTa inference", unit="batch"):
    batch = texts[i : i + BATCH_SIZE]
    batch_results = classifier(batch)
    results.extend(batch_results)

df["roberta_sentiment"] = [LABEL_MAP[r["label"]] for r in results]
# Puntuación de confianza: probabilidad de la etiqueta ganadora (0-1)
df["roberta_score"] = [round(r["score"], 4) for r in results]

print("    Distribución RoBERTa:")
print(df["roberta_sentiment"].value_counts().to_string())


# ── 4. Guardar CSV enriquecido ────────────────────────────────────────────────
print(f"[4/6] Guardando CSV en: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)


# ── 5. Comparación VADER vs RoBERTa ──────────────────────────────────────────
print("[5/6] Calculando métricas comparativas...")

# Binarizamos: positive=True, neutral/negative=False
# Así podemos comparar directamente con voted_up (True/False)
vader_binary   = df["vader_sentiment"].map({"positive": True}).fillna(False)
roberta_binary = df["roberta_sentiment"].map({"positive": True}).fillna(False)
voted_binary   = df["voted_up"]

vader_acc   = (vader_binary == voted_binary).mean()
roberta_acc = (roberta_binary == voted_binary).mean()

# Coincidencia entre los dos modelos (independientemente de voted_up)
agree    = int((df["vader_sentiment"] == df["roberta_sentiment"]).sum())
disagree = len(df) - agree

# ── Métricas completas de cada modelo ────────────────────────────────────────
def _compute_metrics(pred_binary, true_binary):
    tp = int(((pred_binary == True)  & (true_binary == True)).sum())
    tn = int(((pred_binary == False) & (true_binary == False)).sum())
    fp = int(((pred_binary == True)  & (true_binary == False)).sum())
    fn = int(((pred_binary == False) & (true_binary == True)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}

vader_m   = _compute_metrics(vader_binary, voted_binary)
roberta_m = _compute_metrics(roberta_binary, voted_binary)

# ── Resumen lado a lado ───────────────────────────────────────────────────────
w = 10  # ancho de columna
print("\n╔══════════════════════════════════════════════════════╗")
print("║           Comparación VADER vs RoBERTa              ║")
print(f"╠{'═'*28}╦{'═'*w}╦{'═'*10}╣")
print(f"║{'Métrica':28s}║{'VADER':^{w}}║{'RoBERTa':^10}║")
print(f"╠{'═'*28}╬{'═'*w}╬{'═'*10}╣")
for key, label in [("accuracy","Accuracy"), ("precision","Precision"),
                   ("recall","Recall"), ("f1_score","F1-score")]:
    v = vader_m[key]
    r = roberta_m[key]
    print(f"║{label:28s}║{v:^{w}.4f}║{r:^10.4f}║")
print(f"╚{'═'*28}╩{'═'*w}╩{'═'*10}╝")
print(f"\n  Coincidencia VADER↔RoBERTa: {agree}/{len(df)} ({agree/len(df)*100:.1f}%)")
print(f"  Diferencias VADER↔RoBERTa : {disagree}/{len(df)} ({disagree/len(df)*100:.1f}%)\n")

# Guardar métricas en JSON
metrics_out = {
    "vader":   {k: round(v, 4) for k, v in vader_m.items()},
    "roberta": {k: round(v, 4) for k, v in roberta_m.items()},
    "agreement": {
        "agree": agree,
        "disagree": disagree,
        "agree_pct": round(agree / len(df) * 100, 2),
    },
}
metrics_path = METRICS_DIR / "vader_vs_roberta_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=2, ensure_ascii=False)
print(f"    Métricas guardadas en: {metrics_path}")


# ── 6. Gráficos ───────────────────────────────────────────────────────────────
print("[6/6] Generando gráficos...")

# ── 6.1 Accuracy VADER vs RoBERTa ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
models = ["VADER", "RoBERTa"]
accs = [vader_m["accuracy"], roberta_m["accuracy"]]
bars = ax.bar(models, accs, color=["#2196F3", "#FF5722"], edgecolor="white", width=0.5)
for bar, val in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.4f}",
        ha="center", va="bottom", fontsize=12, fontweight="bold",
    )
ax.set_title("Accuracy vs voted_up de Steam\nVADER vs RoBERTa", fontsize=12)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.15)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "17_vader_vs_roberta_accuracy.png", dpi=150)
plt.close(fig)

# ── 6.2 Coincidencia entre modelos ───────────────────────────────────────────
# Este gráfico muestra cuántas reseñas etiquetan igual ambos modelos,
# independientemente de cuál sea correcto
fig, ax = plt.subplots(figsize=(6, 5))
labels = ["Coinciden", "Difieren"]
values = [agree, disagree]
colors = ["#4CAF50", "#F44336"]
bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + len(df) * 0.005,
        f"{val}\n({val / len(df) * 100:.1f}%)",
        ha="center", va="bottom", fontsize=10,
    )
ax.set_title("Coincidencia entre VADER y RoBERTa\n(etiqueta de sentimiento)", fontsize=12)
ax.set_ylabel("Número de reseñas")
ax.set_ylim(0, max(values) * 1.2)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "18_vader_roberta_agreement.png", dpi=150)
plt.close(fig)

print(f"    Gráficos guardados en: {FIGURES_DIR}")
print("\n  Análisis RoBERTa completado.")
print(f"    CSV con RoBERTa : {OUTPUT_CSV}")
print(f"    Gráficos        : {FIGURES_DIR}")
print(f"    Métricas        : {metrics_path}")
