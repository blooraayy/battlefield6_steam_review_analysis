"""
Genera una muestra de 45 reseñas para validación manual del modelo de sentimiento.

La muestra está estratificada en tres grupos de 15:
  - 15 donde RoBERTa y voted_up COINCIDEN en positivo
  - 15 donde RoBERTa y voted_up COINCIDEN en negativo
  - 15 donde RoBERTa y voted_up NO COINCIDEN (casos más difíciles: ironía, sarcasmo...)

El CSV de salida incluye columnas vacías 'human_label' y 'notes' para rellenar a mano.
"""

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]
INPUT_CSV  = ROOT / "data" / "battlefield6_reviews_sentiment.csv"
OUTPUT_CSV = ROOT / "outputs" / "manual_validation_sample.csv"

N_PER_GROUP = 15
RANDOM_SEED = 42

# ── 1. Carga ──────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df["voted_up"] = df["voted_up"].map(
    lambda x: True if str(x).strip().lower() == "true" else False
)

# ── 2. Clasificar cada fila ───────────────────────────────────────────────────
# roberta_positive = True si el modelo predice "positive"
roberta_pos = df["roberta_sentiment"] == "positive"

agree_pos  = df[ roberta_pos  &  df["voted_up"]]   # modelo + / Steam +
agree_neg  = df[~roberta_pos  & ~df["voted_up"]]   # modelo - o neutral / Steam -
disagree   = df[ roberta_pos  & ~df["voted_up"] |  # modelo + / Steam -
               (~roberta_pos  &  df["voted_up"])]  # modelo - o neutral / Steam +

print(f"Coinciden positivo  : {len(agree_pos)}")
print(f"Coinciden negativo  : {len(agree_neg)}")
print(f"Desacuerdo          : {len(disagree)}")

# ── 3. Muestreo estratificado ─────────────────────────────────────────────────
rng = random.Random(RANDOM_SEED)

def sample_group(group_df, n):
    n = min(n, len(group_df))
    return group_df.sample(n=n, random_state=RANDOM_SEED)

s_pos      = sample_group(agree_pos, N_PER_GROUP)
s_neg      = sample_group(agree_neg, N_PER_GROUP)
s_disagree = sample_group(disagree,  N_PER_GROUP)

sample = pd.concat([s_pos, s_neg, s_disagree]).reset_index(drop=True)
sample.index += 1  # numerar desde 1

# ── 4. Columnas de salida ─────────────────────────────────────────────────────
output = pd.DataFrame({
    "id":                sample.index,
    "text":              sample["text_cleaned"],
    "voted_up_steam":    sample["voted_up"].map({True: "positive", False: "negative"}),
    "roberta_sentiment": sample["roberta_sentiment"],
    "roberta_score":     sample["roberta_score"].round(3),
    "grupo":             (
        ["coincide_positivo"] * len(s_pos) +
        ["coincide_negativo"] * len(s_neg) +
        ["desacuerdo"]        * len(s_disagree)
    ),
    "human_label":       "",   # rellenar: positive / neutral / negative
    "notes":             "",   # observaciones: ironía, sarcasmo, ambigüedad...
})

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\nMuestra guardada en: {OUTPUT_CSV}")
print(f"Total reseñas       : {len(output)}")
print("\nInstrucciones:")
print("  1. Abre el CSV en Excel o cualquier editor.")
print("  2. Lee cada reseña en la columna 'text'.")
print("  3. Escribe tu etiqueta en 'human_label': positive / neutral / negative")
print("  4. Anota en 'notes' si detectas ironía, sarcasmo, ambigüedad, etc.")
print("  5. Compara tu etiqueta con 'roberta_sentiment' y comenta diferencias.")
