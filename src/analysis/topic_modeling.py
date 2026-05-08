"""
Modelado de temas (LDA) sobre reseñas de Steam usando Gensim.
Carga el CSV con sentimiento, entrena LDA con 5 temas y genera
gráficos y una visualización interactiva con pyLDAvis.
"""

import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora, models

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Rutas del proyecto ────────────────────────────────────────────────────────
# Usamos Path(__file__) para que las rutas sean correctas independientemente
# del directorio desde el que se ejecute el script.
ROOT = Path(__file__).parents[2]

# Cargamos el CSV de sentimiento porque ya incluye text_lemmatized y vader_sentiment.
# Si solo existe el CSV limpio, las columnas de VADER simplemente no estarán.
INPUT_CSV = ROOT / "data" / "battlefield6_reviews_sentiment.csv"
OUTPUT_CSV = ROOT / "data" / "battlefield6_reviews_topics.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

NUM_TOPICS = 5          # número de temas a descubrir
RANDOM_STATE = 42       # semilla para reproducibilidad
NO_BELOW = 5            # ignorar tokens que aparezcan en menos de 5 documentos
NO_ABOVE = 0.70         # ignorar tokens que aparezcan en más del 70% de los docs
EXAMPLES_PER_TOPIC = 3  # reseñas representativas a mostrar por tema
SECONDARY_THRESHOLD = 0.20  # probabilidad mínima para considerar un tema secundario


# ── 1. Carga de datos ─────────────────────────────────────────────────────────
print("[1/8] Cargando CSV...")
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
df["voted_up"] = df["voted_up"].map(
    lambda x: True if str(x).strip().lower() == "true" else False
)
print(f"    Filas cargadas: {len(df)}")


# ── 2. Preparación de textos ──────────────────────────────────────────────────
# text_lemmatized ya está lematizado y sin stopwords; basta con separar por espacios.
# Usamos esta columna porque es la representación lingüísticamente más limpia.
print("[2/8] Tokenizando text_lemmatized...")
tokenized = df["text_lemmatized"].fillna("").apply(str.split).tolist()


# ── 3. Diccionario y corpus numérico ─────────────────────────────────────────
# filter_extremes elimina ruido: tokens muy raros (erratas) y ubicuos (sin valor discriminativo).
print("[3/8] Construyendo diccionario y corpus...")
dictionary = corpora.Dictionary(tokenized)
print(f"    Tokens antes del filtrado: {len(dictionary)}")
dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
print(f"    Tokens después del filtrado: {len(dictionary)}")

# bag-of-words: cada documento se convierte en lista de (token_id, frecuencia)
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]


# ── 4. Entrenamiento del modelo LDA ──────────────────────────────────────────
# chunksize y passes equilibran velocidad y convergencia para datasets medianos.
print(f"[4/8] Entrenando LDA con {NUM_TOPICS} temas...")
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=RANDOM_STATE,
    passes=10,          # número de pasadas completas sobre el corpus
    chunksize=100,      # documentos procesados por iteración
    alpha="auto",       # aprende la distribución de temas por documento
    eta="auto",         # aprende la distribución de palabras por tema
)
print("    Entrenamiento completado.")


# ── 5. Palabras clave por tema ────────────────────────────────────────────────
print("\n[5/8] Palabras clave por tema:")

topic_labels = {
    0: "Rendimiento",
    1: "Mecánicas de combate",
    2: "Valoración general positiva",
    3: "Ambientación y universo del juego",
    4: "Historia y personajes"
}

for topic_id in range(NUM_TOPICS):
    words = lda_model.show_topic(topic_id, topn=10)
    keywords = ", ".join(word for word, _ in words)
    print(f"    {topic_labels[topic_id]}: {keywords}")
print()


# ── 6. Asignación de tema dominante y temas secundarios ──────────────────────
print("[6/8] Asignando tema dominante y temas secundarios a cada reseña...")

def get_topic_assignment(bow):
    """
    Devuelve (dominant_id, [secondary_labels]) donde secondary_labels son los
    temas distintos del dominante cuya probabilidad supera SECONDARY_THRESHOLD.
    """
    distribution = lda_model.get_document_topics(bow, minimum_probability=0.0)
    if not distribution:
        return -1, []
    dominant_id = max(distribution, key=lambda x: x[1])[0]
    secondaries = [
        topic_labels[tid]
        for tid, prob in distribution
        if tid != dominant_id and prob >= SECONDARY_THRESHOLD
    ]
    return dominant_id, secondaries

assignments = [get_topic_assignment(bow) for bow in corpus]
df["dominant_topic"] = [a[0] for a in assignments]
df["secondary_topics"] = [", ".join(a[1]) if a[1] else "" for a in assignments]

mixed_count = (df["secondary_topics"] != "").sum()
print("    Distribución por tema dominante:")
print(df["dominant_topic"].value_counts().sort_index().to_string())
print(f"\n    Reseñas con temas secundarios (prob ≥ {SECONDARY_THRESHOLD}): {mixed_count} "
      f"({mixed_count / len(df) * 100:.1f} %)")


# ── 7. Ejemplos representativos por tema ─────────────────────────────────────
# Para cada tema ordenamos por probabilidad descendente y mostramos los top-N.
print(f"\n[7/8] {EXAMPLES_PER_TOPIC} ejemplos representativos por tema:")

def topic_probability(bow, topic_id):
    """Probabilidad de un tema concreto para un documento bow."""
    dist = dict(lda_model.get_document_topics(bow, minimum_probability=0.0))
    return dist.get(topic_id, 0.0)

for topic_id in range(NUM_TOPICS):
    # Calculamos la probabilidad del tema para cada doc y tomamos los mejores
    probs = [topic_probability(bow, topic_id) for bow in corpus]
    df_tmp = df.copy()
    df_tmp["_prob"] = probs
    top_rows = df_tmp.nlargest(EXAMPLES_PER_TOPIC, "_prob")

    words = lda_model.show_topic(topic_id, topn=8)
    keywords = ", ".join(word for word, _ in words)
    print(f"\n  ── {topic_labels[topic_id]} [{keywords}] ──")
    for _, row in top_rows.iterrows():
        print(f"    [{row.get('vader_sentiment', 'N/A')}] {row['text_cleaned'][:120]}")


# ── 8. Gráficos ───────────────────────────────────────────────────────────────
print("\n[8/8] Generando gráficos...")
TOPIC_COLORS = ["#5C6BC0", "#26A69A", "#FFA726", "#EF5350", "#66BB6A"]

# ── 8.1 Distribución de reseñas por tema dominante ───────────────────────────
counts = df["dominant_topic"].value_counts().sort_index()
labels = [topic_labels[i] for i in counts.index]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, counts.values,
              color=[TOPIC_COLORS[i] for i in counts.index], edgecolor="white")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
ax.set_title("Distribución de reseñas por tema dominante", fontsize=13)
ax.set_xlabel("Tema")
ax.set_ylabel("Número de reseñas")
ax.set_ylim(0, counts.max() * 1.15)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "06_topic_distribution.png", dpi=150)
plt.close(fig)

# ── 8.2 Tema dominante vs vader_sentiment ────────────────────────────────────
# Solo si existe la columna vader_sentiment (generada por sentiment_analysis.py).
if "vader_sentiment" in df.columns:
    cross_vs = (
        df.groupby(["dominant_topic", "vader_sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(range(NUM_TOPICS))
    )
    sentiment_colors = {"negative": "#EF5350", "neutral": "#FFC107", "positive": "#4CAF50"}
    fig, ax = plt.subplots(figsize=(10, 5))
    cross_vs.plot(
        kind="bar", ax=ax, rot=0,
        color=[sentiment_colors.get(c, "gray") for c in cross_vs.columns],
        edgecolor="white",
    )
    ax.set_title("Tema dominante vs vader_sentiment", fontsize=13)
    ax.set_xlabel("Tema dominante")
    ax.set_ylabel("Número de reseñas")
    ax.set_xticklabels([topic_labels[i] for i in cross_vs.index], rotation=15, ha="right")
    ax.legend(title="vader_sentiment")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_topic_vs_vader_sentiment.png", dpi=150)
    plt.close(fig)

# ── 8.3 Temas dominantes por categoría de sentimiento ────────────────────────
if "vader_sentiment" in df.columns:
    sentiment_colors = {"negative": "#EF5350", "neutral": "#FFC107", "positive": "#4CAF50"}
    for sentiment in ["negative", "neutral", "positive"]:
        subset = df[df["vader_sentiment"] == sentiment]
        if subset.empty:
            continue

        all_topic_counts = Counter()
        for _, row in subset.iterrows():
            all_topic_counts[topic_labels[row["dominant_topic"]]] += 1
            for t in row["secondary_topics"].split(", "):
                if t:
                    all_topic_counts[t] += 1
        all_labels = list(topic_labels.values())
        s_values = [all_topic_counts.get(lbl, 0) for lbl in all_labels]
        title = f"Todos los temas en reseñas {sentiment} (dominante + secundarios)"
        ylabel = "Apariciones"

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(all_labels, s_values,
                      color=sentiment_colors[sentiment], edgecolor="white", alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Tema")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(s_values) * 1.15 if max(s_values) > 0 else 1)
        ax.set_xticklabels(all_labels, rotation=15, ha="right")
        fig.tight_layout()
        fname = {"negative": "14", "neutral": "15", "positive": "16"}[sentiment]
        fig.savefig(FIGURES_DIR / f"{fname}_topics_{sentiment}.png", dpi=150)
        plt.close(fig)

print(f"    Gráficos guardados en: {FIGURES_DIR}")


# ── 9. Visualización interactiva con pyLDAvis ─────────────────────────────────
# pyLDAvis genera un HTML interactivo que permite explorar los temas visualmente.
print("[9/9] Generando visualización pyLDAvis...")
try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis  # API moderna (pyLDAvis >= 3.3)

    # prepare() calcula las distancias entre temas y las frecuencias de palabras
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    html_path = FIGURES_DIR / "lda_visualization.html"
    pyLDAvis.save_html(vis_data, str(html_path))
    print(f"    Visualización guardada en: {html_path}")
except ImportError:
    print("    AVISO: pyLDAvis no está instalado. Omitiendo visualización interactiva.")
    print("           Instala con: pip install pyldavis")
except Exception as exc:
    print(f"    AVISO: No se pudo generar la visualización pyLDAvis: {exc}")


# ── 10. Guardar CSV enriquecido ───────────────────────────────────────────────
print(f"[10/10] Guardando CSV con temas en: {OUTPUT_CSV}")
df["topic_label"] = df["dominant_topic"].map(topic_labels)
# secondary_topics: cadena vacía si la reseña es monotemática, o lista separada por comas
df.to_csv(OUTPUT_CSV, index=False)

print("\n── Proceso completado ──────────────────────────────────────")
print(f"  CSV de temas   : {OUTPUT_CSV}")
print(f"  Gráficos       : {FIGURES_DIR}")
print("────────────────────────────────────────────────────────────")
