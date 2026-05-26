# Battlefield 6 — Análisis de Reseñas de Steam

> **Trabajo académico** | Minería de texto y análisis de sentimiento sobre reseñas de la plataforma Steam

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E)](https://huggingface.co/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Descripción

Este proyecto implementa un **pipeline completo de procesamiento de lenguaje natural (NLP)** para analizar las reseñas de usuarios del videojuego *Battlefield 6* (App ID: `2807960`) extraídas de Steam. El objetivo es identificar patrones de sentimiento, descubrir los temas más debatidos por la comunidad y correlacionar el compromiso de los jugadores con su opinión sobre el juego.

El análisis combina técnicas clásicas de NLP (VADER, LDA) con modelos de *deep learning* basados en Transformers (RoBERTa), permitiendo una comparación robusta entre enfoques y una interpretación más rica de los datos.

---

## Estructura del proyecto

```
battlefied6_steam_review_analysis/
│
├── data/                                    # Datos en distintas etapas del pipeline
│   ├── battlefield6_reviews_raw.csv         # Extracción directa de la API de Steam
│   ├── battlefield6_reviews_clean.csv       # Texto limpio y lematizado
│   ├── battlefield6_reviews_sentiment.csv   # Enriquecido con puntuaciones RoBERTa
│   └── battlefield6_reviews_topics.csv      # Enriquecido con temas LDA
│
├── src/
│   ├── data_processing/
│   │   ├── fetch_reviews.py                 # Extracción paginada vía API de Steam
│   │   └── data_cleaning.py                 # Limpieza, normalización y lematización
│   └── analysis/
│       ├── sentiment_analysis.py            # Análisis de sentimiento con VADER
│       ├── roberta_sentiment.py             # Análisis de sentimiento con RoBERTa
│       ├── topic_modeling.py                # Modelado de temas con LDA (Gensim)
│       ├── complementary_analysis.py        # Análisis temporal y de engagement
│       └── manual_validation.py             # Validación manual de resultados
│
├── outputs/
│   ├── figures/                             # Gráficos y visualizaciones generadas (~25+)
│   │   └── lda_visualization.html           # Visualización interactiva pyLDAvis
│   └── metrics/                             # Métricas de rendimiento en JSON
│       ├── vader_metrics.json
│       └── roberta_metrics.json
│
├── test/
│   ├── check_gpu.py                         # Verifica disponibilidad de GPU (CUDA)
│   ├── test_ensemble.py                     # Pruebas del ensemble VADER + RoBERTa
│   └── count_params.ipynb                   # Notebook: conteo de parámetros del modelo
│
├── ldavis.py                                # Generación de visualización LDA estática
├── topic_figures.py                         # Generación de figuras adicionales de temas
├── main.py                                  # Orquestador del pipeline completo
├── requirements.txt                         # Dependencias del proyecto
└── README.md
```

---

## Pipeline de análisis

El pipeline se ejecuta en cuatro pasos secuenciales, todos coordinados desde `main.py`:

```
fetch  →  clean  →  sentiment  →  topics  →  extra
  ↓          ↓           ↓            ↓          ↓
API       spaCy      RoBERTa        LDA      Temporal +
Steam   + NLTK      (+ VADER)    Gensim     Engagement
```

| Paso        | Script                              | Descripción                                                   |
|-------------|-------------------------------------|---------------------------------------------------------------|
| `fetch`     | `src/data_processing/fetch_reviews.py` | Extrae reseñas paginadas de la Steam Web API                  |
| `clean`     | `src/data_processing/data_cleaning.py` | Limpieza, normalización, tokenización y lematización (spaCy)  |
| `sentiment` | `src/analysis/roberta_sentiment.py`    | Inferencia con `cardiffnlp/twitter-roberta-base-sentiment`    |
| `topics`    | `src/analysis/topic_modeling.py`       | Entrenamiento LDA con 5 temas + coherence score (c_v)         |
| `extra`     | `src/analysis/complementary_analysis.py` | Análisis temporal, engagement y correlaciones estadísticas  |

---

## Metodología

### 1. Extracción de datos
Las reseñas se obtienen directamente de la **Steam Web API** (`/appreviews/{app_id}`) mediante paginación por cursor. Se capturan campos como texto de la reseña, `voted_up`, horas jugadas (`playtime_forever`) y puntuación ponderada de utilidad (`weighted_vote_score`).

### 2. Limpieza y preprocesamiento
- Eliminación de HTML, caracteres especiales y URLs.
- Lematización con **spaCy** (`en_core_web_sm`).
- Filtrado de *stop words* genéricas y específicas del dominio gaming.
- Control del desbalanceo positivo/negativo mediante `--max-positives`.

### 3. Análisis de sentimiento

#### VADER (baseline)
Analizador léxico de **NLTK** calibrado para texto informal en redes sociales. Produce una puntuación compuesta (`vader_compound`) en el rango `[-1, +1]` y una etiqueta triclase (positive / neutral / negative).

#### RoBERTa (modelo principal)
Modelo `cardiffnlp/twitter-roberta-base-sentiment` descargado de Hugging Face. Se ejecuta con inferencia por *batches* (`BATCH_SIZE=32`) y soporte GPU/CPU automático. Produce `roberta_sentiment`, `roberta_score` y `roberta_compound`.

### 4. Modelado de temas (LDA)
Se entrena un modelo **Latent Dirichlet Allocation** (Gensim) con `NUM_TOPICS=5` sobre los textos lematizados, filtrando tokens por frecuencia (`no_below=10`, `no_above=0.40`). Los 5 temas identificados son:

| ID | Tema |
|----|------|
| 0  | Comparación con entregas anteriores |
| 1  | Mecánicas de combate y anti-cheat |
| 2  | Mapas, armas y vehículos |
| 3  | Problemas técnicos y servidores |
| 4  | Monetización y Battlepass |

La calidad del modelo se valida con el **coherence score c_v** (objetivo: > 0.45).

### 5. Análisis complementario
- Evolución temporal semanal del sentimiento y volumen de reseñas.
- Correlaciones de Pearson y Spearman entre `playtime_hours` / `weighted_vote_score` y `roberta_compound`.
- Distribución de temas por grupo (`voted_up=True` vs `voted_up=False`).
- Heatmap de sentimiento por tema dominante.

---

## Instalación y uso

### Requisitos previos
- Python 3.12
- (Opcional) GPU con CUDA 12.1 o 12.4 para inferencia RoBERTa acelerada

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/blooraayy/battlefied6_steam_review_analysis.git
cd battlefied6_steam_review_analysis

# Instalar dependencias (CPU)
pip install -r requirements.txt

# Instalar dependencias (GPU — CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

## Tecnologías

| Categoría | Librería / Herramienta |
|-----------|------------------------|
| Datos y cálculo | `pandas`, `numpy`, `scipy` |
| NLP clásico | `nltk` (VADER), `spacy`, `gensim` (LDA) |
| Deep Learning | `torch`, `transformers` (HuggingFace) |
| Visualización | `matplotlib`, `pyLDAvis`, `wordcloud` |
| Extracción datos | `requests` (Steam Web API) |
| Utilidades | `tqdm`, `pathlib` |

---

## Datos

Los datos en bruto se obtienen de la **Steam Web API pública** sin necesidad de autenticación. El archivo `data/battlefield6_reviews_raw.csv` no se incluye en el repositorio por su tamaño; puedes generarlo ejecutando:

```bash
python src/data_processing/fetch_reviews.py
```

Los archivos CSV intermedios (`_clean`, `_sentiment`, `_topics`) se producen automáticamente al ejecutar el pipeline completo.

---

## Licencia

Este trabajo está publicado bajo la licencia **Creative Commons Atribución-NoComercial 4.0 Internacional (CC BY-NC 4.0)**.

[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

**Eres libre de:**
- **Compartir** — copiar y redistribuir el material en cualquier medio o formato.
- **Adaptar** — remezclar, transformar y construir a partir del material.

**Bajo las siguientes condiciones:**
- **Atribución** — Debes dar crédito apropiado, proporcionar un enlace a la licencia e indicar si se realizaron cambios.
- **No Comercial** — No puedes utilizar el material con fines comerciales.

> El texto completo de la licencia está disponible en: https://creativecommons.org/licenses/by-nc/4.0/legalcode

---

## Cita

Si utilizas este trabajo en tu investigación, por favor cítalo de la siguiente manera:

```
blooraayy (2025). Battlefield 6 Steam Review Analysis:
NLP Pipeline para análisis de sentimiento y modelado de temas en Steam.
GitHub. https://github.com/blooraayy/battlefied6_steam_review_analysis
```

---

## Autor

**blooraayy** — Trabajo académico, 2025.

> *Este proyecto fue desarrollado con fines exclusivamente académicos como parte de un análisis de texto sobre temas de actualidad.*