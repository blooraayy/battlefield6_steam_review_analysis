import csv
import random
import re
from pathlib import Path

import spacy

RAW_PATH = Path(__file__).parents[2] / "data" / "pragmata_reviews_raw.csv"
CLEAN_PATH = Path(__file__).parents[2] / "data" / "pragmata_reviews_clean.csv"

KEEP_FIELDS = ["voted_up", "text_cleaned", "text_lemmatized"]

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPECIAL_RE = re.compile(r"[^a-z0-9\s.,;:?!]")
_SPACES_RE = re.compile(r"\s+")

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _clean_text(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _SPECIAL_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()
    return text


def _lemmatize(text: str) -> str:
    nlp = _get_nlp()
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.lemma_) >= 3
    ]
    return " ".join(tokens)


def clean_reviews(input_path: Path = RAW_PATH, output_path: Path = CLEAN_PATH) -> int:
    print("[1/5] Cargando modelo spaCy...")
    _get_nlp()
    print("[1/5] Modelo cargado.")

    print(f"[2/5] Leyendo CSV de entrada: {input_path}")
    with open(input_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        positives = []
        negatives = []
        for row in reader:
            if row["voted_up"].strip().lower() == "true":
                positives.append(row)
            else:
                negatives.append(row)

    print(f"[2/5] Reseñas positivas encontradas: {len(positives)}")
    print(f"[2/5] Reseñas negativas encontradas: {len(negatives)}")

    print("[3/5] Muestreando 1200 reseñas positivas (random_state=42)...")
    rng = random.Random(42)
    sampled_positives = rng.sample(positives, min(1200, len(positives)))
    dataset = negatives + sampled_positives
    rng.shuffle(dataset)
    print(f"[3/5] Dataset final: {len(negatives)} negativas + {len(sampled_positives)} positivas = {len(dataset)} reseñas.")

    print("[4/5] Procesando y limpiando filas...")
    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=KEEP_FIELDS)
        writer.writeheader()
        count = 0
        for row in dataset:
            text_cleaned = _clean_text(row["text"])
            text_lemmatized = _lemmatize(text_cleaned)
            writer.writerow({
                "voted_up": row["voted_up"],
                "text_cleaned": text_cleaned,
                "text_lemmatized": text_lemmatized,
            })
            count += 1
            if count % 100 == 0:
                print(f"        {count}/{len(dataset)} filas procesadas...")

    print(f"[5/5] Guardando resultado en: {output_path}")
    return count


if __name__ == "__main__":
    processed = clean_reviews()
    print(f"\nFilas procesadas: {processed}")
    print(f"Archivo guardado en: {CLEAN_PATH}")
