import csv
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import spacy

RAW_PATH = Path(__file__).parents[2] / "data" / "battlefield6_reviews_raw.csv"
CLEAN_PATH = Path(__file__).parents[2] / "data" / "battlefield6_reviews_clean.csv"

KEEP_FIELDS = [
    "voted_up",
    "playtime_hours",
    "date",
    "weighted_vote_score",
    "text_cleaned",
    "text_lemmatized",
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPECIAL_RE = re.compile(r"[^a-z0-9\s.,;:?!]")
_SPACES_RE = re.compile(r"\s+")
_ONLY_NUMS_SYMS_RE = re.compile(r"^[^a-z]+$")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F02F"
    "\U0001F0A0-\U0001F0FF"
    "\U0001F100-\U0001F1FF"
    "⌀-⏿⬀-⯿]+",
    flags=re.UNICODE,
)

_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
]

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _is_valid(text_cleaned: str) -> bool:
    """Devuelve False si la reseña debe descartarse."""
    words = text_cleaned.split()
    if len(words) < 5:
        return False
    if _ONLY_NUMS_SYMS_RE.match(text_cleaned):
        return False
    # Descartar si alguna palabra se repite más de 5 veces
    from collections import Counter
    if max(Counter(words).values()) > 5:
        return False
    return True


def _is_only_emojis(raw_text: str) -> bool:
    """True si el texto original es únicamente emojis y espacios."""
    stripped = _EMOJI_RE.sub("", raw_text).strip()
    return stripped == ""


def _clean_text(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _SPECIAL_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()
    return text


def _lemmatize(text: str) -> str:
    nlp = _get_nlp()
    doc = nlp(text)
    tokens = list(doc)
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Detectar bigrama not_siguiente antes de aplicar stopwords/lematización
        if token.lower_ == "not" and i + 1 < len(tokens) and not tokens[i + 1].is_punct:
            next_token = tokens[i + 1]
            bigram = f"not_{next_token.lemma_.lower()}"
            result.append(bigram)
            i += 2
        else:
            if not token.is_stop and not token.is_punct and len(token.lemma_) >= 3:
                result.append(token.lemma_)
            i += 1
    return " ".join(result)


def _parse_date(value: str) -> str:
    value = value.strip()
    # Unix timestamp (entero)
    if value.isdigit():
        return datetime.utcfromtimestamp(int(value)).strftime("%Y-%m-%d %H:%M:%S")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return value  # si no se puede parsear, se deja tal cual


def _to_playtime_hours(minutes_str: str) -> str:
    try:
        return f"{int(minutes_str) / 60:.2f}"
    except (ValueError, ZeroDivisionError):
        return ""


def clean_reviews(
    input_path: Path = RAW_PATH,
    output_path: Path = CLEAN_PATH,
    max_positives: int | None = None,
) -> int:
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

    rng = random.Random(42)
    if max_positives is None:
        sampled_positives = list(positives)
        print("[3/5] Usando todas las reseñas positivas.")
    else:
        print(f"[3/5] Muestreando hasta {max_positives} reseñas positivas (random_state=42)...")
        sampled_positives = rng.sample(positives, min(max_positives, len(positives)))
    dataset = negatives + sampled_positives
    rng.shuffle(dataset)
    print(f"[3/5] Dataset final: {len(negatives)} negativas + {len(sampled_positives)} positivas = {len(dataset)} reseñas.")

    print("[4/5] Procesando y limpiando filas...")
    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=KEEP_FIELDS)
        writer.writeheader()
        count = 0
        skipped = 0
        for row in dataset:
            raw_text = row["text"]
            if _is_only_emojis(raw_text):
                skipped += 1
                continue
            text_cleaned = _clean_text(raw_text)
            if not _is_valid(text_cleaned):
                skipped += 1
                continue
            text_lemmatized = _lemmatize(text_cleaned)
            if not text_lemmatized.strip():
                skipped += 1
                continue
            writer.writerow({
                "voted_up": row["voted_up"],
                "playtime_hours": _to_playtime_hours(row["playtime_forever"]),
                "date": _parse_date(row["date"]),
                "weighted_vote_score": row["weighted_vote_score"],
                "text_cleaned": text_cleaned,
                "text_lemmatized": text_lemmatized,
            })
            count += 1
            if count % 100 == 0:
                print(f"        {count}/{len(dataset)} filas procesadas...")
        print(f"        Filas eliminadas por text_lemmatized vacío: {skipped}")

    print(f"[5/5] Guardando resultado en: {output_path}")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Limpieza de reseñas de Steam")
    parser.add_argument(
        "--max-positives", type=int, default=None,
        help="Número máximo de reseñas positivas a incluir (default: todas).",
    )
    args = parser.parse_args()

    processed = clean_reviews(max_positives=args.max_positives)
    print(f"\nFilas procesadas: {processed}")
    print(f"Archivo guardado en: {CLEAN_PATH}")
