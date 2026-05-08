from pathlib import Path

import requests
import pandas as pd
import time

APP_ID = 2807960
LANGUAGE = "english"
NUM_PER_PAGE = 100
MAX_REVIEWS = None       # cambiar a None para sacar todas

def fetch_all_reviews(app_id, language):
    reviews = []
    cursor = "*"
    page = 1

    print(f"Iniciando extracción de reseñas en {language}...")

    while True:
        params = {
            "json": 1,
            "language": language,
            "filter": "recent",
            "num_per_page": NUM_PER_PAGE,
            "cursor": cursor,
            "purchase_type": "all"
        }

        response = requests.get(
            f"https://store.steampowered.com/appreviews/{app_id}",
            params=params
        )

        if response.status_code != 200:
            print(f"Error en la petición: {response.status_code}")
            break

        data = response.json()

        # Comprobar si hay reseñas en esta página
        batch = data.get("reviews", [])
        if not batch:
            print("No hay más reseñas disponibles.")
            break

        for review in batch:
            reviews.append({
                "review_id":           review.get("recommendationid"),
                "text":                review.get("review"),
                "language":            review.get("language"),
                "voted_up":            review.get("voted_up"),
                "timestamp_created":   review.get("timestamp_created"),
                "playtime_forever":    review.get("author", {}).get("playtime_forever"),
                "votes_helpful":       review.get("votes_helpful", 0),
                "votes_funny":         review.get("votes_funny", 0),
                "weighted_vote_score": review.get("weighted_vote_score", 0)
            })

            if MAX_REVIEWS is not None and len(reviews) >= MAX_REVIEWS:
                print(f"Límite de {MAX_REVIEWS} reseñas alcanzado.")
                return reviews

        print(f"Página {page} — reseñas acumuladas: {len(reviews)}")

        # Actualizar cursor para la siguiente página
        new_cursor = data.get("cursor", "")
        if not new_cursor or new_cursor == cursor:
            print("Cursor sin cambios, fin de la extracción.")
            break

        cursor = new_cursor
        page += 1

        # Pausa para no sobrecargar la API
        time.sleep(1)

    return reviews


def main():
    reviews = fetch_all_reviews(APP_ID, LANGUAGE)

    if not reviews:
        print("No se han obtenido reseñas.")
        return

    df = pd.DataFrame(reviews)

    # Convertir timestamp a fecha legible
    df["date"] = pd.to_datetime(df["timestamp_created"], unit="s")

    # Resumen básico
    print("\n--- RESUMEN ---")
    print(f"Total reseñas extraídas: {len(df)}")
    print(f"Reseñas positivas (voted_up=True):  {df['voted_up'].sum()}")
    print(f"Reseñas negativas (voted_up=False): {(~df['voted_up']).sum()}")
    print(f"Fecha más antigua: {df['date'].min()}")
    print(f"Fecha más reciente: {df['date'].max()}")

    # Guardar CSV
    output_path = Path(__file__).parents[2] / "data" / "battlefield6_reviews_raw.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nDataset guardado en: {output_path}")


if __name__ == "__main__":
    main()