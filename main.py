"""
Pipeline de análisis de reseñas de Steam para Pragmata.

Pasos:
  1. clean     — Limpia y lematiza el texto (controla reseñas positivas con --max-positives)
  2. sentiment — Análisis de sentimiento RoBERTa
  3. topics    — Modelado de temas LDA
  4. extra     — Análisis complementario (temporal + engagement)

Uso rápido:
  python main.py                          # pipeline completo con valores por defecto
  python main.py --max-positives 1000     # limitar a 1000 reseñas positivas
  python main.py --steps clean sentiment  # ejecutar solo esos pasos
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

SCRIPTS = {
    "clean":     ROOT / "src" / "data_processing" / "data_cleaning.py",
    "sentiment": ROOT / "src" / "analysis" / "roberta_sentiment.py",
    "topics":    ROOT / "src" / "analysis" / "topic_modeling.py",
    "extra":     ROOT / "src" / "analysis" / "complementary_analysis.py",
}

STEP_ORDER = ["clean", "sentiment", "topics", "extra"]

STEP_DESCRIPTIONS = {
    "clean":     "Limpieza y lematización del texto",
    "sentiment": "Análisis de sentimiento RoBERTa",
    "topics":    "Modelado de temas LDA",
    "extra":     "Análisis complementario (temporal + engagement)",
}


def _separator(title: str) -> None:
    width = 62
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _run_step(name: str, extra_args: list[str]) -> bool:
    script = SCRIPTS[name]
    cmd = [sys.executable, str(script)] + extra_args
    _separator(f"PASO: {STEP_DESCRIPTIONS[name]}")
    print(f"  Script : {script.relative_to(ROOT)}")
    if extra_args:
        print(f"  Args   : {' '.join(extra_args)}")
    print()

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  ERROR: el paso '{name}' ha fallado (código {result.returncode}).")
        return False

    print(f"\n  Completado en {elapsed:.1f} s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de análisis de reseñas Steam — Pragmata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-positives", type=int, default=None, metavar="N",
        help="Máximo de reseñas positivas a incluir en la limpieza (default: todas).",
    )
    parser.add_argument(
        "--steps", nargs="+", choices=STEP_ORDER, metavar="PASO",
        help=(
            "Pasos a ejecutar. Opciones: fetch clean sentiment topics extra. "
            "Si se omite, se ejecutan todos en orden."
        ),
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Detener el pipeline si un paso falla (por defecto continúa).",
    )
    args = parser.parse_args()

    # Determinar qué pasos ejecutar
    steps = args.steps if args.steps else STEP_ORDER

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        Pipeline — Análisis de Reseñas Steam · Pragmata       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Pasos a ejecutar : {', '.join(steps)}")
    print(f"  Max. positivas   : {args.max_positives if args.max_positives is not None else 'todas'}")

    pipeline_start = time.perf_counter()
    failed = []

    for step in steps:
        # Argumentos extra por paso
        extra: list[str] = []
        if step == "clean" and args.max_positives is not None:
            extra = ["--max-positives", str(args.max_positives)]

        ok = _run_step(step, extra)
        if not ok:
            failed.append(step)
            if args.stop_on_error:
                print("\n  Pipeline detenido por --stop-on-error.")
                break

    total = time.perf_counter() - pipeline_start
    _separator("RESUMEN")
    print(f"  Tiempo total : {total:.1f} s")
    if failed:
        print(f"  Pasos fallidos: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("  Todos los pasos completados correctamente.")


if __name__ == "__main__":
    main()
