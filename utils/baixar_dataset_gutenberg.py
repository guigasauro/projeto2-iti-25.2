import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

try:
    from preparar_dataset import process_file
except ModuleNotFoundError:
    from utils.preparar_dataset import process_file


WORKS = [
    {
        "class": "romantismo",
        "title": "Iracema",
        "author": "José Martiniano de Alencar",
        "ebook_id": 67740,
    },
    {
        "class": "romantismo",
        "title": "O Guarany Vol. 1",
        "author": "José Martiniano de Alencar",
        "ebook_id": 67724,
    },
    {
        "class": "romantismo",
        "title": "O Guarany Vol. 2",
        "author": "José Martiniano de Alencar",
        "ebook_id": 67725,
    },
    {
        "class": "romantismo",
        "title": "Ubirajara",
        "author": "José Martiniano de Alencar",
        "ebook_id": 38496,
    },
    {
        "class": "romantismo",
        "title": "Cinco Minutos",
        "author": "José Martiniano de Alencar",
        "ebook_id": 44540,
    },
    {
        "class": "romantismo",
        "title": "A Pata da Gazella",
        "author": "José Martiniano de Alencar",
        "ebook_id": 67831,
    },
    {
        "class": "realismo",
        "title": "Memorias Posthumas de Braz Cubas",
        "author": "Machado de Assis",
        "ebook_id": 54829,
    },
    {
        "class": "realismo",
        "title": "Quincas Borba",
        "author": "Machado de Assis",
        "ebook_id": 55682,
    },
    {
        "class": "realismo",
        "title": "Dom Casmurro",
        "author": "Machado de Assis",
        "ebook_id": 55752,
    },
    {
        "class": "realismo",
        "title": "Esau e Jacob",
        "author": "Machado de Assis",
        "ebook_id": 56737,
    },
    {
        "class": "realismo",
        "title": "Memorial de Ayres",
        "author": "Machado de Assis",
        "ebook_id": 55797,
    },
    {
        "class": "realismo",
        "title": "O Cortico",
        "author": "Aluisio Azevedo",
        "ebook_id": 69187,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Baixa um dataset novo do Project Gutenberg em uma pasta separada "
            "e gera os trechos no formato usado pelo projeto."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_gutenberg",
        help="Pasta de destino do novo dataset.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3000,
        help="Tamanho aproximado dos trechos gerados em bytes.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="So baixa os textos completos, sem gerar os trechos.",
    )
    return parser.parse_args()


def sanitize_filename(name):
    safe = name.strip().lower()
    safe = safe.replace(" ", "_")
    safe = re.sub(r"[^a-z0-9_]+", "", safe)
    return safe or "obra"


def gutenberg_text_url(ebook_id):
    return f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"


def strip_gutenberg_boilerplate(text):
    text = text.lstrip("\ufeff")
    start_match = re.search(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
    end_match = re.search(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)

    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]

    return text.strip() + "\n"


def normalize_local_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()

    cleaned_text = strip_gutenberg_boilerplate(text)

    with open(path, "w", encoding="utf-8") as file:
        file.write(cleaned_text)


def download_text(url, retries=3, timeout=120):
    last_error = None

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (dataset-gutenberg-builder)"},
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return raw.decode("utf-8", errors="ignore")
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2)

    raise last_error


def ensure_dirs(base_dir):
    for class_name in ("romantismo", "realismo"):
        Path(base_dir, class_name, "completos").mkdir(parents=True, exist_ok=True)


def write_sources_json(base_dir, sources):
    path = Path(base_dir, "sources.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(sources, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    ensure_dirs(output_dir)

    sources = []
    failures = []

    print(f"Baixando obras do Project Gutenberg em: {output_dir}")

    for work in WORKS:
        url = gutenberg_text_url(work["ebook_id"])
        filename = sanitize_filename(work["title"]) + ".txt"
        output_path = Path(output_dir, work["class"], "completos", filename)

        print(f"  - [{work['class']}] {work['title']} ({work['ebook_id']})")

        try:
            if output_path.exists():
                print(f"    reaproveitando arquivo ja baixado: {output_path.name}")
                normalize_local_text_file(output_path)
            else:
                text = download_text(url)
                cleaned_text = strip_gutenberg_boilerplate(text)

                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(cleaned_text)

            if not args.skip_prepare:
                class_dir = Path(output_dir, work["class"])
                process_file(str(output_path), str(class_dir), chunk_size=args.chunk_size)

            sources.append(
                {
                    **work,
                    "url": url,
                    "output_file": str(output_path.relative_to(output_dir)),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    **work,
                    "url": url,
                    "error": str(exc),
                }
            )
            print(f"    falha ao baixar/processar: {exc}")

    write_sources_json(output_dir, sources)

    print("\nDataset Gutenberg criado com sucesso.")
    print(f"Pasta raiz: {output_dir}")
    print(f"Fontes: {os.path.join(output_dir, 'sources.json')}")
    if args.skip_prepare:
        print("Trechos: nao gerados (--skip-prepare).")
    else:
        print("Trechos: gerados nas pastas de cada classe.")

    if failures:
        failures_path = os.path.join(output_dir, "failures.json")
        with open(failures_path, "w", encoding="utf-8") as file:
            json.dump(failures, file, ensure_ascii=False, indent=2)
        print(f"Falhas registradas em: {failures_path}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nOperacao interrompida.", file=sys.stderr)
        raise SystemExit(130)
