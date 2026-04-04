import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict

from modelo_ppm_utils import (
    DEFAULT_MODEL_DIR,
    DEFAULT_PPM_CMD,
    PPMConfig,
    compression_stats_for_bytes,
    read_text_bytes,
    read_text_for_chunking,
    split_text_into_chunks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classifica um texto usando modelos por classe e o criterio "
            "de menor custo incremental no compressor PPM."
        )
    )
    parser.add_argument(
        "input_file",
        help="Arquivo de texto a ser classificado.",
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=DEFAULT_MODEL_DIR,
        help="Pasta onde estao os modelos gerados e o manifesto.",
    )
    parser.add_argument(
        "--ppm-cmd",
        default=DEFAULT_PPM_CMD,
        help="Caminho para o executavel ppm.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help=(
            "Se maior que zero, divide o texto em trechos desse tamanho "
            "aproximado e faz votacao entre os trechos."
        ),
    )
    parser.add_argument(
        "--min-chunk-bytes",
        type=int,
        default=500,
        help="Tamanho minimo, em bytes, para um trecho ser considerado.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Quantidade de classes exibidas no ranking final.",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Mostra o vencedor e o ranking de cada trecho.",
    )
    return parser.parse_args()


def load_manifest(model_dir):
    manifest_path = os.path.join(model_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Nao encontrei o manifesto em '{manifest_path}'. "
            "Rode primeiro o script de treino dos modelos."
        )

    with open(manifest_path, "r", encoding="utf-8") as file:
        return json.load(file), manifest_path


def load_models(model_dir, manifest):
    models = []

    for entry in manifest.get("classes", []):
        model_path = os.path.join(model_dir, entry["model_file"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo ausente para a classe '{entry['name']}': {model_path}"
            )

        with open(model_path, "rb") as file:
            model_bytes = file.read()

        models.append(
            {
                "name": entry["name"],
                "model_path": model_path,
                "model_bytes": model_bytes,
                "compressed_bytes": int(entry["compressed_bytes"]),
                "compressed_bits": int(entry.get("compressed_bits", entry["compressed_bytes"] * 8)),
                "raw_bytes": int(entry["raw_bytes"]),
            }
        )

    if not models:
        raise RuntimeError("O manifesto nao possui modelos de classe cadastrados.")

    return models


def load_chunks(input_file, lowercase, chunk_size, min_chunk_bytes):
    if chunk_size > 0:
        text = read_text_for_chunking(input_file, lowercase=lowercase)
        chunks = split_text_into_chunks(
            text,
            chunk_size=chunk_size,
            min_chunk_bytes=min_chunk_bytes,
        )
        return chunks

    data = read_text_bytes(input_file, lowercase=lowercase)
    return [data] if data else []


def classify_chunks(chunks, models, ppm_cmd, ppm_config, separator, show_chunks):
    separator_bytes = separator.encode("utf-8")
    votes = Counter()
    delta_bits_by_class = defaultdict(int)
    chunk_results = []

    with tempfile.TemporaryDirectory(prefix="classificar_por_modelo_") as temp_dir:
        for index, chunk in enumerate(chunks, start=1):
            ranking = []

            for model in models:
                combined = model["model_bytes"] + separator_bytes + chunk
                compression_stats = compression_stats_for_bytes(
                    ppm_cmd,
                    ppm_config,
                    combined,
                    temp_dir=temp_dir,
                    suffix=".txt",
                )
                delta_bits = compression_stats["compressed_bits"] - model["compressed_bits"]
                bits_per_byte = delta_bits / max(1, len(chunk))

                ranking.append(
                    {
                        "class": model["name"],
                        "delta_bits": delta_bits,
                        "bits_per_byte": bits_per_byte,
                    }
                )

                delta_bits_by_class[model["name"]] += delta_bits

            ranking.sort(key=lambda item: (item["delta_bits"], item["class"]))
            winner = ranking[0]["class"]
            votes[winner] += 1

            chunk_result = {
                "index": index,
                "size_bytes": len(chunk),
                "winner": winner,
                "ranking": ranking,
            }
            chunk_results.append(chunk_result)

            if show_chunks:
                print(
                    f"Trecho {index}: vencedor = {winner} "
                    f"({len(chunk)} bytes)"
                )
                for item in ranking:
                    print(
                        f"  - {item['class']}: "
                        f"delta={item['delta_bits']} bits, "
                        f"bits/byte={item['bits_per_byte']:.4f}"
                    )

    summary = []
    total_bytes = sum(len(chunk) for chunk in chunks)

    for model in models:
        class_name = model["name"]
        delta_bits = delta_bits_by_class[class_name]
        bits_per_byte = delta_bits / max(1, total_bytes)
        summary.append(
            {
                "class": class_name,
                "votes": votes[class_name],
                "delta_bits": delta_bits,
                "bits_per_byte": bits_per_byte,
            }
        )

    summary.sort(key=lambda item: (-item["votes"], item["bits_per_byte"], item["class"]))
    return summary, chunk_results


def main():
    args = parse_args()
    input_file = os.path.abspath(args.input_file)
    model_dir = os.path.abspath(args.model_dir)
    ppm_cmd = os.path.abspath(args.ppm_cmd)

    if not os.path.exists(input_file):
        print(f"Erro: arquivo de entrada '{input_file}' nao existe.", file=sys.stderr)
        return 1

    if not os.path.exists(ppm_cmd):
        print(f"Erro: nao encontrei o executavel ppm em '{ppm_cmd}'.", file=sys.stderr)
        print("Compile antes com: make", file=sys.stderr)
        return 1

    try:
        manifest, manifest_path = load_manifest(model_dir)
        models = load_models(model_dir, manifest)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    ppm_config = PPMConfig.from_dict(manifest["ppm_config"])
    lowercase = bool(manifest.get("lowercase", False))
    separator = manifest.get("separator", "\n\n")

    chunks = load_chunks(
        input_file,
        lowercase=lowercase,
        chunk_size=args.chunk_size,
        min_chunk_bytes=args.min_chunk_bytes,
    )

    if not chunks:
        print("Erro: nenhum conteudo util foi encontrado no arquivo de entrada.", file=sys.stderr)
        return 1

    print(f"Manifesto carregado: {manifest_path}")
    print(f"Texto de entrada: {input_file}")
    print(f"Quantidade de modelos: {len(models)}")
    print(f"Quantidade de trechos avaliados: {len(chunks)}")

    summary, _ = classify_chunks(
        chunks,
        models,
        ppm_cmd=ppm_cmd,
        ppm_config=ppm_config,
        separator=separator,
        show_chunks=args.show_chunks,
    )

    predicted = summary[0]["class"]

    print("\nRanking final:")
    for item in summary[: max(1, args.top)]:
        print(
            f"  - {item['class']}: "
            f"votos={item['votes']}, "
            f"delta_total={item['delta_bits']} bits, "
            f"bits/byte={item['bits_per_byte']:.4f}"
        )

    print(f"\nClasse prevista: {predicted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
