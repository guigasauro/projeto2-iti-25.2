import argparse
import json
import os
import sys
import tempfile
from datetime import datetime

from modelo_ppm_utils import (
    DEFAULT_DATASET,
    DEFAULT_MODEL_DIR,
    DEFAULT_PPM_CMD,
    DEFAULT_SEPARATOR,
    PPMConfig,
    build_model_bytes,
    collect_class_files,
    compression_stats_for_bytes,
    measure_concatenated_size,
    sanitize_label,
    write_bytes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera um modelo textual por classe e precomputa o tamanho "
            "comprimido base para classificacao por modelo PPM."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DEFAULT_DATASET,
        help="Pasta raiz do dataset rotulado.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_MODEL_DIR,
        help="Pasta onde os modelos e o manifesto serao salvos.",
    )
    parser.add_argument(
        "--ppm-cmd",
        default=DEFAULT_PPM_CMD,
        help="Caminho para o executavel ppm.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Busca recursivamente arquivos .txt dentro de cada classe.",
    )
    parser.add_argument(
        "--include-completos",
        action="store_true",
        help="Tambem considera subpastas chamadas completos.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Limita a quantidade de arquivos usados por classe.",
    )
    parser.add_argument(
        "--max-bytes-per-class",
        type=int,
        default=0,
        help="Limita o tamanho bruto do modelo gerado por classe.",
    )
    parser.add_argument(
        "--balance-bytes",
        action="store_true",
        help=(
            "Balanceia os modelos usando o mesmo teto de bytes para todas "
            "as classes, com base na menor classe."
        ),
    )
    parser.add_argument(
        "--separator",
        default=DEFAULT_SEPARATOR,
        help="Separador inserido entre os textos concatenados.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Converte todo o texto dos modelos para minusculas.",
    )
    parser.add_argument(
        "--ppm-kmax",
        type=int,
        default=5,
        help="Parametro kmax do compressor PPM.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1000,
        help="Janela do reset adaptativo do compressor.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Limiar percentual do reset adaptativo.",
    )
    parser.add_argument(
        "--reset",
        type=int,
        choices=(0, 1),
        default=0,
        help="Ativa (1) ou desativa (0) o reset adaptativo.",
    )
    return parser.parse_args()


def build_manifest(args, dataset_dir, grouped_files, ppm_config):
    return {
        "version": 1,
        "strategy": "ppm_class_model_delta",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": dataset_dir,
        "recursive": args.recursive,
        "include_completos": args.include_completos,
        "limit_per_class": args.limit_per_class,
        "max_bytes_per_class": args.max_bytes_per_class,
        "balance_bytes": args.balance_bytes,
        "separator": args.separator,
        "lowercase": args.lowercase,
        "ppm_config": ppm_config.to_dict(),
        "classes": [],
    }


def main():
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset)
    output_dir = os.path.abspath(args.output_dir)
    ppm_cmd = os.path.abspath(args.ppm_cmd)
    ppm_config = PPMConfig(
        kmax=args.ppm_kmax,
        window=args.window,
        threshold=args.threshold,
        reset=args.reset,
    )

    if not os.path.isdir(dataset_dir):
        print(f"Erro: a pasta de dataset '{dataset_dir}' nao existe.", file=sys.stderr)
        return 1

    if not os.path.exists(ppm_cmd):
        print(f"Erro: nao encontrei o executavel ppm em '{ppm_cmd}'.", file=sys.stderr)
        print("Compile antes com: make", file=sys.stderr)
        return 1

    grouped_files = collect_class_files(
        dataset_dir,
        recursive=args.recursive,
        include_completos=args.include_completos,
        limit_per_class=args.limit_per_class,
    )

    if not grouped_files:
        print(
            "Nenhum arquivo .txt foi encontrado para gerar os modelos.\n"
            "Dica: gere os trechos com utils/preparar_dataset.py ou use "
            "--recursive/--include-completos se isso fizer sentido."
        )
        return 1

    os.makedirs(output_dir, exist_ok=True)
    manifest = build_manifest(args, dataset_dir, grouped_files, ppm_config)
    effective_max_bytes = args.max_bytes_per_class

    if args.balance_bytes:
        class_sizes = {
            class_name: measure_concatenated_size(
                grouped_files[class_name],
                separator=args.separator,
                lowercase=args.lowercase,
            )
            for class_name in grouped_files
        }
        balanced_limit = min(class_sizes.values())
        if effective_max_bytes > 0:
            balanced_limit = min(balanced_limit, effective_max_bytes)
        effective_max_bytes = balanced_limit
        manifest["balanced_target_bytes"] = balanced_limit

        print(
            "Balanceamento ativado: "
            f"cada modelo usara ate {balanced_limit} bytes brutos."
        )

    print("Gerando modelos por classe:")

    with tempfile.TemporaryDirectory(prefix="treinar_modelos_ppm_") as temp_dir:
        for class_name in sorted(grouped_files):
            model_bytes, selected_paths = build_model_bytes(
                grouped_files[class_name],
                separator=args.separator,
                lowercase=args.lowercase,
                max_bytes=effective_max_bytes,
            )

            if not model_bytes:
                print(f"  - {class_name}: ignorada, nenhum texto entrou no modelo.")
                continue

            model_file = sanitize_label(class_name) + ".txt"
            model_path = os.path.join(output_dir, model_file)
            write_bytes(model_path, model_bytes)

            compression_stats = compression_stats_for_bytes(
                ppm_cmd,
                ppm_config,
                model_bytes,
                temp_dir=temp_dir,
                suffix=".txt",
            )

            relative_sources = [
                os.path.relpath(path, dataset_dir) for path in selected_paths
            ]

            manifest["classes"].append(
                {
                    "name": class_name,
                    "model_file": model_file,
                    "raw_bytes": len(model_bytes),
                    "compressed_bytes": compression_stats["compressed_bytes"],
                    "compressed_bits": compression_stats["compressed_bits"],
                    "source_count": len(selected_paths),
                    "source_files": relative_sources,
                }
            )

            print(
                f"  - {class_name}: "
                f"{len(selected_paths)} arquivos, "
                f"{len(model_bytes)} bytes brutos, "
                f"{compression_stats['compressed_bytes']} bytes comprimidos, "
                f"{compression_stats['compressed_bits']} bits"
            )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print("\nModelos prontos.")
    print(f"Manifesto salvo em: {manifest_path}")
    print(f"Diretorio dos modelos: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
