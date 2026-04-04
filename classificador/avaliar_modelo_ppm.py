import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from classificar_por_modelo import classify_chunks
from modelo_ppm_utils import (
    DEFAULT_DATASET,
    DEFAULT_PPM_CMD,
    DEFAULT_SEPARATOR,
    PPMConfig,
    build_model_bytes,
    collect_class_files,
    compression_stats_for_bytes,
    measure_concatenated_size,
    read_text_bytes,
    read_text_for_chunking,
    split_text_into_chunks,
)


@dataclass(frozen=True)
class Amostra:
    indice: int
    classe: str
    caminho: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Avalia o classificador por modelo de classe com PPM usando "
            "validacao leave-one-out."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DEFAULT_DATASET,
        help="Pasta raiz do dataset rotulado.",
    )
    parser.add_argument(
        "--ppm-cmd",
        default=DEFAULT_PPM_CMD,
        help="Caminho para o executavel ppm.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Numero de amostras avaliadas em paralelo.",
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
        help="Limita a quantidade de amostras por classe para testes rapidos.",
    )
    parser.add_argument(
        "--max-bytes-per-class",
        type=int,
        default=0,
        help="Limita o tamanho bruto dos modelos de classe em cada fold.",
    )
    parser.add_argument(
        "--balance-bytes",
        action="store_true",
        help=(
            "Balanceia os modelos usando o mesmo teto de bytes para todas "
            "as classes em cada fold."
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
        help="Converte os textos para minusculas antes da avaliacao.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help=(
            "Se maior que zero, divide cada amostra em trechos desse tamanho "
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
        "--csv",
        default="",
        help="Se informado, salva o resumo final em CSV.",
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


def collect_samples(grouped_files):
    samples = []
    for class_name in sorted(grouped_files):
        for path in grouped_files[class_name]:
            samples.append(Amostra(len(samples), class_name, os.path.abspath(path)))
    return samples


def print_dataset_summary(grouped_files):
    print("Resumo do dataset:")
    for class_name in sorted(grouped_files):
        print(f"  - {class_name}: {len(grouped_files[class_name])} amostras")


def load_sample_chunks(sample_path, lowercase, chunk_size, min_chunk_bytes):
    if chunk_size > 0:
        text = read_text_for_chunking(sample_path, lowercase=lowercase)
        return split_text_into_chunks(
            text,
            chunk_size=chunk_size,
            min_chunk_bytes=min_chunk_bytes,
        )

    data = read_text_bytes(sample_path, lowercase=lowercase)
    return [data] if data else []


def resolve_effective_max_bytes(grouped_training_files, args):
    effective_max_bytes = args.max_bytes_per_class

    if args.balance_bytes:
        class_sizes = {
            class_name: measure_concatenated_size(
                grouped_training_files[class_name],
                separator=args.separator,
                lowercase=args.lowercase,
            )
            for class_name in grouped_training_files
        }
        balanced_limit = min(class_sizes.values())
        if effective_max_bytes > 0:
            balanced_limit = min(balanced_limit, effective_max_bytes)
        effective_max_bytes = balanced_limit

    return effective_max_bytes


def build_fold_models(sample, grouped_files, ppm_cmd, ppm_config, args):
    grouped_training_files = {}

    for class_name in sorted(grouped_files):
        training_files = grouped_files[class_name]
        if class_name == sample.classe:
            training_files = [path for path in training_files if path != sample.caminho]

        if not training_files:
            raise RuntimeError(
                f"A classe '{class_name}' ficou sem amostras de treino no fold atual."
            )

        grouped_training_files[class_name] = training_files

    effective_max_bytes = resolve_effective_max_bytes(grouped_training_files, args)

    models = []
    for class_name in sorted(grouped_training_files):
        model_bytes, selected_paths = build_model_bytes(
            grouped_training_files[class_name],
            separator=args.separator,
            lowercase=args.lowercase,
            max_bytes=effective_max_bytes,
        )

        if not model_bytes:
            raise RuntimeError(
                f"Nao foi possivel montar o modelo da classe '{class_name}' "
                "no fold atual."
            )

        stats = compression_stats_for_bytes(
            ppm_cmd,
            ppm_config,
            model_bytes,
            suffix=".txt",
        )

        models.append(
            {
                "name": class_name,
                "model_bytes": model_bytes,
                "compressed_bytes": stats["compressed_bytes"],
                "compressed_bits": stats["compressed_bits"],
                "raw_bytes": len(model_bytes),
                "source_count": len(selected_paths),
            }
        )

    return models


def evaluate_sample(sample, grouped_files, ppm_cmd, ppm_config, args):
    models = build_fold_models(sample, grouped_files, ppm_cmd, ppm_config, args)
    chunks = load_sample_chunks(
        sample.caminho,
        lowercase=args.lowercase,
        chunk_size=args.chunk_size,
        min_chunk_bytes=args.min_chunk_bytes,
    )

    if not chunks:
        raise RuntimeError(
            f"A amostra '{sample.caminho}' nao gerou nenhum trecho valido para avaliar."
        )

    summary, _ = classify_chunks(
        chunks,
        models,
        ppm_cmd=ppm_cmd,
        ppm_config=ppm_config,
        separator=args.separator,
        show_chunks=False,
    )

    predicted = summary[0]["class"]
    winning_bits_per_byte = summary[0]["bits_per_byte"]

    return {
        "index": sample.indice,
        "real_class": sample.classe,
        "predicted_class": predicted,
        "path": sample.caminho,
        "chunks": len(chunks),
        "bits_per_byte": winning_bits_per_byte,
    }


def print_progress(done, total):
    step = max(1, total // 10)
    if done == total or done == 1 or done % step == 0:
        print(f"Avaliacao PPM: {done}/{total}")


def evaluate_leave_one_out(samples, grouped_files, ppm_cmd, ppm_config, args):
    results = []

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(
                evaluate_sample,
                sample,
                grouped_files,
                ppm_cmd,
                ppm_config,
                args,
            )
            for sample in samples
        ]

        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print_progress(done, len(futures))

    results.sort(key=lambda item: item["index"])
    return results


def build_confusion_matrix(samples, results):
    classes = sorted({sample.classe for sample in samples})
    confusion = {class_name: {other: 0 for other in classes} for class_name in classes}

    for result in results:
        confusion[result["real_class"]][result["predicted_class"]] += 1

    return confusion


def compute_metrics(samples, results):
    total = len(samples)
    correct = sum(1 for result in results if result["real_class"] == result["predicted_class"])
    accuracy = correct / total if total else 0.0

    confusion = build_confusion_matrix(samples, results)
    recalls = []
    for class_name in sorted(confusion):
        total_class = sum(confusion[class_name].values())
        recall = confusion[class_name][class_name] / total_class if total_class else 0.0
        recalls.append(recall)

    balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "confusion": confusion,
    }


def print_metrics(metrics):
    print("\nResultados da validacao leave-one-out do modelo PPM:")
    print(
        f"Accuracy: {metrics['accuracy'] * 100.0:.2f}%  "
        f"| Balanced Accuracy: {metrics['balanced_accuracy'] * 100.0:.2f}%  "
        f"| Acertos: {metrics['correct']}/{metrics['total']}"
    )


def print_confusion_matrix(metrics):
    confusion = metrics["confusion"]
    classes = sorted(confusion)

    print("\nMatriz de confusao:")
    header = "real\\pred".ljust(14) + " ".join(class_name.ljust(12) for class_name in classes)
    print(header)

    for real_class in classes:
        line = real_class.ljust(14)
        for predicted_class in classes:
            line += str(confusion[real_class][predicted_class]).ljust(12)
        print(line)


def save_csv(csv_path, metrics):
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write("accuracy,balanced_accuracy,acertos,total\n")
        file.write(
            f"{metrics['accuracy']:.6f},"
            f"{metrics['balanced_accuracy']:.6f},"
            f"{metrics['correct']},"
            f"{metrics['total']}\n"
        )


def main():
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset)
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
            "Nenhuma amostra .txt foi encontrada para avaliar.\n"
            "Dica: gere os trechos com utils/preparar_dataset.py ou use "
            "--recursive/--include-completos se isso fizer sentido."
        )
        return 1

    samples = collect_samples(grouped_files)
    if len(samples) < 2:
        print("Sao necessarias pelo menos 2 amostras para avaliar o modelo.", file=sys.stderr)
        return 1

    print_dataset_summary(grouped_files)

    classes_small = [class_name for class_name, files in grouped_files.items() if len(files) < 2]
    if classes_small:
        print(
            "\nErro: para leave-one-out do modelo PPM, cada classe precisa ter "
            "pelo menos 2 amostras."
        )
        for class_name in classes_small:
            print(f"  - {class_name}")
        return 1

    if args.balance_bytes:
        print("\nBalanceamento de bytes: ativo")

    results = evaluate_leave_one_out(samples, grouped_files, ppm_cmd, ppm_config, args)
    metrics = compute_metrics(samples, results)

    print_metrics(metrics)
    print_confusion_matrix(metrics)

    if args.csv:
        csv_path = os.path.abspath(args.csv)
        save_csv(csv_path, metrics)
        print(f"\nResumo salvo em: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
