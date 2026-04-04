import argparse
import os
import sys
import time
from itertools import product
from types import SimpleNamespace

from avaliar_modelo_ppm import (
    collect_samples,
    compute_metrics,
    evaluate_leave_one_out,
    print_dataset_summary,
    save_csv as save_summary_csv,
)
from modelo_ppm_utils import (
    DEFAULT_DATASET,
    DEFAULT_PPM_CMD,
    DEFAULT_SEPARATOR,
    PPMConfig,
    collect_class_files,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Testa varias combinacoes de parametros do classificador por "
            "modelo PPM e aponta a melhor configuracao."
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
        help="Numero de amostras avaliadas em paralelo por configuracao.",
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
        help="Limita o tamanho bruto dos modelos em cada fold.",
    )
    parser.add_argument(
        "--balance-bytes",
        action="store_true",
        help="Balanceia os modelos usando o mesmo teto de bytes por classe.",
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
        help="Se maior que zero, divide cada amostra em trechos e vota.",
    )
    parser.add_argument(
        "--min-chunk-bytes",
        type=int,
        default=500,
        help="Tamanho minimo, em bytes, para um trecho ser considerado.",
    )
    parser.add_argument(
        "--kmax-values",
        default="4,5,6",
        help="Lista de kmax separados por virgula. Ex: 4,5,6,7",
    )
    parser.add_argument(
        "--window-values",
        default="500,1000,2000",
        help="Lista de janelas separados por virgula. Ex: 500,1000,2000",
    )
    parser.add_argument(
        "--threshold-values",
        default="5,10,15",
        help="Lista de limiares separados por virgula. Ex: 5,10,15",
    )
    parser.add_argument(
        "--reset-values",
        default="0,1",
        help="Lista de valores de reset separados por virgula. Ex: 0,1",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Se informado, salva o ranking completo das configuracoes em CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Se informado, salva o resumo da melhor configuracao em CSV.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Quantas configuracoes melhores mostrar no ranking final.",
    )
    return parser.parse_args()


def parse_int_list(raw):
    values = []
    for value in raw.split(","):
        value = value.strip()
        if value:
            values.append(int(value))
    return values


def parse_float_list(raw):
    values = []
    for value in raw.split(","):
        value = value.strip()
        if value:
            values.append(float(value))
    return values


def parse_reset_values(raw):
    values = []
    for value in parse_int_list(raw):
        if value not in (0, 1):
            raise ValueError("Os valores de reset devem ser 0 ou 1.")
        values.append(value)
    return values


def build_search_configs(args):
    kmax_values = parse_int_list(args.kmax_values)
    window_values = parse_int_list(args.window_values)
    threshold_values = parse_float_list(args.threshold_values)
    reset_values = parse_reset_values(args.reset_values)

    if not kmax_values:
        raise ValueError("Informe pelo menos um valor em --kmax-values.")
    if not window_values:
        raise ValueError("Informe pelo menos um valor em --window-values.")
    if not threshold_values:
        raise ValueError("Informe pelo menos um valor em --threshold-values.")
    if not reset_values:
        raise ValueError("Informe pelo menos um valor em --reset-values.")

    configs = []
    seen = set()

    for kmax, reset in product(kmax_values, reset_values):
        if reset == 0:
            config = PPMConfig(
                kmax=kmax,
                window=window_values[0],
                threshold=threshold_values[0],
                reset=0,
            )
            key = (config.kmax, config.window, config.threshold, config.reset)
            if key not in seen:
                seen.add(key)
                configs.append(config)
            continue

        for window, threshold in product(window_values, threshold_values):
            config = PPMConfig(
                kmax=kmax,
                window=window,
                threshold=threshold,
                reset=1,
            )
            key = (config.kmax, config.window, config.threshold, config.reset)
            if key not in seen:
                seen.add(key)
                configs.append(config)

    return configs


def build_eval_args(args):
    return SimpleNamespace(
        jobs=args.jobs,
        max_bytes_per_class=args.max_bytes_per_class,
        balance_bytes=args.balance_bytes,
        separator=args.separator,
        lowercase=args.lowercase,
        chunk_size=args.chunk_size,
        min_chunk_bytes=args.min_chunk_bytes,
    )


def validate_dataset(grouped_files):
    if not grouped_files:
        raise RuntimeError(
            "Nenhuma amostra .txt foi encontrada para avaliar.\n"
            "Dica: gere os trechos com utils/preparar_dataset.py ou use "
            "--recursive/--include-completos se isso fizer sentido."
        )

    classes_small = [
        class_name for class_name, files in grouped_files.items() if len(files) < 2
    ]
    if classes_small:
        details = "\n".join(f"  - {class_name}" for class_name in classes_small)
        raise RuntimeError(
            "Cada classe precisa ter pelo menos 2 amostras para o leave-one-out.\n"
            + details
        )


def evaluate_config(index, total, config, samples, grouped_files, ppm_cmd, eval_args):
    start = time.time()
    print(
        f"\n[{index}/{total}] Testando "
        f"kmax={config.kmax}, window={config.window}, "
        f"threshold={config.threshold:g}, reset={config.reset}"
    )

    results = evaluate_leave_one_out(
        samples,
        grouped_files,
        ppm_cmd=ppm_cmd,
        ppm_config=config,
        args=eval_args,
    )
    metrics = compute_metrics(samples, results)
    elapsed = time.time() - start

    row = {
        "kmax": config.kmax,
        "window": config.window,
        "threshold": config.threshold,
        "reset": config.reset,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "correct": metrics["correct"],
        "total": metrics["total"],
        "elapsed_sec": elapsed,
        "confusion": metrics["confusion"],
    }

    print(
        f"Resultado: accuracy={row['accuracy'] * 100.0:.2f}% | "
        f"balanced={row['balanced_accuracy'] * 100.0:.2f}% | "
        f"acertos={row['correct']}/{row['total']} | "
        f"tempo={row['elapsed_sec']:.1f}s"
    )
    return row


def sort_results(rows):
    return sorted(
        rows,
        key=lambda row: (
            -row["balanced_accuracy"],
            -row["accuracy"],
            row["kmax"],
            row["reset"],
            row["window"],
            row["threshold"],
        ),
    )


def print_results_table(rows, top):
    print("\nRanking das configuracoes:")
    print("Pos | Kmax | Window | Threshold | Reset | Accuracy | Balanced | Acertos | Tempo(s)")
    print("----|------|--------|-----------|-------|----------|----------|---------|--------")

    for position, row in enumerate(rows[: max(1, top)], start=1):
        print(
            f"{position:>3} | "
            f"{row['kmax']:>4} | "
            f"{row['window']:>6} | "
            f"{row['threshold']:>9g} | "
            f"{row['reset']:>5} | "
            f"{row['accuracy'] * 100.0:>7.2f}% | "
            f"{row['balanced_accuracy'] * 100.0:>7.2f}% | "
            f"{row['correct']}/{row['total']:<7} | "
            f"{row['elapsed_sec']:>6.1f}"
        )


def print_best_confusion(best_row):
    confusion = best_row["confusion"]
    classes = sorted(confusion)

    print("\nMatriz de confusao da melhor configuracao:")
    header = "real\\pred".ljust(14) + " ".join(class_name.ljust(12) for class_name in classes)
    print(header)

    for real_class in classes:
        line = real_class.ljust(14)
        for predicted_class in classes:
            line += str(confusion[real_class][predicted_class]).ljust(12)
        print(line)


def save_ranking_csv(csv_path, rows):
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write(
            "kmax,window,threshold,reset,accuracy,balanced_accuracy,acertos,total,elapsed_sec\n"
        )
        for row in rows:
            file.write(
                f"{row['kmax']},"
                f"{row['window']},"
                f"{row['threshold']},"
                f"{row['reset']},"
                f"{row['accuracy']:.6f},"
                f"{row['balanced_accuracy']:.6f},"
                f"{row['correct']},"
                f"{row['total']},"
                f"{row['elapsed_sec']:.4f}\n"
            )


def main():
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset)
    ppm_cmd = os.path.abspath(args.ppm_cmd)

    if not os.path.isdir(dataset_dir):
        print(f"Erro: a pasta de dataset '{dataset_dir}' nao existe.", file=sys.stderr)
        return 1

    if not os.path.exists(ppm_cmd):
        print(f"Erro: nao encontrei o executavel ppm em '{ppm_cmd}'.", file=sys.stderr)
        print("Compile antes com: make", file=sys.stderr)
        return 1

    try:
        configs = build_search_configs(args)
    except ValueError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    grouped_files = collect_class_files(
        dataset_dir,
        recursive=args.recursive,
        include_completos=args.include_completos,
        limit_per_class=args.limit_per_class,
    )

    try:
        validate_dataset(grouped_files)
    except RuntimeError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    samples = collect_samples(grouped_files)
    eval_args = build_eval_args(args)

    print_dataset_summary(grouped_files)
    print(f"\nTotal de configuracoes a testar: {len(configs)}")
    if args.balance_bytes:
        print("Balanceamento de bytes: ativo")

    rows = []
    overall_start = time.time()
    for index, config in enumerate(configs, start=1):
        row = evaluate_config(
            index,
            len(configs),
            config,
            samples,
            grouped_files,
            ppm_cmd,
            eval_args,
        )
        rows.append(row)

    rows = sort_results(rows)
    best_row = rows[0]

    print_results_table(rows, top=args.top)
    print(
        "\nMelhor configuracao: "
        f"kmax={best_row['kmax']}, "
        f"window={best_row['window']}, "
        f"threshold={best_row['threshold']:g}, "
        f"reset={best_row['reset']} "
        f"(balanced accuracy = {best_row['balanced_accuracy'] * 100.0:.2f}%, "
        f"accuracy = {best_row['accuracy'] * 100.0:.2f}%)"
    )
    print_best_confusion(best_row)
    print(f"\nTempo total da busca: {time.time() - overall_start:.1f}s")

    if args.csv:
        csv_path = os.path.abspath(args.csv)
        save_ranking_csv(csv_path, rows)
        print(f"Ranking salvo em: {csv_path}")

    if args.summary_csv:
        summary_path = os.path.abspath(args.summary_csv)
        save_summary_csv(
            summary_path,
            {
                "accuracy": best_row["accuracy"],
                "balanced_accuracy": best_row["balanced_accuracy"],
                "correct": best_row["correct"],
                "total": best_row["total"],
            },
        )
        print(f"Resumo da melhor configuracao salvo em: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
