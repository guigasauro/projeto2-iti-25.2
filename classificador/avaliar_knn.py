import argparse
import os
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
DEFAULT_PPM_CMD = os.path.abspath(os.path.join(BASE_DIR, "..", "ppm"))


@dataclass(frozen=True)
class Amostra:
    indice: int
    classe: str
    caminho: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Avalia varios valores de K para o classificador KNN usando "
            "leave-one-out no dataset rotulado."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DEFAULT_DATASET,
        help="Pasta raiz do dataset (padrao: ../dataset).",
    )
    parser.add_argument(
        "--ppm-cmd",
        default=DEFAULT_PPM_CMD,
        help="Caminho para o executavel ppm.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=15,
        help="Maior valor de K a ser testado.",
    )
    parser.add_argument(
        "--k-values",
        default="",
        help="Lista explicita de K separados por virgula. Ex: 1,3,5,7",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Numero de tarefas em paralelo para calcular as distancias.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Busca recursivamente arquivos .txt dentro de cada classe.",
    )
    parser.add_argument(
        "--include-completos",
        action="store_true",
        help="Tambem considera arquivos dentro de subpastas chamadas completos.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Limita a quantidade de amostras por classe para testes rapidos.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Se informado, salva o resumo dos resultados em CSV.",
    )
    parser.add_argument(
        "--ppm-kmax",
        type=int,
        default=5,
        help="Parametro kmax usado no compressor para calcular o NCD.",
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
        help="Limiar percentual do reset adaptativo do compressor.",
    )
    parser.add_argument(
        "--reset",
        type=int,
        choices=(0, 1),
        default=0,
        help="Ativa (1) ou desativa (0) o reset adaptativo no compressor.",
    )
    return parser.parse_args()


def compressor_args(args):
    return (
        str(args.ppm_kmax),
        str(args.window),
        str(args.threshold),
        str(args.reset),
    )


def coletar_amostras(dataset_dir, recursive=False, include_completos=False, limit_per_class=0):
    agrupadas = {}

    for classe in sorted(os.listdir(dataset_dir)):
        caminho_classe = os.path.join(dataset_dir, classe)
        if not os.path.isdir(caminho_classe):
            continue

        arquivos = []

        if recursive:
            for raiz, dirs, nomes in os.walk(caminho_classe):
                if not include_completos:
                    dirs[:] = [d for d in dirs if d.lower() != "completos"]

                for nome in sorted(nomes):
                    caminho = os.path.join(raiz, nome)
                    if nome.lower().endswith(".txt") and os.path.isfile(caminho):
                        arquivos.append(caminho)
        else:
            for nome in sorted(os.listdir(caminho_classe)):
                caminho = os.path.join(caminho_classe, nome)
                if nome.lower().endswith(".txt") and os.path.isfile(caminho):
                    arquivos.append(caminho)

        if limit_per_class > 0:
            arquivos = arquivos[:limit_per_class]

        if arquivos:
            agrupadas[classe] = arquivos

    amostras = []
    for classe, arquivos in agrupadas.items():
        for caminho in arquivos:
            amostras.append(Amostra(len(amostras), classe, os.path.abspath(caminho)))

    return agrupadas, amostras


def apagar_artefatos(*caminhos):
    for caminho in caminhos:
        if os.path.exists(caminho):
            os.remove(caminho)


def tamanho_comprimido_arquivo(ppm_cmd, args_ppm, caminho_arquivo, temp_dir):
    descritor, arquivo_saida = tempfile.mkstemp(suffix=".ppm", dir=temp_dir)
    os.close(descritor)

    comando = [ppm_cmd, "encode", caminho_arquivo, arquivo_saida, *args_ppm]

    try:
        subprocess.run(
            comando,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return os.path.getsize(arquivo_saida)
    finally:
        apagar_artefatos(
            arquivo_saida,
            arquivo_saida + ".rate.csv",
            arquivo_saida + ".reset.csv",
        )


def tamanho_comprimido_bytes(ppm_cmd, args_ppm, conteudo, temp_dir):
    descritor, arquivo_entrada = tempfile.mkstemp(suffix=".txt", dir=temp_dir)
    os.close(descritor)

    with open(arquivo_entrada, "wb") as arquivo:
        arquivo.write(conteudo)

    arquivo_saida = arquivo_entrada + ".ppm"
    comando = [ppm_cmd, "encode", arquivo_entrada, arquivo_saida, *args_ppm]

    try:
        subprocess.run(
            comando,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return os.path.getsize(arquivo_saida)
    finally:
        apagar_artefatos(
            arquivo_entrada,
            arquivo_saida,
            arquivo_saida + ".rate.csv",
            arquivo_saida + ".reset.csv",
        )


def mostrar_progresso(rotulo, concluidos, total):
    passo = max(1, total // 10)
    if concluidos == total or concluidos == 1 or concluidos % passo == 0:
        print(f"{rotulo}: {concluidos}/{total}")


def calcular_tamanhos_base(amostras, ppm_cmd, args_ppm, temp_dir, jobs):
    tamanhos = {}
    total = len(amostras)

    def tarefa(amostra):
        tamanho = tamanho_comprimido_arquivo(ppm_cmd, args_ppm, amostra.caminho, temp_dir)
        return amostra.indice, tamanho

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futuros = [executor.submit(tarefa, amostra) for amostra in amostras]
        for concluidos, futuro in enumerate(as_completed(futuros), start=1):
            indice, tamanho = futuro.result()
            tamanhos[indice] = tamanho
            mostrar_progresso("Tamanhos base", concluidos, total)

    return tamanhos


def calcular_distancias(amostras, tamanhos_base, ppm_cmd, args_ppm, temp_dir, jobs):
    total_pares = (len(amostras) * (len(amostras) - 1)) // 2
    if total_pares == 0:
        return {}

    conteudos = {}
    for amostra in amostras:
        with open(amostra.caminho, "rb") as arquivo:
            conteudos[amostra.indice] = arquivo.read()

    pares = []
    for i in range(len(amostras)):
        for j in range(i + 1, len(amostras)):
            pares.append((i, j))

    def tarefa(par):
        i, j = par
        combinado = conteudos[i] + conteudos[j]
        cxy = tamanho_comprimido_bytes(ppm_cmd, args_ppm, combinado, temp_dir)
        minimo = min(tamanhos_base[i], tamanhos_base[j])
        maximo = max(tamanhos_base[i], tamanhos_base[j])
        ncd = (cxy - minimo) / maximo
        return (i, j), ncd

    distancias = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futuros = [executor.submit(tarefa, par) for par in pares]
        for concluidos, futuro in enumerate(as_completed(futuros), start=1):
            chave, ncd = futuro.result()
            distancias[chave] = ncd
            mostrar_progresso("Distancias NCD", concluidos, total_pares)

    return distancias


def prever_classe(amostras, distancias, indice_alvo, k):
    vizinhos = []

    for outro_indice, outra_amostra in enumerate(amostras):
        if outro_indice == indice_alvo:
            continue

        chave = (min(indice_alvo, outro_indice), max(indice_alvo, outro_indice))
        vizinhos.append((distancias[chave], outra_amostra.classe))

    vizinhos.sort(key=lambda item: (item[0], item[1]))
    top_k = vizinhos[:k]

    votos = Counter(classe for _, classe in top_k)
    maior_votacao = max(votos.values())
    empatadas = [classe for classe, total in votos.items() if total == maior_votacao]

    if len(empatadas) == 1:
        return empatadas[0]

    medias = {}
    for classe in empatadas:
        distancias_classe = [dist for dist, rotulo in top_k if rotulo == classe]
        medias[classe] = sum(distancias_classe) / len(distancias_classe)

    return min(empatadas, key=lambda classe: (medias[classe], classe))


def avaliar_k(amostras, distancias, k):
    classes = sorted({amostra.classe for amostra in amostras})
    matriz = {classe: {outra: 0 for outra in classes} for classe in classes}
    acertos = 0

    for indice, amostra in enumerate(amostras):
        predita = prever_classe(amostras, distancias, indice, k)
        matriz[amostra.classe][predita] += 1
        if predita == amostra.classe:
            acertos += 1

    total = len(amostras)
    acuracia = acertos / total if total else 0.0

    recalls = []
    for classe in classes:
        total_classe = sum(matriz[classe].values())
        recall = matriz[classe][classe] / total_classe if total_classe else 0.0
        recalls.append(recall)

    acuracia_balanceada = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "k": k,
        "acertos": acertos,
        "total": total,
        "accuracy": acuracia,
        "balanced_accuracy": acuracia_balanceada,
        "confusion": matriz,
    }


def resolver_ks(args, total_amostras):
    limite_superior = max(1, min(args.k_max, total_amostras - 1))

    if args.k_values.strip():
        ks = []
        for valor in args.k_values.split(","):
            valor = valor.strip()
            if not valor:
                continue
            ks.append(int(valor))
    else:
        ks = list(range(1, limite_superior + 1))

    ks = sorted({k for k in ks if 1 <= k <= total_amostras - 1})
    return ks


def imprimir_resumo_dataset(agrupadas):
    print("Resumo do dataset:")
    for classe in sorted(agrupadas):
        arquivos = agrupadas[classe]
        print(f"  - {classe}: {len(arquivos)} amostras")


def imprimir_resultados(resultados):
    print("\nResultados da validacao leave-one-out:")
    print("K | Accuracy | Balanced Accuracy | Acertos")
    print("--|----------|-------------------|--------")
    for resultado in resultados:
        accuracy = resultado["accuracy"] * 100.0
        bal = resultado["balanced_accuracy"] * 100.0
        print(
            f"{resultado['k']:>2} | "
            f"{accuracy:>7.2f}% | "
            f"{bal:>17.2f}% | "
            f"{resultado['acertos']}/{resultado['total']}"
        )


def imprimir_matriz_confusao(resultado):
    matriz = resultado["confusion"]
    classes = sorted(matriz.keys())

    print("\nMatriz de confusao do melhor K:")
    cabecalho = "real\\pred".ljust(14) + " ".join(classe.ljust(12) for classe in classes)
    print(cabecalho)

    for classe_real in classes:
        linha = classe_real.ljust(14)
        for classe_predita in classes:
            linha += str(matriz[classe_real][classe_predita]).ljust(12)
        print(linha)


def salvar_csv(caminho_csv, resultados):
    with open(caminho_csv, "w", encoding="utf-8") as arquivo:
        arquivo.write("k,accuracy,balanced_accuracy,acertos,total\n")
        for resultado in resultados:
            arquivo.write(
                f"{resultado['k']},"
                f"{resultado['accuracy']:.6f},"
                f"{resultado['balanced_accuracy']:.6f},"
                f"{resultado['acertos']},"
                f"{resultado['total']}\n"
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

    agrupadas, amostras = coletar_amostras(
        dataset_dir,
        recursive=args.recursive,
        include_completos=args.include_completos,
        limit_per_class=args.limit_per_class,
    )

    if not amostras:
        print(
            "Nenhuma amostra .txt foi encontrada para avaliar.\n"
            "Dica: gere os trechos com o script utils/preparar_dataset.py "
            "ou use --recursive/--include-completos se fizer sentido."
        )
        return 1

    if len(amostras) < 2:
        print("Sao necessarias pelo menos 2 amostras para avaliar o K.", file=sys.stderr)
        return 1

    ks = resolver_ks(args, len(amostras))
    if not ks:
        print("Nenhum valor de K valido foi informado.", file=sys.stderr)
        return 1

    imprimir_resumo_dataset(agrupadas)

    classes_pequenas = [classe for classe, arquivos in agrupadas.items() if len(arquivos) < 2]
    if classes_pequenas:
        print(
            "\nAviso: ha classes com menos de 2 amostras. "
            "O leave-one-out pode ficar enviesado:"
        )
        for classe in classes_pequenas:
            print(f"  - {classe}")

    args_ppm = compressor_args(args)

    with tempfile.TemporaryDirectory(prefix="avaliar_knn_") as temp_dir:
        tamanhos_base = calcular_tamanhos_base(amostras, ppm_cmd, args_ppm, temp_dir, args.jobs)
        distancias = calcular_distancias(amostras, tamanhos_base, ppm_cmd, args_ppm, temp_dir, args.jobs)

    resultados = [avaliar_k(amostras, distancias, k) for k in ks]
    resultados.sort(key=lambda item: item["k"])

    imprimir_resultados(resultados)

    melhor = sorted(
        resultados,
        key=lambda item: (-item["balanced_accuracy"], -item["accuracy"], item["k"]),
    )[0]

    print(
        "\nMelhor K sugerido: "
        f"{melhor['k']} "
        f"(balanced accuracy = {melhor['balanced_accuracy'] * 100.0:.2f}%, "
        f"accuracy = {melhor['accuracy'] * 100.0:.2f}%)"
    )
    imprimir_matriz_confusao(melhor)

    if args.csv:
        caminho_csv = os.path.abspath(args.csv)
        salvar_csv(caminho_csv, resultados)
        print(f"\nResumo salvo em: {caminho_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
