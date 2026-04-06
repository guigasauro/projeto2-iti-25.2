import os
import shutil
import tempfile
from avaliar_alfabetos_tamanhos import avaliar_cenario_acuracia

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Workflow de avaliação PPM")
    parser.add_argument("--por-taxa", action="store_true", help="Faz a classificação baseada na variação de taxa de compressão (bits/simb)")
    parser.add_argument("--limiar", type=float, default=0.10, help="Limiar percentual para aceite na taxa (ex: 0.10 para 10%)")
    parser.add_argument("--tamanho-chunk", type=str, help="Executar teste apenas para o chunk especificado (ex: '9' ou '9kb')")
    parser.add_argument("--multithread", action="store_true", help="Habilita execução multithreading para acelerar a inferência PPM")
    parser.add_argument("--csv", type=str, default="resultados.csv", help="Nome do arquivo CSV para exportação dos dados (Padrão: resultados.csv)")
    args = parser.parse_args()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    
    # Identificar todas as classes para teste e pra montar o cabecalho
    pastas = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    classes = [d for d in pastas if d not in ["modelos", "modelos_temp"]]
    classes.sort()
    
    import csv
    # Configurar caminho do CSV
    os.makedirs("resultados", exist_ok=True)
    if args.csv == "resultados.csv":
        chunk_str = str(args.tamanho_chunk).upper().replace("KB","") if args.tamanho_chunk else "TODOS"
        limiar_str = str(args.limiar)
        caminho_csv = os.path.join("resultados", f"tabela-{len(classes)}-{chunk_str}KB-{limiar_str}.csv")
    else:
        caminho_csv = args.csv
        
    csv_file = open(caminho_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    header = ["tamanho_chunk", "kmax", "tratamento", "resultado", "nome_arquivo", "classe_real", "classe_predicao"]
    for c in classes:
        header.append(f"custo_extra_{c}")
    for c in classes:
        header.append(f"var_taxa_{c}")
    for c in classes:
        header.append(f"taxa_base_{c}")
    csv_writer.writerow(header)
    
    tamanhos = {
        '3kb': {'treino': 80, 'teste': 20},
        '6kb': {'treino': 48, 'teste': 12},
        '9kb': {'treino': 32, 'teste': 8}
    }
    
    if args.tamanho_chunk:
        t_key = f"{args.tamanho_chunk}kb" if not args.tamanho_chunk.lower().endswith("kb") else args.tamanho_chunk.lower()
        if t_key in tamanhos:
            tamanhos = {t_key: tamanhos[t_key]}
        else:
            print(f"[!] Erro: Tamanho de chunk inválido: {args.tamanho_chunk}. Use 3, 6 ou 9.")
            sys.exit(1)
    
    alfabetos = ['untreated', 'lower', 'no_accents', 'no_punctuation', 'full_treated']
    kVals = list(range(1, 11))
    
    print(f"=== INICIANDO AVALIAÇÃO MATRIX ===\n")
    
    for tamanho, limits in tamanhos.items():
        treino = limits['treino']
        teste = limits['teste']
        
        print("\n" + "="*80)
        print(f"============= AVALIANDO CHUNKS DE {tamanho.upper()} =============")
        print(f" Config -> Treino: {treino} recortes | Teste: {teste} recortes por classe")
        
        # Dicionário: resultados[alfabeto][k] = acuracia
        resultados = {alf: {} for alf in alfabetos}
        
        for alfabeto in alfabetos:
            print(f"\n--- Preparando Dataset ({tamanho}) para Alfabeto: {alfabeto} ---")
            
            temp_dataset_dir = tempfile.mkdtemp(prefix=f"dataset_{tamanho}_{alfabeto}_")
            
            try:
                todas_existem = True
                for classe in classes:
                    origem = os.path.join(base_dir, classe, 'data', tamanho, alfabeto)
                    destino = os.path.join(temp_dataset_dir, classe)
                    if not os.path.exists(origem):
                        print(f"  [!] Aviso: Pasta de dados não encontrada: {origem}")
                        todas_existem = False
                        continue
                    os.symlink(origem, destino)
                    
                if todas_existem:
                    for k in kVals:
                        print(f" -> Rodando com K={k}...")
                        m = "taxa" if args.por_taxa else "custo"
                        acuracia, registros = avaliar_cenario_acuracia(temp_dataset_dir, treino, teste, kmax=k, metodo=m, limiar=args.limiar, multithread=args.multithread)
                        resultados[alfabeto][k] = acuracia
                        
                        for reg in registros:
                            row = [tamanho.replace("kb", ""), k, alfabeto, reg["resultado"], reg["nome_arquivo"], reg["classe_real"], reg["classe_predicao"]]
                            for c in classes:
                                row.append(reg["custos"].get(c, ""))
                            for c in classes:
                                row.append(reg["taxas"].get(c, ""))
                            for c in classes:
                                row.append(reg["taxas_base"].get(c, ""))
                            csv_writer.writerow(row)
                        csv_file.flush()
                else:
                    for k in kVals:
                        resultados[alfabeto][k] = 0.0
                
            finally:
                # Limpeza
                shutil.rmtree(temp_dataset_dir, ignore_errors=True)
                
        print("\n" + "-"*80)
        print(f"RESUMO DAS ACURÁCIAS - MÉTODO: {'TAXA' if args.por_taxa else 'CUSTO'} - LIMIAR: {args.limiar*100}% - CHUNKS DE {tamanho.upper()} (ALFABETO x KMAX):")
        
        # Cabeçalho da tabela
        header = f"{'Alfabeto':<15} | " + " | ".join([f"K={k:<3}" for k in kVals])
        print(header)
        print("-" * len(header))
        
        for alfabeto in alfabetos:
            linha = f"{alfabeto:<15} | "
            valores = []
            for k in kVals:
                acc = resultados[alfabeto][k]
                valores.append(f"{acc:>4.0f}%")
            linha += " | ".join(valores)
            print(linha)
        print("-"*80)
            
    csv_file.close()

if __name__ == '__main__':
    main()
