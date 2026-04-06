import os
import glob
import random
import shutil
import tempfile
from treinar_ppm import criar_modelos_ppm
from classificar_ppm import classificar_ppm_direto

def avaliar_cenario_acuracia(pasta_dataset_preparado, num_amostras_treino, num_amostras_teste, kmax=5, metodo="custo", limiar=0.10, multithread=False):
    """
    Simula uma avaliação variando a quantidade de dados e o valor kmax do PPM.
    """
    classes = [d for d in os.listdir(pasta_dataset_preparado) 
               if os.path.isdir(os.path.join(pasta_dataset_preparado, d))]
               
    if not classes:
        print("Erro: Dataset preparado não contém pastas de classe.")
        return

    # Preparar sub-dataset de treino e lista de testes
    temp_dir = tempfile.mkdtemp(prefix="ppm_avaliacao_")
    pasta_treino_temp = os.path.join(temp_dir, "treino")
    pasta_modelos_temp = os.path.join(temp_dir, "modelos")
    os.makedirs(pasta_treino_temp, exist_ok=True)
    
    testes_a_fazer = [] # (caminho_arquivo, classe_real)

    print(f"\n>> Iniciando Avaliação | Treino: {num_amostras_treino} chunks/classe | Teste: {num_amostras_teste} chunks/classe")

    for classe in classes:
        caminho_classe = os.path.join(pasta_dataset_preparado, classe)
        arquivos = glob.glob(os.path.join(caminho_classe, "*.txt"))
        
        # Embaralha aleatoriamente para simular test set real
        random.shuffle(arquivos)
        
        treino_arquivos = arquivos[:num_amostras_treino]
        # Pega para teste apenas os que não caíram no treino
        teste_arquivos = arquivos[num_amostras_treino : num_amostras_treino + num_amostras_teste]
        
        # Mover treino para temp
        pasta_classe_treino = os.path.join(pasta_treino_temp, classe)
        os.makedirs(pasta_classe_treino, exist_ok=True)
        for arq in treino_arquivos:
            shutil.copy(arq, pasta_classe_treino)
            
        # Ppopular a lita de testes
        for arq in teste_arquivos:
            testes_a_fazer.append((arq, classe))

    # Treina modelos PPM
    criar_modelos_ppm(pasta_treino_temp, pasta_modelos_temp)
    
    # Validação (Ocultar prints internos do classificador)
    import sys
    sys_stdout = sys.stdout # Salva para restaurar depois
    
    acertos = 0
    total = len(testes_a_fazer)
    
    print(f"\nRealizando inferência de {total} amostras simultaneamente...")
    # Desativa print padrão para não sujar o log com saídas internas
    sys.stdout = open(os.devnull, 'w')
    
    import concurrent.futures
    
    try:
        processados = 0
        linhas_csv = []
        def processar_item(item):
            arquivo_alvo, classe_real_alvo = item
            pred_alvo, detalhes = classificar_ppm_direto(arquivo_alvo, pasta_modelos_temp, kmax=kmax, metodo=metodo, limiar=limiar, retornar_detalhes=True)
            acertou = pred_alvo and pred_alvo.lower() == classe_real_alvo.lower()
            return acertou, os.path.basename(arquivo_alvo), classe_real_alvo, pred_alvo, detalhes

        def renderizar_log(res, acertos_locais, processados_locais):
            acertou, arq_nome, real, pred, det = res
            if acertou:
                acertos_locais += 1
            processados_locais += 1
            
            detalhes_csv = {}
            taxas_csv = {}
            taxas_base_csv = {}
            for r in det:
                c_name = r["classe"]
                detalhes_csv[c_name] = r["custo_bytes"]
                taxas_csv[c_name] = r["variacao_taxa"]
                taxas_base_csv[c_name] = r["taxa_base"]
                
            linhas_csv.append({
                "resultado": 1 if acertou else 0,
                "nome_arquivo": arq_nome,
                "classe_real": real,
                "classe_predicao": pred if pred else "N/A",
                "custos": detalhes_csv,
                "taxas": taxas_csv,
                "taxas_base": taxas_base_csv
            })
            
            pred_str = str(pred).upper() if pred else "N/A"
            status = "✓ ACERTO" if acertou else "x ERRO  "
            
            sys_stdout.write(f"\n{status} | Arquivo: {arq_nome} | Real: {real.upper()} | Predição: {pred_str}\n")
            for r in det:
                sys_stdout.write(f"  └─ Corpus: {r['classe'].title():<12} | Custo Extra: {r['custo_bytes']:>5} B | Variação T.: {r['variacao_taxa']:.4f} b/s | Taxa B.: {r['taxa_base']:.3f}\n")
            
            pct = (acertos_locais / processados_locais) * 100
            barra_len = 30
            progresso = int((processados_locais / total) * barra_len)
            barra = "█" * progresso + "-" * (barra_len - progresso)
            sys_stdout.write(f"[{barra}] {processados_locais}/{total} | Score: {acertos_locais}/{processados_locais} ({pct:.1f}%)\n")
            sys_stdout.flush()
            
            return acertos_locais, processados_locais

        if multithread:
            workers = (os.cpu_count() or 1) * 2
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futuros = [executor.submit(processar_item, t) for t in testes_a_fazer]
                
                for futuro in concurrent.futures.as_completed(futuros):
                    acertos, processados = renderizar_log(futuro.result(), acertos, processados)
        else:
            for item in testes_a_fazer:
                acertos, processados = renderizar_log(processar_item(item), acertos, processados)
                
    finally:
        # Restaura print
        sys.stdout.close()
        sys.stdout = sys_stdout

    print("\n") # Quebra a linha após o término da barra
    acuracia = (acertos / total) * 100 if total > 0 else 0
    print(f"Resultado do Cenário -> Acertos: {acertos}/{total} ({acuracia:.2f}%)")
    
    # Limpa disco
    shutil.rmtree(temp_dir)
    return acuracia, linhas_csv

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Script para avaliar acuracia sob diferentes tamanhos")
    parser.add_argument("dataset", help="Pasta raiz contendo o dataset já preparado/limpo, contendo subpastas por classe.")
    parser.add_argument("--treino", type=int, default=50, help="Nº max de recortes (arquivos txt) agrupados para o treino da classe")
    parser.add_argument("--teste", type=int, default=20, help="Nº max de amostras para teste por classe")
    
    args = parser.parse_args()
    avaliar_cenario_acuracia(args.dataset, args.treino, args.teste)
