import os
import subprocess
import tempfile
import sys
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PPM_CMD = os.path.abspath(os.path.join(BASE_DIR, "..", "ppm"))

def tamanho_comprimido(caminho_arquivo, kmax="5"):
    """
    Comprime o arquivo indicado usando PPM-C e retorna o tamanho.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ppm") as temp_saida:
        arquivo_saida = temp_saida.name

    # Comando PPM encode padrão para textos
    cmd = [PPM_CMD, "encode", caminho_arquivo, arquivo_saida, str(kmax), "1000", "10", "0"]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return os.path.getsize(arquivo_saida)
    finally:
        for artefato in (arquivo_saida, arquivo_saida + ".rate.csv", arquivo_saida + ".reset.csv"):
            if os.path.exists(artefato):
                try: os.remove(artefato)
                except: pass

def classificar_ppm_direto(caminho_desconhecido, pasta_corpora, kmax="5", metodo="custo", limiar=0.10, retornar_detalhes=False):
    """
    Classificador usando Cross-Entropy / Entropia Condicional PPM.
    Mede quantos bytes o texto desconhecido adiciona a um corpo já consolidado.
    L(T | C) = C(Corpo + T) - C(Corpo)
    A classe com menor acréscimo "vence".
    """
    if not os.path.exists(PPM_CMD):
        raise RuntimeError(f"Executável PPM não encontrado em {PPM_CMD}")
        
    corpora = glob.glob(os.path.join(pasta_corpora, "corpus_*.txt"))
    if not corpora:
        raise RuntimeError(f"Nenhum corpus encontrado em {pasta_corpora}. Execute o treinar_ppm.py primeiro.")

    # Ler bytes do arquivo desconhecido
    with open(caminho_desconhecido, 'rb') as f:
        bytes_desconhecidos = f.read()

    resultados = []
    
    print("\n--- Analisando Entropia Condicional (PPM) ---")
    for caminho_corpus in corpora:
        # Pega a classe a partir de "corpus_romantismo.txt" -> "romantismo"
        nome_arquivo = os.path.basename(caminho_corpus)
        classe = nome_arquivo.replace("corpus_", "").replace(".txt", "")
        
        # 1. Comprime apenas o Corpus Base (C_Base)
        tamanho_base = tamanho_comprimido(caminho_corpus, kmax)
        tamanho_base_orig = os.path.getsize(caminho_corpus)
        taxa_base = (tamanho_base * 8) / tamanho_base_orig if tamanho_base_orig > 0 else 0
        
        # 2. Concatena Corpus + Desconhecido
        with tempfile.NamedTemporaryFile(delete=False) as temp_combo:
            with open(caminho_corpus, 'rb') as fc:
                temp_combo.write(fc.read())
                temp_combo.write(b" ") # Espaçador
            temp_combo.write(bytes_desconhecidos)
            caminho_combo = temp_combo.name
            
        tamanho_combo_orig = os.path.getsize(caminho_combo)
            
        # 3. Comprime o Combo (C_Combo)
        tamanho_combo = tamanho_comprimido(caminho_combo, kmax)
        os.remove(caminho_combo)
        
        taxa_combo = (tamanho_combo * 8) / tamanho_combo_orig if tamanho_combo_orig > 0 else 0
        
        # 4. Calcula os bytes necessários e variação de taxa
        entropia_cruzada_bytes = tamanho_combo - tamanho_base
        variacao_taxa = taxa_combo - taxa_base
        
        resultados.append({
            "classe": classe,
            "custo_bytes": entropia_cruzada_bytes,
            "variacao_taxa": variacao_taxa,
            "taxa_base": taxa_base
        })
        print(f"[{classe.title():>12}] -> Custo: {entropia_cruzada_bytes} B | Var Taxa: {variacao_taxa:.4f} b/s")

    vencedor = None
    if not resultados:
        pass
    elif metodo == "taxa":
        # Metologia Alternativa: taxa bits/simbolo com limiar restrito de melhoria
        # 1. Se for positiva a variação, já é rejeitada automaticamente (pois é maior que o limiar negativo)
        # 2. Exige ganho de compressão (valor negativo) igual ou mais forte que o threshold
        candidatos = [r for r in resultados if r["variacao_taxa"] <= -(r["taxa_base"] * limiar)]
        if len(candidatos) == 1:
            vencedor = candidatos[0]["classe"]
    else:
        # Metodologia Padrão: Ordena pelo MENOR custo em bytes
        resultados.sort(key=lambda x: x["custo_bytes"])
        vencedor = resultados[0]["classe"]
        
    if retornar_detalhes:
        return vencedor, resultados
    return vencedor

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classificador via Compressão PPM Direta")
    parser.add_argument("desconhecido", help="Arquivo .txt a ser classificado")
    parser.add_argument("--corpora", default="modelos_corpus", help="Pasta com os corpora treinados")
    args = parser.parse_args()
    
    try:
        resultado = classificar_ppm_direto(args.desconhecido, args.corpora)
        print(f"\n=> VEREDITO: O texto pertence a classe '{resultado.upper()}'!")
    except Exception as e:
        print(f"ERRO: {e}")
