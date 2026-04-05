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

def classificar_ppm_direto(caminho_desconhecido, pasta_corpora, kmax="5"):
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
        
        # 2. Concatena Corpus + Desconhecido
        with tempfile.NamedTemporaryFile(delete=False) as temp_combo:
            with open(caminho_corpus, 'rb') as fc:
                temp_combo.write(fc.read())
                temp_combo.write(b" ") # Espaçador
            temp_combo.write(bytes_desconhecidos)
            caminho_combo = temp_combo.name
            
        # 3. Comprime o Combo (C_Combo)
        tamanho_combo = tamanho_comprimido(caminho_combo, kmax)
        os.remove(caminho_combo)
        
        # 4. Calcula os bytes necessários apenas para codificar o desconhecido
        entropia_cruzada_bytes = tamanho_combo - tamanho_base
        
        resultados.append({
            "classe": classe,
            "custo_bytes": entropia_cruzada_bytes
        })
        print(f"[{classe.title():>12}] -> Custou {entropia_cruzada_bytes} bytes adicionais.")

    if not resultados:
        return None

    # Ordena pelo MENOR custo
    resultados.sort(key=lambda x: x["custo_bytes"])
    vencedor = resultados[0]["classe"]
    
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
