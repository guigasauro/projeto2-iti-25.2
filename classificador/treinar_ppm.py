import os
import sys
import glob

def criar_modelos_ppm(pasta_dataset, pasta_saida):
    """
    Concatena todos os arquivos de treinamento de cada classe em um 
    único "corpus", que servirá como base para classificação PPM pura.
    """
    if not os.path.exists(pasta_dataset):
        print(f"ERRO: Pasta do dataset '{pasta_dataset}' não existe.")
        return

    os.makedirs(pasta_saida, exist_ok=True)
    print(f"Gerando corpora (modelos) baseados no dataset: {pasta_dataset}\n")

    for classe in os.listdir(pasta_dataset):
        caminho_classe = os.path.join(pasta_dataset, classe)
        if not os.path.isdir(caminho_classe) or classe.lower() == "completos":
            continue

        arquivos_txt = glob.glob(os.path.join(caminho_classe, "*.txt"))
        if not arquivos_txt:
            continue
            
        print(f"-> Concatenando {len(arquivos_txt)} amostras (TREINO) da classe: [{classe.title()}]")
        
        caminho_corpus = os.path.join(pasta_saida, f"corpus_{classe}.txt")
        with open(caminho_corpus, 'wb') as corpus_file:
            for txt_file in sorted(arquivos_txt):
                with open(txt_file, 'rb') as f:
                    # Lê o byte stream e anexa ao corpus geral
                    corpus_file.write(f.read())
                    # Anexar um espaço para interligar palavras nas bordas sem quebrar bytes (opcional)
                    corpus_file.write(b" ")
                    
        print(f"   Corpus '{classe}' salvo: {caminho_corpus} ({os.path.getsize(caminho_corpus)} bytes)")

    print("\n✓ TREINAMENTO FINALIZADO.")
    print(f"✓ Corpora salvos em: {pasta_saida}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pasta_dataset", nargs="?", default="../dataset", help="Pasta com as amostras de treino (.txt)")
    parser.add_argument("pasta_saida", nargs="?", default="modelos_corpus", help="Onde salvar os arquivos concatenados")
    args = parser.parse_args()
    
    criar_modelos_ppm(args.pasta_dataset, args.pasta_saida)
