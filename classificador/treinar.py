import os
import json
import subprocess
import sys

# Caminho para o seu executável PPM (ajustado considerando que o script roda na pasta classificador/)
PPM_CMD = "../ppm" 

def treinar_modelo(pasta_dataset, arquivo_saida_json):
    """
    Roda a compressão base para todos e cada um dos arquivos `.txt` 
    contidos no dataset e gera um 'Cache' (modelo de treinamento) em JSON.
    """
    if not os.path.exists(PPM_CMD):
        print(f"ERRO CRÍTICO: Não encontrei o executável do compressor no caminho '{PPM_CMD}'.")
        print("Você compilou o projeto C++ com o Makefile na pasta raiz?")
        return

    if not os.path.exists(pasta_dataset):
        print(f"ERRO: Pasta do dataset '{pasta_dataset}' não existe.")
        return

    # Um dicionário que vai armazenar (Ex: {'dataset/romantismo/trecho1.txt': 850 bytes})
    cache_treinamento = {}

    print(f"Iniciando o treinamento na pasta: {pasta_dataset}\n")

    # Varre as pastas internas (ex: 'romantismo', 'modernismo')
    for pasta_literaria in os.listdir(pasta_dataset):
        caminho_classe = os.path.join(pasta_dataset, pasta_literaria)
        
        # Pula se for "completos" ou arquivos avulsos
        if not os.path.isdir(caminho_classe) or pasta_literaria.lower() == "completos":
            continue
            
        arquivos_txt = [arq for arq in os.listdir(caminho_classe) if arq.endswith(".txt")]
        if not arquivos_txt:
            continue
            
        print(f"-> Analisando {len(arquivos_txt)} amostras da classe: [{pasta_literaria.title()}]")

        for arquivo_txt in arquivos_txt:
            caminho_completo = os.path.join(caminho_classe, arquivo_txt)
            arquivo_temporario_ppm = caminho_completo + ".ppm"

            # Invocando o programa C++ (usaremos a configuração NCD boa: Reset Desativado = 0)
            # cmd: ../ppm encode <entrada> <saida.ppm> 5 1000 10 0
            comando = [PPM_CMD, "encode", caminho_completo, arquivo_temporario_ppm, "5", "1000", "10", "0"]
            
            try:
                # Oculta o stdout massivo do print C++ para não sujar o terminal do python
                subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                # Coleta o tamanho gravado gerado pelo algoritmo
                tamanho_bytes = os.path.getsize(arquivo_temporario_ppm)
                
                # Salva no dicionário / cache
                cache_treinamento[caminho_completo] = tamanho_bytes
                
            except subprocess.CalledProcessError:
                print(f"   [!] Falha grave ao tentar comprimir: {arquivo_txt}")
            finally:
                # Segurança: Garante que o PPM gerado será apagado do disco mesmo se der tela azul!
                if os.path.exists(arquivo_temporario_ppm):
                    os.remove(arquivo_temporario_ppm)

    # Escreve (salva) o Dicionário em Disco no formato JSON
    with open(arquivo_saida_json, 'w', encoding='utf-8') as f:
        json.dump(cache_treinamento, f, indent=4)

    print("\n========================================================")
    print(f"✓ TREINAMENTO FINALIZADO COM SUCESSO!")
    print(f"✓ {len(cache_treinamento)} amostras foram calculadas e comprimidas.")
    print(f"✓ O Modelo Padrão foi salvo em: {arquivo_saida_json}")
    print("========================================================")

if __name__ == "__main__":
    
    # O Python chamará por ex: python treinar.py ../dataset/
    pasta_do_dataset = sys.argv[1] if len(sys.argv) > 1 else "../dataset"
    
    # Nome do Arquivo Modelo de destino
    modelo_json = "modelo_treinado.json"
    
    # Inicia o treino
    treinar_modelo(pasta_do_dataset, modelo_json)
