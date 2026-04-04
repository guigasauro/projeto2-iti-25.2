import os
import subprocess
import tempfile
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho para o seu executável PPM compilado
PPM_CMD = os.path.abspath(os.path.join(BASE_DIR, "..", "ppm"))

def tamanho_comprimido(caminho_arquivo):
    """
    Usa o compressor C++ como caixa preta, comprime e
    retorna o tamanho (em bytes) do arquivo de saída gerado.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ppm") as temp_saida:
        arquivo_saida = temp_saida.name

    # Executa o seu compressor (exemplo configurando para kmax=5)
    # A flag reset=0 é boa para NCD se voce quiser manter a memoria
    cmd = [PPM_CMD, "encode", caminho_arquivo, arquivo_saida, "5", "1000", "10", "0"]

    try:
        # Chama o programa silenciando a saida do stdout
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return os.path.getsize(arquivo_saida)
    finally:
        for artefato in (
            arquivo_saida,
            arquivo_saida + ".rate.csv",
            arquivo_saida + ".reset.csv",
        ):
            if os.path.exists(artefato):
                os.remove(artefato)

def calcular_ncd(caminho_x, caminho_y):
    """
    Calcula a NCD entre dois arquivos.
    """
    # 1. Comprime X e Y isoladamente
    Cx = tamanho_comprimido(caminho_x)
    Cy = tamanho_comprimido(caminho_y)
    
    # 2. Concatena os dois em um arquivo temporário
    # Junta os bytes X e Y num mesmo arquivo
    with tempfile.NamedTemporaryFile(delete=False) as temp_xy:
        with open(caminho_x, 'rb') as fx: temp_xy.write(fx.read())
        with open(caminho_y, 'rb') as fy: temp_xy.write(fy.read())
        caminho_xy = temp_xy.name
        
    # 3. Comprime o arquivo combinado
    Cxy = tamanho_comprimido(caminho_xy)
    os.remove(caminho_xy) # Limpa o temporário
    
    # 4. Aplica a fórmula matemática
    minimo = min(Cx, Cy)
    maximo = max(Cx, Cy)
    
    ncd = (Cxy - minimo) / maximo
    return ncd

def classificar_knn(caminho_desconhecido, pasta_dataset, k_vizinhos=5):
    """
    Varre a pasta dataset, mede o NCD entre o arquivo_desconhecido
    e cada uma das amostras para achar as classes mais parecidas.
    """
    distancias = []
    
    # Itera sobre todas as pastas (classes literárias: "romantismo", "modernismo")
    for classe in os.listdir(pasta_dataset):
        caminho_classe = os.path.join(pasta_dataset, classe)
        if not os.path.isdir(caminho_classe): continue
            
        arquivos_treino = [
            arquivo
            for arquivo in sorted(os.listdir(caminho_classe))
            if arquivo.lower().endswith(".txt")
            and os.path.isfile(os.path.join(caminho_classe, arquivo))
        ]

        # Mede a distancia para todos os arquivos daquela classe
        for arquivo_treino in arquivos_treino:
            caminho_treino = os.path.join(caminho_classe, arquivo_treino)
            
            # O processamento "pesado" ocorre aqui
            distancia_ncd = calcular_ncd(caminho_desconhecido, caminho_treino)
            
            distancias.append({
                "classe": classe,
                "arquivo": arquivo_treino,
                "ncd": distancia_ncd
            })
            
    if not distancias:
        raise RuntimeError(
            "Nenhuma amostra .txt foi encontrada no dataset. "
            "Prepare os trechos antes de classificar."
        )

    # Ordena as distancias da menor (mais parecida) para a maior (mais diferente)
    distancias.sort(key=lambda x: x["ncd"])
    
    # Pega apenas os "K" primeiros do topo
    top_k = distancias[:k_vizinhos]
    print(f"\n--- Top {k_vizinhos} arquivos métricos ---")
    for vizinho in top_k:
        print(f"[{vizinho['classe'].upper()}] - {vizinho['arquivo']} (NCD: {vizinho['ncd']:.4f})")
    
    # Votação (Acha qual foi a classe majoritária no Top-K)
    votos = [v['classe'] for v in top_k]
    classe_vencedora = Counter(votos).most_common(1)[0][0]
    
    return classe_vencedora

if __name__ == "__main__":
    desconhecido = os.path.join(BASE_DIR, "desconhecidos", "poema_misterioso.txt")
    if os.path.exists(desconhecido):
        pasta_dataset = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
        resultado = classificar_knn(desconhecido, pasta_dataset, k_vizinhos=3)
        print(f"\n=> Veredito Final: O texto pertence ao {resultado.upper()}!")
