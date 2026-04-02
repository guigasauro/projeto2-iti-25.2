import os
import glob
import sys

def process_file(input_path, output_dir, chunk_size=3000):
    """
    Lê um arquivo de texto, converte para minúsculas, e divide em 
    subarquivos contendo um número de bytes aproximado ao `chunk_size`,
    evitando cortar palavras ao meio.
    """
    try:
        # Usa errors='ignore' para evitar travamentos com caracteres quebrados
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print(f"Erro ao ler {input_path}: {e}")
        return

    # 1. Alterar todo o texto para LOWRECASE (minúsculo)
    text = text.lower()
    
    # 2. Separar por palavras para garantir que nenhuma palavra será truncada
    words = text.split()
    
    chunks = []
    current_chunk = []
    current_bytes = 0
    
    for word in words:
        # Pega o tamanho real em bytes da palavra (caracteres acentudos = 2 bytes)
        word_bytes = len(word.encode('utf-8'))
        
        # Considera +1 byte para o espaço da junção (salvo se for a primeira palavra do trecho)
        added_bytes = word_bytes + (1 if current_chunk else 0)
        
        if current_bytes + added_bytes > chunk_size and current_chunk:
            # Se for ultrapassar os ~3KB, fecha o bloco atual e salva na lista
            chunks.append(" ".join(current_chunk))
            # O próximo bloco agora se inicia com a palavra atual
            current_chunk = [word]
            current_bytes = word_bytes
        else:
            # Senão, vai acumulando a palavra no bloco e somando o tamanho
            current_chunk.append(word)
            current_bytes += added_bytes
            
    # Guarda o último bloco remanescente, se sobrar algo
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    base_name = os.path.basename(input_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Cria a pasta de saída (se já existir, não faz nada)
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for i, chunk in enumerate(chunks):
        # Ignora blocos finais do livro ou sobras que ficaram muito curtas (< 500 bytes)
        # Textos minúsculos destroem a matemática da distância (NCD)
        if len(chunk.encode('utf-8')) < 500:
            continue
            
        # Gera os nomes: "iracema_p0001.txt", "iracema_p0002.txt"...
        output_filename = f"{name_without_ext}_p{i+1:04d}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(chunk)
        count += 1
        
    print(f"  -> [{base_name}] gerou {count} subarquivos.")

if __name__ == "__main__":
    # Verifica se os caminhos foram passados no terminal
    if len(sys.argv) != 3:
        print("===== UTILITÁRIO DE DATASET =============================================")
        print("Uso: python preparar_dataset.py <pasta_entrada> <pasta_saida>")
        print("Exemplo de Uso no seu projeto:")
        print("     python preparar_dataset.py ../dataset/romantismo/completos ../dataset/romantismo")
        print("=========================================================================")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Verifica se a pasta existe
    if not os.path.isdir(input_dir):
        print(f"Erro: A pasta de entrada '{input_dir}' não existe.")
        sys.exit(1)
        
    print(f"Buscando arquivos .txt na pasta '{input_dir}'...")
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not txt_files:
        print("Nenhum arquivo .txt encontrado nesta pasta!")
        sys.exit(0)
        
    print(f"Processando {len(txt_files)} arquivos. Particionando em ~3KB (lowercase):")
    for txt in txt_files:
        process_file(txt, output_dir, chunk_size=3000)
        
    print("\nTratamento concluído com sucesso!")
