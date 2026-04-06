import os
import glob
import sys
import argparse
import unicodedata
import string

def process_file(input_path, output_dir, chunk_size=3000, keep_case=False, remove_accents=False, remove_punctuation=False):
    """
    Lê um arquivo de texto, aplica pré-processamentos, e divide em 
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

    # 1. Transformações de Alfabeto
    if keep_case:
        text = text.lower()
        
    if remove_accents:
        # Normaliza retirando os diacríticos
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        
    if remove_punctuation:
        # Substitui a pontuação por espaços (para não fundir palavras, ex: "fim.O" -> "fim O")
        trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(trans)
    
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
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    periodos = ['modernismo', 'pré-modernismo', 'realismo', 'romantismo']
    
    tamanhos_chunk = {
        '3kb': 3000,
        '6kb': 6000,
        '9kb': 9000
    }
    
    tipos_dataset = {
        'full_treated':   {'keep_case': True,  'remove_accents': True,  'remove_punctuation': True},
        'lower':          {'keep_case': True,  'remove_accents': False, 'remove_punctuation': False},
        'no_accents':     {'keep_case': False, 'remove_accents': True,  'remove_punctuation': False},
        'no_punctuation': {'keep_case': False, 'remove_accents': False, 'remove_punctuation': True},
        'untreated':      {'keep_case': False, 'remove_accents': False, 'remove_punctuation': False}
    }
    
    for periodo in periodos:
        input_dir = os.path.join(BASE_DIR, periodo, 'completos')
        if not os.path.exists(input_dir):
            print(f"Aviso: Pasta de originais não encontrada: {input_dir}")
            continue
            
        txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
        if not txt_files:
            continue
            
        print(f"\n=========================================")
        print(f"[{periodo.upper()}] - Processando {len(txt_files)} arquivos originais...")
        
        for nome_tamanho, bytes_tamanho in tamanhos_chunk.items():
            for nome_tipo, flags in tipos_dataset.items():
                output_dir = os.path.join(BASE_DIR, periodo, 'data', nome_tamanho, nome_tipo)
                
                if os.path.exists(output_dir) and any(f.endswith('.txt') for f in os.listdir(output_dir)):
                    continue
                
                print(f" -> Gerando recortes de {nome_tamanho} | Filtro: {nome_tipo}")
                for txt in txt_files:
                    process_file(
                        txt, 
                        output_dir, 
                        chunk_size=bytes_tamanho, 
                        keep_case=flags['keep_case'], 
                        remove_accents=flags['remove_accents'], 
                        remove_punctuation=flags['remove_punctuation']
                    )
                    
    print("\nTratamento em lote concluído com sucesso!")
