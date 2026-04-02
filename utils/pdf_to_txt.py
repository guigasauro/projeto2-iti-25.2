import os
import glob
import sys
from pypdf import PdfReader

def convert_pdf_to_txt(pdf_path, output_dir):
    """
    Lê um arquivo PDF página por página e extrai todo o texto
    para um novo arquivo .txt correspondente na pasta de saída.
    """
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"Erro ao abrir {pdf_path}: {e}")
        return

    text_content = []
    
    # Extrai o texto página por página
    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
        except Exception as e:
            print(f"Aviso: Não foi possível extrair a página {page_num} de {pdf_path}")

    # Junta as páginas com quebras de linha
    full_text = "\n".join(text_content)
    
    # Se o PDF for só imagens, ele ficará vazio
    if not full_text.strip():
        print(f"Atenção: Nenhum texto foi extraído de {pdf_path}. Era um PDF escaneado como imagem?")
        return
        
    base_name = os.path.basename(pdf_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{name_without_ext}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
        
    print(f"  -> [{base_name}] convertido com sucesso! Salvou em '{output_filename}'")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("===== CONVERSOR PDF -> TXT ==============================================")
        print("Uso: python pdf_to_txt.py <pasta_entrada_pdfs> <pasta_saida_txts>")
        print("Exemplo:")
        print("     python pdf_to_txt.py ../dataset/pdf_originais ../dataset/romantismo/completos")
        print("=========================================================================")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.isdir(input_dir):
        print(f"Erro: A pasta de entrada '{input_dir}' não existe.")
        sys.exit(1)
        
    print(f"Buscando arquivos .pdf na pasta '{input_dir}'...")
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    # Também busca com extensão maiúscula caso haja algum .PDF
    pdf_files.extend(glob.glob(os.path.join(input_dir, "*.PDF")))
    
    if not pdf_files:
        print("Nenhum arquivo PDF encontrado nesta pasta!")
        sys.exit(0)
        
    print(f"Processando e convertendo {len(pdf_files)} arquivos PDF para TXT:")
    for pdf in pdf_files:
        convert_pdf_to_txt(pdf, output_dir)
        
    print("\nConversão concluída! Agora você já pode rodar o seu script `preparar_dataset.py` nos txts gerados.")
