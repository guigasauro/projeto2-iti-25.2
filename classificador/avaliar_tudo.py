import os
import shutil
import tempfile
from avaliar_alfabetos_tamanhos import avaliar_cenario_acuracia

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    classes = ['modernismo', 'realismo', 'romantismo']
    
    # Devido à diferença na quantidade gerada por cada tamanho de chunk (arquivos maiores geram menos partes),
    # definimos um limite seguro de treino e teste para que não falte dados pra teste.
    tamanhos = {
        '3kb': {'treino': 40, 'teste': 5},
        '6kb': {'treino': 20, 'teste': 5},
        '9kb': {'treino': 10, 'teste': 5}
    }
    
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
                        acuracia = avaliar_cenario_acuracia(temp_dataset_dir, treino, teste, kmax=k)
                        resultados[alfabeto][k] = acuracia
                else:
                    for k in kVals:
                        resultados[alfabeto][k] = 0.0
                
            finally:
                # Limpeza
                shutil.rmtree(temp_dataset_dir, ignore_errors=True)
                
        print("\n" + "-"*80)
        print(f"RESUMO DAS ACURÁCIAS - CHUNKS DE {tamanho.upper()} (ALFABETO x KMAX):")
        
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

if __name__ == '__main__':
    main()
