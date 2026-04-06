# Classificador de Períodos Literários via PPM-C

Este projeto implementa um sistema de classificação de períodos literários (Romantismo, Realismo, Modernismo, etc.) baseado em entropia cruzada, utilizando o algoritmo de compactação **PPM-C** (Prediction by Partial Matching).

Através da variação da taxa de compactação (bits/símbolo) de arquivos literários aglutinados, o sistema consegue identificar correlações morfológicas profundas e inferir a classe de um arquivo misterioso sem a necessidade do aprendizado de deep learning convencional.

---

## 🛠 Pré-requisitos & Compilação

Certifique-se de que o algoritmo base em C++ responsável pelo compressor foi compilado.

```bash
# Na raiz do projeto, acione o Makefile para construir o executável `ppm`
make
```
*(Certifique-se que você possua o Python 3.12+ e as dependências padrão, opcionalmente instale a biblioteca `matplotlib` via Pip para exibir gráficos dos relatórios subsequentes).*

## 📊 Fluxo de Execução

### 1. Preparação do Dataset
Antes do benchmark, os livros contidos na pasta `dataset` precisam ser rigorosamente fatiados e padronizados contra ruídos nas respectivas pastas de classe.

Execute a construção local:
```bash
python3 utils/preparar_dataset.py
```
* **O que faz:** Rasga os arquivos base em dezenas de recortes literais ("chunks") limitados em tamanho exato (3KB, 6KB, 9KB) e os espelha aplicando gradualmente filtros semânticos e morfológicos (`untreated`, `lower`, `no_punctuation`, `no_accents`, `full_treated`).
* *(O script economiza processamento ignorando pastas nas quais a divisão já fora realizada com êxito previamente).*

### 2. Rodando o Classificador

Acesse a pasta interna e execute o orquestrador (Exemplo ideal):
```bash
cd classificador/
python3 avaliar_tudo.py --tamanho-chunk 9 --por-taxa --limiar 0.001 --multithread
```

**Principais Flags:**
* `--tamanho-chunk {3,6,9}`: Força o teste contra arquivos fracionados de um agrupamento específico de KB.
* `--multithread`: Inicia processamento paralelo utilizando o número de threads disponíveis.
* `--por-taxa`: A avaliação prioriza *bits/símbolo* invertendo a predição para ganho natural ao invés do tamanho resultante em bytes. 
* `--limiar`: Desce o limiar de sensibilidade do controle de ruído de variações negativas (ex: `0.0005` ou `0.001`).

### 3. Resultados
Por padrão, para cada lote rodado, o programa gera uma tabela em `.csv` baseada em acurácia contendo os micro-cenários em `classificador/resultados/`.
