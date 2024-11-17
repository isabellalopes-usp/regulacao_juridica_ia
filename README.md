## Regulação Jurídica de IA
## Descrição
Este projeto visa coletar, pré-processar e estruturar decisões de segunda instância do Tribunal de Justiça do Estado de São Paulo (TJSP), utilizando técnicas de raspagem de dados e Processamento de Linguagem Natural (PLN). Outro objetivo é classificar sentenças favoráveis ao banco ou ao cliente, utilizando o modelo JurisBERT.

## Estrutura do Projeto

### 1. Coleta de Dados
- **`scraping.r`**: Realiza a raspagem das decisões do portal e-SAJ com base em uma lista de termos de busca.
  - **Uso**: Insira suas credenciais do TJSP para acessar o portal.
  - **Entrada**: `termos_utilizados.xlsx` (lista de termos de busca).
  - **Saída**: Arquivo CSV contendo as decisões raspadas.

### 2. Pré-processamento
- **`preprocessing.py`**: Limpa os textos, extrai excertos relevantes e identifica referências cruzadas entre as decisões.
- **`duplicates.py`**: Gera, para cada processo, uma lista dos outros termos nos quais ele aparece.

### 3. Aplicação do Modelo JurisBERT
- **`jurisbert_application.py`**: Aplica o modelo BERT pré-treinado (JurisBERT) em um dataset de decisões sobre a validade de contratos bancários firmados por meio de reconhecimento facial.
  - **Classificação**: Sentenças favoráveis ao banco ou ao cliente.

### 4. Manipulação dos Datasets
- **`jurisbert_split_datasets.py`**: Divide explicitamente os datasets de treino e teste e aplica o modelo JurisBERT.
- **`jurisbert_label_shuffling.py`**: Embaralha os labels do dataset de treino para análise da robustez do modelo.

## Dados

- **`termos_utilizados.xlsx`**: Lista de termos utilizados para a raspagem.
- **`dataset.csv`**: Dataset original com 524 instâncias relacionadas à validade de contratos bancários.
- **`balanced_dataset.csv`**: Dataset balanceado com 362 instâncias utilizado para as aplicações do modelo JurisBERT.
