# Japan-Real-Estate-Prices
Trabalho realizado na disciplina de Aprendizado de Máquina (CMP263) na UFRGS no semestre 2026/1. O objetivo é realizar a
predição do valor de imóveis a partir de dados coletados pelo MLIT (Ministério da Terra, Infraestrutura, Transporte e
Turismo do Japão) durante o período de 2005 à 2019;

# Dataset
Disponível no link: https://www.kaggle.com/datasets/nishiodens/japan-real-estate-transaction-prices, ou instalado
através da API do Kaggle no executável `main.py`;

# Reprodução
Instalar ambiente virtual:
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
``` 

### Execução local
Após instalar as dependências, execute:
```bash
python main.py
```

### Executar Docker
```bash
docker build -t japan-real-estate-prices .
docker run --rm japan-real-estate-prices
```

# Pipeline
O projeto baixa e carrega os dados, faz limpeza e engenharia de atributos, separa treino/validação/teste por tempo,
treina modelos de regressão e compara os resultados com métricas como MAPE, RMSE, MAE e R².

# Estrutura
- `main.py`: ponto de entrada do pipeline;
- `src/`: módulos de dados, pré-processamento, modelos, avaliação e visualização;
- `parameters/`: parâmetros e resultados gerados durante os experimentos.