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

# Sobre o Dataset
| FieldName | Description | Values |
| --- | --- | --- |
| No |  |  |
| Type | Real Estate Type | Residential Land(Land Only), Agricultural Land, Residential Land(Land and Building), Pre-owned Condominiums, etc., Forest Land |
| Region | The characteristics of surrounding areas | Residential Area, Potential Residential Area, Commercial Area, Industrial Area |
| MunicipalityCode | City code of japan |  |
| Prefecture | Prefecture Name of japan |  |
| Municipality | City |  |
| DistrictName | District name |  |
| NearestStation | Nearest station name |  |
| TimeToNearestStation | Time to the nearest station (original). The original data contains non-numeric values(ex. 1H-1H30, 30-60minuts). It is difficult to use for data analysis, I added min/max time to nearrest station columns. |  |
| MinTimeToNearestStation | Min time to the nearest station (minutes) |  |
| MaxTimeToNearestStation | Max time to the nearest station (minutes) |  |
| TradePrice | Trade prices (Yen) |  |
| FloorPlan | Floor plan | 3LDK' '4DK' '2LDK' '4LDK' '2DK' '1K' '3LDK+S' '5LDK' '3DK' '1LDK' '2DK+S' 'Open Floor' '1DK' '1R' '4LDK+S' '2K' '2LDK+S' '6DK' '1LDK+S' '5DK' '1R+S' '1LK' '1K+S' '3K' '7LDK' '4K' '3DK+S' '3D' '1DK+S' '6LDK' 'Studio Apartment' '6LDK+S' '4L+K' '5LDK+S' '7DK' '3LK' '5K' '2K+S' '8LDK' '3LDK+K' '3LD' '1L' '4DK+S' '2LK' 'Duplex' '7LDK+S' '4LDK+K' '3LD+S' '2LD+S' '8LDK+S' '4L' '2L' '2LDK+K' '2LK+S' '5LDK+K' '1LD+S' '2L+S' '3K+S' '1DK+K' '2LD' '1L+S' '2D' '4D' |
| Area | The surveyed area (m^2) |  |
| AreaIsGreaterFlag | An area of 2000 m^2 or greater, the area data are displayed 2000, and this flag is true. |  |
| UnitPrice | Unit Land Price(Yen) per m^2 |  |
| PricePerTsubo | Unit Land Price(Yen) per Tsubo. Tsubo is a japanese units, see https://en.wikipedia.org/wiki/Pyeong#Tsubo |  |
| LandShape | Land shape | Semi-rectangular Shaped' 'Semi-trapezoidal Shaped' 'Irregular Shaped' 'Trapezoidal Shaped' 'Rectangular Shaped' 'Semi-shaped' 'Semi-square Shaped' 'Square Shaped' 'Flag-shaped etc.' |
| Frontage | Frontage(m). |  |
| FrontageIsGreaterFlag | Some original Frontage data records "50.0m or longer." I converted Frontage to a number and added this flag. |  |
| TotalFloorArea | Total Floor Area (m^2). |  |
| TotalFloorAreaIsGreaterFlag | Some data are displayed as "2,000 m^2 or greater”, I converted Total Floor Area to a number and added this flag. |  |
| BuildingYear | Construction Year of Building |  |
| PrewarBuilding | Buildings built before 1945, the construction year data are displayed as “before the war.” I converted BuildYear to 1945, and added this flag. |  |
| Structure | Building Structure. SRC= Steel frame reinforced concrete, RC= Reinforced concrete, S = Steel frame, LS = Light steel structure, B = Concrete block, W = Wooden | RC' 'W' 'SRC' 'S' 'LS' 'S, W, LS' 'B' 'W, LS' 'RC, W' 'S, W' 'RC, S' 'W, B' 'RC, LS' 'RC, W, LS' 'RC, W, B' 'RC, S, LS' 'SRC, W' 'S, B' 'SRC, RC' 'S, LS' 'S, W, B' 'B, LS' 'RC, B' 'SRC, S' 'RC, S, W' 'SRC, B' 'W, B, LS' 'S, B, LS' 'RC, B, LS' 'RC, S, B' 'SRC, RC, S' 'SRC, LS' 'SRC, W, B' 'S, RC, W' 'RC, S, W, B' |
| Use | Current Usage. For example, House, office, shop, factory, warehouse, workshop, parking lot, and other |  |
| Purpose | The purpose of future use. For example, House, shop, office, factory, warehouse, and other. | Other' 'House' 'Warehouse' 'Office' 'Factory' 'Shop' |
| Direction | Frontage road Direction | Southwest' 'Northwest' 'East' 'No facing road' 'Northeast' 'Southeast' 'South' 'West' 'North' |
| Classification | Frontage road Classification | Road' 'City Road' 'Prefectural Road' nan 'Village Road' 'Private Road' 'National Highway' 'Access Road' 'Agricultural Road' 'Ward Road' 'Town Road' 'Kyoto/ Osaka Prefectural Road' 'Forest Road' 'Hokkaido Prefectural Road' 'Tokyo Metropolitan Road' |
| Breadth | Frontage road Breadth(m) |  |
| CityPlanning | The use districts designated by the City Planning Act | Category I Exclusively Low-story Residential Zone' 'Urbanization Control Area' nan 'Category I Residential Zone' 'Category I Exclusively Medium-high Residential Zone' 'Category II Exclusively Medium-high Residential Zone' 'Quasi-industrial Zone' 'Neighborhood Commercial Zone' 'Commercial Zone' 'Category II Residential Zone' 'Exclusively Industrial Zone' 'Industrial Zone' 'Category II Exclusively Low-story Residential Zone' 'Outside City Planning Area' 'Quasi-residential Zone' 'Non-divided City Planning Area' 'Quasi-city Planning Area' |
| CoverageRatio | Maximus Building Coverage Ratio(%) |  |
| FloorAreaRatio | Maximus Floor-area Ratio(%) |  |
| Period | Time of transaction |  |
| Year | Time of transaction year |  |
| Quarter | Time of transaction year-quarter |  |
| Renovation | Renovation? | Not yet' 'Done' |
| Remarks | Note |  |