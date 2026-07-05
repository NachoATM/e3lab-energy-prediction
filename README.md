# TFM — Modelos Predictivos para la Optimización del Consumo Energético en Climatización

Trabajo de Fin de Máster · Máster en Ciencia de Datos e Inteligencia Artificial, Universidad de Navarra  
Predicción del consumo energético del sistema HVAC del edificio **E3Lab (ETSA)** a partir de sensores IoT y variables meteorológicas.

---

## Estructura del proyecto

```
e3lab-energy-prediction/
├── src/
│   ├── config.py                        # Constantes globales (rutas, umbrales, hiperparámetros)
│   ├── preprocessing/
│   │   ├── data_loader.py               # DataLoader — carga y unión de CSVs Grafana + meteo
│   │   ├── cleaner.py                   # DataCleaner — nulos, outliers, interpolación
│   │   ├── feature_engineer.py          # FeatureEngineer — variable objetivo, lags, agregados
│   │   └── visualizer.py                # PreprocessingVisualizer — guarda PNGs por planta/estación
│   └── models/
│       ├── base_model.py                # BaseModel abstracta + helpers (evaluar, plot_prediccion)
│       ├── ridge.py                     # RidgeModel
│       ├── random_forest.py             # RandomForestModel
│       ├── xgboost_model.py             # XGBoostModel (early stopping + búsqueda de hiperparámetros)
│       ├── lstm_model.py                # LSTMModel + LSTMDataset (PyTorch)
│       ├── prophet_model.py             # ProphetModel (Meta)
│       └── evaluator.py                 # ModelEvaluator — métricas, comparativas, SHAP, selección
├── output/
│   ├── data/                            # CSVs intermedios generados por run_preprocessing.py
│   └── figures/
│       ├── preprocessing/               # PNGs del análisis exploratorio
│       └── models/                      # PNGs de predicciones, residuos, importancias, SHAP
├── notebooks/
│   ├── TFM_E3Lab_Preprocesamiento.ipynb # Notebook original — referencia
│   └── TFM_E3Lab_Modelos5.ipynb         # Notebook original — referencia
├── run_preprocessing.py                 # Punto de entrada — bloque 1
└── run_models.py                        # Punto de entrada — bloque 2
```

> `E3Lab_modelo.csv` (~29 MB) no se incluye en el repositorio por superar el límite de GitHub.

---

## Cómo ejecutar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch prophet shap joblib
```

```bash
# Bloque 1 — ETL, limpieza, feature engineering y visualización exploratoria
python run_preprocessing.py

# Bloque 2 — Entrenamiento, evaluación y comparativa de modelos
python run_models.py
```

Todas las figuras se guardan automáticamente en `output/figures/`. No se requiere Jupyter.

---

## Pipeline de preprocesamiento

| Paso | Clase | Descripción |
|------|-------|-------------|
| Carga | `DataLoader` | Une múltiples CSVs Grafana; parsea metadatos embebidos en el nombre de columna (planta, zona, tipo de sensor) |
| Limpieza | `DataCleaner` | Elimina columnas con >60% de nulos, filtra outliers físicos por rango, interpola huecos ≤2 h, fusiona datos meteorológicos |
| Feature engineering | `FeatureEngineer` | Construye la variable objetivo (ΔEnergía HVAC 15 min), features temporales con codificación sin/cos, lags (15 min–1 semana), medias móviles, agregados del edificio |
| Visualización | `PreprocessingVisualizer` | Series diarias, boxplots mensuales, heatmaps día×hora y comparativas por planta, zona y estación; guarda ~25 PNGs |

---

## Modelos entrenados

Comparativa con split temporal cronológico **70 / 15 / 15** (train / validación / test):

| Modelo | Clase | Notas |
|--------|-------|-------|
| Baseline naive | — | lag-24 h como referencia mínima |
| Ridge Regression | `RidgeModel` | RobustScaler + búsqueda de alpha en logspace |
| Random Forest | `RandomForestModel` | 300 árboles, max_depth=12 |
| XGBoost | `XGBoostModel` | Búsqueda manual de hiperparámetros + early stopping=50 |
| LSTM | `LSTMModel` | PyTorch · hidden=128, layers=2, seq_len=24 (6 h), HuberLoss |
| Prophet | `ProphetModel` | Meta · estacionalidades semanal, anual e intradiaria (Fourier-10) |

Adicionalmente se entrenan variantes **por estación** (invierno / primavera / verano / otoño) y por **grupos climáticos** (Frío: Invierno+Otoño · Cálido: Primavera+Verano).

### Selección de variables (`ModelEvaluator`)

1. Correlación de Pearson con el target
2. Feature importance intrínseca (RF y XGBoost)
3. Permutation importance sobre validación
4. SHAP values (TreeExplainer)

Se entrena una versión optimizada de RF y XGBoost con el subconjunto consensuado y se compara contra los modelos originales.

---

## Datos

| Campo | Detalle |
|-------|---------|
| Fuente | Sensores IoT del edificio E3Lab exportados desde Grafana + estación meteorológica exterior |
| Variables | Temperatura, humedad relativa, CO₂, temperatura de impulsión/retorno, potencia y energía HVAC |
| Resolución | 15 minutos |
| Período | 2023–2025 |
| Dataset de modelado | `E3Lab_modelo.csv` (generado por `run_preprocessing.py`) |
