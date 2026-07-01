# TFM — Modelos Predictivos para la Optimización del Consumo Energético en Climatización

Trabajo de Fin de Máster (Master en Ciencia de Datos e Inteligencia Artificial, UNAV).  
Objetivo: predecir el consumo energético del sistema HVAC del edificio **E3Lab** usando datos de sensores IoT y variables meteorológicas.

---

## Estructura del repositorio

```
modelos_pred/
├── TFM_E3Lab_Preprocesamiento.ipynb   # ETL, limpieza y feature engineering
├── TFM_E3Lab_Modelos5.ipynb           # Entrenamiento y evaluación de modelos
├── html_exports/
│   ├── TFM_E3Lab_Preprocesamiento.html  # Vista renderizada (con gráficas)
│   └── TFM_E3Lab_Modelos5.html          # Vista renderizada (con gráficas)
└── README.md
```

> **Nota:** El fichero `E3Lab_modelo.csv` (~29 MB) no se incluye en el repositorio por superar el límite de GitHub. Puedes descargarlo desde [aquí](#) o solicitarlo al autor.

---

## Notebooks

### 1. Preprocesamiento — [`TFM_E3Lab_Preprocesamiento.ipynb`](TFM_E3Lab_Preprocesamiento.ipynb)

Pipeline completo de preparación de datos:

| Bloque | Contenido |
|--------|-----------|
| Carga | Unión de múltiples CSVs Grafana con metadatos embebidos en el nombre de columna |
| Limpieza | Duplicados temporales, outliers físicos, imputación por interpolación lineal |
| EDA | Series temporales, boxplots mensuales, heatmaps día×hora, matriz de correlación |
| Segmentación | Análisis por planta, zona (A/B/C) y estación del año |
| Meteorología | Fusión con estación meteorológica (resampleo 1 min → 15 min) |
| Feature engineering | Variable objetivo (potencia HVAC), lag features, medias móviles, agregados del edificio |

### 2. Modelos — [`TFM_E3Lab_Modelos5.ipynb`](TFM_E3Lab_Modelos5.ipynb)

Comparativa de cinco familias de modelos con split temporal 70/15/15:

| Modelo | Descripción |
|--------|-------------|
| Baseline naive | Predicción por el valor del día anterior (lag-24h) |
| Ridge Regression | Regresión lineal regularizada como referencia |
| Random Forest | Ensemble de árboles con selección de variables |
| XGBoost | Gradient boosting optimizado con Optuna |
| LSTM | Red neuronal recurrente (PyTorch) |
| Prophet | Modelo aditivo de series temporales (Meta) |

Además se evalúan modelos **por estación** (primavera/verano/otoño/invierno) y por **grupos climáticos** (Invierno+Otoño vs. Primavera+Verano), con selección de variables mediante importancia RF/XGBoost.

---

## Cómo visualizar los notebooks

Los notebooks con todas las gráficas renderizadas están en [`html_exports/`](html_exports/).  
Descarga el `.html` y ábrelo en tu navegador, o usa [nbviewer.org](https://nbviewer.org) pegando la URL del `.ipynb` en GitHub.

Para ejecutarlos localmente:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch prophet optuna
jupyter notebook
```

---

## Datos

- **Fuente:** Sensores IoT del edificio E3Lab (temperatura, humedad, CO₂, potencia HVAC) exportados desde Grafana, más estación meteorológica exterior.
- **Resolución temporal:** 15 minutos.
- **Período:** 2022–2024.
- **Dataset de modelado:** `E3Lab_modelo.csv` (generado por el notebook de preprocesamiento).

---

## Autor

**José Ignacio Esteban González**  
Máster en Ciencia de Datos e Inteligencia Artificial — Universidad de Navarra
