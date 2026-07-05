"""
config.py — Constantes globales del proyecto E3Lab Energy Prediction.
Todas las rutas, umbrales, semillas y listas de columnas en un único lugar.
"""

from pathlib import Path

# ── Rutas base ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "E3Lab Grafana Datos"
METEO_DIR    = PROJECT_ROOT / "Estacion Meteo"
OUTPUT_DIR   = PROJECT_ROOT / "output"
FIGURES_DIR  = OUTPUT_DIR / "figures"
DATA_OUT_DIR = OUTPUT_DIR / "data"
MODEL_DIR    = OUTPUT_DIR / "modelos_guardados"

FIGURES_PREPROCESSING = FIGURES_DIR / "preprocessing"
FIGURES_MODELS        = FIGURES_DIR / "models"

# ── Archivos CSV de Grafana ───────────────────────────────────────────────────
CSV_FILES = [
    # 2023
    DATA_DIR / "E3Lab Grafana 2023-05-12-2023-09-30.csv",
    DATA_DIR / "E3Lab Grafana 2023-09-30-2023-12-31.csv",
    # 2024
    DATA_DIR / "E3Lab Grafana 2024-01-01-2024-05-31.csv",
    DATA_DIR / "E3Lab Grafana 2024-05-31-2024-09-30.csv",
    DATA_DIR / "E3Lab Grafana 2024-09-30-2024-12-31.csv",
    # 2025
    DATA_DIR / "E3Lab Grafana 2025-01-01-2025-05-31.csv",
    DATA_DIR / "E3Lab Grafana 2025-05-31-2025-10-07.csv",
    DATA_DIR / "E3Lab Grafana 2025-10-07-2025-12-08.csv",
]

# ── Limpieza ──────────────────────────────────────────────────────────────────
UMBRAL_NULOS   = 60.0   # porcentaje; columnas con más nulos se eliminan
MAX_GAP_PERIODS = 8      # máximo de períodos de 15 min a interpolar

RANGOS_FISICOS = {
    "CO2":  (300, 5000),
    "HUM":  (0,   100),
    "TEMP": (10,   35),
    "TIMP": (3,    80),
    "TRET": (3,    80),
    "TRAD": (-40,  80),
    "TTER": (-20,  40),
    "POT":  (0,  None),
    "ENER": (0,  None),
    "VOL":  (0,  None),
    "CAUD": (0,  None),
}

RANGOS_METEO = {
    "temp_ext":   (-30, 50),
    "hr_ext":     (0, 100),
    "dew_point":  (-40, 35),
    "wind_speed": (0, 60),
    "solar_rad":  (0, 1400),
    "solar_diff": (0, 800),
    "rain":       (0, 100),
    "wind_dir":   (0, 360),
}

# ── Índices de columnas en los CSV de la estación meteorológica ───────────────
METEO_COL_IDX = {
    "timestamp":  1,
    "temp_ext":   2,
    "hr_ext":     3,
    "dew_point":  4,
    "wind_speed": 5,
    "solar_rad":  6,
    "solar_diff": 7,
    "rain":       8,
    "wind_dir":   12,
}

# ── Mapeo de tipos de sensor a código corto ───────────────────────────────────
TIPO_CORTO = {
    "CO2":                    "CO2",
    "Humedad Relativa":       "HUM",
    "Temperatura":            "TEMP",
    "Temperatura Impulsion":  "TIMP",
    "Temperatura Retorno":    "TRET",
    "Temperatura Radiante":   "TRAD",
    "Temperatura Terreno":    "TTER",
    "Diferencia Temperatura": "TDIF",
    "Caudal":                 "CAUD",
    "Potencia":               "POT",
    "Energia":                "ENER",
    "Volumen":                "VOL",
}

# ── Features del modelo ───────────────────────────────────────────────────────
FEATURES = [
    # Temporales cíclicas
    "hora_sin", "hora_cos", "dia_sin", "dia_cos", "mes_sin", "mes_cos",
    # Temporales categóricas
    "calendario_lectivo", "estacion_num",
    # Lags y rolling de la variable objetivo
    "energia_lag_1", "energia_lag_4", "energia_lag_8",
    "energia_lag_96", "energia_lag_672",
    "energia_roll_1h", "energia_roll_6h", "energia_roll_24h",
    # Meteorología
    "meteo_temp_ext", "meteo_hr_ext", "meteo_solar_rad",
    "meteo_solar_diff", "meteo_rain", "meteo_wind_speed",
    # Edificio
    "temp_int_media", "hum_int_media", "co2_medio", "timp_media", "delta_temp",
]

TARGET = "target_energia_15min_hvac"

# ── Lags y ventanas rolling ───────────────────────────────────────────────────
LAGS = {
    "energia_lag_1":   1,
    "energia_lag_4":   4,
    "energia_lag_8":   8,
    "energia_lag_96":  96,
    "energia_lag_672": 672,
}

ROLLING = {
    "energia_roll_1h":  4,
    "energia_roll_6h":  24,
    "energia_roll_24h": 96,
}

# ── Calendario académico UNAV ─────────────────────────────────────────────────
PERIODOS_OPERACION_NORMAL = [
    ("2023-09-04", "2024-06-29"),
    ("2024-09-02", "2025-06-27"),
]

PERIODOS_VACACIONES = [
    ("2023-12-22", "2024-01-05"),
    ("2024-03-25", "2024-04-06"),
    ("2024-07-01", "2024-08-31"),
    ("2024-12-20", "2025-01-07"),
    ("2025-04-14", "2025-04-26"),
    ("2025-06-28", "2025-08-30"),
]

FESTIVOS_UNAV_PAMPLONA = [
    "2023-10-12", "2023-11-01", "2023-11-29", "2023-12-04",
    "2023-12-06", "2023-12-08", "2024-01-28", "2024-03-19",
    "2024-05-01", "2024-06-26",
    "2024-10-12", "2024-11-01", "2024-11-29", "2024-12-03",
    "2024-12-06", "2024-12-08", "2025-01-28", "2025-03-19",
    "2025-05-01", "2025-06-26",
]

# ── Visualización ─────────────────────────────────────────────────────────────
SEED = 42

PLANTAS = ["P0", "P1", "P2", "P3", "PS"]

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

ESTACIONES = ["invierno", "primavera", "verano", "otoño"]

COLORES_ZONA = {
    "A":        "#e63946",
    "B":        "#457b9d",
    "C":        "#2a9d8f",
    "Interior": "#f4a261",
    "Exterior": "#6d6875",
}

COLORES_EST = {
    "invierno":  "#4e9af1",
    "primavera": "#57cc99",
    "verano":    "#f4a261",
    "otoño":     "#c77dff",
}

# Grupos estacionales para modelos agrupados
GRUPO_A_ESTACIONES = ["invierno", "otono"]   # Frío
GRUPO_B_ESTACIONES = ["primavera", "verano"]  # Calor

# ── Hiperparámetros de modelos ────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":    300,
    "max_depth":       12,
    "min_samples_leaf": 5,
    "max_features":    0.5,
    "n_jobs":          -1,
}

RF_SEASONAL_PARAMS = {
    "n_estimators":    200,
    "max_depth":       12,
    "min_samples_leaf": 5,
    "max_features":    0.5,
    "n_jobs":          -1,
}

XGB_PARAM_GRID = [
    {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 3,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.0,  "reg_lambda": 1.0},
    {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 5,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.05, "reg_lambda": 1.5},
    {"max_depth": 5, "learning_rate": 0.04, "min_child_weight": 5,
     "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.10, "reg_lambda": 2.0},
    {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 7,
     "subsample": 0.80, "colsample_bytree": 0.75, "reg_alpha": 0.10, "reg_lambda": 3.0},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 3,
     "subsample": 0.90, "colsample_bytree": 0.80, "reg_alpha": 0.00, "reg_lambda": 1.0},
    {"max_depth": 5, "learning_rate": 0.02, "min_child_weight": 8,
     "subsample": 0.90, "colsample_bytree": 0.90, "reg_alpha": 0.20, "reg_lambda": 2.5},
]

XGB_SEASONAL_PARAMS = {
    "objective":           "reg:squarederror",
    "n_estimators":        1000,
    "max_depth":           4,
    "learning_rate":       0.04,
    "subsample":           0.85,
    "colsample_bytree":    0.85,
    "min_child_weight":    5,
    "reg_alpha":           0.05,
    "reg_lambda":          1.5,
    "early_stopping_rounds": 40,
    "eval_metric":         "rmse",
    "tree_method":         "hist",
    "n_jobs":              -1,
}

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.2
LSTM_SEQ_LEN     = 24
LSTM_BATCH_SIZE  = 256
LSTM_EPOCHS      = 40
LSTM_LR          = 1e-3
LSTM_PATIENCE    = 8

RIDGE_ALPHAS = list(__import__("numpy").logspace(-3, 3, 13))

# ── Análisis de features ──────────────────────────────────────────────────────
UMBRAL_FI   = 0.005   # importancia normalizada mínima (media RF+XGB)
UMBRAL_PERM = 0.0     # permutation importance > 0
N_REPEATS_PERM = 10   # repeticiones Monte Carlo para permutation importance
N_SHAP_SAMPLES = 2000 # muestra de train para SHAP
