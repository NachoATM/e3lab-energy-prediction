"""
run_preprocessing.py — Pipeline completo de preprocesamiento E3Lab.

Ejecutar desde la raíz del proyecto:
    python run_preprocessing.py

Pasos:
  1. Carga de CSVs Grafana + estación meteorológica.
  2. Limpieza: reindexado, filtrado nulos, rangos físicos, interpolación, merge meteo.
  3. Ingeniería de variables: target delta-energía, features temporales, lags, agregados.
  4. Generación de figuras exploatorias (PNG en output/figures/preprocessing/).
  5. Exportación de artefactos a output/data/.
"""

import sys
import time
from pathlib import Path

# Asegurar que src/ es importable cuando se lanza desde la raíz del proyecto
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import (
    DATA_OUT_DIR,
    FIGURES_PREPROCESSING,
    FEATURES,
    TARGET,
)
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.visualizer import PreprocessingVisualizer


def main():
    t0 = time.time()

    # Crear directorios de salida
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_PREPROCESSING.mkdir(parents=True, exist_ok=True)

    # ── 1. Carga ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 1: Carga de datos")
    print("=" * 60)
    loader = DataLoader()
    df_raw, df_meta, df_meteo = loader.load()

    # Guardar metadatos
    meta_path = DATA_OUT_DIR / "E3Lab_metadata_sensores.csv"
    df_meta.to_csv(meta_path, index=False)
    print(f"  Metadatos guardados: {meta_path}")

    # ── 2. Limpieza ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 2: Limpieza")
    print("=" * 60)
    cleaner = DataCleaner()
    df_clean, df_final = cleaner.clean(df_raw, df_meteo)

    # Guardar dataset limpio (sin meteo) y con meteo
    clean_path = DATA_OUT_DIR / "E3Lab_clean.csv"
    final_path = DATA_OUT_DIR / "E3Lab_final.csv"
    df_clean.to_csv(clean_path)
    df_final.to_csv(final_path)
    print(f"  Dataset limpio guardado: {clean_path}")
    print(f"  Dataset con meteo guardado: {final_path}")

    if not cleaner.outliers_fisicos.empty:
        out_path = DATA_OUT_DIR / "E3Lab_outliers_fisicos.csv"
        cleaner.outliers_fisicos.to_csv(out_path, index=False)
        print(f"  Outliers físicos guardados: {out_path}")

    # ── 3. Ingeniería de variables ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 3: Ingeniería de variables")
    print("=" * 60)
    fe = FeatureEngineer()
    df_final_enr, df_model = fe.build(df_final)

    model_path = DATA_OUT_DIR / "E3Lab_modelo.csv"
    df_model.to_csv(model_path)
    print(f"\n  Dataset de modelado guardado: {model_path}")
    print(f"  Shape: {df_model.shape}")

    # Copiar también a la raíz del proyecto (compatibilidad con el notebook)
    root_model_path = ROOT / "E3Lab_modelo.csv"
    df_model.to_csv(root_model_path)
    print(f"  Copia en raíz del proyecto: {root_model_path}")

    # ── 4. Visualizaciones ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 4: Generación de figuras (puede tardar varios minutos)")
    print("=" * 60)
    viz = PreprocessingVisualizer(output_dir=FIGURES_PREPROCESSING)
    viz.plot_all(
        df_clean   = df_clean,
        df_model   = df_model,
        features   = fe.features,
        target     = TARGET,
    )

    # ── Resumen ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"  Tiempo total:              {elapsed:.0f} s")
    print(f"  Shape df_clean:            {df_clean.shape}")
    print(f"  Shape df_final (con meteo): {df_final.shape}")
    print(f"  Shape df_model:            {df_model.shape}")
    print(f"  Features en modelo:        {len(fe.features)}")
    print(f"  Target cobertura:          "
          f"{df_model[TARGET].notna().mean() * 100:.1f} %")
    print(f"\n  Artefactos en:  {DATA_OUT_DIR}")
    print(f"  Figuras en:     {FIGURES_PREPROCESSING}")


if __name__ == "__main__":
    main()
