"""
cleaner.py — Limpieza del dataset crudo de Grafana.

Pasos:
  1. Reindexado a rejilla completa de 15 min.
  2. Eliminación de columnas con demasiados nulos (>= UMBRAL_NULOS).
  3. Filtrado de valores fuera de rango físico (RANGOS_FISICOS).
  4. Interpolación lineal limitada (MAX_GAP_PERIODS) sobre variables interpolables.
  5. Merge con datos meteorológicos (left join).
"""

import pandas as pd
import numpy as np

from src.config import (
    UMBRAL_NULOS,
    MAX_GAP_PERIODS,
    RANGOS_FISICOS,
)


class DataCleaner:
    """Aplica el pipeline de limpieza sobre df_raw (Grafana) y lo une con df_meteo.

    Atributos públicos tras llamar a clean():
        df_clean — Dataset limpio con índice 15 min.
        df_final — Dataset limpio + columnas meteo.
        outliers_fisicos — DataFrame con detalle de valores fuera de rango reemplazados.
    """

    def __init__(
        self,
        umbral_nulos: float = UMBRAL_NULOS,
        max_gap_periods: int = MAX_GAP_PERIODS,
        rangos_fisicos: dict = None,
    ):
        self.umbral_nulos    = umbral_nulos
        self.max_gap_periods = max_gap_periods
        self.rangos_fisicos  = rangos_fisicos if rangos_fisicos is not None else RANGOS_FISICOS
        self.df_clean         = None
        self.df_final         = None
        self.outliers_fisicos = None

    # ── Paso 1: reindexar a rejilla 15 min ──────────────────────────────────

    def reindex_15min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rellena los timestamps faltantes con NaN para tener rejilla uniforme."""
        idx_completo = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq="15min",
        )
        huecos = len(idx_completo) - len(df)
        print(
            f"  Registros esperados: {len(idx_completo):,} | "
            f"Presentes: {len(df):,} | Huecos: {huecos:,} "
            f"({huecos/len(idx_completo)*100:.1f} %)"
        )
        df_reindexed = df.reindex(idx_completo)
        df_reindexed.index.name = "timestamp"
        return df_reindexed

    # ── Paso 2: eliminar columnas con demasiados nulos ────────────────────────

    def drop_high_null_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina columnas con porcentaje de nulos superior a umbral_nulos."""
        nulos_pct     = df.isna().mean() * 100
        cols_eliminar = nulos_pct[nulos_pct > self.umbral_nulos].index.tolist()
        print(
            f"  Columnas eliminadas (> {self.umbral_nulos:.0f} % nulos): "
            f"{len(cols_eliminar)}"
        )
        return df.drop(columns=cols_eliminar), nulos_pct

    # ── Paso 3: filtrado por rango físico ────────────────────────────────────

    def apply_physical_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reemplaza por NaN los valores fuera del rango físico esperado."""
        outliers_detalle = []
        outliers_total   = 0

        for col in df.columns:
            tipo = col.split("_")[0]
            if tipo not in self.rangos_fisicos:
                continue
            vmin, vmax = self.rangos_fisicos[tipo]
            mask = pd.Series(False, index=df.index)
            if vmin is not None:
                mask = mask | (df[col] < vmin)
            if vmax is not None:
                mask = mask | (df[col] > vmax)
            n_out = int(mask.sum())
            if n_out > 0:
                valores_fuera = df.loc[mask, col]
                n_validos     = int(df[col].notna().sum())
                outliers_detalle.append({
                    "variable":         col,
                    "tipo":             tipo,
                    "rango_min":        vmin,
                    "rango_max":        vmax,
                    "n_fuera_rango":    n_out,
                    "pct_fuera_rango":  np.nan if n_validos == 0 else 100 * n_out / n_validos,
                    "min_observado":    valores_fuera.min(),
                    "max_observado":    valores_fuera.max(),
                    "primer_timestamp": valores_fuera.index.min(),
                    "ultimo_timestamp": valores_fuera.index.max(),
                })
                df.loc[mask, col] = np.nan
                outliers_total += n_out

        self.outliers_fisicos = (
            pd.DataFrame(outliers_detalle)
            .sort_values(["tipo", "n_fuera_rango"], ascending=[True, False])
            .reset_index(drop=True)
            if outliers_detalle
            else pd.DataFrame()
        )

        print(f"  Valores fuera de rango reemplazados por NaN: {outliers_total:,}")
        return df

    # ── Paso 4: interpolación ────────────────────────────────────────────────

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolación lineal limitada; ENER y VOL no se interpolan."""
        cols_no_interpolar = [c for c in df.columns if c.startswith(("ENER", "VOL"))]
        cols_interpolar    = [c for c in df.columns if c not in cols_no_interpolar]

        df[cols_interpolar] = df[cols_interpolar].interpolate(
            method="linear",
            limit=self.max_gap_periods,
            limit_direction="both",
            axis=0,
        )

        nulos_restantes = df.isna().sum().sum()
        print(f"  NaN restantes tras interpolación: {nulos_restantes:,}")
        return df

    # ── Paso 5: merge con meteo ───────────────────────────────────────────────

    def merge_meteo(self, df_clean: pd.DataFrame, df_meteo: pd.DataFrame) -> pd.DataFrame:
        """Left join del dataset limpio con la meteo a 15 min."""
        if df_meteo is None or df_meteo.empty:
            print("  [SKIP] No hay datos meteorológicos — se omite el merge.")
            return df_clean

        df_final  = df_clean.join(df_meteo, how="left")
        cobertura = df_final["meteo_temp_ext"].notna().mean() * 100 if "meteo_temp_ext" in df_final.columns else 0
        print(f"  Cobertura meteo: {cobertura:.1f} %")
        print(f"  Shape final con meteo: {df_final.shape}")
        return df_final

    # ── API principal ─────────────────────────────────────────────────────────

    def clean(
        self, df_raw: pd.DataFrame, df_meteo: pd.DataFrame = None
    ) -> tuple:
        """Ejecuta el pipeline completo. Devuelve (df_clean, df_final)."""
        print("=== Paso 1: Reindexado a 15 min ===")
        df = self.reindex_15min(df_raw)

        print("\n=== Paso 2: Eliminación de columnas con muchos nulos ===")
        df, nulos_pct = self.drop_high_null_cols(df)

        print("\n=== Paso 3: Filtrado por rangos físicos ===")
        df = self.apply_physical_ranges(df)

        print("\n=== Paso 4: Interpolación lineal ===")
        df = self.interpolate(df)

        self.df_clean = df
        print(f"\nShape df_clean: {df.shape}")

        print("\n=== Paso 5: Merge con datos meteorológicos ===")
        df_final = self.merge_meteo(df, df_meteo)
        self.df_final = df_final

        return self.df_clean, self.df_final
