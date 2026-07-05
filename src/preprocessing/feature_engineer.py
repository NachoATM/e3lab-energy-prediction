"""
feature_engineer.py — Ingeniería de variables para el modelo predictivo.

Construye sobre df_final (datos limpios + meteo):
  - Variable objetivo: delta de energía HVAC en intervalos de 15 min.
  - Features temporales cíclicas (sin/cos) + calendario lectivo UNAV.
  - Lag features y medias móviles de la energía.
  - Agregados del edificio (temperatura interior media, humedad, CO2, etc.).
  - Dataset de modelado final (df_model) con solo filas con target válido.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    FEATURES,
    TARGET,
    LAGS,
    ROLLING,
    PERIODOS_OPERACION_NORMAL,
    PERIODOS_VACACIONES,
    FESTIVOS_UNAV_PAMPLONA,
)


class FeatureEngineer:
    """Construye la variable objetivo y las features del modelo.

    Atributos públicos tras llamar a build():
        df_final — DataFrame enriquecido con todas las features y el target.
        df_model — Subconjunto de df_final con filas con target válido.
        features — Lista de features disponibles en df_model.
    """

    def __init__(self):
        self.df_final = None
        self.df_model = None
        self.features = None

    # ── Target: delta de energía HVAC ────────────────────────────────────────

    def build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula el consumo energético en cada intervalo de 15 min."""
        cols_ener = [c for c in df.columns if c.startswith("ENER_")]
        if not cols_ener:
            print("  [WARN] No hay columnas ENER_* — el target quedará vacío.")
            df[TARGET] = np.nan
            return df

        print(f"  Contadores de energía acumulada: {len(cols_ener)}")
        energia_acumulada = df[cols_ener].sort_index()
        delta_energia     = energia_acumulada.diff()

        # Descartar consumos artificiales (saltos temporales o reinicios)
        intervalo_ok = pd.Series(
            df.index.to_series().diff().eq(pd.Timedelta(minutes=15)).to_numpy(),
            index=df.index,
        )
        delta_energia = delta_energia.where(intervalo_ok, np.nan)
        delta_energia = delta_energia.mask(delta_energia < 0)

        df[TARGET] = delta_energia.sum(axis=1, min_count=1)

        n_reinicios = int((energia_acumulada.diff() < 0).sum().sum())
        pct_target  = df[TARGET].notna().mean() * 100
        print(f"  Cobertura del target: {pct_target:.1f} %")
        print(f"  Lecturas descartadas por reinicio: {n_reinicios:,}")
        return df

    # ── Features temporales ───────────────────────────────────────────────────

    @staticmethod
    def _mes_a_estacion_num(mes: int) -> int:
        if mes in [3, 4, 5]:   return 1  # primavera
        if mes in [6, 7, 8]:   return 2  # verano
        if mes in [9, 10, 11]: return 3  # otoño
        return 0                          # invierno

    def build_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade features temporales: hora, día semana, mes, día año, calendario, estación, sin/cos."""
        df["hora"]       = df.index.hour
        df["dia_semana"] = df.index.dayofweek
        df["mes"]        = df.index.month
        df["dia_anyo"]   = df.index.dayofyear

        # Calendario académico UNAV
        festivos = pd.to_datetime(FESTIVOS_UNAV_PAMPLONA).normalize()
        fecha    = pd.Series(df.index.normalize(), index=df.index)
        es_laborable = pd.Series(df.index.dayofweek < 5, index=df.index)

        en_periodo_operacion = pd.Series(False, index=df.index)
        for inicio, fin in PERIODOS_OPERACION_NORMAL:
            en_periodo_operacion |= fecha.between(
                pd.Timestamp(inicio), pd.Timestamp(fin)
            )

        en_vacaciones = pd.Series(False, index=df.index)
        for inicio, fin in PERIODOS_VACACIONES:
            en_vacaciones |= fecha.between(
                pd.Timestamp(inicio), pd.Timestamp(fin)
            )

        es_festivo = fecha.isin(festivos)
        df["calendario_lectivo"] = (
            es_laborable & en_periodo_operacion & ~en_vacaciones & ~es_festivo
        ).astype(int)

        df["estacion_num"] = df["mes"].map(self._mes_a_estacion_num)

        # Codificación cíclica
        df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
        df["dia_sin"]  = np.sin(2 * np.pi * df["dia_semana"] / 7)
        df["dia_cos"]  = np.cos(2 * np.pi * df["dia_semana"] / 7)
        df["mes_sin"]  = np.sin(2 * np.pi * df["mes"] / 12)
        df["mes_cos"]  = np.cos(2 * np.pi * df["mes"] / 12)

        print(
            f"  Días de operación normal: "
            f"{df['calendario_lectivo'].mean() * 100:.1f} % de los registros"
        )
        return df

    # ── Lag features y rolling means ─────────────────────────────────────────

    def build_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade lags y medias móviles de la variable objetivo."""
        energia = df[TARGET]

        for nombre, n in LAGS.items():
            df[nombre] = energia.shift(n)

        for nombre, ventana in ROLLING.items():
            df[nombre] = energia.shift(1).rolling(ventana, min_periods=1).mean()

        print(f"  Lags creados:          {list(LAGS.keys())}")
        print(f"  Rolling means creados: {list(ROLLING.keys())}")
        return df

    # ── Agregados del edificio ────────────────────────────────────────────────

    def build_building_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula variables agregadas del edificio (medias de sensores del mismo tipo)."""
        cols_ti  = [c for c in df.columns
                    if c.startswith("TEMP_") and "Exterior" not in c
                    and "Terreno" not in c and not c.startswith("meteo_")]
        cols_hum = [c for c in df.columns if c.startswith("HUM_")]
        cols_co2 = [c for c in df.columns if c.startswith("CO2_")]
        cols_timp = [c for c in df.columns if c.startswith("TIMP_")]

        if cols_ti:
            df["temp_int_media"] = df[cols_ti].mean(axis=1)
        if cols_hum:
            df["hum_int_media"]  = df[cols_hum].mean(axis=1)
        if cols_co2:
            df["co2_medio"]      = df[cols_co2].mean(axis=1)
        if cols_timp:
            df["timp_media"]     = df[cols_timp].mean(axis=1)

        if "meteo_temp_ext" in df.columns and cols_ti:
            df["delta_temp"] = df["temp_int_media"] - df["meteo_temp_ext"]

        feats_edif = ["temp_int_media", "hum_int_media", "co2_medio", "timp_media", "delta_temp"]
        for f in feats_edif:
            if f in df.columns:
                pct = df[f].notna().mean() * 100
                print(f"    {f:<20} {pct:.1f} % datos")
        return df

    # ── Dataset de modelado ───────────────────────────────────────────────────

    def build_model_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra a columnas FEATURES + TARGET y elimina filas sin target."""
        # Solo features que existan en el DataFrame
        feats_ok = [f for f in FEATURES if f in df.columns]
        self.features = feats_ok

        df_model = df[feats_ok + [TARGET]].dropna(subset=[TARGET])
        self.df_model = df_model

        print(f"\n  Features seleccionadas: {len(feats_ok)}")
        print(f"  Filas con target válido: {len(df_model):,}")
        nan_feats = df_model[feats_ok].isna().any(axis=1).sum()
        print(f"  Filas con algún NaN en features: {nan_feats:,}")
        return df_model

    # ── API principal ─────────────────────────────────────────────────────────

    def build(self, df_final: pd.DataFrame) -> tuple:
        """Ejecuta el pipeline completo. Devuelve (df_final_enriquecido, df_model)."""
        df = df_final.sort_index().copy()

        print("=== Target: delta energía HVAC ===")
        df = self.build_target(df)

        print("\n=== Features temporales ===")
        df = self.build_temporal_features(df)

        print("\n=== Lag features y rolling means ===")
        df = self.build_lag_features(df)

        print("\n=== Agregados del edificio ===")
        df = self.build_building_aggregates(df)

        self.df_final = df

        print("\n=== Dataset de modelado ===")
        self.build_model_dataset(df)

        return self.df_final, self.df_model
