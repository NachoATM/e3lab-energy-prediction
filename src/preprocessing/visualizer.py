"""
visualizer.py — Visualizaciones del pipeline de preprocesamiento.
Todas las figuras se guardan como PNG; nunca se llama a plt.show().
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla; debe ir antes de importar pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import (
    PLANTAS,
    DIAS,
    ESTACIONES,
    COLORES_ZONA,
    COLORES_EST,
    FIGURES_PREPROCESSING,
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


class PreprocessingVisualizer:
    """Genera y guarda las figuras del análisis exploratorio y de limpieza."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else FIGURES_PREPROCESSING
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, filename: str) -> None:
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Guardado: {path.name}")

    # ── Análisis de nulos ─────────────────────────────────────────────────────

    def plot_null_heatmap(self, df: pd.DataFrame) -> None:
        """Heatmap de valores nulos (muestreo temporal para eficiencia)."""
        sample_step = max(1, len(df) // 1500)
        df_sample   = df.iloc[::sample_step]
        null_matrix = df_sample.isna().T

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(
            null_matrix,
            cmap=["#F5F5F5", "#B22222"],
            cbar=True,
            xticklabels=False,
            yticklabels=10,
            linewidths=0,
            ax=ax,
        )
        ax.set_title("Mapa de Valores Nulos por Sensor", fontsize=18, pad=20, weight="bold")
        ax.set_xlabel("Tiempo", fontsize=13)
        ax.set_ylabel("Sensores", fontsize=13)
        ax.tick_params(axis="y", labelsize=9)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(["Valor válido", "Valor nulo"])
        fig.tight_layout()
        self._save(fig, "nulos_heatmap.png")

    def plot_null_by_sensor(self, df: pd.DataFrame) -> None:
        """Barplot del porcentaje de nulos por sensor."""
        null_percent = df.isna().mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(12, 14))
        sns.barplot(x=null_percent.values, y=null_percent.index,
                    palette="Reds_r", ax=ax)
        ax.set_title("Porcentaje de Valores Nulos por Sensor", fontsize=18, weight="bold")
        ax.set_xlabel("Porcentaje de nulos (%)")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "nulos_por_sensor.png")

    def plot_null_by_type(self, df: pd.DataFrame) -> None:
        """Barplot del porcentaje medio de nulos por tipo de variable."""
        nulos_pct = df.isna().mean() * 100
        df_nulos = nulos_pct.reset_index()
        df_nulos.columns = ["sensor", "pct_nulos"]
        df_nulos["tipo"] = df_nulos["sensor"].str.split("_").str[0]

        fig, ax = plt.subplots(figsize=(10, 4))
        df_nulos.groupby("tipo")["pct_nulos"].mean().sort_values().plot.barh(
            ax=ax, color="steelblue"
        )
        ax.set_xlabel("% medio de valores nulos")
        ax.set_title("Porcentaje medio de nulos por tipo de variable")
        ax.axvline(50, color="red", linestyle="--", alpha=0.7, label="Umbral 50 %")
        ax.legend()
        fig.tight_layout()
        self._save(fig, "nulos_por_tipo.png")

    # ── Series temporales por tipo de variable ────────────────────────────────

    def plot_temp_interior_plants(self, df: pd.DataFrame) -> None:
        """Temperatura interior media diaria agrupada por planta."""
        cols = [c for c in df.columns if c.startswith("TEMP_")
                and "Exterior" not in c and "Terreno" not in c]
        plantas = sorted(set(c.split("_")[2] for c in cols))

        fig, ax = plt.subplots(figsize=(15, 5))
        for planta in plantas:
            cols_pl = [c for c in cols if c.split("_")[2] == planta]
            if cols_pl:
                serie = df[cols_pl].mean(axis=1).resample("D").mean()
                ax.plot(serie.index, serie.values, label=planta, linewidth=0.9)

        ax.set_title("Temperatura interior media diaria por planta", fontsize=13)
        ax.set_ylabel("Temperatura (°C)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.legend(title="Planta", bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        self._save(fig, "temp_interior_plantas.png")

    def plot_co2_by_uso(self, df: pd.DataFrame) -> None:
        """CO2 medio diario por tipo de uso."""
        cols_co2  = [c for c in df.columns if c.startswith("CO2_")]
        usos_co2  = sorted(set(c.split("_")[3] for c in cols_co2))

        fig, ax = plt.subplots(figsize=(15, 5))
        for uso in usos_co2:
            cols_uso = [c for c in cols_co2 if c.split("_")[3] == uso]
            if cols_uso:
                serie = df[cols_uso].mean(axis=1).resample("D").mean()
                ax.plot(serie.index, serie.values, label=uso, linewidth=0.9)

        ax.axhline(1000, color="red", linestyle="--", alpha=0.5, label="Límite 1000 ppm")
        ax.set_title("CO2 medio diario por tipo de uso", fontsize=13)
        ax.set_ylabel("CO2 (ppm)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.legend(title="Uso", bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        self._save(fig, "co2_por_uso.png")

    def plot_hvac_power(self, df: pd.DataFrame) -> None:
        """Potencia HVAC total media diaria."""
        cols_pot = [c for c in df.columns if c.startswith("POT_")]
        if not cols_pot:
            print("  [SKIP] Sin columnas POT_* para graficar.")
            return

        pot_diaria = df[cols_pot].sum(axis=1).resample("D").mean()
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.fill_between(pot_diaria.index, pot_diaria.values, alpha=0.6, color="darkorange")
        ax.plot(pot_diaria.index, pot_diaria.values, color="darkorange", linewidth=0.8)
        ax.set_title("Potencia HVAC total media diaria", fontsize=13)
        ax.set_ylabel("Potencia (kW)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        fig.tight_layout()
        self._save(fig, "potencia_hvac_diaria.png")

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Heatmap de correlación entre variables representativas."""
        def primera_col_tipo(prefijo):
            cols = [c for c in df.columns if c.startswith(prefijo)]
            return cols[0] if cols else None

        cols_corr = [
            primera_col_tipo("TEMP_"),
            primera_col_tipo("HUM_"),
            primera_col_tipo("CO2_"),
            primera_col_tipo("TIMP_"),
            primera_col_tipo("CAUD_"),
            primera_col_tipo("POT_"),
            primera_col_tipo("TRAD_"),
        ]
        cols_corr = [c for c in cols_corr if c is not None]
        if len(cols_corr) < 2:
            print("  [SKIP] Insuficientes variables para la correlación.")
            return

        corr_df = df[cols_corr].resample("h").mean().corr()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
        ax.set_title("Correlación entre variables representativas (horaria)", fontsize=12)
        labels = [c.split("_")[0] + "\n" + c.split("_")[1] for c in cols_corr]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels, rotation=0)
        fig.tight_layout()
        self._save(fig, "correlacion_variables.png")

    # ── Análisis por planta ───────────────────────────────────────────────────

    @staticmethod
    def _zona_de(col: str) -> str:
        parts = col.split("_")
        return parts[4] if len(parts) > 4 else "N/A"

    def plot_temp_by_plant(self, df: pd.DataFrame) -> None:
        """Serie diaria, boxplot mensual, comparativa por zona y perfil horario para cada planta."""
        for planta in PLANTAS:
            cols = [c for c in df.columns
                    if c.startswith("TEMP_") and c.split("_")[2] == planta
                    and "Exterior" not in c and "Terreno" not in c]
            if not cols:
                continue

            # Serie diaria
            fig, ax = plt.subplots(figsize=(15, 4))
            for col in cols:
                zona  = self._zona_de(col)
                color = COLORES_ZONA.get(zona, None)
                ax.plot(df[col].resample("D").mean(), linewidth=0.8, alpha=0.75,
                        color=color, label=f"{col.split('_')[1]} ({zona})")
            ax.set_title(f"Temperatura interior — {planta} — serie diaria por sensor", fontsize=12)
            ax.set_ylabel("Temperatura (°C)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend(fontsize=7, ncol=3, bbox_to_anchor=(1.01, 1))
            fig.tight_layout()
            self._save(fig, f"temp_{planta}_serie_diaria.png")

            # Boxplot mensual
            serie = df[cols].mean(axis=1).resample("h").mean()
            df_bp = pd.DataFrame({"temp": serie.values, "mes": serie.index.strftime("%Y-%m")})
            meses = sorted(df_bp["mes"].unique())
            fig, ax = plt.subplots(figsize=(14, 4))
            sns.boxplot(data=df_bp, x="mes", y="temp", order=meses,
                        color="steelblue",
                        flierprops=dict(marker=".", alpha=0.3, markersize=2), ax=ax)
            ax.set_title(f"Temperatura interior — {planta} — distribución mensual", fontsize=12)
            ax.set_xlabel("")
            ax.set_ylabel("Temperatura (°C)")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            self._save(fig, f"temp_{planta}_boxplot_mensual.png")

            # Comparativa por zona
            zonas = sorted(set(self._zona_de(c) for c in cols))
            fig, ax = plt.subplots(figsize=(15, 4))
            for zona in zonas:
                cols_z = [c for c in cols if self._zona_de(c) == zona]
                serie  = df[cols_z].mean(axis=1).resample("D").mean()
                color  = COLORES_ZONA.get(zona, "gray")
                ax.plot(serie.index, serie.values, label=f"Zona {zona}",
                        color=color, linewidth=1.4)
                if len(cols_z) > 1:
                    s_min = df[cols_z].min(axis=1).resample("D").mean()
                    s_max = df[cols_z].max(axis=1).resample("D").mean()
                    ax.fill_between(serie.index, s_min, s_max, alpha=0.12, color=color)
            ax.set_title(f"Temperatura interior — {planta} — comparativa por zona", fontsize=12)
            ax.set_ylabel("Temperatura (°C)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend(title="Zona")
            fig.tight_layout()
            self._save(fig, f"temp_{planta}_zonas.png")

            # Perfil horario
            media = df[cols].mean(axis=1)
            df_hm = pd.DataFrame({
                "temp":       media.values,
                "hora":       media.index.hour,
                "dia_semana": media.index.dayofweek,
            })
            pivot = df_hm.groupby(["dia_semana", "hora"])["temp"].mean().unstack()
            fig, ax = plt.subplots(figsize=(14, 4))
            sns.heatmap(pivot, cmap="RdYlBu_r", ax=ax,
                        xticklabels=range(0, 24), yticklabels=DIAS,
                        cbar_kws={"label": "°C"}, linewidths=0.15)
            ax.set_title(f"Temperatura interior — {planta} — perfil horario (°C)", fontsize=12)
            ax.set_xlabel("Hora del día")
            ax.set_ylabel("")
            fig.tight_layout()
            self._save(fig, f"temp_{planta}_perfil_horario.png")

    def plot_hum_by_plant(self, df: pd.DataFrame) -> None:
        """Serie y comparativa por zona de humedad para cada planta."""
        for planta in PLANTAS:
            cols = [c for c in df.columns
                    if c.startswith("HUM_") and c.split("_")[2] == planta]
            if not cols:
                continue

            zonas = sorted(set(self._zona_de(c) for c in cols))
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))

            for col in cols:
                zona  = self._zona_de(col)
                color = COLORES_ZONA.get(zona, None)
                axes[0].plot(df[col].resample("D").mean(), linewidth=0.8, alpha=0.75,
                             color=color, label=f"{col.split('_')[1]} ({zona})")
            axes[0].axhspan(30, 60, alpha=0.08, color="green", label="Rango confort")
            axes[0].set_title(f"Humedad — {planta} — serie diaria")
            axes[0].set_ylabel("Humedad relativa (%)")
            axes[0].set_ylim(0, 100)
            axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            axes[0].xaxis.set_major_locator(mdates.MonthLocator())
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].legend(fontsize=7, ncol=2)

            for zona in zonas:
                cols_z = [c for c in cols if self._zona_de(c) == zona]
                serie  = df[cols_z].mean(axis=1).resample("D").mean()
                color  = COLORES_ZONA.get(zona, "gray")
                axes[1].plot(serie.index, serie.values, label=f"Zona {zona}",
                             color=color, linewidth=1.4)
                if len(cols_z) > 1:
                    s_min = df[cols_z].min(axis=1).resample("D").mean()
                    s_max = df[cols_z].max(axis=1).resample("D").mean()
                    axes[1].fill_between(serie.index, s_min, s_max, alpha=0.12, color=color)
            axes[1].axhspan(30, 60, alpha=0.08, color="green", label="Rango confort")
            axes[1].set_title(f"Humedad — {planta} — por zona")
            axes[1].set_ylim(0, 100)
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            axes[1].xaxis.set_major_locator(mdates.MonthLocator())
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].legend(title="Zona")

            fig.suptitle(f"Humedad relativa — {planta}", fontsize=13, y=1.01)
            fig.tight_layout()
            self._save(fig, f"hum_{planta}_serie_zonas.png")

    def plot_co2_by_plant(self, df: pd.DataFrame) -> None:
        """Serie y comparativa por zona de CO2 para cada planta."""
        for planta in PLANTAS:
            cols = [c for c in df.columns
                    if c.startswith("CO2_") and c.split("_")[2] == planta]
            if not cols:
                continue

            zonas = sorted(set(self._zona_de(c) for c in cols))
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))

            for col in cols:
                zona  = self._zona_de(col)
                color = COLORES_ZONA.get(zona, None)
                axes[0].plot(df[col].resample("D").mean(), linewidth=0.8, alpha=0.75,
                             color=color, label=f"{col.split('_')[1]} ({zona})")
            axes[0].axhline(1000, color="red", linestyle="--", alpha=0.6, linewidth=1, label="1000 ppm")
            axes[0].set_title(f"CO₂ — {planta} — serie diaria")
            axes[0].set_ylabel("CO₂ (ppm)")
            axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            axes[0].xaxis.set_major_locator(mdates.MonthLocator())
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].legend(fontsize=7, ncol=2)

            for zona in zonas:
                cols_z = [c for c in cols if self._zona_de(c) == zona]
                serie  = df[cols_z].mean(axis=1).resample("D").mean()
                color  = COLORES_ZONA.get(zona, "gray")
                axes[1].plot(serie.index, serie.values, label=f"Zona {zona}",
                             color=color, linewidth=1.4)
                if len(cols_z) > 1:
                    s_min = df[cols_z].min(axis=1).resample("D").mean()
                    s_max = df[cols_z].max(axis=1).resample("D").mean()
                    axes[1].fill_between(serie.index, s_min, s_max, alpha=0.12, color=color)
            axes[1].axhline(1000, color="red", linestyle="--", alpha=0.6, linewidth=1, label="1000 ppm")
            axes[1].set_title(f"CO₂ — {planta} — por zona")
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            axes[1].xaxis.set_major_locator(mdates.MonthLocator())
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].legend(title="Zona")

            fig.suptitle(f"CO₂ — {planta}", fontsize=13, y=1.01)
            fig.tight_layout()
            self._save(fig, f"co2_{planta}_serie_zonas.png")

    def plot_timp_by_plant(self, df: pd.DataFrame) -> None:
        """Temperatura de impulsión (y retorno) por zona y planta."""
        for planta in PLANTAS:
            cols_imp = [c for c in df.columns
                        if c.startswith("TIMP_") and c.split("_")[2] == planta]
            cols_ret = [c for c in df.columns
                        if c.startswith("TRET_") and c.split("_")[2] == planta]
            if not cols_imp:
                continue

            zonas_imp = sorted(set(self._zona_de(c) for c in cols_imp))
            fig, ax = plt.subplots(figsize=(15, 4))

            for zona in zonas_imp:
                cols_z = [c for c in cols_imp if self._zona_de(c) == zona]
                serie  = df[cols_z].mean(axis=1).resample("D").mean()
                color  = COLORES_ZONA.get(zona, "gray")
                ax.plot(serie.index, serie.values, label=f"Imp. Zona {zona}",
                        color=color, linewidth=1.3, linestyle="-")

            if cols_ret:
                zonas_ret = sorted(set(self._zona_de(c) for c in cols_ret))
                for zona in zonas_ret:
                    cols_z = [c for c in cols_ret if self._zona_de(c) == zona]
                    serie  = df[cols_z].mean(axis=1).resample("D").mean()
                    color  = COLORES_ZONA.get(zona, "gray")
                    ax.plot(serie.index, serie.values, label=f"Ret. Zona {zona}",
                            color=color, linewidth=1.0, linestyle="--", alpha=0.7)

            ax.set_title(f"T. Impulsión (y Retorno) — {planta} — por zona", fontsize=12)
            ax.set_ylabel("Temperatura (°C)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend(fontsize=8, ncol=3, bbox_to_anchor=(1.01, 1))
            fig.tight_layout()
            self._save(fig, f"timp_{planta}_zonas.png")

    # ── Análisis por estación ─────────────────────────────────────────────────

    @staticmethod
    def _mes_a_estacion(mes: int) -> str:
        if mes in [3, 4, 5]:   return "primavera"
        if mes in [6, 7, 8]:   return "verano"
        if mes in [9, 10, 11]: return "otoño"
        return "invierno"

    def _build_dfs_est(self, df: pd.DataFrame):
        df2 = df.copy()
        df2["estacion"] = df2.index.month.map(self._mes_a_estacion)
        dfs_est = {
            est: df2[df2["estacion"] == est].drop(columns="estacion")
            for est in ESTACIONES
        }
        return dfs_est

    def plot_seasonal_temp(self, df: pd.DataFrame) -> None:
        """Boxplot + violinplot de temperatura interior por estación."""
        dfs_est = self._build_dfs_est(df)
        cols_ti = [c for c in df.columns if c.startswith("TEMP_")
                   and "Exterior" not in c and "Terreno" not in c]
        if not cols_ti:
            print("  [SKIP] Sin columnas TEMP_* para análisis estacional.")
            return

        df_bp_est = pd.concat([
            pd.DataFrame({
                "temp":      dfs_est[est][cols_ti].mean(axis=1).resample("h").mean().values,
                "estacion":  est,
                "timestamp": dfs_est[est][cols_ti].mean(axis=1).resample("h").mean().index,
            })
            for est in ESTACIONES
            if est in dfs_est and len(dfs_est[est]) > 0
        ], ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        sns.boxplot(data=df_bp_est, x="estacion", y="temp", order=ESTACIONES,
                    palette=COLORES_EST, ax=axes[0],
                    flierprops=dict(marker=".", alpha=0.2, markersize=2))
        axes[0].set_title("Distribución de temperatura interior por estación", fontsize=12)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Temperatura (°C)")

        sns.violinplot(data=df_bp_est, x="estacion", y="temp", order=ESTACIONES,
                       palette=COLORES_EST, ax=axes[1], inner="quartile", cut=0)
        axes[1].set_title("Densidad de temperatura interior por estación", fontsize=12)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Temperatura (°C)")

        fig.suptitle("Temperatura interior — comparativa por estación", fontsize=13)
        fig.tight_layout()
        self._save(fig, "est_temp_interior_comparativa.png")

    def plot_seasonal_power_co2(self, df: pd.DataFrame) -> None:
        """Boxplot de potencia HVAC y CO2 por estación."""
        dfs_est  = self._build_dfs_est(df)
        cols_pot = [c for c in df.columns if c.startswith("POT_")]
        cols_co2 = [c for c in df.columns if c.startswith("CO2_")]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        if cols_pot:
            pot_data = []
            for est in ESTACIONES:
                if est not in dfs_est:
                    continue
                df_e    = dfs_est[est]
                cols_ok = [c for c in cols_pot if c in df_e.columns]
                if cols_ok:
                    vals = df_e[cols_ok].sum(axis=1).resample("h").mean().dropna().values
                    pot_data.append(pd.DataFrame({"valor": vals, "estacion": est}))
            if pot_data:
                df_pot_bp = pd.concat(pot_data, ignore_index=True)
                sns.boxplot(data=df_pot_bp, x="estacion", y="valor", order=ESTACIONES,
                            palette=COLORES_EST, ax=axes[0],
                            flierprops=dict(marker=".", alpha=0.2, markersize=2))
                axes[0].set_title("Potencia HVAC total por estación", fontsize=12)
                axes[0].set_ylabel("Potencia (kW)")
                axes[0].set_xlabel("")
        else:
            axes[0].text(0.5, 0.5, "Sin datos de Potencia", ha="center",
                         va="center", transform=axes[0].transAxes)

        if cols_co2:
            co2_data = []
            for est in ESTACIONES:
                if est not in dfs_est:
                    continue
                df_e = dfs_est[est]
                vals = df_e[cols_co2].mean(axis=1).resample("h").mean().dropna().values
                co2_data.append(pd.DataFrame({"valor": vals, "estacion": est}))
            if co2_data:
                df_co2_bp = pd.concat(co2_data, ignore_index=True)
                sns.boxplot(data=df_co2_bp, x="estacion", y="valor", order=ESTACIONES,
                            palette=COLORES_EST, ax=axes[1],
                            flierprops=dict(marker=".", alpha=0.2, markersize=2))
                axes[1].axhline(1000, color="red", linestyle="--", alpha=0.6, label="1000 ppm")
                axes[1].set_title("CO2 interior por estación", fontsize=12)
                axes[1].set_ylabel("CO2 (ppm)")
                axes[1].set_xlabel("")
                axes[1].legend()

        fig.suptitle("Demanda energética y calidad del aire por estación", fontsize=13)
        fig.tight_layout()
        self._save(fig, "est_potencia_co2.png")

    def plot_seasonal_hourly_heatmap(self, df: pd.DataFrame) -> None:
        """Heatmap 2D del perfil horario por estación (temperatura interior)."""
        dfs_est = self._build_dfs_est(df)
        cols_ti = [c for c in df.columns if c.startswith("TEMP_")
                   and "Exterior" not in c and "Terreno" not in c]
        if not cols_ti:
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        axes = axes.flatten()
        pivots = {}
        for est in ESTACIONES:
            if est not in dfs_est or len(dfs_est[est]) == 0:
                continue
            df_e  = dfs_est[est]
            media = df_e[cols_ti].mean(axis=1)
            df_hm = pd.DataFrame({"temp": media.values, "hora": media.index.hour,
                                   "dia": media.index.dayofweek})
            pivots[est] = df_hm.groupby(["dia", "hora"])["temp"].mean().unstack()

        if not pivots:
            return

        vmin = min(p.min().min() for p in pivots.values())
        vmax = max(p.max().max() for p in pivots.values())

        for i, est in enumerate(ESTACIONES):
            if est not in pivots:
                axes[i].set_visible(False)
                continue
            sns.heatmap(pivots[est], cmap="RdYlBu_r", ax=axes[i],
                        vmin=vmin, vmax=vmax,
                        xticklabels=range(24), yticklabels=DIAS,
                        cbar_kws={"label": "°C"}, linewidths=0.1)
            axes[i].set_title(f"{COLORES_EST.get(est, '')} {est.capitalize()}", fontsize=12)
            axes[i].set_xlabel("Hora del día")
            axes[i].set_ylabel("")

        fig.suptitle("Perfil horario de temperatura interior por estación (°C)", fontsize=14)
        fig.tight_layout()
        self._save(fig, "est_perfil_horario_temp.png")

    def plot_feature_correlation(self, df_model: pd.DataFrame, features: list, target: str) -> None:
        """Barplot de correlación absoluta de cada feature con el target."""
        corr_target = df_model[features + [target]].corr()[target].drop(target)
        corr_target = corr_target.abs().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(5, len(corr_target) * 0.3)))
        colors = [
            "#e63946" if v > 0.5 else "#457b9d" if v > 0.2 else "#adb5bd"
            for v in corr_target.values
        ]
        corr_target.plot.barh(ax=ax, color=colors, edgecolor="white")
        ax.axvline(0.2, color="#457b9d", linestyle="--", alpha=0.5, label="|r| = 0.2")
        ax.axvline(0.5, color="#e63946", linestyle="--", alpha=0.5, label="|r| = 0.5")
        ax.set_title("Correlación absoluta de cada feature con la Energía HVAC", fontsize=12)
        ax.set_xlabel("|Correlación de Pearson|")
        ax.legend()
        fig.tight_layout()
        self._save(fig, "feat_correlacion_target.png")

    # ── API de conveniencia: genera todas las figuras ─────────────────────────

    def plot_all(self, df_clean: pd.DataFrame, df_model: pd.DataFrame = None,
                 features: list = None, target: str = None) -> None:
        """Genera todas las figuras del preprocesamiento."""
        print("\n--- Generando figuras de nulos ---")
        self.plot_null_heatmap(df_clean)
        self.plot_null_by_sensor(df_clean)
        self.plot_null_by_type(df_clean)

        print("\n--- Generando figuras de series temporales ---")
        self.plot_temp_interior_plants(df_clean)
        self.plot_co2_by_uso(df_clean)
        self.plot_hvac_power(df_clean)
        self.plot_correlation_heatmap(df_clean)

        print("\n--- Generando figuras por planta ---")
        self.plot_temp_by_plant(df_clean)
        self.plot_hum_by_plant(df_clean)
        self.plot_co2_by_plant(df_clean)
        self.plot_timp_by_plant(df_clean)

        print("\n--- Generando figuras estacionales ---")
        self.plot_seasonal_temp(df_clean)
        self.plot_seasonal_power_co2(df_clean)
        self.plot_seasonal_hourly_heatmap(df_clean)

        if df_model is not None and features is not None and target is not None:
            print("\n--- Generando figura de correlación de features ---")
            self.plot_feature_correlation(df_model, features, target)

        print(f"\nTodas las figuras guardadas en: {self.output_dir}")
