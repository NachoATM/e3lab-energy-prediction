"""
evaluator.py — Recolecta resultados de todos los modelos, genera comparativas y rankings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.config import (
    SEED,
    N_REPEATS_PERM,
    N_SHAP_SAMPLES,
    UMBRAL_FI,
    UMBRAL_PERM,
    FIGURES_MODELS,
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130


class ModelEvaluator:
    """Agrega resultados de múltiples modelos y genera figuras comparativas.

    Uso:
        ev = ModelEvaluator(output_dir)
        ev.add_result(metrics_dict)
        ev.save_comparison_plots()
        ev.save_ranking_heatmap()
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else FIGURES_MODELS
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: list = []

    def add_result(self, result: dict) -> None:
        """Añade un dict de métricas (debe contener 'modelo', 'MAE', 'RMSE', 'R2', etc.)."""
        self._results.append(result)

    def add_results(self, results: list) -> None:
        for r in results:
            self.add_result(r)

    @property
    def df_results(self) -> pd.DataFrame:
        """DataFrame con todos los resultados, sin duplicados, indexado por 'modelo'."""
        if not self._results:
            return pd.DataFrame()
        return (
            pd.DataFrame(self._results)
            .drop_duplicates(subset="modelo", keep="last")
            .set_index("modelo")
        )

    # ── Figuras de comparativa global ─────────────────────────────────────────

    def save_comparison_plots(self, filename: str = "04_comparativa_global.png") -> None:
        """Barplot de MAE, RMSE y R2 para todos los modelos registrados."""
        df = self.df_results
        if df.empty:
            print("  [SKIP] No hay resultados para comparar.")
            return

        metricas = ["MAE", "RMSE", "R2"]
        colores  = ["#457b9d", "#e63946", "#2a9d8f"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (metrica, color) in enumerate(zip(metricas, colores)):
            vals = df[metrica]
            bars = axes[i].bar(
                vals.index, vals.values,
                color=color, edgecolor="white", alpha=0.85,
            )
            axes[i].set_title(metrica, fontsize=12)
            axes[i].tick_params(axis="x", rotation=20)
            for label in axes[i].get_xticklabels():
                label.set_ha("right")
            offset = max(abs(vals).max() * 0.01, 1e-6)
            for bar, val in zip(bars, vals.values):
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{val:.4f}", ha="center", fontsize=9,
                )

        if df["R2"].min() >= 0:
            axes[2].set_ylim(0, min(1.1, max(1.0, df["R2"].max() * 1.1)))

        fig.suptitle("Comparativa de métricas — modelos globales", fontsize=13)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {filename}")

    # ── Heatmap de ranking ────────────────────────────────────────────────────

    @staticmethod
    def _normalizar(df_sub: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Normaliza métricas 0-1 donde 0=peor, 1=mejor para colormap RdYlGn_r."""
        dn = df_sub[cols].copy()
        for col in ["MAE", "RMSE", "WAPE", "sMAPE"]:
            if col not in dn.columns:
                continue
            rng = dn[col].max() - dn[col].min() + 1e-9
            dn[col] = (dn[col] - dn[col].min()) / rng
        if "R2" in dn.columns:
            rng_r2 = dn["R2"].max() - dn["R2"].min() + 1e-9
            dn["R2"] = 1 - ((dn["R2"] - dn["R2"].min()) / rng_r2)
        return dn

    def save_ranking_heatmap(
        self,
        filename: str = "06_ranking_final.png",
        title: str = "Ranking de modelos — métricas normalizadas\n(verde = mejor)",
    ) -> None:
        """Heatmap con valores reales en las celdas y coloración por percentil."""
        df = self.df_results.sort_values("RMSE")
        if df.empty:
            return

        cols_m = [c for c in ["MAE", "RMSE", "R2", "WAPE", "sMAPE"] if c in df.columns]
        dn     = self._normalizar(df, cols_m)

        fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.5)))
        sns.heatmap(
            dn[cols_m],
            annot=df[cols_m].round(4),
            fmt="",
            cmap="RdYlGn_r",
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Peor -> Mejor (normalizado)"},
        )
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {filename}")

    # ── Comparativa estacional ────────────────────────────────────────────────

    def save_seasonal_comparison(
        self,
        df_results_global: pd.DataFrame,
        df_results_estacional: pd.DataFrame,
        estaciones: list,
        nombres_est: dict,
        filename: str = "05b_comparativa_modelos_estacionales.png",
    ) -> None:
        """Barplot de métricas por estación para cada tipo de modelo."""
        if df_results_estacional.empty:
            print("  [SKIP] Sin resultados estacionales.")
            return

        df_plot = df_results_estacional.reset_index().copy()
        split   = df_plot["modelo"].str.rsplit(" - ", n=1, expand=True)
        df_plot["tipo"]     = split[0]
        df_plot["Estacion"] = split[1].map(nombres_est)

        tipos_ordenados = ["Ridge", "Random Forest", "XGBoost", "LSTM", "Prophet"]
        tipos_presentes = [t for t in tipos_ordenados if t in df_plot["tipo"].values]
        est_orden       = [nombres_est[e] for e in estaciones
                           if nombres_est.get(e) in df_plot["Estacion"].values]

        palette    = plt.cm.tab10.colors
        color_tipo = {t: palette[j] for j, t in enumerate(tipos_presentes)}
        metricas   = ["MAE", "RMSE", "R2"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        global_keys = {
            "Ridge":         "Ridge (global)",
            "Random Forest": "Random Forest (global)",
            "XGBoost":       "XGBoost (global)",
            "LSTM":          "LSTM (global)",
            "Prophet":       "Prophet (global)",
        }

        for i, met in enumerate(metricas):
            ax  = axes[i]
            n_t = len(tipos_presentes)
            n_e = len(est_orden)
            x   = np.arange(n_e)
            w   = 0.8 / max(n_t, 1)

            for j, tipo in enumerate(tipos_presentes):
                vals = []
                for est in est_orden:
                    sub = df_plot[(df_plot["tipo"] == tipo) & (df_plot["Estacion"] == est)]
                    vals.append(float(sub[met].iloc[0]) if len(sub) > 0 else np.nan)
                offset = (j - n_t / 2 + 0.5) * w
                ax.bar(x + offset, vals, w * 0.92,
                       label=tipo, color=color_tipo[tipo], alpha=0.85, edgecolor="white")

            # Referencia: rendimiento global
            for tipo, gkey in global_keys.items():
                if tipo in tipos_presentes and gkey in df_results_global.index:
                    ax.axhline(
                        df_results_global.loc[gkey, met],
                        linestyle="--", linewidth=1.4,
                        color=color_tipo[tipo], alpha=0.55,
                    )

            ax.set_title(met, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(est_orden, rotation=15)
            if i == 0:
                ax.legend(fontsize=8, loc="upper right")

        fig.suptitle(
            "Comparativa por estación — todos los modelos\n"
            "(líneas discontinuas = rendimiento del modelo global de referencia)",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {filename}")

    # ── Feature importance comparativa (RF vs XGB) ────────────────────────────

    def save_feature_importance_comparison(
        self,
        df_imp_joint: pd.DataFrame,
        filename: str = "11_feature_importance_comparativa.png",
    ) -> None:
        """Barplot comparativo de importancias normalizadas RF vs XGB."""
        top_n = min(25, len(df_imp_joint))
        df_p  = df_imp_joint.head(top_n).sort_values("mean_norm")

        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_n * 0.35)), sharey=True)
        axes[0].barh(df_p["feature"], df_p["RF_norm"],
                     color="steelblue", edgecolor="white", alpha=0.9)
        axes[0].set_title("Random Forest — importancia norm.", fontsize=11)
        axes[0].set_xlabel("Importancia normalizada")

        axes[1].barh(df_p["feature"], df_p["XGB_norm"],
                     color="darkorange", edgecolor="white", alpha=0.9)
        axes[1].set_title("XGBoost — importancia norm.", fontsize=11)
        axes[1].set_xlabel("Importancia normalizada")

        fig.suptitle(f"Comparativa Feature Importance — Top {top_n} (media RF+XGB)", fontsize=12)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {filename}")

    # ── Permutation importance ────────────────────────────────────────────────

    def compute_permutation_importance(
        self,
        rf_model,
        xgb_model,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        features: list,
    ) -> tuple:
        """Calcula permutation importance para RF y XGB sobre el conjunto de validación."""
        print(f"  Calculando Permutation Importance (n_repeats={N_REPEATS_PERM})...")

        perm_rf = permutation_importance(
            rf_model.model, X_val, y_val,
            n_repeats=N_REPEATS_PERM,
            scoring="neg_root_mean_squared_error",
            random_state=SEED, n_jobs=-1,
        )
        df_perm_rf = pd.DataFrame({
            "feature":   features,
            "perm_mean": perm_rf.importances_mean,
            "perm_std":  perm_rf.importances_std,
        }).sort_values("perm_mean", ascending=False).reset_index(drop=True)

        perm_xgb = permutation_importance(
            xgb_model.model, X_val.values, y_val,
            n_repeats=N_REPEATS_PERM,
            scoring="neg_root_mean_squared_error",
            random_state=SEED, n_jobs=-1,
        )
        df_perm_xgb = pd.DataFrame({
            "feature":   features,
            "perm_mean": perm_xgb.importances_mean,
            "perm_std":  perm_xgb.importances_std,
        }).sort_values("perm_mean", ascending=False).reset_index(drop=True)

        return df_perm_rf, df_perm_xgb

    def save_permutation_importance_plot(
        self,
        df_perm_rf: pd.DataFrame,
        df_perm_xgb: pd.DataFrame,
        filename: str = "12_permutation_importance.png",
    ) -> None:
        top_n = min(20, len(df_perm_rf))
        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_n * 0.38)))
        for ax, df_p, titulo, color in zip(
            axes,
            [df_perm_rf.head(top_n).sort_values("perm_mean"),
             df_perm_xgb.head(top_n).sort_values("perm_mean")],
            ["RF — Permutation Importance (val)", "XGBoost — Permutation Importance (val)"],
            ["steelblue", "darkorange"],
        ):
            ax.barh(df_p["feature"], df_p["perm_mean"],
                    xerr=df_p["perm_std"], color=color,
                    edgecolor="white", alpha=0.9, capsize=3)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(titulo, fontsize=11)
            ax.set_xlabel("Caída media RMSE al permutar (mayor = más relevante)")

        fig.suptitle(f"Permutation Importance — Top {top_n} features", fontsize=12)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {filename}")

    # ── SHAP ─────────────────────────────────────────────────────────────────

    def compute_and_save_shap(
        self,
        xgb_model,
        X_train: pd.DataFrame,
        features: list,
    ) -> pd.DataFrame:
        """Calcula y guarda SHAP summary plots para XGBoost."""
        try:
            import shap
        except ImportError:
            print("  [SKIP] SHAP no disponible — instalar con: pip install shap")
            return pd.DataFrame()

        n_shap = min(N_SHAP_SAMPLES, len(X_train))
        idx_shap = np.random.RandomState(SEED).choice(len(X_train), n_shap, replace=False)
        X_shap   = X_train.iloc[idx_shap]

        explainer  = shap.TreeExplainer(xgb_model.model)
        shap_values = explainer.shap_values(X_shap)

        # Summary plot
        fig = plt.figure(figsize=(10, max(6, min(len(features), 20) * 0.35)))
        shap.summary_plot(shap_values, X_shap, max_display=20, show=False)
        plt.title("SHAP Summary Plot — XGBoost (muestra train)", fontsize=12)
        plt.tight_layout()
        fig.savefig(self.output_dir / "13_shap_summary_xgb.png", bbox_inches="tight")
        plt.close(fig)

        # Bar plot
        fig = plt.figure(figsize=(8, max(6, min(len(features), 20) * 0.35)))
        shap.summary_plot(shap_values, X_shap, max_display=20, plot_type="bar", show=False)
        plt.title("SHAP — Importancia global media |SHAP| (XGBoost)", fontsize=12)
        plt.tight_layout()
        fig.savefig(self.output_dir / "13b_shap_bar_xgb.png", bbox_inches="tight")
        plt.close(fig)

        df_shap_imp = pd.DataFrame({
            "feature":        features,
            "mean_abs_shap":  np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        print("  SHAP guardado.")
        return df_shap_imp

    # ── Selección consensuada de features ─────────────────────────────────────

    @staticmethod
    def select_features(
        features: list,
        df_imp_joint: pd.DataFrame,
        df_perm_rf: pd.DataFrame,
        df_perm_xgb: pd.DataFrame,
        df_shap_imp: pd.DataFrame = None,
        umbral_fi: float = UMBRAL_FI,
        umbral_perm: float = UMBRAL_PERM,
    ) -> list:
        """Selección OR: feature se incluye si supera AL MENOS una señal."""
        fi_signal = set(df_imp_joint.loc[df_imp_joint["mean_norm"] >= umbral_fi, "feature"])
        perm_signal = (
            set(df_perm_rf.loc[df_perm_rf["perm_mean"] > umbral_perm, "feature"]) |
            set(df_perm_xgb.loc[df_perm_xgb["perm_mean"] > umbral_perm, "feature"])
        )

        if df_shap_imp is not None and len(df_shap_imp) > 0:
            shap_median = df_shap_imp["mean_abs_shap"].median()
            shap_signal = set(df_shap_imp.loc[df_shap_imp["mean_abs_shap"] >= shap_median, "feature"])
        else:
            shap_signal = set()

        candidates   = fi_signal | perm_signal | shap_signal
        features_opt = [f for f in features if f in candidates]

        # Garantizar mínimo razonable
        if len(features_opt) < 10:
            features_opt = df_imp_joint["feature"].head(10).tolist()
            print("  Umbral demasiado restrictivo — se usan las Top-10 por importancia media.")

        print(f"  Features originales:    {len(features)}")
        print(f"  Features seleccionadas: {len(features_opt)}  "
              f"({len(features_opt)/len(features)*100:.1f}%)")
        return features_opt

    # ── Comparativa original vs optimizado ───────────────────────────────────

    def save_orig_vs_opt_comparison(
        self,
        res_rf_test: dict,
        res_rf_opt_test: dict,
        res_xgb_test: dict,
        res_xgb_opt_test: dict,
        n_orig: int,
        n_opt: int,
        filename: str = "15_comparativa_orig_vs_opt.png",
    ) -> None:
        modelos_cmp = {
            f"RF Original ({n_orig} feat)":  res_rf_test,
            f"RF Opt ({n_opt} feat)":         res_rf_opt_test,
            f"XGB Original ({n_orig} feat)":  res_xgb_test,
            f"XGB Opt ({n_opt} feat)":        res_xgb_opt_test,
        }
        df_cmp = pd.DataFrame(
            [{**v, "modelo": k} for k, v in modelos_cmp.items()]
        ).set_index("modelo")

        metricas  = ["MAE", "RMSE", "R2"]
        colores   = ["steelblue", "#7ec8e3", "darkorange", "#f4a261"]
        etiquetas = [n.replace(" ", "\n") for n in df_cmp.index]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for i, met in enumerate(metricas):
            ax   = axes[i]
            vals = df_cmp[met].values
            bars = ax.bar(etiquetas, vals, color=colores, edgecolor="white", alpha=0.9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01, f"{val:.4f}",
                        ha="center", va="bottom", fontsize=8)
            ax.set_title(met, fontsize=12)
            ax.set_ylabel(met)
            ax.tick_params(axis="x", labelsize=8)

        fig.suptitle(
            "Comparativa Original vs Optimizado — conjunto test\n"
            "(mismas métricas y mismo split temporal)",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {filename}")

    # ── Split temporal ────────────────────────────────────────────────────────

    def save_split_plot(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        target: str,
        filename: str = "00_split_temporal.png",
    ) -> None:
        import matplotlib.dates as mdates
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(df_train.index, df_train[target], color="steelblue",
                linewidth=0.7, label=f"Train ({len(df_train):,})")
        ax.plot(df_val.index,   df_val[target],   color="darkorange",
                linewidth=0.7, label=f"Val ({len(df_val):,})")
        ax.plot(df_test.index,  df_test[target],  color="crimson",
                linewidth=0.7, label=f"Test ({len(df_test):,})")
        ax.set_title("Split temporal — Energía HVAC 15 min (kWh)", fontsize=12)
        ax.set_ylabel("Energía 15 min (kWh)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {filename}")

    # ── Correlación features ──────────────────────────────────────────────────

    def save_correlation_plots(
        self,
        df_train: pd.DataFrame,
        features: list,
        target: str,
    ) -> tuple:
        """Barplot de correlación con target y heatmap top-20."""
        corr_matrix = df_train[features + [target]].corr()
        corr_target = corr_matrix[target].drop(target).abs().sort_values(ascending=False)

        top_n = min(25, len(features))
        fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
        corr_target.head(top_n).sort_values().plot.barh(
            ax=ax, color="steelblue", edgecolor="white"
        )
        ax.set_title(f"|Correlación Pearson| con el target — Top {top_n}", fontsize=11)
        ax.set_xlabel("|Correlación|")
        fig.tight_layout()
        fig.savefig(self.output_dir / "10_correlacion_target.png", bbox_inches="tight")
        plt.close(fig)
        print("  Guardado: 10_correlacion_target.png")

        # Heatmap top-20
        top20_feat = corr_target.head(20).index.tolist()
        corr_sub   = corr_matrix.loc[top20_feat, top20_feat]
        mask_tri   = np.triu(np.ones_like(corr_sub, dtype=bool))
        fig, ax = plt.subplots(figsize=(10, 9))
        sns.heatmap(corr_sub, mask=mask_tri, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, linewidths=0.4,
                    annot_kws={"size": 7}, ax=ax)
        ax.set_title("Correlación entre el Top-20 features más relacionadas con el target", fontsize=11)
        fig.tight_layout()
        fig.savefig(self.output_dir / "10b_correlacion_heatmap_top20.png", bbox_inches="tight")
        plt.close(fig)
        print("  Guardado: 10b_correlacion_heatmap_top20.png")

        # Pares de features con correlación mutua elevada
        high_corr_pairs = [
            {"feature_a": features[i], "feature_b": features[j],
             "correlacion": round(abs(corr_matrix.loc[features[i], features[j]]), 4)}
            for i in range(len(features))
            for j in range(i + 1, len(features))
            if abs(corr_matrix.loc[features[i], features[j]]) > 0.92
        ]
        df_high_corr = (
            pd.DataFrame(high_corr_pairs).sort_values("correlacion", ascending=False)
            if high_corr_pairs
            else pd.DataFrame(columns=["feature_a", "feature_b", "correlacion"])
        )
        print(f"  Pares con |correlación| > 0.92: {len(df_high_corr)}")
        return corr_target, df_high_corr

    # ── Comparativa enfoques estacionales ────────────────────────────────────

    def save_seasonal_approaches_comparison(
        self,
        df_res_est: pd.DataFrame,
        df_res_grupos: pd.DataFrame,
        filename: str = "17_comparativa_enfoques_estacionales.png",
    ) -> None:
        """Compara el rendimiento medio entre 4 estaciones individuales y 2 grupos climáticos."""
        if df_res_est.empty or df_res_grupos.empty:
            print("  [SKIP] Datos insuficientes para comparar enfoques estacionales.")
            return

        mask_rf_est  = df_res_est.index.str.contains("Random Forest")
        mask_xgb_est = df_res_est.index.str.contains("XGBoost")
        mask_rf_grp  = df_res_grupos.index.str.contains("RF")
        mask_xgb_grp = df_res_grupos.index.str.contains("XGBoost")

        resumen_enf = {}
        if mask_rf_est.any():
            resumen_enf["RF — 4 estaciones"]  = df_res_est[mask_rf_est][["MAE", "RMSE", "R2"]].mean()
        if mask_rf_grp.any():
            resumen_enf["RF — 2 grupos"]       = df_res_grupos[mask_rf_grp][["MAE", "RMSE", "R2"]].mean()
        if mask_xgb_est.any():
            resumen_enf["XGB — 4 estaciones"] = df_res_est[mask_xgb_est][["MAE", "RMSE", "R2"]].mean()
        if mask_xgb_grp.any():
            resumen_enf["XGB — 2 grupos"]      = df_res_grupos[mask_xgb_grp][["MAE", "RMSE", "R2"]].mean()

        if not resumen_enf:
            return

        df_resumen = pd.DataFrame(resumen_enf).T
        colores    = ["steelblue", "#7ec8e3", "darkorange", "#f4a261"]
        etiquetas  = [n.replace(" — ", "\n") for n in df_resumen.index]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for i, met in enumerate(["MAE", "RMSE", "R2"]):
            ax   = axes[i]
            vals = df_resumen[met].values
            bars = ax.bar(etiquetas, vals,
                          color=colores[:len(vals)], edgecolor="white", alpha=0.9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01, f"{val:.4f}",
                        ha="center", va="bottom", fontsize=8)
            ax.set_title(met, fontsize=12)
            ax.set_ylabel(met)
            ax.tick_params(axis="x", labelsize=8)

        fig.suptitle(
            "Media por enfoque de agrupación estacional\n"
            "(4 estaciones individuales vs 2 grupos climáticos)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {filename}")

    # ── Ranking final ampliado ────────────────────────────────────────────────

    def save_extended_ranking_heatmap(
        self,
        df_globales: pd.DataFrame,
        df_estacionales: pd.DataFrame,
        df_grupos: pd.DataFrame,
        filename: str = "18_ranking_final_ampliado.png",
    ) -> pd.DataFrame:
        """Heatmap del ranking que incluye globales, estacionales y grupos."""
        todos = list(df_globales.reset_index().to_dict("records"))
        if not df_estacionales.empty:
            todos += list(df_estacionales.reset_index().to_dict("records"))
        if not df_grupos.empty:
            todos += list(df_grupos.reset_index().to_dict("records"))

        df_final = (
            pd.DataFrame(todos)
            .drop_duplicates(subset="modelo", keep="last")
            .set_index("modelo")
            .sort_values("RMSE")
        )

        cols_m = [c for c in ["MAE", "RMSE", "R2", "WAPE", "sMAPE"] if c in df_final.columns]
        dn_amp = self._normalizar(df_final, cols_m)

        fig, ax = plt.subplots(figsize=(12, max(6, len(df_final) * 0.42)))
        sns.heatmap(
            dn_amp[cols_m],
            annot=df_final[cols_m].round(4),
            fmt="",
            cmap="RdYlGn_r",
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Peor → Mejor (normalizado)"},
        )
        ax.set_title(
            "Ranking final ampliado — todos los modelos\n"
            "(globales, optimizados con FEATURES_OPT y grupos estacionales)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {filename}")
        return df_final
