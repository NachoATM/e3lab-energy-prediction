"""
ridge.py — Modelo Ridge Regression con escalado robusto y selección de alpha por validación.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

from src.config import RIDGE_ALPHAS, SEED
from src.models.base_model import BaseModel, recortar_predicciones, plot_prediccion, plot_residuos


class RidgeModel(BaseModel):
    """Ridge Regression con RobustScaler y búsqueda de alpha sobre validación temporal.

    El escalado se ajusta exclusivamente en train para evitar data leakage.
    """

    def __init__(self, features: list, target: str, alphas: list = None):
        super().__init__("Ridge (global)", features, target)
        self.alphas  = alphas if alphas is not None else RIDGE_ALPHAS
        self.scaler  = RobustScaler()
        self.model   = None
        self._best_alpha = None
        self._search_rows = []

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
    ) -> "RidgeModel":
        """Ajusta el scaler y selecciona el mejor alpha por RMSE de validación."""
        X_tr_sc = self.scaler.fit_transform(X_train)

        if X_val is not None and y_val is not None:
            X_vl_sc  = self.scaler.transform(X_val)
            best_model, best_rmse = None, np.inf
            self._search_rows = []

            for alpha in self.alphas:
                m = Ridge(alpha=alpha)
                m.fit(X_tr_sc, y_train)
                pred_val = recortar_predicciones(m.predict(X_vl_sc))
                rmse     = np.sqrt(mean_squared_error(y_val, pred_val))
                self._search_rows.append({"alpha": alpha, "RMSE_val": rmse})
                if rmse < best_rmse:
                    best_rmse  = rmse
                    best_model = m

            self.model       = best_model
            self._best_alpha = self.model.alpha
            print(f"  Ridge — mejor alpha: {self._best_alpha:g}  (RMSE val: {best_rmse:.4f})")
        else:
            # Sin validación: usar alpha por defecto (1.0)
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_tr_sc, y_train)
            self._best_alpha = 1.0
            print("  Ridge — ajustado con alpha=1.0 (sin validación disponible)")

        self._fitted = True
        return self

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_sc = self.scaler.transform(X)
        return recortar_predicciones(self.model.predict(X_sc))

    # ── Figuras ───────────────────────────────────────────────────────────────

    def save_plots(
        self,
        output_dir: Path,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Coeficientes absolutos (top 15)
        coef_df = pd.DataFrame({
            "feature": self.features,
            "coef":    np.abs(self.model.coef_),
        }).sort_values("coef", ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(8, 5))
        coef_df.plot.barh(x="feature", y="coef", ax=ax, color="steelblue", legend=False)
        ax.set_title("Ridge — Top 15 features por coeficiente absoluto", fontsize=11)
        ax.set_xlabel("|Coeficiente|")
        fig.tight_layout()
        fig.savefig(output_dir / "01_ridge_coeficientes.png", bbox_inches="tight")
        plt.close(fig)

        # Predicción vs real
        plot_prediccion(y_true, y_pred, "Ridge Regression",
                        output_dir / "01_ridge_prediccion.png")

        # Residuos
        plot_residuos(y_true, y_pred, "Ridge Regression",
                      output_dir / "01_ridge_residuos.png")

    def get_search_results(self) -> pd.DataFrame:
        """Devuelve la tabla de búsqueda de alpha."""
        return pd.DataFrame(self._search_rows)
