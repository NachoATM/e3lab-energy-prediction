"""
xgboost_model.py — Modelo XGBoost con búsqueda manual de hiperparámetros y early stopping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception as exc:
    xgb = None
    XGB_AVAILABLE = False
    _XGB_ERROR = exc

from sklearn.metrics import mean_squared_error

from src.config import XGB_PARAM_GRID, XGB_SEASONAL_PARAMS, SEED
from src.models.base_model import BaseModel, recortar_predicciones, plot_prediccion, plot_residuos


class XGBoostModel(BaseModel):
    """XGBoost Regressor con búsqueda manual de hiperparámetros sobre validación temporal.

    Si se pasa params directamente se salta la búsqueda (útil para modelos estacionales).
    """

    def __init__(
        self,
        features: list,
        target: str,
        name: str = "XGBoost (global)",
        param_grid: list = None,
        params: dict = None,
    ):
        if not XGB_AVAILABLE:
            raise RuntimeError(
                "XGBoost no está disponible. "
                "En Mac instala libomp: brew install libomp. "
                f"Detalle: {_XGB_ERROR}"
            )
        super().__init__(name, features, target)
        self.param_grid   = param_grid if param_grid is not None else XGB_PARAM_GRID
        self._fixed_params = params   # si se pasan, no se hace búsqueda
        self.model        = None
        self._search_rows = []

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> "XGBoostModel":
        X_tr = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        X_vl = X_val.values   if (X_val is not None and hasattr(X_val, "values")) else (
            np.asarray(X_val) if X_val is not None else None
        )

        # Modo con parámetros fijos (estacionales)
        if self._fixed_params is not None:
            m = xgb.XGBRegressor(random_state=SEED, n_jobs=-1, **self._fixed_params)
            eval_set = [(X_vl, y_val)] if X_vl is not None else None
            m.fit(X_tr, y_train, eval_set=eval_set, verbose=False)
            self.model   = m
            self._fitted = True
            print(f"  {self.name} — entrenado con parámetros fijos.")
            return self

        # Búsqueda manual sobre validación temporal
        if X_vl is None or y_val is None:
            raise ValueError("XGBoostModel.fit() requiere X_val e y_val para la búsqueda de hiperparámetros.")

        best_model, best_rmse = None, np.inf
        self._search_rows = []

        for i, params in enumerate(self.param_grid, start=1):
            m = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=1500,
                early_stopping_rounds=50,
                eval_metric="rmse",
                tree_method="hist",
                random_state=SEED,
                n_jobs=-1,
                **params,
            )
            m.fit(X_tr, y_train, eval_set=[(X_vl, y_val)], verbose=False)
            pred_val       = recortar_predicciones(m.predict(X_vl))
            rmse           = np.sqrt(mean_squared_error(y_val, pred_val))
            best_iteration = getattr(m, "best_iteration", None)
            row = {**params, "RMSE_val": rmse, "best_iteration": best_iteration}
            self._search_rows.append(row)
            print(f"  Config {i}/{len(self.param_grid)} | RMSE val={rmse:.4f} | best_iter={best_iteration}")
            if rmse < best_rmse:
                best_rmse  = rmse
                best_model = m

        self.model   = best_model
        self._fitted = True
        print(f"  {self.name} — mejor RMSE val: {best_rmse:.4f}")
        return self

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        return recortar_predicciones(self.model.predict(X_arr))

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save_model(self, path: Path) -> None:
        self._check_fitted()
        self.model.save_model(str(path))
        print(f"  Modelo XGBoost guardado: {Path(path).name}")

    # ── Figuras ───────────────────────────────────────────────────────────────

    def save_plots(
        self,
        output_dir: Path,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "02_xgb",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Feature importance (top 20)
        imp_df = pd.DataFrame({
            "feature":    self.features,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=True).tail(20)

        fig, ax = plt.subplots(figsize=(8, 7))
        imp_df.plot.barh(x="feature", y="importance", ax=ax,
                         color="darkorange", legend=False)
        ax.set_title(f"{self.name} — Top 20 features por importancia", fontsize=11)
        ax.set_xlabel("Feature importance")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_importance.png", bbox_inches="tight")
        plt.close(fig)

        # Predicción vs real
        plot_prediccion(y_true, y_pred, self.name,
                        output_dir / f"{prefix}_prediccion.png")

        # Residuos
        plot_residuos(y_true, y_pred, self.name,
                      output_dir / f"{prefix}_residuos.png")

    def feature_importances_df(self) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame({
            "feature":    self.features,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def get_search_results(self) -> pd.DataFrame:
        return pd.DataFrame(self._search_rows).sort_values("RMSE_val") if self._search_rows else pd.DataFrame()
