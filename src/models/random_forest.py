"""
random_forest.py — Modelo Random Forest Regressor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from src.config import RF_PARAMS, SEED
from src.models.base_model import BaseModel, recortar_predicciones, plot_prediccion, plot_residuos


class RandomForestModel(BaseModel):
    """Random Forest para predicción de energía HVAC.

    Usa los hiperparámetros definidos en config.RF_PARAMS salvo que se sobreescriban.
    """

    def __init__(
        self,
        features: list,
        target: str,
        name: str = "Random Forest (global)",
        params: dict = None,
    ):
        super().__init__(name, features, target)
        rf_params = params if params is not None else RF_PARAMS
        self.model = RandomForestRegressor(
            random_state=SEED,
            **rf_params,
        )

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        **kwargs,
    ) -> "RandomForestModel":
        print(f"  Entrenando {self.name}...")
        self.model.fit(X_train, y_train)
        self._fitted = True
        print(f"  {self.name} — entrenado. n_estimators={self.model.n_estimators}")
        return self

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return recortar_predicciones(self.model.predict(X))

    # ── Figuras ───────────────────────────────────────────────────────────────

    def save_plots(
        self,
        output_dir: Path,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "02_rf",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Feature importance (top 20)
        imp_rf = pd.DataFrame({
            "feature":    self.features,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=True).tail(20)

        fig, ax = plt.subplots(figsize=(8, 7))
        imp_rf.plot.barh(x="feature", y="importance", ax=ax,
                         color="steelblue", legend=False)
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
        """Devuelve DataFrame con las importancias de las features."""
        self._check_fitted()
        return pd.DataFrame({
            "feature":    self.features,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
