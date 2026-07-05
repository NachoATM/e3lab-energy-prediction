"""
prophet_model.py — Modelo Prophet para predicción de energía HVAC.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    try:
        from fbprophet import Prophet  # nombre antiguo < 1.0
        PROPHET_AVAILABLE = True
    except Exception as exc:
        Prophet = None
        PROPHET_AVAILABLE = False
        _PROPHET_ERROR = exc

from src.models.base_model import BaseModel, recortar_predicciones, plot_prediccion, plot_residuos

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetModel(BaseModel):
    """Modelo Prophet para predicción de series temporales de energía HVAC.

    Prophet solo usa la serie temporal del target (ds + y); los regressores
    externos no se incluyen para mantener la comparabilidad con los demás modelos.

    Para modelos estacionales se desactiva yearly_seasonality y se usa
    un orden de Fourier reducido para la estacionalidad intra-diaria.
    """

    def __init__(
        self,
        features: list,
        target: str,
        name: str = "Prophet (global)",
        yearly_seasonality: bool = True,
        n_changepoints: int = 25,
        daily_fourier_order: int = 10,
    ):
        if not PROPHET_AVAILABLE:
            raise RuntimeError(
                "Prophet no está disponible. Instalar con: pip install prophet. "
                f"Detalle: {_PROPHET_ERROR}"
            )
        super().__init__(name, features, target)
        self.yearly_seasonality    = yearly_seasonality
        self.n_changepoints        = n_changepoints
        self.daily_fourier_order   = daily_fourier_order
        self.model                 = None
        self._forecast_components  = None

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        index_train: pd.DatetimeIndex = None,
        **kwargs,
    ) -> "ProphetModel":
        """
        Entrena Prophet. Requiere el índice temporal de train para construir ds+y.
        Si X_train tiene un DatetimeIndex, se usa directamente.
        """
        if index_train is not None:
            idx = index_train
        elif isinstance(X_train.index, pd.DatetimeIndex):
            idx = X_train.index
        else:
            raise ValueError(
                "ProphetModel.fit() necesita un DatetimeIndex. "
                "Pasa index_train=df_train.index."
            )

        df_prophet_train = pd.DataFrame({"ds": idx, "y": y_train})

        self.model = Prophet(
            seasonality_mode="additive",
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=self.yearly_seasonality,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            n_changepoints=self.n_changepoints,
            interval_width=0.95,
        )
        self.model.add_seasonality(
            name="diaria_15min",
            period=1,
            fourier_order=self.daily_fourier_order,
        )

        print(f"  Entrenando {self.name} (puede tardar unos minutos)...")
        self.model.fit(df_prophet_train)
        print(f"  {self.name} — entrenamiento completado.")
        self._fitted = True
        return self

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones para los timestamps en el índice de X.
        """
        self._check_fitted()
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("ProphetModel.predict() requiere X con DatetimeIndex.")

        future = pd.DataFrame({"ds": X.index})
        forecast = self.model.predict(future)
        self._forecast_components = forecast
        return recortar_predicciones(forecast["yhat"].values)

    def predict_from_index(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Predicción directamente desde un DatetimeIndex."""
        self._check_fitted()
        future   = pd.DataFrame({"ds": index})
        forecast = self.model.predict(future)
        self._forecast_components = forecast
        return recortar_predicciones(forecast["yhat"].values)

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save_model(self, path: Path) -> None:
        """Serializa el modelo Prophet como JSON."""
        self._check_fitted()
        import json
        try:
            from prophet.serialize import model_to_json
            with open(str(path), "w") as fh:
                json.dump(model_to_json(self.model), fh)
            print(f"  Prophet guardado: {Path(path).name}")
        except Exception as exc:
            print(f"  [WARN] No se pudo serializar Prophet: {exc}")

    # ── Figuras ───────────────────────────────────────────────────────────────

    def save_plots(
        self,
        output_dir: Path,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "04_prophet",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Componentes estacionales (si hay forecast almacenado)
        if self._forecast_components is not None:
            try:
                fig_comp = self.model.plot_components(self._forecast_components)
                fig_comp.suptitle(f"{self.name} — Componentes estacionales", fontsize=12, y=1.01)
                fig_comp.tight_layout()
                fig_comp.savefig(output_dir / f"{prefix}_componentes.png", bbox_inches="tight")
                plt.close(fig_comp)
                print(f"  Guardado: {prefix}_componentes.png")
            except Exception as exc:
                print(f"  [WARN] No se pudieron guardar los componentes de Prophet: {exc}")

        # Predicción vs real
        plot_prediccion(y_true, y_pred, self.name,
                        output_dir / f"{prefix}_prediccion.png")

        # Residuos
        plot_residuos(y_true, y_pred, self.name,
                      output_dir / f"{prefix}_residuos.png")
