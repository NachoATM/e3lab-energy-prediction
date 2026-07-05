"""
base_model.py — Clase base abstracta y funciones auxiliares compartidas por todos los modelos.
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130


# ── Funciones auxiliares ──────────────────────────────────────────────────────

def recortar_predicciones(y_pred: np.ndarray, lower: float = 0) -> np.ndarray:
    """La energía HVAC no puede ser negativa; recorta predicciones no físicas."""
    return np.maximum(np.asarray(y_pred, dtype=float), lower)


def evaluar(y_true: np.ndarray, y_pred: np.ndarray, nombre: str = "Modelo") -> dict:
    """Calcula MAE, RMSE, R2, WAPE y sMAPE. Imprime el resumen y devuelve el dict."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    wape  = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100
    smape = np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100

    print(f"-- {nombre} --")
    print(f"  MAE:   {mae:.4f} kWh")
    print(f"  RMSE:  {rmse:.4f} kWh")
    print(f"  R2:    {r2:.4f}")
    print(f"  WAPE:  {wape:.2f} %")
    print(f"  sMAPE: {smape:.2f} %")

    return {"modelo": nombre, "MAE": mae, "RMSE": rmse, "R2": r2,
            "WAPE": wape, "sMAPE": smape}


def imputar_segmento_temporal(
    df_segmento: pd.DataFrame,
    features: list,
    fill_values: pd.Series,
    contexto_previo: pd.DataFrame = None,
) -> pd.DataFrame:
    """Imputa NaN usando forward-fill del pasado disponible y medianas de train."""
    if contexto_previo is not None and len(contexto_previo) > 0:
        combinado   = pd.concat([contexto_previo.tail(1), df_segmento], axis=0)
        features_imp = combinado[features].ffill().iloc[1:]
    else:
        features_imp = df_segmento[features].ffill()

    df_imp = df_segmento.copy()
    df_imp[features] = features_imp.fillna(fill_values)
    return df_imp


def plot_prediccion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nombre: str,
    filepath: Path,
    n_dias: int = 7,
) -> None:
    """Guarda figura con predicción vs real (últimos n_dias) y scatter plot."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n  = min(n_dias * 24 * 4, len(y_true))
    yt = y_true[-n:]
    yp = y_pred[-n:]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    axes[0].plot(range(len(yt)), yt, color="steelblue", linewidth=1.0, label="Real")
    axes[0].plot(range(len(yt)), yp, color="crimson",   linewidth=1.0, alpha=0.8, label="Predicción")
    axes[0].set_title(f"{nombre} — Predicción vs real (últimos {n_dias} días)", fontsize=12)
    axes[0].set_ylabel("Energía HVAC (kWh)")
    axes[0].legend()

    axes[1].scatter(yt, yp, alpha=0.3, s=8, color="steelblue")
    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    axes[1].plot(lim, lim, "r--", linewidth=1.5, label="Predicción perfecta")
    axes[1].set_xlabel("Real (kWh)")
    axes[1].set_ylabel("Predicho (kWh)")
    axes[1].set_title("Real vs predicho — scatter")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {Path(filepath).name}")


def plot_residuos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nombre: str,
    filepath: Path,
) -> None:
    """Guarda figura con residuos en el tiempo e histograma."""
    residuos = np.asarray(y_true) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(residuos, linewidth=0.7, color="slategray")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"{nombre} - residuos en el tiempo")
    axes[0].set_ylabel("Error (kWh)")

    sns.histplot(residuos, bins=50, kde=True, ax=axes[1], color="steelblue")
    axes[1].set_title("Distribución de residuos")
    axes[1].set_xlabel("Error (kWh)")

    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {Path(filepath).name}")


# ── Clase base abstracta ──────────────────────────────────────────────────────

class BaseModel(abc.ABC):
    """Interfaz común para todos los modelos predictivos del proyecto.

    Subclases deben implementar:
        fit(X_train, y_train)
        predict(X)
        save_plots(output_dir)
    """

    def __init__(self, name: str, features: list, target: str):
        self.name     = name
        self.features = features
        self.target   = target
        self._fitted  = False

    @abc.abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> "BaseModel":
        """Entrena el modelo."""
        ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Genera predicciones (no negativas)."""
        ...

    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> dict:
        """Evalúa el modelo sobre X. Devuelve dict de métricas."""
        y_pred = self.predict(X)
        return evaluar(y_true, y_pred, self.name)

    @abc.abstractmethod
    def save_plots(self, output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Guarda las figuras específicas del modelo en output_dir."""
        ...

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predict().")
