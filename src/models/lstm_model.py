"""
lstm_model.py — Modelo LSTM (PyTorch) para predicción de energía HVAC.

Arquitectura: 2 capas LSTM (hidden=128, dropout=0.2) + FC(128→64→1).
Escalado RobustScaler aprendido solo en train.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import RobustScaler

from src.config import (
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    LSTM_SEQ_LEN,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LR,
    LSTM_PATIENCE,
    SEED,
)
from src.models.base_model import BaseModel, recortar_predicciones, plot_prediccion, plot_residuos


# ── Dataset ───────────────────────────────────────────────────────────────────

class LSTMDataset(Dataset):
    """Dataset de secuencias temporales para el LSTM.

    Cada muestra: X = datos[idx : idx+seq_len, :-1]  (features)
                  y = datos[idx+seq_len, -1]           (target, última columna)
    """

    def __init__(self, data: np.ndarray, seq_len: int):
        self.data    = torch.as_tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.seq_len, :-1]
        y = self.data[idx + self.seq_len, -1]
        return x, y


# ── Red neuronal ──────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(
        self,
        input_size:  int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers:  int = LSTM_NUM_LAYERS,
        dropout:     float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])   # último paso temporal
        return self.fc(out).squeeze(-1)


# ── Funciones de entrenamiento y evaluación ───────────────────────────────────

def _entrenar_epoch(
    model: _LSTMNet,
    loader: TorchDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def _evaluar_loader(
    model: _LSTMNet,
    loader: TorchDataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item() * len(x)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(preds),
        np.concatenate(targets),
    )


# ── Clase principal ───────────────────────────────────────────────────────────

class LSTMModel(BaseModel):
    """LSTM Regressor con early stopping y escalado RobustScaler.

    El scaler se ajusta con features + target de train para poder escalar/desescalar
    las predicciones del espacio normalizado al espacio original.
    """

    def __init__(
        self,
        features: list,
        target: str,
        name: str = "LSTM (global)",
        seq_len: int = LSTM_SEQ_LEN,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers:  int = LSTM_NUM_LAYERS,
        dropout:     float = LSTM_DROPOUT,
        batch_size:  int = LSTM_BATCH_SIZE,
        epochs:      int = LSTM_EPOCHS,
        lr:          float = LSTM_LR,
        patience:    int = LSTM_PATIENCE,
    ):
        super().__init__(name, features, target)
        self.seq_len    = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.lr          = lr
        self.patience    = patience

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler  = RobustScaler()
        self.net     = None
        self._hist_train: list = []
        self._hist_val:   list = []
        self._best_val_loss = float("inf")

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _make_loader(self, data: np.ndarray, shuffle: bool) -> TorchDataLoader:
        ds = LSTMDataset(data, self.seq_len)
        return TorchDataLoader(ds, batch_size=self.batch_size,
                               shuffle=shuffle, drop_last=False)

    def _inverse_target(self, scaled_values: np.ndarray, n: int) -> np.ndarray:
        """Desescala solo la columna del target."""
        dummy = np.zeros((n, len(self.features) + 1))
        dummy[:, -1] = scaled_values
        return self.scaler.inverse_transform(dummy)[:, -1]

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
        df_train: pd.DataFrame = None,
        df_val: pd.DataFrame = None,
        **kwargs,
    ) -> "LSTMModel":
        """
        El LSTM necesita el DataFrame completo (features + target) para escalar.
        Puedes pasar df_train / df_val directamente, o X_train+y_train / X_val+y_val.
        """
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Construir arrays escalados
        if df_train is not None:
            arr_train = df_train[self.features + [self.target]].values
        else:
            arr_train = np.column_stack([X_train, y_train])

        self.scaler.fit(arr_train)
        train_scaled = self.scaler.transform(arr_train)

        if df_val is not None:
            val_scaled = self.scaler.transform(
                df_val[self.features + [self.target]].values
            )
        elif X_val is not None and y_val is not None:
            val_scaled = self.scaler.transform(np.column_stack([X_val, y_val]))
        else:
            val_scaled = None

        dl_train = self._make_loader(train_scaled, shuffle=True)
        dl_val   = self._make_loader(val_scaled, shuffle=False) if val_scaled is not None else None

        # Inicializar red
        input_size = len(self.features)
        self.net = _LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.HuberLoss()

        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"  {self.name} | parámetros: {total_params:,} | device: {self.device}")
        print(f"  Entrenando {self.epochs} épocas máx. | patience={self.patience}")

        best_weights  = None
        patience_cnt  = 0
        self._hist_train = []
        self._hist_val   = []
        self._best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            loss_train = _entrenar_epoch(self.net, dl_train, optimizer, criterion, self.device)
            self._hist_train.append(loss_train)

            if dl_val is not None:
                loss_val, _, _ = _evaluar_loader(self.net, dl_val, criterion, self.device)
                self._hist_val.append(loss_val)
                scheduler.step(loss_val)

                improved = loss_val < self._best_val_loss - 1e-5
                if improved:
                    self._best_val_loss = loss_val
                    best_weights = {
                        k: v.detach().cpu().clone()
                        for k, v in self.net.state_dict().items()
                    }
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if epoch == 1 or epoch % 5 == 0 or improved:
                    marca = " <- mejor" if improved else ""
                    print(
                        f"  Época {epoch:3d} | train: {loss_train:.6f} | "
                        f"val: {loss_val:.6f}{marca}"
                    )

                if patience_cnt >= self.patience:
                    print(f"\n  Early stopping en época {epoch}")
                    break
            else:
                if epoch == 1 or epoch % 5 == 0:
                    print(f"  Época {epoch:3d} | train: {loss_train:.6f}")

        if best_weights is not None:
            self.net.load_state_dict(best_weights)
            print(f"  Mejor val loss: {self._best_val_loss:.6f}")

        self._fitted = True
        return self

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones en escala original.
        Nota: las primeras seq_len filas no tienen predicción (sin contexto previo suficiente).
        Se devuelve un array de longitud max(0, len(X) - seq_len).
        """
        self._check_fitted()

        # Necesitamos escalar X junto a un target ficticio para usar el scaler
        n = len(X)
        dummy_target  = np.zeros(n)
        arr = np.column_stack([X.values if hasattr(X, "values") else np.asarray(X), dummy_target])
        arr_scaled = self.scaler.transform(arr)

        dl = self._make_loader(arr_scaled, shuffle=False)
        criterion = nn.HuberLoss()

        _, pred_scaled, _ = _evaluar_loader(self.net, dl, criterion, self.device)
        n_pred = len(pred_scaled)
        return recortar_predicciones(self._inverse_target(pred_scaled, n_pred))

    def predict_from_scaled_loader(self, dl: TorchDataLoader):
        """Predicción desde un DataLoader ya escalado (devuelve pred_scaled, y_scaled)."""
        self._check_fitted()
        criterion = nn.HuberLoss()
        _, pred_sc, y_sc = _evaluar_loader(self.net, dl, criterion, self.device)
        return pred_sc, y_sc

    def inverse_target(self, scaled_values: np.ndarray) -> np.ndarray:
        return self._inverse_target(scaled_values, len(scaled_values))

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path) -> None:
        self._check_fitted()
        torch.save({
            "model_state_dict": self.net.state_dict(),
            "scaler":           self.scaler,
            "seq_len":          self.seq_len,
            "input_size":       len(self.features),
            "features":         self.features,
            "target":           self.target,
            "hidden_size":      self.hidden_size,
            "num_layers":       self.num_layers,
            "dropout":          self.dropout,
        }, str(path))
        print(f"  LSTM guardado: {Path(path).name}")

    # ── Figuras ───────────────────────────────────────────────────────────────

    def save_plots(
        self,
        output_dir: Path,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "03_lstm",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Curva de aprendizaje
        if self._hist_train:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self._hist_train, label="Train loss", color="steelblue")
            if self._hist_val:
                ax.plot(self._hist_val, label="Val loss", color="crimson")
            ax.set_title("LSTM — Curva de aprendizaje", fontsize=12)
            ax.set_xlabel("Época")
            ax.set_ylabel("Huber Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"{prefix}_learning_curve.png", bbox_inches="tight")
            plt.close(fig)

        # Predicción vs real
        plot_prediccion(y_true, y_pred, self.name,
                        output_dir / f"{prefix}_prediccion.png")

        # Residuos
        plot_residuos(y_true, y_pred, self.name,
                      output_dir / f"{prefix}_residuos.png")
