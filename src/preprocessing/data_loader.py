"""
data_loader.py — Carga y concatenación de los CSV de Grafana y la estación meteo.
Construye df_raw, df_meta y df_meteo_15 con nombres de columna normalizados.
"""

import io
import re
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    CSV_FILES,
    METEO_DIR,
    TIPO_CORTO,
    METEO_COL_IDX,
    RANGOS_METEO,
)


class DataLoader:
    """Carga los CSV de Grafana y la estación meteorológica.

    Atributos públicos tras llamar a load():
        df_raw   — DataFrame con todas las filas brutas, índice 'timestamp'.
        df_meta  — DataFrame con metadatos de cada sensor (col_original, col_corta, dir, …).
        df_meteo — DataFrame con variables meteo resampleadas a 15 min.
    """

    def __init__(self, csv_files=None, meteo_dir=None):
        self.csv_files = csv_files if csv_files is not None else CSV_FILES
        self.meteo_dir = Path(meteo_dir) if meteo_dir is not None else METEO_DIR
        self.df_raw   = None
        self.df_meta  = None
        self.df_meteo = None

    # ── Grafana ───────────────────────────────────────────────────────────────

    @staticmethod
    def parse_grafana_col(col: str):
        """Devuelve (nombre_corto, dict_metadatos) para una columna Grafana."""
        m = re.match(r"^(.+?)\s*\{(.+)\}$", col)
        if not m:
            return col, {}
        tipo_raw = m.group(1).strip()
        meta_str = m.group(2)
        meta = {}
        for part in re.split(r",\s*(?=[a-z_]+=)", meta_str):
            kv = part.split("=")
            if len(kv) >= 2:
                meta[kv[0].strip()] = kv[1].strip().strip('"')

        tipo_corto = TIPO_CORTO.get(tipo_raw, tipo_raw.replace(" ", "_"))
        dir_num    = meta.get("dir", "").replace("dir_", "")
        planta     = meta.get("planta", "")
        uso        = meta.get("uso", "").replace(" ", "_")[:8]
        zona       = meta.get("zona", "")

        nombre_corto = f"{tipo_corto}_d{dir_num.zfill(2)}_{planta}_{uso}_{zona}"
        return nombre_corto, meta

    def _build_col_map(self, df: pd.DataFrame):
        """Construye el mapeo columna_original -> nombre_corto y la tabla de metadatos."""
        col_map   = {}
        meta_rows = []
        for col in df.columns:
            if col == "Time":
                col_map[col] = "timestamp"
                continue
            nombre_corto, meta = self.parse_grafana_col(col)
            base = nombre_corto
            n = 1
            while nombre_corto in col_map.values():
                nombre_corto = f"{base}_v{n}"
                n += 1
            col_map[col] = nombre_corto
            meta_rows.append({"col_original": col, "col_corta": nombre_corto, **meta})
        return col_map, pd.DataFrame(meta_rows)

    def load_grafana(self) -> pd.DataFrame:
        """Carga y concatena los CSV de Grafana. Devuelve df_raw con índice timestamp."""
        dfs = []
        for path in self.csv_files:
            if not Path(path).exists():
                print(f"  [SKIP] No encontrado: {path}")
                continue
            df_tmp = pd.read_csv(path, parse_dates=["Time"])
            print(
                f"  {Path(path).name}: {df_tmp.shape[0]:,} filas | "
                f"{df_tmp['Time'].min().date()} -> {df_tmp['Time'].max().date()}"
            )
            dfs.append(df_tmp)

        if not dfs:
            raise FileNotFoundError(
                "No se encontró ningún CSV de Grafana. Comprueba DATA_DIR en config.py."
            )

        df_raw = pd.concat(dfs, axis=0, ignore_index=True)
        df_raw = df_raw.sort_values("Time").reset_index(drop=True)

        col_map, df_meta = self._build_col_map(df_raw)
        df_raw = df_raw.rename(columns=col_map)
        df_raw = df_raw.set_index("timestamp")

        # Eliminar duplicados de timestamp
        n_dupl = df_raw.index.duplicated().sum()
        if n_dupl > 0:
            print(f"  Timestamps duplicados eliminados: {n_dupl}")
            df_raw = df_raw[~df_raw.index.duplicated(keep="first")]

        self.df_raw  = df_raw
        self.df_meta = df_meta

        print(
            f"\nDataset Grafana unificado: {df_raw.shape[0]:,} filas, "
            f"{df_raw.shape[1]} columnas"
        )
        print(f"  Periodo: {df_raw.index.min()} -> {df_raw.index.max()}")
        return df_raw

    # ── Meteorología ─────────────────────────────────────────────────────────

    @staticmethod
    def _leer_meteo_robusto(path: Path) -> pd.DataFrame:
        """Lee un CSV de la estación meteo con cabecera malformada y encoding variable."""
        raw = None
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(path, encoding=enc) as f:
                    raw = f.read()
                break
            except UnicodeDecodeError:
                continue
        if raw is None:
            raise ValueError(f"No se pudo leer {path} con ningún encoding conocido.")

        lines = raw.splitlines(keepends=True)
        hdr = lines[0]
        if hdr.startswith('"'):
            hdr = hdr[1:]
            if hdr.rstrip("\n").endswith('"'):
                hdr = hdr.rstrip("\n")[:-1] + "\n"
            hdr = hdr.replace('""', '"')
        lines[0] = hdr
        return pd.read_csv(io.StringIO("".join(lines)), low_memory=False)

    @staticmethod
    def _parsear_timestamp(serie: pd.Series) -> pd.Series:
        """Parsea timestamps y elimina zona horaria si la tuviera."""
        ts = pd.to_datetime(serie, dayfirst=True, errors="coerce", format="mixed")
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        return ts

    def load_meteo(self) -> pd.DataFrame:
        """Carga todos los CSV meteo y los resamplea a 15 min. Devuelve df_meteo."""
        if not self.meteo_dir.exists():
            print(f"  [SKIP] Directorio meteo no encontrado: {self.meteo_dir}")
            self.df_meteo = pd.DataFrame()
            return self.df_meteo

        meteo_files = sorted(self.meteo_dir.glob("*.csv"))
        print(f"Archivos meteo encontrados: {len(meteo_files)}")

        dfs_meteo = []
        for f in meteo_files:
            try:
                df_raw = self._leer_meteo_robusto(f)
                ncols  = len(df_raw.columns)
                idx    = {k: v for k, v in METEO_COL_IDX.items() if v < ncols}
                df_sel = df_raw.iloc[:, list(idx.values())].copy()
                df_sel.columns = list(idx.keys())
                df_sel["timestamp"] = self._parsear_timestamp(df_sel["timestamp"])
                df_sel = df_sel.dropna(subset=["timestamp"])
                for col in df_sel.columns[1:]:
                    df_sel[col] = pd.to_numeric(df_sel[col], errors="coerce")
                dfs_meteo.append(df_sel)
                print(
                    f"  {f.name}: {len(df_sel):,} filas | "
                    f"{df_sel['timestamp'].min().date()} -> {df_sel['timestamp'].max().date()}"
                )
            except Exception as e:
                print(f"  [ERROR] {f.name}: {e}")

        if not dfs_meteo:
            print("  No se cargaron datos meteo.")
            self.df_meteo = pd.DataFrame()
            return self.df_meteo

        df_meteo_raw = (
            pd.concat(dfs_meteo, ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
            .set_index("timestamp")
        )

        # Eliminar duplicados
        n_dupl = df_meteo_raw.index.duplicated().sum()
        if n_dupl > 0:
            print(f"  Duplicados meteo eliminados: {n_dupl}")
            df_meteo_raw = df_meteo_raw[~df_meteo_raw.index.duplicated(keep="first")]

        # Rango físico
        for col, (vmin, vmax) in RANGOS_METEO.items():
            if col not in df_meteo_raw.columns:
                continue
            mask = (df_meteo_raw[col] < vmin) | (df_meteo_raw[col] > vmax)
            df_meteo_raw.loc[mask, col] = np.nan

        # Resampleado a 15 min
        cols_media = ["temp_ext", "hr_ext", "dew_point", "wind_speed",
                      "solar_rad", "solar_diff", "wind_dir"]
        cols_suma  = ["rain"]

        df_meteo_15 = pd.concat([
            df_meteo_raw[
                [c for c in cols_media if c in df_meteo_raw.columns]
            ].resample("15min").mean(),
            df_meteo_raw[
                [c for c in cols_suma if c in df_meteo_raw.columns]
            ].resample("15min").sum(),
        ], axis=1)

        df_meteo_15.columns = ["meteo_" + c for c in df_meteo_15.columns]

        print(
            f"\nMeteo a 15 min: {df_meteo_15.shape[0]:,} filas, "
            f"{df_meteo_15.shape[1]} columnas"
        )
        self.df_meteo = df_meteo_15
        return df_meteo_15

    # ── API principal ─────────────────────────────────────────────────────────

    def load(self):
        """Carga Grafana + Meteo. Devuelve (df_raw, df_meta, df_meteo)."""
        print("=== Cargando datos de Grafana ===")
        self.load_grafana()
        print("\n=== Cargando datos meteorológicos ===")
        self.load_meteo()
        return self.df_raw, self.df_meta, self.df_meteo
