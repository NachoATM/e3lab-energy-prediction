"""
run_models.py — Pipeline completo de entrenamiento y evaluación de modelos E3Lab.

Ejecutar desde la raíz del proyecto:
    python run_models.py

Pasos:
  1. Carga de E3Lab_modelo.csv (output/data/ o raíz del proyecto).
  2. Split temporal 70/15/15 + imputación sin data leakage.
  3. Modelos globales: Naive, Ridge, Random Forest, XGBoost, LSTM, Prophet.
  4. Comparativa global + ranking.
  5. Análisis de features: correlación, permutation importance, SHAP.
  6. Modelos con features optimizadas (RF_opt, XGB_opt).
  7. Modelos estacionales por estación (XGBoost, Ridge, RF, Prophet).
  8. Modelos de grupos climáticos (Grupo A = Inv+Oto, Grupo B = Pri+Ver).
  9. Rankings finales y exportación de métricas.
"""

import sys
import json
import time
import random
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import joblib

warnings.filterwarnings("ignore")

from src.config import (
    DATA_OUT_DIR,
    MODEL_DIR,
    FIGURES_MODELS,
    TARGET,
    SEED,
    RIDGE_ALPHAS,
    RF_PARAMS,
    RF_SEASONAL_PARAMS,
    XGB_PARAM_GRID,
    XGB_SEASONAL_PARAMS,
    ESTACIONES,
    GRUPO_A_ESTACIONES,
    GRUPO_B_ESTACIONES,
)
from src.models.base_model import (
    recortar_predicciones,
    evaluar,
    imputar_segmento_temporal,
)
from src.models.ridge import RidgeModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel, LSTMDataset
from src.models.prophet_model import ProphetModel
from src.models.evaluator import ModelEvaluator

from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader as TorchDataLoader

# Reproducibilidad
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mes_a_estacion(mes: int) -> str:
    if mes in [3, 4, 5]:   return "primavera"
    if mes in [6, 7, 8]:   return "verano"
    if mes in [9, 10, 11]: return "otono"
    return "invierno"


def _split_temporal(df: pd.DataFrame, ratios=(0.70, 0.15)):
    """Split temporal 70/15/15 sobre un DataFrame ya ordenado."""
    n    = len(df)
    n_tr = int(n * ratios[0])
    n_vl = int(n * ratios[1])
    return df.iloc[:n_tr], df.iloc[n_tr:n_tr + n_vl], df.iloc[n_tr + n_vl:]


def _entrenar_grupo(
    df_grp: pd.DataFrame,
    nombre_grp: str,
    features: list,
    target: str,
    evaluator: ModelEvaluator,
    output_dir: Path,
):
    """Entrena RF + XGBoost para un grupo estacional."""
    df_tr, df_vl, df_te = _split_temporal(df_grp)
    if min(len(df_tr), len(df_vl), len(df_te)) < 200:
        print(f"  {nombre_grp}: datos insuficientes.")
        return {}, []

    print(f"  Split: train={len(df_tr):,}  val={len(df_vl):,}  test={len(df_te):,}")
    X_tr, y_tr = df_tr[features], df_tr[target].values
    X_vl, y_vl = df_vl[features], df_vl[target].values
    X_te, y_te = df_te[features], df_te[target].values

    resultado_grp = {}
    res_list      = []

    # Random Forest
    rf_g = RandomForestModel(features, target, name=f"RF - {nombre_grp}",
                              params=RF_SEASONAL_PARAMS)
    rf_g.fit(X_tr, y_tr)
    pred_rf = rf_g.predict(X_te)
    res_rf  = evaluar(y_te, pred_rf, f"RF - {nombre_grp}")
    res_list.append(res_rf)
    resultado_grp["rf"] = {"model": rf_g, "y_test": y_te, "pred_test": pred_rf}

    # XGBoost
    xgb_g = XGBoostModel(features, target, name=f"XGBoost - {nombre_grp}",
                          params=XGB_SEASONAL_PARAMS)
    xgb_g.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
    pred_xgb = xgb_g.predict(X_te)
    res_xgb  = evaluar(y_te, pred_xgb, f"XGBoost - {nombre_grp}")
    res_list.append(res_xgb)
    resultado_grp["xgb"] = {"model": xgb_g, "y_test": y_te, "pred_test": pred_xgb}

    return resultado_grp, res_list


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── Directorios ───────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_MODELS.mkdir(parents=True, exist_ok=True)

    # ── 1. Carga del dataset de modelado ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 1: Carga del dataset de modelado")
    print("=" * 60)

    DATA_CANDIDATES = [
        DATA_OUT_DIR / "E3Lab_modelo.csv",
        ROOT / "E3Lab_modelo.csv",
    ]
    existing = []
    seen     = set()
    for path in DATA_CANDIDATES:
        if path.exists():
            rp = path.resolve()
            if rp not in seen:
                seen.add(rp)
                preview = pd.read_csv(path, index_col=0, parse_dates=True)
                existing.append({"path": path, "rows": len(preview),
                                  "last_date": preview.index.max()})

    if not existing:
        raise FileNotFoundError(
            "No se encontró E3Lab_modelo.csv. "
            "Ejecuta primero: python run_preprocessing.py"
        )

    selected   = sorted(existing, key=lambda x: (x["last_date"], x["rows"]), reverse=True)[0]
    DATA_PATH  = selected["path"]
    df_model   = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df_model   = df_model.sort_index()
    df_model   = df_model[~df_model.index.duplicated(keep="first")]

    if TARGET not in df_model.columns:
        raise ValueError(f"No existe la variable objetivo esperada: {TARGET}")

    FEATURES = [c for c in df_model.columns if c != TARGET]
    print(f"  CSV: {DATA_PATH}")
    print(f"  Shape: {df_model.shape}")
    print(f"  Periodo: {df_model.index.min()} -> {df_model.index.max()}")
    print(f"  Features: {len(FEATURES)}")
    print(f"\n  Variable objetivo — estadísticas:")
    print(df_model[TARGET].describe().round(3).to_string())

    # ── 2. Split + imputación ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 2: Split temporal 70/15/15 + imputación")
    print("=" * 60)
    df_clean_model = df_model.dropna(subset=[TARGET]).copy()
    df_train_raw, df_val_raw, df_test_raw = _split_temporal(df_clean_model)

    train_medians = df_train_raw[FEATURES].median(numeric_only=True)
    df_train = imputar_segmento_temporal(df_train_raw, FEATURES, train_medians)
    df_val   = imputar_segmento_temporal(df_val_raw,   FEATURES, train_medians, df_train)
    df_test  = imputar_segmento_temporal(df_test_raw,  FEATURES, train_medians, df_val)

    print(f"  Train:      {len(df_train):,}  {df_train.index.min().date()} -> {df_train.index.max().date()}")
    print(f"  Validación: {len(df_val):,}  {df_val.index.min().date()} -> {df_val.index.max().date()}")
    print(f"  Test:       {len(df_test):,}  {df_test.index.min().date()} -> {df_test.index.max().date()}")

    X_train = df_train[FEATURES]
    X_val   = df_val[FEATURES]
    X_test  = df_test[FEATURES]
    y_train = df_train[TARGET].values
    y_val   = df_val[TARGET].values
    y_test  = df_test[TARGET].values

    # Evaluador global
    ev = ModelEvaluator(output_dir=FIGURES_MODELS)

    # Plot del split temporal
    ev.save_split_plot(df_train, df_val, df_test, TARGET)

    resultados    = []
    resultados_gl = []   # solo globales

    # ── 3a. Naive baseline ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 3a: Baseline Naive lag-1")
    print("=" * 60)
    if "energia_lag_1" in FEATURES:
        pred_test_naive = recortar_predicciones(df_test["energia_lag_1"].values)
    else:
        pred_test_naive = np.r_[y_val[-1], y_test[:-1]]
    res_naive = evaluar(y_test, pred_test_naive, "Naive lag-1")
    resultados.append(res_naive)
    resultados_gl.append(res_naive)

    # ── 3b. Ridge ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 3b: Ridge Regression")
    print("=" * 60)
    ridge = RidgeModel(FEATURES, TARGET, alphas=RIDGE_ALPHAS)
    ridge.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    pred_val_lr   = ridge.predict(X_val)
    pred_test_lr  = ridge.predict(X_test)
    _             = evaluar(y_val, pred_val_lr, "Ridge - Validación")
    res_lr_test   = evaluar(y_test, pred_test_lr, "Ridge - Test")
    res_lr_test["modelo"] = "Ridge (global)"
    resultados.append(res_lr_test)
    resultados_gl.append(res_lr_test)

    ridge.save_plots(FIGURES_MODELS, y_test, pred_test_lr)
    ridge.get_search_results().to_csv(MODEL_DIR / "ridge_validacion_alphas.csv", index=False)
    joblib.dump({"model": ridge.model, "scaler": ridge.scaler, "features": FEATURES,
                 "target": TARGET, "alpha": ridge._best_alpha},
                MODEL_DIR / "ridge_global.pkl")

    # ── 3c. Random Forest ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 3c: Random Forest")
    print("=" * 60)
    rf = RandomForestModel(FEATURES, TARGET, params=RF_PARAMS)
    rf.fit(X_train, y_train)
    pred_val_rf   = rf.predict(X_val)
    pred_test_rf  = rf.predict(X_test)
    _             = evaluar(y_val, pred_val_rf, "Random Forest - Validación")
    res_rf_test   = evaluar(y_test, pred_test_rf, "Random Forest - Test")
    res_rf_test["modelo"] = "Random Forest (global)"
    resultados.append(res_rf_test)
    resultados_gl.append(res_rf_test)

    rf.save_plots(FIGURES_MODELS, y_test, pred_test_rf)
    joblib.dump({"model": rf.model, "features": FEATURES, "target": TARGET},
                MODEL_DIR / "randomforest_global.pkl")

    # ── 3d. XGBoost ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 3d: XGBoost")
    print("=" * 60)
    xgb_m = None
    res_xgb_test = None
    pred_test_xgb = None
    try:
        xgb_m = XGBoostModel(FEATURES, TARGET, param_grid=XGB_PARAM_GRID)
        xgb_m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        pred_val_xgb  = xgb_m.predict(X_val)
        pred_test_xgb = xgb_m.predict(X_test)
        _             = evaluar(y_val, pred_val_xgb, "XGBoost - Validación")
        res_xgb_test  = evaluar(y_test, pred_test_xgb, "XGBoost - Test")
        res_xgb_test["modelo"] = "XGBoost (global)"
        resultados.append(res_xgb_test)
        resultados_gl.append(res_xgb_test)

        xgb_m.save_plots(FIGURES_MODELS, y_test, pred_test_xgb)
        xgb_m.save_model(MODEL_DIR / "xgboost_global.json")
        xgb_m.get_search_results().to_csv(MODEL_DIR / "xgb_busqueda_validacion.csv", index=False)
    except RuntimeError as exc:
        print(f"  [SKIP] XGBoost no disponible: {exc}")

    # ── 3e. LSTM ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 3e: LSTM")
    print("=" * 60)
    lstm_m = LSTMModel(FEATURES, TARGET)
    lstm_m.fit(None, None, df_train=df_train, df_val=df_val)

    # Construir DataLoader de test escalado
    test_arr    = lstm_m.scaler.transform(df_test[FEATURES + [TARGET]].values)
    ds_test     = LSTMDataset(test_arr, lstm_m.seq_len)
    dl_test     = TorchDataLoader(ds_test, batch_size=lstm_m.batch_size, shuffle=False)
    pred_sc, y_sc = lstm_m.predict_from_scaled_loader(dl_test)

    pred_test_lstm = lstm_m.inverse_target(pred_sc)
    pred_test_lstm = recortar_predicciones(pred_test_lstm)
    y_test_lstm    = lstm_m.inverse_target(y_sc)

    res_lstm_test  = evaluar(y_test_lstm, pred_test_lstm, "LSTM - Test")
    res_lstm_test["modelo"] = "LSTM (global)"
    resultados.append(res_lstm_test)
    resultados_gl.append(res_lstm_test)

    lstm_m.save_plots(FIGURES_MODELS, y_test_lstm, pred_test_lstm)
    lstm_m.save_checkpoint(MODEL_DIR / "lstm_global.pt")

    # ── 3f. Prophet ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 3f: Prophet")
    print("=" * 60)
    prophet_m = None
    res_prophet_test = None
    pred_test_prophet = None
    try:
        prophet_m = ProphetModel(FEATURES, TARGET)
        prophet_m.fit(X_train, y_train, index_train=df_train.index)
        pred_val_prophet  = prophet_m.predict_from_index(df_val.index)
        pred_test_prophet = prophet_m.predict_from_index(df_test.index)
        _                 = evaluar(y_val, pred_val_prophet, "Prophet - Validación")
        res_prophet_test  = evaluar(y_test, pred_test_prophet, "Prophet - Test")
        res_prophet_test["modelo"] = "Prophet (global)"
        resultados.append(res_prophet_test)
        resultados_gl.append(res_prophet_test)

        prophet_m.save_plots(FIGURES_MODELS, y_test, pred_test_prophet)
        prophet_m.save_model(MODEL_DIR / "prophet_global.json")
    except RuntimeError as exc:
        print(f"  [SKIP] Prophet no disponible: {exc}")

    # ── 4. Comparativa global ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 4: Comparativa global")
    print("=" * 60)
    ev.add_results(resultados_gl)
    df_resultados = ev.df_results
    print(df_resultados.round(4).to_string())

    ev.save_comparison_plots()
    ev.save_ranking_heatmap(filename="06a_ranking_globales.png",
                             title="Ranking modelos globales — métricas normalizadas\n(verde = mejor)")

    mejor_global = df_resultados["RMSE"].idxmin()
    print(f"\n  Mejor modelo global (menor RMSE): {mejor_global}")
    print(f"    RMSE: {df_resultados.loc[mejor_global, 'RMSE']:.4f} kWh")
    print(f"    R2:   {df_resultados.loc[mejor_global, 'R2']:.4f}")

    # ── 5. Análisis de features ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 5: Análisis de features (correlación, permutation importance, SHAP)")
    print("=" * 60)

    # Correlación con target
    corr_target, _ = ev.save_correlation_plots(df_train, FEATURES, TARGET)

    # Feature importance comparativa (RF vs XGB, si disponibles)
    FEATURES_OPT = FEATURES  # fallback si no hay XGB
    if xgb_m is not None:
        df_imp_rf  = pd.DataFrame({"feature": FEATURES, "RF":  rf.model.feature_importances_})
        df_imp_xgb = pd.DataFrame({"feature": FEATURES, "XGB": xgb_m.model.feature_importances_})
        df_imp_joint = df_imp_rf.merge(df_imp_xgb, on="feature")
        df_imp_joint["RF_norm"]   = df_imp_joint["RF"]  / (df_imp_joint["RF"].sum()  + 1e-12)
        df_imp_joint["XGB_norm"]  = df_imp_joint["XGB"] / (df_imp_joint["XGB"].sum() + 1e-12)
        df_imp_joint["mean_norm"] = (df_imp_joint["RF_norm"] + df_imp_joint["XGB_norm"]) / 2
        df_imp_joint = df_imp_joint.sort_values("mean_norm", ascending=False).reset_index(drop=True)
        ev.save_feature_importance_comparison(df_imp_joint)

        # Permutation importance
        df_perm_rf, df_perm_xgb = ev.compute_permutation_importance(
            rf, xgb_m, X_val, y_val, FEATURES
        )
        ev.save_permutation_importance_plot(df_perm_rf, df_perm_xgb)

        # SHAP
        df_shap_imp = ev.compute_and_save_shap(xgb_m, X_train, FEATURES)

        # Selección de features
        FEATURES_OPT = ModelEvaluator.select_features(
            FEATURES, df_imp_joint, df_perm_rf, df_perm_xgb,
            df_shap_imp if not df_shap_imp.empty else None,
        )
        print(f"\n  FEATURES_OPT seleccionadas: {len(FEATURES_OPT)}")

    # ── 6. Modelos optimizados ────────────────────────────────────────────────
    if xgb_m is not None and FEATURES_OPT != FEATURES:
        print("\n" + "=" * 60)
        print("PASO 6: Modelos con features optimizadas")
        print("=" * 60)
        X_tr_opt = df_train[FEATURES_OPT]
        X_vl_opt = df_val[FEATURES_OPT]
        X_te_opt = df_test[FEATURES_OPT]

        # RF optimizado
        rf_opt = RandomForestModel(FEATURES_OPT, TARGET,
                                   name=f"RF Opt ({len(FEATURES_OPT)} feat)",
                                   params=RF_PARAMS)
        rf_opt.fit(X_tr_opt, y_train)
        pred_test_rf_opt = rf_opt.predict(X_te_opt)
        res_rf_opt_test  = evaluar(y_test, pred_test_rf_opt,
                                   f"RF Opt ({len(FEATURES_OPT)} feat) - Test")
        res_rf_opt_test["modelo"] = f"RF Opt ({len(FEATURES_OPT)} feat)"
        resultados.append(res_rf_opt_test)
        rf_opt.save_plots(FIGURES_MODELS, y_test, pred_test_rf_opt,
                          prefix=f"14_rf_opt")

        # XGB optimizado
        _valid_xgb_keys = {
            "objective", "n_estimators", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "min_child_weight",
            "reg_alpha", "reg_lambda", "tree_method",
        }
        best_xgb_params = {k: v for k, v in xgb_m.model.get_params().items()
                           if k in _valid_xgb_keys and v is not None}
        xgb_opt_params = {**best_xgb_params, "early_stopping_rounds": 50,
                          "eval_metric": "rmse"}
        xgb_opt = XGBoostModel(FEATURES_OPT, TARGET,
                                name=f"XGB Opt ({len(FEATURES_OPT)} feat)",
                                params=xgb_opt_params)
        xgb_opt.fit(X_tr_opt, y_train, X_val=X_vl_opt, y_val=y_val)
        pred_test_xgb_opt = xgb_opt.predict(X_te_opt)
        res_xgb_opt_test  = evaluar(y_test, pred_test_xgb_opt,
                                    f"XGB Opt ({len(FEATURES_OPT)} feat) - Test")
        res_xgb_opt_test["modelo"] = f"XGB Opt ({len(FEATURES_OPT)} feat)"
        resultados.append(res_xgb_opt_test)
        xgb_opt.save_plots(FIGURES_MODELS, y_test, pred_test_xgb_opt,
                           prefix="14d_xgb_opt")

        # Comparativa original vs optimizado
        ev.save_orig_vs_opt_comparison(
            res_rf_test, res_rf_opt_test,
            res_xgb_test, res_xgb_opt_test,
            n_orig=len(FEATURES), n_opt=len(FEATURES_OPT),
        )

    # ── 7. Modelos estacionales (4 estaciones) ────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 7: Modelos estacionales (4 estaciones individuales)")
    print("=" * 60)
    df_estacional = pd.concat([df_train, df_val, df_test], axis=0).copy()
    df_estacional["estacion"] = df_estacional.index.month.map(_mes_a_estacion)

    ESTACIONES_MOD = ["invierno", "primavera", "verano", "otono"]
    resultados_est = []

    for est in ESTACIONES_MOD:
        df_est = df_estacional[df_estacional["estacion"] == est].copy()
        if len(df_est) < 500:
            print(f"  {est}: datos insuficientes ({len(df_est)} filas)")
            continue

        df_tr_e, df_vl_e, df_te_e = _split_temporal(df_est)
        if min(len(df_tr_e), len(df_vl_e), len(df_te_e)) == 0:
            continue

        X_tr_e, y_tr_e = df_tr_e[FEATURES].values, df_tr_e[TARGET].values
        X_vl_e, y_vl_e = df_vl_e[FEATURES].values, df_vl_e[TARGET].values
        X_te_e, y_te_e = df_te_e[FEATURES].values, df_te_e[TARGET].values

        print(f"\n  --- Estación: {est} ---")

        # XGBoost
        if xgb_m is not None:
            try:
                xgb_e = XGBoostModel(FEATURES, TARGET, name=f"XGBoost - {est}",
                                     params=XGB_SEASONAL_PARAMS)
                xgb_e.fit(pd.DataFrame(X_tr_e, columns=FEATURES), y_tr_e,
                          X_val=pd.DataFrame(X_vl_e, columns=FEATURES), y_val=y_vl_e)
                pred_te = xgb_e.predict(pd.DataFrame(X_te_e, columns=FEATURES))
                res_e   = evaluar(y_te_e, pred_te, f"XGBoost - {est}")
                resultados_est.append(res_e)
                xgb_e.save_model(MODEL_DIR / f"xgboost_{est}.json")
            except Exception as exc:
                print(f"  [WARN] XGBoost {est}: {exc}")

        # Ridge
        sc_e  = RobustScaler()
        X_tr_sc = sc_e.fit_transform(X_tr_e)
        X_vl_sc = sc_e.transform(X_vl_e)
        X_te_sc = sc_e.transform(X_te_e)
        best_r, best_rmse_r = None, np.inf
        from sklearn.linear_model import Ridge as SkRidge
        from sklearn.metrics import mean_squared_error as _mse
        for alpha in RIDGE_ALPHAS:
            m = SkRidge(alpha=alpha)
            m.fit(X_tr_sc, y_tr_e)
            pred = recortar_predicciones(m.predict(X_vl_sc))
            rmse = np.sqrt(_mse(y_vl_e, pred))
            if rmse < best_rmse_r:
                best_rmse_r, best_r = rmse, m
        pred_te = recortar_predicciones(best_r.predict(X_te_sc))
        res_e   = evaluar(y_te_e, pred_te, f"Ridge - {est}")
        resultados_est.append(res_e)

        # Random Forest
        rf_e = RandomForestModel(FEATURES, TARGET, name=f"Random Forest - {est}",
                                 params=RF_SEASONAL_PARAMS)
        rf_e.fit(pd.DataFrame(X_tr_e, columns=FEATURES), y_tr_e)
        pred_te = rf_e.predict(pd.DataFrame(X_te_e, columns=FEATURES))
        res_e   = evaluar(y_te_e, pred_te, f"Random Forest - {est}")
        resultados_est.append(res_e)

        # Prophet
        if prophet_m is not None or ProphetModel is not None:
            try:
                p_e = ProphetModel(FEATURES, TARGET, name=f"Prophet - {est}",
                                   yearly_seasonality=False, n_changepoints=15,
                                   daily_fourier_order=8)
                p_e.fit(df_tr_e, y_tr_e, index_train=df_tr_e.index)
                pred_te = p_e.predict_from_index(df_te_e.index)
                res_e   = evaluar(y_te_e, pred_te, f"Prophet - {est}")
                resultados_est.append(res_e)
                print(f"  Prophet {est} completado.")
            except Exception as exc:
                print(f"  [WARN] Prophet {est}: {exc}")

    df_res_est = (pd.DataFrame(resultados_est).set_index("modelo")
                  if resultados_est else pd.DataFrame())

    if not df_res_est.empty:
        print("\n=== RESULTADOS ESTACIONALES ===")
        print(df_res_est.round(4).to_string())

        NOMBRES_EST = {
            "invierno":  "Invierno",
            "primavera": "Primavera",
            "verano":    "Verano",
            "otono":     "Otoño",
        }
        ev.save_seasonal_comparison(
            df_resultados, df_res_est,
            estaciones=ESTACIONES_MOD,
            nombres_est=NOMBRES_EST,
        )
        ev.save_ranking_heatmap(
            filename="06b_ranking_estacionales.png",
            title="Ranking modelos estacionales — métricas normalizadas\n(verde = mejor)",
        )

    # ── 8. Grupos climáticos (A y B) ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 8: Grupos climáticos (Inv+Oto vs Pri+Ver)")
    print("=" * 60)
    df_grupo_a = df_estacional[df_estacional["estacion"].isin(GRUPO_A_ESTACIONES)].copy()
    df_grupo_b = df_estacional[df_estacional["estacion"].isin(GRUPO_B_ESTACIONES)].copy()

    # Figura de series por grupo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax, df_g, titulo, color in zip(
        axes,
        [df_grupo_a, df_grupo_b],
        ["Grupo A — Invierno + Otoño", "Grupo B — Primavera + Verano"],
        ["#4e9af1", "#f4a261"],
    ):
        ax.plot(df_g.index, df_g[TARGET], color=color, linewidth=0.5, alpha=0.8)
        ax.set_title(titulo, fontsize=11)
        ax.set_ylabel("Energía HVAC (kWh)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    fig.suptitle("Serie temporal del target por grupo estacional", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_MODELS / "16_series_grupos_estacionales.png", bbox_inches="tight")
    plt.close(fig)

    resultados_grupos = []
    modelos_grupos    = {}

    for nombre_grp, df_grp in [("Grupo A (Inv+Oto)", df_grupo_a),
                                ("Grupo B (Pri+Ver)", df_grupo_b)]:
        print(f"\n  --- {nombre_grp} ---")
        res_grp, res_list = _entrenar_grupo(
            df_grp, nombre_grp, FEATURES, TARGET, ev, FIGURES_MODELS
        )
        modelos_grupos[nombre_grp] = res_grp
        resultados_grupos.extend(res_list)

    df_res_grupos = (pd.DataFrame(resultados_grupos).set_index("modelo")
                     if resultados_grupos else pd.DataFrame())

    if not df_res_grupos.empty:
        print("\n=== RESULTADOS GRUPOS ESTACIONALES ===")
        print(df_res_grupos[["MAE", "RMSE", "R2", "WAPE", "sMAPE"]].round(4).to_string())
        ev.save_seasonal_approaches_comparison(df_res_est, df_res_grupos)

    # ── 9. Ranking final ampliado ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 9: Ranking final ampliado")
    print("=" * 60)

    # Actualizar df_resultados con los modelos optimizados
    df_resultados_final = (
        pd.DataFrame(resultados)
        .drop_duplicates(subset="modelo", keep="last")
        .set_index("modelo")
    )

    df_final_ampliado = ev.save_extended_ranking_heatmap(
        df_globales=df_resultados_final,
        df_estacionales=df_res_est,
        df_grupos=df_res_grupos,
    )

    print("\n=== RANKING FINAL AMPLIADO (ordenado por RMSE) ===")
    cols_m = [c for c in ["MAE", "RMSE", "R2", "WAPE", "sMAPE"] if c in df_final_ampliado.columns]
    print(df_final_ampliado[cols_m].round(4).to_string())

    # Exportar métricas
    df_final_ampliado.round(4).to_csv(MODEL_DIR / "resultados_metricas.csv")
    print(f"\n  Métricas guardadas: {MODEL_DIR / 'resultados_metricas.csv'}")

    ganador = df_final_ampliado["RMSE"].idxmin()
    print(f"\n  MEJOR MODELO: {ganador}")
    print(f"    RMSE: {df_final_ampliado.loc[ganador, 'RMSE']:.4f} kWh")
    print(f"    R2:   {df_final_ampliado.loc[ganador, 'R2']:.4f}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"  Tiempo total: {elapsed:.0f} s  ({elapsed/60:.1f} min)")
    print(f"  Modelos guardados en:  {MODEL_DIR}")
    print(f"  Figuras guardadas en:  {FIGURES_MODELS}")
    print(f"  Modelos globales evaluados: {len(df_resultados_final)}")
    print(f"  Modelos estacionales:       {len(df_res_est)}")
    print(f"  Modelos de grupos:          {len(df_res_grupos)}")


if __name__ == "__main__":
    main()
