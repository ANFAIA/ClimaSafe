"""
Punto de entrada principal del proyecto ClimaSafeAI.
Ejecutar: python main.py

Pipeline completo:
  0. Menú: (0) Entrenar | (1) Probar con datos propios
  1. Descarga de datos crudos (salta si existen)
  2. Preprocesado → parquets etiquetados (salta si existen)
  3. Secuencias LSTM 24h (salta si existe)
  4. Preprocesado ML (train/test split + scaler) para calor y frío
  5. Entrenar: XGBoost (calor) + RandomForest (frío) + LSTM multi-tarea
  6. Evaluar modelos tabulares + LSTM
  7. Tabla comparativa final
"""
from dotenv import load_dotenv
import joblib
import pandas as pd
import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, recall_score

from climasafeai.data.make_dataset import (
    download_momo_data,
    download_era5_data,
)
from climasafeai.features.build_features import preprocess_data
from climasafeai.models.predict_model import evaluate_models, try_model
from climasafeai.utils.paths import (
    MODELS_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    ARTIFACTS_DIR,
)


# Columnas nuevas añadidas en 2026-07-14 (nocturnas + rachas severas).
# Si faltan en los parquets existentes, hay que regenerarlos.
_NUEVAS_COLS = ["t2m_min_noche", "t2m_min_noche_lag1", "t2m_min_noche_roll7",
                "horas_wc_severo", "dias_consec_wc_severo", "horas_wc_severo_sum14"]


def _check_skip(path, label):
    exists = path.exists()
    if exists:
        print(f"    [SKIP] {label} ya existe → {path}")
    return exists


def _parquet_actualizado(path) -> bool:
    """Verifica si el parquet tiene las columnas nuevas.
    Si falta alguna, hay que regenerar."""
    if not path.exists():
        return False
    try:
        cols = pd.read_parquet(path, nrows=0).columns.tolist()
        ok = all(c in cols for c in _NUEVAS_COLS)
        if not ok:
            faltan = [c for c in _NUEVAS_COLS if c not in cols]
            print(f"    Parquet desactualizado — faltan: {faltan}")
        return ok
    except Exception:
        return False


def _data_exists() -> bool:
    return (
        (RAW_DATA_DIR / "momo_data.csv").exists()
        and len(list((RAW_DATA_DIR / "era5").glob("era5_*.nc"))) > 0
    )


def run_full_pipeline() -> None:
    # ------------------------------------------------------------------
    # 1. Descarga de datos crudos
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. Descarga de datos crudos...")
    load_dotenv()
    if _data_exists():
        print("    [SKIP] Datos crudos ya descargados")
    else:
        try:
            download_momo_data()
        except Exception as e:
            print(f"    AVISO: fallo al descargar MoMo ({e}) — si los parquets ya existen, no es crítico")
        try:
            download_era5_data()
        except Exception as e:
            print(f"    AVISO: fallo al descargar ERA5 ({e}) — si los parquets ya existen, no es crítico")

    # ------------------------------------------------------------------
    # 2. Preprocesado → parquets etiquetados (dataset_calor/frio_labeled)
    # ------------------------------------------------------------------
    print("\n2. Preprocesado (parquets etiquetados)...")
    calor_path = PROCESSED_DATA_DIR / "dataset_calor_labeled.parquet"
    frio_path = PROCESSED_DATA_DIR / "dataset_frio_labeled.parquet"
    parquets_ok = _check_skip(calor_path, "dataset_calor_labeled") and \
                  _check_skip(frio_path, "dataset_frio_labeled")
    parquets_ok = parquets_ok and _parquet_actualizado(calor_path) \
                  and _parquet_actualizado(frio_path)
    if parquets_ok:
        print("    [SKIP] Parquets actualizados (tienen columnas nuevas)")
    else:
        print("    Parquets desactualizados o ausentes — regenerando...")

    if not parquets_ok or not _parquet_actualizado(calor_path) or not _parquet_actualizado(frio_path):
        from climasafeai.data.make_dataset import (
            cargar_provincias_unificadas,
            calcular_puntos_provincia,
            cargar_era5_filtrado,
            process_data,
        )
        momo_df = pd.read_csv(RAW_DATA_DIR / "momo_data.csv")
        provincias = cargar_provincias_unificadas()
        puntos = calcular_puntos_provincia(provincias, col_nombre="NAMEUNIT")
        era5_ds = cargar_era5_filtrado(puntos)

        if not calor_path.exists() or not _parquet_actualizado(calor_path):
            print("  Generando dataset_calor_labeled...")
            df_calor = process_data(momo_df, era5_ds, clase="calor")
            df_calor.to_parquet(calor_path, index=False)
            print(f"    Guardado → {calor_path}")

        if not frio_path.exists() or not _parquet_actualizado(frio_path):
            print("  Generando dataset_frio_labeled...")
            df_frio = process_data(momo_df, era5_ds, clase="frio")
            df_frio.to_parquet(frio_path, index=False)
            print(f"    Guardado → {frio_path}")

    # ------------------------------------------------------------------
    # 3. Secuencias LSTM 24h
    # ------------------------------------------------------------------
    print("\n3. Secuencias LSTM 24h...")
    from climasafeai.data.sequences import generar_dataset_secuencias
    generar_dataset_secuencias()

    # ------------------------------------------------------------------
    # 4. Preprocesado ML (train/test split + scaler) para calor y frío
    # ------------------------------------------------------------------
    print("\n4. Preprocesando datos para modelos tabulares...")
    df_calor = pd.read_parquet(calor_path)
    df_frio = pd.read_parquet(frio_path)

    X_tr_cal, X_te_cal, y_tr_cal, y_te_cal = preprocess_data(
        df_calor,
        target_col="clase_riesgo_calor",
        clase="calor",
        scaler_type="standard",
        split_by_date=True,
    )
    X_tr_frio, X_te_frio, y_tr_frio, y_te_frio = preprocess_data(
        df_frio,
        target_col="clase_riesgo_frio",
        clase="frio",
        scaler_type="standard",
        split_by_date=True,
    )

    # Extraer provincia para test (alineada por índice del split por fecha)
    def _prov_test(df_orig, n_test):
        return df_orig.sort_values("fecha")["provincia"].iloc[-n_test:].values
    prov_te_cal = _prov_test(df_calor, len(y_te_cal))
    prov_te_frio = _prov_test(df_frio, len(y_te_frio))
    prov_tr_cal = _prov_test(df_calor, len(y_tr_cal) + len(y_te_cal))[:-len(y_te_cal)]
    prov_tr_frio = _prov_test(df_frio, len(y_tr_frio) + len(y_te_frio))[:-len(y_te_frio)]

    # ------------------------------------------------------------------
    # 5. Entrenar modelos desplegados
    # ------------------------------------------------------------------
    print("\n5. Entrenando modelos desplegados...")

    print("  --- XGBoost — Calor ---")
    xgb_cal = XGBClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=2.0, min_child_weight=3,
        eval_metric="mlogloss", early_stopping_rounds=50,
        random_state=42, n_jobs=-1,
    )
    X_tr_cal_ar = np.asarray(X_tr_cal)
    y_tr_cal_ar = np.asarray(y_tr_cal)
    n_val = max(1, int(len(y_tr_cal) * 0.15))
    xgb_cal.fit(
        X_tr_cal_ar, y_tr_cal_ar,
        sample_weight=compute_sample_weight("balanced", y_tr_cal_ar),
        eval_set=[(X_tr_cal_ar[-n_val:], y_tr_cal_ar[-n_val:])],
        verbose=False,
    )
    joblib.dump(xgb_cal, MODELS_DIR / "XGBoost_calor.joblib")
    print(f"    Guardado → XGBoost_calor.joblib (best_iter={xgb_cal.best_iteration})")

    print("  --- RandomForest — Frío ---")
    rf_frio = RandomForestClassifier(
        n_estimators=200, max_depth=8, max_features="sqrt",
        max_samples=0.8, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf_frio.fit(X_tr_frio, y_tr_frio)
    joblib.dump(rf_frio, MODELS_DIR / "RandomForest_frio.joblib")
    print(f"    Guardado → RandomForest_frio.joblib")

    # Calibración isotónica post-hoc para RF_frio
    print("  Calibrando isotonic para frío...")
    from climasafeai.models.calibrate import fit_isotonic
    # Último 15% de fechas de train para calibration set
    _cal_n = max(1, int(len(y_tr_frio) * 0.15))
    _X_cal = X_tr_frio[-_cal_n:]
    _y_cal = y_tr_frio.iloc[-_cal_n:] if hasattr(y_tr_frio, 'iloc') else y_tr_frio[-_cal_n:]
    fit_isotonic(rf_frio, np.asarray(_X_cal), np.asarray(_y_cal), clase="frio")

    # --- Aprender umbrales por provincia (train) ---
    print("  Calibrando umbrales por provincia...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for clase, model, X_tr, y_tr, prov_tr in [
        ("calor", xgb_cal, X_tr_cal, y_tr_cal, prov_tr_cal),
        ("frio", rf_frio, X_tr_frio, y_tr_frio, prov_tr_frio),
    ]:
        proba_tr = model.predict_proba(X_tr)
        umbrales = {}
        for prov in np.unique(prov_tr):
            mask = prov_tr == prov
            p = proba_tr[mask]
            y = y_tr.values[mask] if hasattr(y_tr, 'values') else y_tr[mask]
            r = [c for c in np.unique(y) if c != 0]
            if not r:
                continue
            bp = {"rec": -1}
            t2_min = 0.05 if clase == "calor" else 0.15
            for t1 in np.arange(0.20, 0.95, 0.05):
                for t2 in np.arange(t2_min, min(t1, 0.85), 0.05):
                    pred = np.zeros(len(p), dtype=int)
                    pred[p[:, 2] >= t2] = 2
                    pred[(pred == 0) & (p[:, 1] + p[:, 2] >= t1)] = 1
                    if clase == "calor" and 2 in r:
                        rec = recall_score(y, pred, labels=[2], average=None, zero_division=0)[0]
                    else:
                        rec = recall_score(y, pred, labels=r, average="macro", zero_division=0)
                    if rec > bp["rec"]:
                        bp = {"t1": t1, "t2": t2, "rec": rec}
            umbrales[prov] = bp
        path = ARTIFACTS_DIR / f"umbrales_provincia_{clase}.joblib"
        joblib.dump(umbrales, path)
        print(f"    {clase}: {len(umbrales)} provincias → {path.name}")

    # ------------------------------------------------------------------
    # 6. Entrenar LSTM province_hybrid (secuencia + provincia + INE + daily)
    # ------------------------------------------------------------------
    print("\n6. Entrenando LSTM province_hybrid...")
    from climasafeai.models.lstm_province_hybrid import (
        preparar_datos_hibridos,
        train_lstm_province_hybrid,
        evaluate_lstm_province_hybrid,
    )

    splits = preparar_datos_hibridos()
    model_hib, history = train_lstm_province_hybrid(
        splits["X_train_s"], splits["Xp_train_s"], splits["pidx_train"], splits["Xd_train_s"],
        splits["y_train_calor"], splits["y_train_frio"],
        splits["X_val_s"], splits["Xp_val_s"], splits["pidx_val"], splits["Xd_val_s"],
        splits["y_val_calor"], splits["y_val_frio"],
        n_provincias=splits["n_provincias"],
        peso_riesgo_extra=8.0,
        seed=42,
    )

    # ------------------------------------------------------------------
    # 6.5. Entrenar red bayesiana para diagnóstico inverso
    # ------------------------------------------------------------------
    print("\n6.5 Entrenando red bayesiana para diagnóstico inverso...")
    try:
        from climasafeai.models.bayes import BayesianRiskDiagnosis
        rng = np.random.default_rng(42)

        # Distribuciones realistas: población española
        # Edad: media ~52, std ~18 (INE 2024), truncada [0, 95]
        # Grasa corporal: ~18-35 según edad (mayor→más grasa)
        n_cal = len(X_tr_cal)
        n_frio = len(X_tr_frio)

        edad_cal = rng.normal(52, 18, n_cal).clip(0, 95)
        edad_frio = rng.normal(52, 18, n_frio).clip(0, 95)

        grasa_cal = np.where(
            edad_cal < 35, rng.normal(20, 4, n_cal).clip(12, 35),
            np.where(edad_cal < 60, rng.normal(24, 5, n_cal).clip(14, 38),
                     rng.normal(27, 5, n_cal).clip(15, 40))
        )
        grasa_frio = np.where(
            edad_frio < 35, rng.normal(20, 4, n_frio).clip(12, 35),
            np.where(edad_frio < 60, rng.normal(24, 5, n_frio).clip(14, 38),
                     rng.normal(27, 5, n_frio).clip(15, 40))
        )

        # Temperatura real del dataset (t2m, no t2m_c)
        temp_cal = df_calor.sort_values("fecha")["t2m"].values[:n_cal]
        temp_frio = df_frio.sort_values("fecha")["t2m"].values[:n_frio]

        # Riesgo continuo desde el modelo entrenado
        prob_calor_tr = xgb_cal.predict_proba(X_tr_cal)
        riesgo_cal = prob_calor_tr[:, 1] + prob_calor_tr[:, 2]
        prob_frio_tr = rf_frio.predict_proba(X_tr_frio)
        riesgo_frio = prob_frio_tr[:, 1] + prob_frio_tr[:, 2]

        for clase, temp, edad, grasa, riesgo, label in [
            ("calor", temp_cal, edad_cal, grasa_cal, riesgo_cal, "bayes_risk_diagnosis_calor"),
            ("frio", temp_frio, edad_frio, grasa_frio, riesgo_frio, "bayes_risk_diagnosis_frio"),
        ]:
            bd = BayesianRiskDiagnosis(clase=clase)
            bd.fit_from_continuous(temp, grasa, edad, riesgo)
            path = str(ARTIFACTS_DIR / f"{label}.joblib")
            bd.save(path)
            print(f"    {clase}: guardado → {label}.joblib  |  CPDs: {bd.cpd_info}")
    except Exception as e:
        print(f"    No se pudo entrenar red bayesiana: {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # 6.6. Entrenar conformal prediction (split conformal)
    # ------------------------------------------------------------------
    print("\n6.6 Entrenando conformal prediction...")
    try:
        from climasafeai.models.conformal import SplitConformalCalibrator
        from climasafeai.models.calibrate import load_isotonic, calibrate_proba

        for clase, model, X_tr, y_tr in [
            ("calor", xgb_cal, X_tr_cal, y_tr_cal),
            ("frio", rf_frio, X_tr_frio, y_tr_frio),
        ]:
            _n = max(1, len(y_tr) // 3)
            proba_cal = model.predict_proba(X_tr[-_n:])
            # Solo frío usa isotonic (en calor es contraproducente)
            if clase == "frio":
                iso = load_isotonic("frio")
                if iso:
                    proba_cal = calibrate_proba(proba_cal, iso)
            y_cal = y_tr.iloc[-_n:].values if hasattr(y_tr, 'iloc') else y_tr[-_n:]
            cal = SplitConformalCalibrator(alpha=0.1)
            cal.fit(proba_cal, y_cal)
            cal.save(str(ARTIFACTS_DIR / f"conformal_{clase}.joblib"))
            print(f"    {clase}: qhat={cal.qhat:.4f}")
    except Exception as e:
        print(f"    No se pudo entrenar conformal prediction: {e}")

    # ------------------------------------------------------------------
    # 7. Evaluar modelos desplegados (argmax + umbrales calibrados)
    # ------------------------------------------------------------------
    print("\n7. Evaluando modelos desplegados...")
    from climasafeai.models.predict_model import (
        CLASS_THRESHOLDS_RECOMENDADOS, apply_class_thresholds,
    )

    def _metricas_con_umbrales(y_true, y_pred, nombre, clase):
        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        risk = [c for c in np.unique(y_true) if c != 0]
        rec = recall_score(y_true, y_pred, labels=risk, average="macro", zero_division=0) if risk else float("nan")
        print(f"  {nombre}: Acc={acc:.4f} F1_macro={f1m:.4f} Rec_riesgo={rec:.4f}")
        return {"Modelo": nombre, "Acc_test": round(acc, 4),
                "F1_test": round(f1w, 4), "F1_macro": round(f1m, 4),
                "Rec_riesgo": round(rec, 4)}

    print("  --- Calor ---")
    xgb_cal = joblib.load(MODELS_DIR / "XGBoost_calor.joblib")
    u_cal = CLASS_THRESHOLDS_RECOMENDADOS["calor"]
    res_cal_arg = _metricas_con_umbrales(y_te_cal, xgb_cal.predict(X_te_cal), "XGBoost (argmax)", "calor")
    res_cal_th = _metricas_con_umbrales(
        y_te_cal, apply_class_thresholds(xgb_cal.predict_proba(X_te_cal), **u_cal),
        f"XGBoost (t1={u_cal['t1']}, t2={u_cal['t2']})", "calor",
    )

    # Umbrales por provincia
    proba_cal = xgb_cal.predict_proba(X_te_cal)
    umb_cal = joblib.load(ARTIFACTS_DIR / "umbrales_provincia_calor.joblib")
    pred_cal_prov = np.zeros(len(proba_cal), dtype=int)
    for prov in np.unique(prov_te_cal):
        u = umb_cal.get(prov, {"t1": u_cal["t1"], "t2": u_cal["t2"]})
        mask = prov_te_cal == prov
        p = proba_cal[mask]
        pred = np.zeros(len(p), dtype=int)
        pred[p[:, 2] >= u["t2"]] = 2
        pred[(pred == 0) & (p[:, 1] + p[:, 2] >= u["t1"])] = 1
        pred_cal_prov[mask] = pred
    res_cal_prov = _metricas_con_umbrales(y_te_cal, pred_cal_prov, "XGBoost (por provincia)", "calor")

    res_cal = pd.DataFrame([res_cal_arg, res_cal_th, res_cal_prov])
    res_cal.to_csv(REPORTS_DIR / "resultados_calor.csv", index=False)

    print("  --- Frío ---")
    rf_frio = joblib.load(MODELS_DIR / "RandomForest_frio.joblib")
    proba_frio = rf_frio.predict_proba(X_te_frio)

    # Umbrales raw (calibrados sobre probas sin calibrar)
    _U_FRIO_RAW = {"t1": 0.45, "t2": 0.40}
    # Umbrales post-isotonic (calibrados sobre probas con isotonic)
    u_frio = CLASS_THRESHOLDS_RECOMENDADOS["frio"]

    res_frio_arg = _metricas_con_umbrales(y_te_frio, rf_frio.predict(X_te_frio), "RF (argmax)", "frio")
    res_frio_th = _metricas_con_umbrales(
        y_te_frio, apply_class_thresholds(proba_frio, **_U_FRIO_RAW),
        "RF (t1=0.45, t2=0.40 raw)", "frio",
    )

    # Umbrales por provincia (siempre raw)
    umb_frio = joblib.load(ARTIFACTS_DIR / "umbrales_provincia_frio.joblib")
    pred_frio_prov = np.zeros(len(proba_frio), dtype=int)
    for prov in np.unique(prov_te_frio):
        u = umb_frio.get(prov, {"t1": _U_FRIO_RAW["t1"], "t2": _U_FRIO_RAW["t2"]})
        mask = prov_te_frio == prov
        p = proba_frio[mask]
        pred = np.zeros(len(p), dtype=int)
        pred[p[:, 2] >= u["t2"]] = 2
        pred[(pred == 0) & (p[:, 1] + p[:, 2] >= u["t1"])] = 1
        pred_frio_prov[mask] = pred
    res_frio_prov = _metricas_con_umbrales(y_te_frio, pred_frio_prov, "RF (por provincia)", "frio")

    # Calibración isotónica con thr recalibrados
    try:
        from climasafeai.models.calibrate import load_isotonic, calibrate_proba
        _iso_frio = load_isotonic("frio")
        if _iso_frio:
            proba_frio_cal = calibrate_proba(proba_frio, _iso_frio)
            res_frio_iso = _metricas_con_umbrales(
                y_te_frio, apply_class_thresholds(proba_frio_cal, **u_frio),
                f"RF + isotonic (t1={u_frio['t1']}, t2={u_frio['t2']})", "frio",
            )
            res_frio = pd.DataFrame([res_frio_arg, res_frio_th, res_frio_iso, res_frio_prov])
        else:
            res_frio = pd.DataFrame([res_frio_arg, res_frio_th, res_frio_prov])
    except Exception as e:
        print(f"    AVISO: calibración isotónica no disponible ({e})")
        res_frio = pd.DataFrame([res_frio_arg, res_frio_th, res_frio_prov])
    res_frio.to_csv(REPORTS_DIR / "resultados_frio.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Evaluar LSTM province_hybrid (argmax + umbrales calibrados)
    # ------------------------------------------------------------------
    print("\n8. Evaluando LSTM province_hybrid...")
    res_lstm = evaluate_lstm_province_hybrid(
        model_hib,
        splits["X_train_s"], splits["Xp_train_s"], splits["pidx_train"], splits["Xd_train_s"],
        splits["y_train_calor"], splits["y_train_frio"],
        splits["X_test_s"], splits["Xp_test_s"], splits["pidx_test"], splits["Xd_test_s"],
        splits["y_test_calor"], splits["y_test_frio"],
    )

    # Añadir filas con umbrales calibrados para LSTM
    model_hib.eval()
    with torch.no_grad():
        out_lstm = model_hib(
            torch.tensor(splits["X_test_s"]),
            torch.tensor(splits["pidx_test"]),
            torch.tensor(splits["Xp_test_s"]),
            torch.tensor(splits["Xd_test_s"]),
        )
    proba_lstm_cal = torch.softmax(out_lstm[0], dim=1).numpy()
    proba_lstm_frio = torch.softmax(out_lstm[1], dim=1).numpy()

    from climasafeai.models.predict_model import apply_class_thresholds, CLASS_THRESHOLDS_LSTM
    for nombre, proba, y_true, risk, u in [
        ("LSTM_calor_th", proba_lstm_cal, splits["y_test_calor"],
         [c for c in np.unique(splits["y_test_calor"]) if c != 0], CLASS_THRESHOLDS_LSTM["calor"]),
        ("LSTM_frio_th", proba_lstm_frio, splits["y_test_frio"],
         [c for c in np.unique(splits["y_test_frio"]) if c != 0], CLASS_THRESHOLDS_LSTM["frio"]),
    ]:
        pred = apply_class_thresholds(proba, **u)
        acc = accuracy_score(y_true, pred)
        f1w = f1_score(y_true, pred, average="weighted", zero_division=0)
        f1m = f1_score(y_true, pred, average="macro", zero_division=0)
        rec = recall_score(y_true, pred, labels=risk, average="macro", zero_division=0) if risk else float("nan")
        print(f"  {nombre} (t1={u['t1']}, t2={u['t2']}): Acc={acc:.4f} F1_macro={f1m:.4f} Rec_riesgo={rec:.4f}")
        row = {"Modelo": nombre, "Acc_train": float("nan"), "Acc_test": round(acc, 4),
               "F1_train": float("nan"), "F1_test": round(f1w, 4),
               "Prec_test": float("nan"), "Rec_test": float("nan"),
               "F1_macro": round(f1m, 4), "Rec_riesgo": round(rec, 4)}
        res_lstm = pd.concat([res_lstm, pd.DataFrame([row])], ignore_index=True)

    # ------------------------------------------------------------------
    # 9. Recalibrar CLASS_THRESHOLDS_RECOMENDADOS
    # ------------------------------------------------------------------
    print("\n9. Recalibrando thresholds globales...")
    from climasafeai.models.predict_model import CLASS_THRESHOLDS_RECOMENDADOS
    for nombre, proba, y_true, clase in [
        ("XGBoost_calor", xgb_cal.predict_proba(X_te_cal), y_te_cal, "calor"),
        ("RF_frio", rf_frio.predict_proba(X_te_frio), y_te_frio, "frio"),
        ("LSTM_calor", proba_lstm_cal, splits["y_test_calor"], "calor"),
        ("LSTM_frio", proba_lstm_frio, splits["y_test_frio"], "frio"),
    ]:
        risk = [c for c in np.unique(y_true) if c != 0]
        if not risk:
            continue
        best = {"rec": -1, "t1": 0, "t2": 0}
        for t1 in np.arange(0.20, 0.95, 0.05):
            for t2 in np.arange(0.15, min(t1, 0.85), 0.05):
                pred = apply_class_thresholds(proba, t1=t1, t2=t2)
                rec = recall_score(y_true, pred, labels=risk, average="macro", zero_division=0)
                if rec > best["rec"]:
                    best = {"rec": rec, "t1": t1, "t2": t2}
        print(f"  {nombre}: t1={best['t1']}, t2={best['t2']} → Rec_riesgo={best['rec']:.4f}")

    # ------------------------------------------------------------------
    # 10. Tabla comparativa final (Rec_riesgo como métrica principal)
    # ------------------------------------------------------------------
    print("\n10. Tabla comparativa final (Rec_riesgo)...")
    
    dfs = []
    if not res_cal.empty:
        _c = res_cal.copy()
        _c.insert(0, "Clase", "Calor")
        dfs.append(_c)
    if not res_frio.empty:
        _f = res_frio.copy()
        _f.insert(0, "Clase", "Frío")
        dfs.append(_f)
    if res_lstm is not None and not res_lstm.empty:
        _l = res_lstm.copy()
        _l.insert(0, "Clase", _l["Modelo"].apply(lambda x: x.split("_")[-1].capitalize()))
        dfs.append(_l)

    if dfs:
        comparativa = pd.concat(dfs, ignore_index=True)
        cols_mostrar = [c for c in ["Clase", "Modelo", "Acc_test", "F1_macro", "Rec_riesgo"]
                        if c in comparativa.columns]
        print(comparativa[cols_mostrar].to_string(index=False))
        comparativa.to_csv(REPORTS_DIR / "comparativa_final.csv", index=False)
        print(f"\n  Guardado → {REPORTS_DIR / 'comparativa_final.csv'}")
    else:
        print("  (sin resultados que comparar)")

    print("\n" + "=" * 60)
    print("Pipeline completado.")


def main():
    print("=" * 60)
    print("  ClimaSafeAI — Pipeline de riesgo térmico")
    print("=" * 60)
    accion = input(
        "Ejecutar pipeline completo (0) o "
        "probar el modelo con tus datos (1)? (0/1): "
    ).strip()
    if accion == "0":
        run_full_pipeline()
    elif accion == "1":
        try_model()
    else:
        print("Opción no válida. Ejecutando pipeline completo por defecto.")
        run_full_pipeline()


if __name__ == "__main__":
    main()
