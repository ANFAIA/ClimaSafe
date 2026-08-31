"""AutoML básico: selecciona el mejor modelo usando validación temporal expansiva."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import joblib
from climasafeai.features.build_features import preprocess_data
from climasafeai.models.xgboost_calor import train_xgboost_calor
from climasafeai.models.randomforest_frio import train_randomforest_frio
from climasafeai.models.lstm_province_hybrid import train_lstm_province_hybrid

def automl_temporal(X, y, n_splits=3):
    """Validación temporal expansiva."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}
    
    # XGBoost calor
    print("Probando XGBoost (calor)...")
    X_calor = X.select_dtypes(include=[np.number]).dropna(axis=0)
    y_calor = y.loc[X_calor.index]
    xgb_scores = []
    for train_idx, test_idx in tscv.split(X_calor):
        X_train, X_test = X_calor.iloc[train_idx], X_cilnor.iloc[test_idx]
        y_train, y_test = y_calor.iloc[train_idx], y_cilnor.iloc[test_idx]
        model = train_xgboost_calor(X_train, y_train)
        preds = model.predict(X_test)
        score = f1_score(y_test, preds, average='weighted')
        xgb_scores.append(score)
    results['xgboost_calor'] = np.mean(xgb_scores)
    print(f"  F1 medio: {results['xgboost_calor']:.4f}")
    
    # RandomForest frio
    print("Probando RandomForest (frio)...")
    # ... similar
    
    # LSTM
    # ... similar
    
    # Seleccionar mejor
    best = max(results, key=results.get)
    return results, best

if __name__ == "__main__":
    import sys
    from climasafeai.data.make_dataset import cargar_provincias_unificadas, cargar_era5_filtrado
    from climasafeai.features.build_features import process_data
    
    provincias = cargar_provincias_unificadas()
    era5_ds = cargar_era5_filtrado(provincias)
    momo_df = pd.read_csv("data/raw/momo_data.csv")
    
    df = process_data(momo_df, era5_ds, clase="calor")
    X = df.drop(columns=["riesgo"])
    y = df["riesgo"]
    
    results, best = automl_temporal(X, y, n_splits=3)
    print(f"\nMejor modelo: {best}")
    print(f"Resultados: {results}")
