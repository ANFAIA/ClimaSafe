"""
Punto de entrada principal del proyecto.
Ejecutar: python main.py
"""

from climasafeai.data.make_dataset import load_data
from climasafeai.features.build_features import preprocess_data
from climasafeai.models.train_model import train_models
from climasafeai.models.predict_model import evaluate_models, DECISION_THRESHOLD, try_model
from climasafeai.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_class_balance,
    plot_categorical_vs_target,
    plot_feature_importance,
    plot_pca_variance,
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
DATA_FILE    = 'dataset.csv'
TARGET_COL   = 'target'
SCALER_TYPE  = 'standard'   # 'standard' | 'minmax'
TEST_SIZE    = 0.2

THRESHOLD    = DECISION_THRESHOLD

# PCA opcional: reducción de dimensionalidad antes del modelado.
# None → sin PCA | 0.95 → conservar 95% varianza | int → nº componentes fijo
USE_PCA      = None   # ← ajusta: None | 0.95 | 10

# Optuna: número de trials por modelo. Más trials → mejor resultado pero más tiempo.
OPTUNA_TRIALS = 50



def run_full_pipeline() -> None:
    print('=' * 60)
    print('1. Cargando datos...')
    df = load_data(DATA_FILE)
    print(f'   Shape: {df.shape}')

    print('\n2. EDA visual...')
    plot_distributions(df, target_col=TARGET_COL)
    plot_correlation_matrix(df)
    plot_class_balance(df, target_col=TARGET_COL)
    plot_categorical_vs_target(df, target_col=TARGET_COL)

    print('\n3. Preprocesando...')
    X_train, X_test, y_train, y_test = preprocess_data(
        df, target_col=TARGET_COL, scaler_type=SCALER_TYPE,
        test_size=TEST_SIZE, use_pca=USE_PCA,
    )


    print('\n4. Optimizando hiperparámetros con Optuna...')
    from tuning.tune_model import tune_models as _tune
    _tune(X_train, y_train, n_trials=OPTUNA_TRIALS)

    print('\n5. Entrenando modelos con mejores params...')

    models = train_models(X_train, y_train, tune_knn=True, cv_evaluate=True)

    print('\n5. Evaluando...')

    df_results = evaluate_models(
        models, X_train, y_train, X_test, y_test, threshold=THRESHOLD
    )


    from climasafeai.utils.paths import PROCESSED_DATA_DIR
    import pandas as pd
    try:
        feature_names = pd.read_csv(PROCESSED_DATA_DIR / 'X_train.csv').columns.tolist()
    except FileNotFoundError:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]


    print('\n6. SHAP — explicabilidad de modelos...')
    from climasafeai.models.predict_model import explain_models
    explain_models(models, X_train, feature_names=feature_names)

    print('\n7. Importancia de variables...')

    plot_feature_importance(models, feature_names)

    if USE_PCA is not None:

        print('\n8. Varianza explicada por PCA...')

        import joblib
        from climasafeai.utils.paths import ARTIFACTS_DIR
        try:
            pca = joblib.load(ARTIFACTS_DIR / 'pca.joblib')
            plot_pca_variance(pca)
        except FileNotFoundError:
            pass

    print('\n' + '=' * 60)
    print('Pipeline completado.')

    best = df_results.sort_values('Acc_test', ascending=False).iloc[0]

    print(f'Mejor modelo: {best.to_dict()}')


def main():
    print('=' * 60)
    accion = input('Ejecutar pipeline completo (0) o probar el modelo con tus datos (1)? (0/1): ').strip()
    if accion == '0':
        run_full_pipeline()

    elif accion == '1':
        try_model()

    else:
        print('Opción no válida. Ejecutando pipeline completo por defecto.')
        run_full_pipeline()


if __name__ == '__main__':
    main()

