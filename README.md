# ClimaSafe

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![ML Type](https://img.shields.io/badge/ML-Random%20Forest-orange)
![Tracking](https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue?logo=mlflow)
![Version](https://img.shields.io/badge/Version-0.0.1-green)
![Author](https://img.shields.io/badge/Author-Alejandro%20Cancelas%20Chapela-blueviolet)
![Template](https://img.shields.io/badge/Generado%20con-dskit-58a6ff?logo=github)

> Prediccion de riesgo climatico personalizado (calor/frio/uv) con ML

**Tipo de ML:** `supervisado`  
**Autor:** Alejandro Cancelas Chapela  
**Versión:** 0.0.1 · RandomForest


ClimaSafe predice el nivel de riesgo climático (calor, frío y radiación UV)
a partir de variables meteorológicas y de ubicación, con el objetivo de anticipar
condiciones potencialmente peligrosas para las personas antes de que ocurran.
El modelo (Random Forest, clasificación supervisada) se entrena con datos históricos
de clima y devuelve una categoría de riesgo interpretable, pensada para
integrarse en alertas o recomendaciones personalizadas.

---

## Estructura del proyecto

```
climasafeai/
├── data/
│   ├── raw/            ← datos originales (nunca modificar)
│   ├── interim/        ← datos en proceso
│   └── processed/      ← datos listos para modelar
├── models/             ← modelos entrenados (.joblib / .pt)
│   └── artifacts/      ← encoders, scalers, etc.
├── notebooks/
│   ├── 0-0-...-Descargadatos.ipynb
│   ├── 0-1-...-ProcesamientoDatos.ipynb
│   └── 0-2-...-Ejecucion.ipynb
├── reports/figures/    ← gráficos generados
├── climasafeai/
│   ├── data/           make_dataset.py
│   ├── features/       build_features.py
│   ├── models/         train_model.py · predict_model.py
│   ├── visualization/  visualize.py
│   └── utils/          paths.py
├── tests/
├── main.py             ← pipeline completo
├── Makefile
└── pyproject.toml
```

## Inicio rápido

Consulta el archivo `documentacion` para más detalles.

---

Template generado con https://github.com/cacelass/dskit