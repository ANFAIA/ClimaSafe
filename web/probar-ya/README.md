# probar-ya — demo ONNX/WASM de ClimaSafeAI (WEB-012)

La demo ejecuta **en el navegador** el pipeline completo de `predict_ensemble`
(python): 3 modelos ONNX (XGBoost_calor, RandomForest_frio, LSTM_province_hybrid)
+ Fórmula (HI/WC) + ensemble conformal-weighted + personalización + overrides
físicos + recomendaciones. Sin backend: todo es estático.

- **Fuente de datos**: Open-Meteo (CORS OK, sin API key). Si la red falla, se
  usa un escenario precargado (`scenarios.json`, el mismo del test de paridad).
- **Paridad**: `tests/test_demo_paridad.py` compara la salida JS con
  `predict_ensemble` Python en 5 escenarios (clase idéntica, % de riesgo ±1
  punto). Actualmente Δ=0.0000 en los 5.

## Estructura

```
web/probar-ya/
  index.html            formulario + resultado (estilo cacelass.github.io)
  js/                   módulos ES: weather, features, modelos, personalizacion,
                        recomendaciones, main, artefactos, ort-runtime
  models/               copia de models/onnx/ (3 .onnx + 19 JSON) + recomendaciones.json
  vendor/               onnxruntime-web (ort.min.js + .wasm) — la demo es autocontenida
  scenarios.json        5 escenarios precargados (fallback offline + paridad)
  test/                 paridad.mjs + package.json (onnxruntime-web) + generar_escenarios.py
README.md
```

## Regenerar los modelos ONNX

Los `.onnx` y los JSON de artefactos salen del script de WEB-011:

```bash
uv run --no-sync python scripts/exportar_onnx.py            # exporta + verifica paridad joblib↔ONNX
uv run --no-sync python scripts/exportar_onnx.py --check-only   # solo verificar
```

Después copia los artefactos a la demo (los `.onnx` de `models/onnx/` están
gitignored; los de la demo no):

```bash
cp models/onnx/* web/probar-ya/models/
cp climasafeai/data/recomendaciones.json web/probar-ya/models/
```

## Probar local

La demo necesita HTTP (los ES modules y fetch no funcionan con `file://`):

```bash
python3 -m http.server 8091 --directory web/probar-ya
# → http://localhost:8091/
```

Primera carga: descarga ~26 MB de modelos y el wasm de onnxruntime; luego todo
es local (salvo el fetch a Open-Meteo, que tiene fallback offline).

## Desplegar en cacelass.github.io/climasafe/probar-ya/

Todos los paths de la demo son relativos a `web/probar-ya/`, así que basta con
copiar la carpeta dentro del repo de GitHub Pages:

```bash
cp -r web/probar-ya ~/Documentos/migithub/cacelass.github.io/climasafe/probar-ya
# commit + push en el repo personal → https://cacelass.github.io/climasafe/probar-ya/
```

El enlace «← index» apunta a `../index.html` (raíz del sitio).

## Test de paridad (node)

```bash
cd web/probar-ya/test && npm install     # onnxruntime-web (solo la primera vez)
uv run --no-sync pytest tests/test_demo_paridad.py -q -s
```

El test corre **sin red**: lee `web/probar-ya/scenarios.json`, ejecuta
`predict_ensemble` en Python (joblib/torch) y el pipeline JS en node, y compara
clase y % de riesgo. Forma parte de `make test`.

## Desviaciones del pipeline Python (documentadas)

- **UV (índice)**: OpenUV requiere API key → `uv_index` siempre `null`. Los
  overrides/downgrades que dependen de UV se comportan como Python con
  `uv_index=None` (el downgrade `UV<6` aplica; el override `HI≥27+UV>3` queda
  en manos del `HI≥32`).
- **AEMET/OpenUV**: no se portan (sin key); Open-Meteo cubre forecast y
  archive.
- **Perfil horario**: la demo usa resolución 60 min (la interpolación 5/15/30
  también está portada en `perfilHorarioDesdeDf` pero no se expone en la UI).
- **`params_estrato.joblib`**: no existe en el proyecto → no aplica (igual que
  Python).
- Los decimales se redondean como en Python (`round(x, 4)` para probas,
  `round(x, 2)` para HI/WC de la fórmula).
