# Estudio previo: el modelo en el navegador con WebAssembly (WEB-002)

> **Fecha:** 2026-08-18
> **Estado:** estudio que decide si se implementa y con qué opción. Criterio 1
> de WEB-002: sin este estudio no se empieza.
> **Punto de partida:** WEB-011 ya exportó los 3 modelos a ONNX
> (`models/onnx/`, paridad verificada <1e-3 con onnxruntime CPU) y 19 JSON de
> artefactos. El objetivo: que la predicción individual corra entera en el
> navegador, sin backend, sin que los datos de salud del usuario salgan de su
> navegador.

---

## 1. Resumen ejecutivo

**Sí se puede.** Los tres modelos del pipeline (`XGBoost_calor`,
`RandomForest_frio`, `LSTM_province_hybrid`) ya están en ONNX (WEB-011) y
onnxruntime-web los ejecuta en WASM con la misma aritmética. El ensamblado
(ensemble conformal-weighted + fórmula HI/WC + personalización + overrides +
recomendaciones) es lógica pura portada a JS sin necesidad de runtime de ML
adicional.

**Opción elegida: ONNX Runtime Web (backend CPU, WASM SIMD + threads).**
Descarga de primera carga ≈ **41 MB** (26,9 MB de modelos + 13,8 MB de runtime
+ 0,3 MB de artefactos JSON y escenarios precargados). Es la opción que WEB-012
ya implementó en `web/probar-ya/` y que este estudio valida y documenta.

**No se lleva a WASM:** la base de datos meteorológica, los perfiles en SQLite
(datos personales de salud), la base de factores mantenida y el pipeline de
entrenamiento. Motivo: son datos vivos que el navegador no debe persistir ni
mantener, y el reentrenamiento no tiene sentido en cliente.

---

## 2. Opciones consideradas

| Opción | Qué es | ¿Ejecuta nuestros modelos? | Veredicto |
|--------|--------|---------------------------|-----------|
| **ONNX Runtime Web** | Runtime de inferencia ONNX compilado a WASM (mismo motor que onnxruntime CPU usado en WEB-011) | Sí: los 3 `.onnx` tal cual salen del exportador | **Elegida** |
| TensorFlow.js | Runtime TF en WASM/WebGL | No directamente: XGBoost y LSTM no tienen conversor TF.js fiable; habría que reentrenar | Descartada |
| Pyodide | Python + pandas + scipy compilados a WASM | No: XGBoost y PyTorch no están en los paquetes oficiales; joblib/torch no cargan | Descartada |
| WebGPU (JSEP) | Backend de ejecución dentro de onnxruntime-web (acelera con GPU) | Sí, mismo ONNX | Opcional futuro, no por defecto |

### 2.1 ONNX Runtime Web — por qué funciona

- El motor es el **mismo** que WEB-011 verificó: `onnxruntime` en CPU. La
  variante web compila el ejecutor a WASM; la aritmética de los operadores
  (TreeEnsemble, LSTM, Gemm, MatMul…) es la misma, por eso la paridad
  joblib↔ONNX (<1e-3, WEB-011) se mantiene en el navegador.
- Los 3 modelos ONNX usan operadores soportados por la build web (verificados
  en la demo WEB-012 y en `tests/test_demo_paridad.py`).
- La demo funciona **sin servidor**: ficheros estáticos servidos por cualquier
  HTTP (GitHub Pages incluido).

### 2.2 TensorFlow.js — descartada

- El catálogo de modelos no está en TF: `XGBoost` (sklearn wrapper) y
  `LSTM` (PyTorch `.pt`). No hay conversor oficial XGBoost→TF.js; el camino
  ONNX→TF (`onnx-tf`) es experimental y rompe operadores TreeEnsemble.
- Reentrenar en TF no es una opción: cambiaría el modelo en producción solo
  para la demo.
- Tamaño: el runtime es menor (~0,5 MB), pero el coste real es inviable por
  conversión.

### 2.3 Pyodide — descartada

- Pyodide ejecuta Python en WASM, pero **no** incluye `xgboost` ni `torch`
  entre sus paquetes oficiales. `joblib` podría leer los `.joblib` de sklearn,
  pero no los de xgboost/torch.
- Tamaño: solo el runtime base son ~10–23 MB; con pandas+scipy+sklearn el
  total supera 50 MB y el arranque es de segundos. Aun así no ejecutaría los
  modelos actuales. No aporta nada que ONNX Runtime Web no dé con menos peso.

### 2.4 WebGPU (JSEP) — opcional futuro

- Es un backend de **onnxruntime-web**, no una opción independiente: acelera
  la inferencia usando la GPU del navegador (requiere WebGPU).
- Coste: ficheros extra `.jsep.mjs`/`.jsep.wasm` (~25,6 MB más).
- Beneficio: para una predicción individual de modelos de 3–23 MB, el backend
  CPU SIMD responde en <1 s. El peso extra no compensa en el caso de uso
  actual. Se deja como mejora futura si la demo pasa a predicción por lotes.

---

## 3. Qué se puede llevar a WASM y qué no

### 3.1 Sí se porta (todo el pipeline de `predict_ensemble`)

| Pieza | Cómo se porta | Fuente |
|-------|---------------|--------|
| XGBoost_calor | `.onnx` (22,7 MB) ejecutado con onnxruntime-web | `models/onnx/XGBoost_calor.onnx` |
| RandomForest_frio | `.onnx` (3,9 MB) ejecutado con onnxruntime-web | `models/onnx/RandomForest_frio.onnx` |
| LSTM_province_hybrid | `.onnx` + `.data` (0,3 MB) ejecutado con onnxruntime-web | `models/onnx/LSTM_province_hybrid.onnx(.data)` |
| Fórmula HI/WC | JS puro (aritmética) | `web/probar-ya/js/features.js` |
| Features diarias + índices | JS puro | `web/probar-ya/js/features.js` |
| Ensemble conformal-weighted | JS puro (usa umbrales/conformal JSON) | `web/probar-ya/js/modelos.js` |
| Personalización (factores de riesgo) | JS puro (usa `factores_riesgo.json`) | `web/probar-ya/js/personalizacion.js` |
| Overrides físicos (HI/WC/UV) | JS puro | `web/probar-ya/js/modelos.js` |
| Recomendaciones | JS puro (usa `recomendaciones.json`) | `web/probar-ya/js/recomendaciones.js` |
| Perfil horario | JS puro | `web/probar-ya/js/features.js` |
| 19 artefactos JSON | copia estática en la demo | `models/onnx/*.json` |

La fórmula y el ensamblado no necesitan runtime de ML: son operaciones
deterministas que se portan a JS y se verifican contra Python (test de
paridad, §5).

### 3.2 No se porta (se queda en el servidor)

| Pieza | Por qué se queda |
|-------|------------------|
| **Base de datos meteorológica** | Es la fuente viva de datos (histórico y forecast). El navegador no la necesita: los datos meteorológicos se obtienen por Open-Meteo (CORS) con fallback a escenario precargado si no hay red. La cache/histórico propio del servidor no se expone. |
| **Perfiles en SQLite** | Son **datos personales de salud** persistidos del usuario. La demo no persiste nada: por privacidad, el perfil se construye en el formulario y solo vive en memoria durante la sesión (localStorage solo guarda la aceptación del aviso médico). El guardado/consulta de perfiles sigue siendo del servidor. |
| **Base de factores mantenida** | La demo usa un snapshot estático (`factores_riesgo.json`). La fuente de verdad y su actualización (papers, revisión médica) viven en el servidor; la demo se reempaqueta cuando cambia. |
| **Modelos joblib/.pt y pipeline de entrenamiento** | El reentrenamiento y la validación son del servidor (y del CI). Al navegador solo llega el artefacto ONNX ya validado. |
| **UV index (OpenUV)** | Requiere API key; la demo usa `uv_index=null` (igual que Python sin key). Si el servidor integra OpenUV, es un dato más que puede inyectarse, no un cálculo que portar. |
| **AEMET / fuentes con API key** | No se portan; Open-Meteo cubre forecast y archive sin key. |

---

## 4. Tamaño de descarga por opción

Medido en disco (bytes reales, 2026-08-18). En red, GitHub Pages/nginx pueden
aplicar gzip/brotli; los `.wasm` y `.onnx` se sirven típicamente sin comprimir
(ya son binarios).

### 4.1 Modelos (iguales en cualquier opción que ejecute ONNX)

| Fichero | Tamaño |
|---------|--------|
| `XGBoost_calor.onnx` | 22 794 102 B (~21,7 MiB) |
| `RandomForest_frio.onnx` | 3 875 580 B (~3,7 MiB) |
| `LSTM_province_hybrid.onnx` + `.data` | 8 475 B + 271 168 B (~0,27 MiB) |
| **Total modelos** | **~26,9 MB** |
| 19 JSON de artefactos (+`recomendaciones.json`) | ~39,5 KB |
| `scenarios.json` (fallback offline, opcional) | 262 KB |
| **Total artefactos** | **~0,3 MB** |

### 4.2 Runtime por opción

| Opción | Runtime | Modelos | **Descarga total aprox.** |
|--------|---------|---------|---------------------------|
| **ONNX Runtime Web (CPU)** | `ort.min.js` 352 KB + `ort-wasm-simd-threaded.wasm` 12,9 MB ≈ **13,8 MB** | 26,9 MB | **≈ 41 MB** |
| ONNX Runtime Web + WebGPU (JSEP) | 13,8 MB + `.jsep.wasm` 25,6 MB ≈ **39,4 MB** | 26,9 MB | ≈ 66 MB |
| TensorFlow.js | ~0,5 MB (core+layers) | no convertible | **inviable** |
| Pyodide | ~10–23 MB base + pandas/scipy/sklearn ≥ 50 MB | no ejecuta xgboost/torch | **inviable** |

> Nota: la carpeta `web/probar-ya/vendor/` pesa ~40,7 MB en disco porque
> incluye el `.jsep.wasm` (WebGPU) aunque la demo no lo carga por defecto; en
> la primera carga real el navegador solo descarga los ficheros que el código
> importa (~13,8 MB de runtime + 26,9 MB de modelos ≈ 41 MB).

### 4.3 Peso de la primera carga frente al modo backend

- **Modo backend actual** (`chat/`): el navegador solo descarga HTML/JS/CSS de
  la web; el peso de los modelos vive en el servidor. La predicción viaja por
  red y los datos del usuario pasan por el servidor.
- **Modo WASM**: +41 MB de descarga única (cacheable por el navegador), pero a
  partir de ahí la predicción es local: sin latencia de red por petición y sin
  que el perfil de salud salga del navegador. 41 MB es aceptable para una
  herramienta consultada de forma repetida (se cachea); es el coste de la
  privacidad por diseño.

---

## 5. Paridad y tolerancia

- **ONNX vs joblib/torch (WEB-011, ya verificado):** diff < 1e-3 en los 5
  escenarios (XGB ≤1,2e-7, RF ≤5,6e-8, LSTM ≤1,1e-5).
- **Ensamblado JS vs Python (WEB-002, criterio 3):** el test
  `tests/test_demo_paridad.py` ejecuta `predict_ensemble` (Python) y el
  pipeline JS (node + onnxruntime-web) sobre los **mismos 5 escenarios
  precargados** (`web/probar-ya/scenarios.json`) y compara:
  - `clase_final`: **idéntica**;
  - % de riesgo (`prob_pers = max(prob_personalizada calor/frío)`):
    **±1 punto** (`TOLERANCIA_PUNTOS = 0.01`).
- La tolerancia declarada es ±1 punto porcentual y clase idéntica porque es lo
  que distingue la decisión operativa (SEGURO/PRECAUCIÓN/PELIGRO); las
  diferencias menores de décimas de punto entre plataformas (redondeo float32
  WASM vs float64 Python) no cambian la clase.

---

## 6. Sin llamadas al backend (criterio 2) — cómo se demuestra

La demo (`web/probar-ya/`) es **estática**: no existe endpoint de predicción
al que llamar. Los únicos `fetch` del código JS son:

1. `api.open-meteo.com` (datos meteorológicos, CORS, con fallback offline a
   `scenarios.json`) — `js/weather.js`.
2. Ficheros locales de la demo (`./models/*`, `./scenarios.json`,
   `./vendor/*`) — `js/artefactos.js`.
3. Tiles del mapa (CARTO, opcional, solo si se usa el mapa) — `js/main.js`.

Demostración:
- **Estática (automatizable):** `tests/test_demo_sin_backend.py` verifica que
  ningún módulo JS de la demo contiene `fetch`/`XMLHttpRequest` a rutas del
  backend (`/api/`, `/predict`, `localhost:<puerto>`, hosts del proyecto).
- **En vivo:** servir la demo (`python3 -m http.server 8091 --directory
  web/probar-ya`), pulsar «Predecir» y abrir la pestaña **Red** de DevTools:
  solo aparecen peticiones a ficheros estáticos locales y a Open-Meteo;
  ninguna a `/api/*`. Equivalente sin navegador: los logs del servidor estático
  solo muestran GET de `.html/.js/.onnx/.json/.wasm`.

---

## 7. La web con backend no se rompe (criterio 5)

La demo WASM es una **mejora progresiva**: convive con `chat/` (la web con
`/api/predict` y SQLite) y con `docs_site/` (documentación). Quien no pueda
ejecutar WASM (navegador antiguo, red lenta, sin ganas de descargar 41 MB)
sigue usando el modo actual. La demo se sirve por separado
(`web/probar-ya/` → `.../climasafe/probar-ya/`); no sustituye ni modifica la
web actual. `make test` cubre ambos: `tests/test_demo_paridad.py` (WASM) y
`tests/test_api.py` + `tests/test_web_predict.py` (backend).

---

## 8. Conclusión y decisión

1. **Se empieza.** El coste es una descarga única de ~41 MB; a cambio la
   predicción corre entera en el navegador, sin servidor y sin que el perfil
   de salud salga del dispositivo.
2. **Opción: ONNX Runtime Web, backend CPU (WASM SIMD + threads).** Sin WebGPU
   por defecto; es el mismo motor que WEB-011 verificó, con paridad ya medida.
3. **Lo que no se porta** (meteorología viva, perfiles SQLite, base de
   factores mantenida, entrenamiento) **se queda en el servidor** y por qué:
   privacidad y ciclo de vida de los datos (§3.2).
4. La implementación de referencia ya existe (`web/probar-ya/`, WEB-012); este
   estudio valida la decisión y WEB-002 la blinda con tests y documentación.
