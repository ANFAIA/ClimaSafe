# LLM en el navegador — redacción local del parte (WEB-016)

La demo `probar-ya` tiene una tarjeta opcional «Redactar el parte con IA
local»: un LLM pequeño descargado **en runtime** desde HuggingFace ejecuta
íntegramente en el navegador (transformers.js) y redacta en prosa el parte a
partir de los resultados **ya calculados** por el pipeline ONNX. El pipeline ML
nunca se sustituye ni se recalcula; si el LLM falla por cualquier motivo, el
parte con plantilla clásica sigue funcionando sin cambios.

## Modelo elegido

| Campo | Valor |
|---|---|
| Repo | [`onnx-community/granite-4.0-1b-ONNX-web`](https://huggingface.co/onnx-community/granite-4.0-1b-ONNX-web) |
| Modelo base | [ibm-granite/granite-4.0-1b](https://huggingface.co/ibm-granite/granite-4.0-1b) (IBM Granite 4.0 1B instruct) |
| Arquitectura | `GraniteMoeHybridForCausalLM` |
| Licencia | Apache 2.0 |
| Conversión | Oficial de HuggingFace (`onnx-community`, `library_name: transformers.js`) |
| Runtime | transformers.js **v4.2.0** vía jsDelivr (ESM, cargado perezosamente solo al activar la IA) |
| Peso de descarga | **≈ 1,25 GB** con WebGPU (`q4f16`) · **≈ 1,8 GB** con WASM (`q4`) |

### ¿Por qué este y no otro?

El criterio pedía «Granite pequeño, ~1 GB». No existe ningún Granite con
soporte transformers.js que pese exactamente ~1 GB; los tamaños siguientes se
verificaron contra la API de HuggingFace (26/08/2026):

| Candidato | dtype | Descarga | Veredicto |
|---|---|---|---|
| **granite-4.0-1b-ONNX-web** | q4f16 / q4 | **1,25 / 1,78 GB** | **elegido**: generación 2025, repo `-web` hecho para navegador, el más cercano a ~1 GB por arriba |
| granite-3.0-2b-instruct | q4 / q4f16 | 1,93 / 1,57 GB | descartado: más pesado y modelo de 2024; misma familia pero peor relación calidad/tamaño hoy |
| granite-4.0-350m-ONNX-web | fp16/q4 | ≈ 0,3–0,7 GB | descartado: demasiado pequeño para redactar bien en español |
| Granite 4.0 h-tiny (7B-A1B) | q4 | > 4 GB | descartado: muy por encima del objetivo |

Notas técnicas:

- Granite 4.0 usa `GraniteMoeHybridForCausalLM`: **requiere transformers.js ≥
  v4** (la v3.x no la carga). Por eso se fija `@huggingface/transformers@4.2.0`.
- Se elige dispositivo en runtime: WebGPU si `navigator.gpu` está disponible
  (`q4f16`, más rápido y ligero); si no, WASM (`q4`). Si WebGPU falla al crear
  el pipeline (drivers, Safari), se reintenta una vez con WASM antes de rendirse.
- GitHub Pages no sirve cabeceras COOP/COEP, así que el WASM corre
  monohilo; es lento (~1–3 min para ~200 tokens) pero funcional. Con WebGPU la
  generación es de segundos.

## Privacidad

- **Nada empaquetado en el repo**: el modelo se baja de huggingface.co la
  primera vez que el usuario pulsa «Activar IA local» y queda cacheado en el
  navegador (Cache API gestionada por transformers.js,
  `env.useBrowserCache = true`). Usos posteriores funcionan sin red.
- **Ejecución 100 % local**: ni los datos del formulario ni el resultado del
  pipeline ni el texto generado salen del dispositivo. La única petición de red
  del modo IA es la descarga inicial (CDN de jsDelivr para la librería + Hub de
  HF para los pesos).
- **Datos mínimos en el prompt**: el LLM recibe solo un resumen del resultado
  (clase, %, provincia, fecha, HI pico, ventana horaria, factores activos);
  ver `contextoDesdeResultado()` en `js/llm.js`. Y como todo ocurre en local,
  ese contexto nunca abandona la máquina.
- Aviso visible en la propia tarjeta (`ia_privacy` en `js/i18n.js`) más un
  disclaimer (`ia_disclaimer`) recordando que la prosa puede contener errores y
  que las cifras válidas son las del pipeline determinista.

## Cadena de fallback

```
pulsar «Activar IA local»
  ├─ import transformers.js desde jsDelivr ──falla──▶ estado error, plantilla intacta
  ├─ pipeline WebGPU/q4f16 ──falla──┐
  ├─ pipeline WASM/q4    ◀──────────┘ (reintento único)
  │        └──────ambos fallan─────▶ estado error, plantilla intacta
  └─ generar(mensajes) ──falla──────▶ estado error, plantilla intacta
```

El resto de la página nunca depende de `llm.js`: el módulo se importa al
arrancar pero no toca la red ni el DOM hasta que hay clic. Sin red, el fallo es
inmediato y limpio.

## Piezas y tests

- `web/probar-ya/js/llm.js` — módulo nuevo. Piezas puras exportadas
  (`MODELO_LLM`, `VERSION_TRANSFORMERS`, `elegirDispositivoYDtype`,
  `contextoDesdeResultado`, `mensajesParaParte`, `limpiarSalidaLlm`) +
  integración (`crearRedactorLocal`, `initParteIA`).
- `web/probar-ya/test/llm_unit.mjs` — 9 pruebas node sin red ni DOM;
  invocadas por `tests/test_demo_llm_units.py` (parte de `make test`).
- `web/probar-ya/index.html` — tarjeta nueva en resultados + clase CSS
  `.parte-ia-salida`; cache-bust de `main.js` subido a `?v=20260826`.
- `web/probar-ya/js/main.js` — guarda `ultimaSalida` tras cada predicción y
  llama a `initParteIA(() => ultimaSalida)` una vez.
- `web/probar-ya/js/i18n.js` — claves `card_ia`/`ia_*` en es y en.

## Verificación manual pendiente (no automatizable aquí)

La ejecución real del modelo exige navegador; comprobar en Chrome/Edge
(WebGPU) y en Firefox/Safari (WASM):

1. `python3 -m http.server 8091 --directory web/probar-ya` → predecir →
   activar la IA → la primera vez debe verse el progreso de descarga y luego
   el parte redactado.
2. DevTools → Network: verificar que solo aparecen peticiones a jsdelivr y
   huggingface.co **durante la activación**, y ninguna durante la generación.
3. Recargar con la red cortada (DevTools → Offline): la segunda activación
   debe servir el modelo desde caché y redactar igual.
4. Recargar sin caché y sin red: error controlado en la tarjeta, parte de
   plantilla normal, consola sin excepciones sin capturar.
