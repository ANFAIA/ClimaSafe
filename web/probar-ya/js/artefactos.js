// artefactos.js — carga de los JSON de models/ (artefactos ONNX) y de los
// ficheros de modelo. Navegador: fetch (rutas relativas). Node (test de
// paridad): fs. Los paths relativos permiten publicar la demo en
// cacelass.github.io/climasafe/probar-ya/ sin absolutos.
let fsMod = null;
try {
  fsMod = await import("node:fs/promises");
} catch {
  fsMod = null;
}
const EN_NODE = fsMod != null;

export function esNode() {
  return EN_NODE;
}

export async function leerBytes(ruta) {
  if (EN_NODE) return new Uint8Array(await fsMod.readFile(ruta));
  const r = await fetch(ruta);
  if (!r.ok) throw new Error(`fetch ${ruta}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

export async function leerJson(ruta) {
  if (EN_NODE) return JSON.parse(await fsMod.readFile(ruta, "utf8"));
  const r = await fetch(ruta);
  if (!r.ok) throw new Error(`fetch ${ruta}: HTTP ${r.status}`);
  return r.json();
}

const NOMBRES_ARTEFACTOS = [
  "feature_names_calor.json", "feature_names_frio.json",
  "scaler_calor.json", "scaler_frio.json",
  "encoders_calor.json", "encoders_frio.json",
  "umbrales_provincia_calor.json", "umbrales_provincia_frio.json",
  "conformal_calor.json", "conformal_frio.json",
  "iso_calib_frio.json",
  "class_thresholds.json", "provincia_mapping.json",
  "ine_features.json", "daily_feature_cols.json",
  "scaler_diarias_lstm.json", "scaler_secuencias_lstm.json",
  "scaler_provincia_features.json", "factores_riesgo.json",
];

// Carga todos los artefactos JSON de una vez: {nombre_sin_ext: contenido}.
export async function cargarArtefactos(dir) {
  const out = {};
  await Promise.all(
    NOMBRES_ARTEFACTOS.map(async (f) => {
      out[f.replace(/\.json$/, "")] = await leerJson(`${dir}/${f}`);
    })
  );
  return out;
}

// Carga las sesiones ONNX (XGBoost_calor, RandomForest_frio, LSTM con su
// external data). Devuelve {xgb, rf, lstm}.
export async function cargarModelosOrt(dir, ort) {
  const [xgb, rf] = await Promise.all([
    ort.InferenceSession.create(`${dir}/XGBoost_calor.onnx`),
    ort.InferenceSession.create(`${dir}/RandomForest_frio.onnx`),
  ]);
  // El LSTM se exportó con external data (LSTM_province_hybrid.onnx.data);
  // onnxruntime-web no la resuelve sola: hay que pasarla explícitamente.
  const modelBuf = await leerBytes(`${dir}/LSTM_province_hybrid.onnx`);
  const extBuf = await leerBytes(`${dir}/LSTM_province_hybrid.onnx.data`);
  const lstm = await ort.InferenceSession.create(modelBuf, {
    externalData: [{ path: "LSTM_province_hybrid.onnx.data", data: extBuf }],
  });
  return { xgb, rf, lstm };
}
