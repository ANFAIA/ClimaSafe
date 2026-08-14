// features.js — port 1:1 a JS del feature engineering Python de ClimaSafeAI
// (climasafeai/features/weather_indices.py, climasafeai/data/make_dataset.py,
//  climasafeai/data/weather_fetcher.py, climasafeai/features/build_features.py)
//
// Convenciones del port:
//   - Una fila horaria es {datetime: "YYYY-MM-DDTHH:00", t2m_c, rh,
//     wind_speed_kmh, sp}; tras procesar lleva además fecha, heat_index_c,
//     wbgt_c y wind_chill_c.
//   - NaN se representa como null (los JSON de Open-Meteo/escenarios nunca
//     traen null; las medias/rezagos sí pueden producirlo).
//   - Los nombres de función replican los del Python para poder cotejar.

// ─────────────────────────────────────────────────────────────────────────────
// Índices meteorológicos (weather_indices.py)
// ─────────────────────────────────────────────────────────────────────────────
const _c2f = (t) => (t * 9) / 5 + 32;
const _f2c = (t) => ((t - 32) * 5) / 9;

export function heatIndex(tC, rh) {
  // Rothfusz (NWS 1990); fuera de T>=80F y RH>=40% usa la aproximación
  // simplificada — idéntico a np.where(valid_range, hi_full, hi_simplified).
  if (!Number.isFinite(tC) || !Number.isFinite(rh)) return null;
  const tF = _c2f(tC);
  const hiFull =
    -42.379 +
    2.04901523 * tF +
    10.14333127 * rh -
    0.22475541 * tF * rh -
    0.00683783 * tF * tF -
    0.05481717 * rh * rh +
    0.00122874 * tF * tF * rh +
    0.00085282 * tF * rh * rh -
    0.00000199 * tF * tF * rh * rh;
  const hiSimplified = 0.5 * (tF + 61.0 + (tF - 68.0) * 1.2 + rh * 0.094);
  const hiF = tF >= 80.0 && rh >= 40.0 ? hiFull : hiSimplified;
  return _f2c(hiF);
}

export function wbgtFromHeatIndex(hiC) {
  if (!Number.isFinite(hiC)) return null;
  const hiF = _c2f(hiC);
  return -0.0034 * hiF * hiF + 0.96 * hiF - 34;
}

export function windChill(tC, vKmh) {
  // NWS 2001; fuera de T<=10C y V>4.8 km/h devuelve la temperatura real.
  if (!Number.isFinite(tC) || !Number.isFinite(vKmh)) return null;
  const v = Math.max(vKmh, 0);
  const wcFull =
    13.12 +
    0.6215 * tC -
    11.37 * Math.pow(v, 0.16) +
    0.3965 * tC * Math.pow(v, 0.16);
  return tC <= 10.0 && vKmh > 4.8 ? wcFull : tC;
}

export function addWeatherIndexColumns(row) {
  // Equivalente a add_weather_index_columns() sobre una fila suelta.
  const out = { ...row };
  if (Number.isFinite(out.t2m_c) && Number.isFinite(out.rh)) {
    out.heat_index_c = heatIndex(out.t2m_c, out.rh);
    out.wbgt_c = wbgtFromHeatIndex(out.heat_index_c);
  } else {
    out.heat_index_c = null;
    out.wbgt_c = null;
  }
  if (Number.isFinite(out.t2m_c) && Number.isFinite(out.wind_speed_kmh)) {
    out.wind_chill_c = windChill(out.t2m_c, out.wind_speed_kmh);
  } else {
    out.wind_chill_c = null;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers numéricos (semántica pandas sobre null=NaN)
// ─────────────────────────────────────────────────────────────────────────────
const _noNan = (a) => a.filter((v) => v != null && Number.isFinite(v));

function _media(a) {
  const v = _noNan(a);
  return v.length ? v.reduce((x, y) => x + y, 0) / v.length : null;
}
function _suma(a) {
  const v = _noNan(a);
  return v.length ? v.reduce((x, y) => x + y, 0) : null;
}
function _min(a) {
  const v = _noNan(a);
  return v.length ? Math.min(...v) : null;
}
function _max(a) {
  const v = _noNan(a);
  return v.length ? Math.max(...v) : null;
}
function _std(a) {
  // pandas std: ddof=1 (muestral), ignora NaN.
  const v = _noNan(a);
  if (v.length < 2) return null;
  const m = v.reduce((x, y) => x + y, 0) / v.length;
  const s = v.reduce((acc, x) => acc + (x - m) ** 2, 0);
  return Math.sqrt(s / (v.length - 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// Agregación diaria (make_dataset._agregar_estadisticas_diarias)
// ─────────────────────────────────────────────────────────────────────────────
const HEAT_INDEX_UMBRAL_C = 32.0;
const WIND_CHILL_UMBRAL_C = 0.0;

export function agregarEstadisticasDiarias(dfHora) {
  // dfHora: filas ya procesadas con fecha/heat_index_c/wind_chill_c/t2m_c.
  const porFecha = new Map();
  for (const r of dfHora) {
    if (!porFecha.has(r.fecha)) porFecha.set(r.fecha, []);
    porFecha.get(r.fecha).push(r);
  }
  const stats = [];
  for (const [fecha, filas] of porFecha) {
    const hi = filas.map((r) => r.heat_index_c);
    const wc = filas.map((r) => r.wind_chill_c);
    const noche = filas.filter((r) => r.hora >= 0 && r.hora <= 8).map((r) => r.t2m_c);
    stats.push({
      fecha,
      heat_index_mean: _media(hi),
      heat_index_std: _std(hi),
      heat_index_min: _min(hi),
      horas_sobre_umbral: hi.filter((v) => v != null && v > HEAT_INDEX_UMBRAL_C).length,
      wind_chill_mean: _media(wc),
      wind_chill_std: _std(wc),
      wind_chill_max: _max(wc),
      horas_bajo_umbral: wc.filter((v) => v != null && v < WIND_CHILL_UMBRAL_C).length,
      t2m_min_noche: _min(noche),
      horas_wc_severo: wc.filter((v) => v != null && v < -5.0).length,
    });
  }
  return stats;
}

// ─────────────────────────────────────────────────────────────────────────────
// Racha previa + rezagos temporales (make_dataset._racha_previa y
// _agregar_rezagos_temporales) — grupo único "__usuario__".
// ─────────────────────────────────────────────────────────────────────────────
function _rachaPrevia(activo) {
  // bloques = (activo != activo.shift()).cumsum()
  const bloques = [];
  let cum = 0;
  for (let i = 0; i < activo.length; i++) {
    if (i === 0 || activo[i] !== activo[i - 1]) cum++;
    bloques.push(cum);
  }
  // incl = groupby(bloques).cumcount()+1 where activo else 0
  const last = new Map();
  const incl = [];
  for (let i = 0; i < activo.length; i++) {
    const b = bloques[i];
    const c = (last.get(b) ?? -1) + 1;
    last.set(b, c);
    incl.push(activo[i] ? c + 1 : 0);
  }
  // shift(1).fillna(0)
  return [0, ...incl.slice(0, activo.length - 1)];
}

function _rollPrev(valores, N, op) {
  // s.shift(1).rolling(N, min_periods=1).mean()|sum()
  const out = [];
  for (let i = 0; i < valores.length; i++) {
    const ventana = valores.slice(Math.max(0, i - N), i); // solo PASADO (shift 1)
    const noNan = _noNan(ventana);
    out.push(noNan.length >= 1 ? op(noNan) : null);
  }
  return out;
}

export function agregarRezagosTemporales(dfDia) {
  // dfDia: una fila por fecha, ordenada cronológicamente.
  const f = dfDia.map((r) => ({ ...r }));
  const n = f.length;
  const col = (k) => f.map((r) => r[k]);

  const setCol = (k, vals) => f.forEach((r, i) => { r[k] = vals[i]; });
  const exceso = (k) => f.map((r) => (r[k] != null ? Math.max(r[k] - HEAT_INDEX_UMBRAL_C, 0) : null));

  // Calor
  setCol("heat_index_c_lag1", [null, ...col("heat_index_c").slice(0, n - 1)]);
  setCol("heat_index_c_roll3", _rollPrev(col("heat_index_c"), 3, _media));
  setCol("heat_index_c_roll7", _rollPrev(col("heat_index_c"), 7, _media));
  setCol("dias_consec_sobre_umbral", _rachaPrevia(col("horas_sobre_umbral").map((v) => v > 0)));
  setCol("grados_dia_calor_roll7", _rollPrev(exceso("heat_index_mean"), 7, _suma));
  setCol("grados_dia_calor_roll14", _rollPrev(exceso("heat_index_mean"), 14, _suma));

  // Frío
  setCol("wind_chill_mean_roll3", _rollPrev(col("wind_chill_mean"), 3, _media));
  setCol("wind_chill_mean_roll7", _rollPrev(col("wind_chill_mean"), 7, _media));
  setCol("wind_chill_mean_roll14", _rollPrev(col("wind_chill_mean"), 14, _media));
  const deficit = f.map((r) => (r.wind_chill_mean != null ? Math.max(WIND_CHILL_UMBRAL_C - r.wind_chill_mean, 0) : null));
  setCol("grados_dia_frio_roll7", _rollPrev(deficit, 7, _suma));
  setCol("grados_dia_frio_roll14", _rollPrev(deficit, 14, _suma));
  setCol("dias_consec_bajo_umbral", _rachaPrevia(col("horas_bajo_umbral").map((v) => v > 0)));

  // Nocturnas y rachas severas (2026-07-14)
  setCol("t2m_min_noche_lag1", [null, ...col("t2m_min_noche").slice(0, n - 1)]);
  setCol("t2m_min_noche_roll7", _rollPrev(col("t2m_min_noche"), 7, _media));
  setCol("dias_consec_wc_severo", _rachaPrevia(col("horas_wc_severo").map((v) => v > 0)));
  setCol("horas_wc_severo_sum14", _rollPrev(col("horas_wc_severo"), 14, _suma));

  return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// procesar_horario_con_indices + _generar_features_completas (weather_fetcher)
// ─────────────────────────────────────────────────────────────────────────────
export function procesarHorarioConIndices(dfHora) {
  return dfHora.map((r) => {
    const out = addWeatherIndexColumns(r);
    out.fecha = String(r.datetime).slice(0, 10);
    out.hora = parseInt(String(r.datetime).slice(11, 13), 10);
    return out;
  });
}

export function generarFeaturesCompletas(dfTarget, dfHist) {
  // dfTarget y dfHist: filas horarias crudas ({datetime, t2m_c, rh, ...}).
  const hist = dfHist && dfHist.length ? dfHist : [];
  let dfHora = procesarHorarioConIndices([...hist, ...dfTarget]);

  // drop_duplicates(subset=["fecha","datetime"]) + sort_values(["fecha","datetime"])
  const vistos = new Set();
  dfHora = dfHora.filter((r) => {
    const k = `${r.fecha}|${r.datetime}`;
    if (vistos.has(k)) return false;
    vistos.add(k);
    return true;
  });
  dfHora.sort((a, b) => (a.fecha === b.fecha ? (a.datetime < b.datetime ? -1 : 1) : a.fecha < b.fecha ? -1 : 1));

  const dfStats = agregarEstadisticasDiarias(dfHora);

  // df_dia = groupby(fecha).first()  (la fila 00:00, primera tras el sort)
  const vistosDia = new Set();
  const dfDia = [];
  for (const r of dfHora) {
    if (!vistosDia.has(r.fecha)) {
      vistosDia.add(r.fecha);
      dfDia.push({ ...r });
    }
  }

  // merge(df_stats, on="fecha", how="left")
  const statsPorFecha = new Map(dfStats.map((s) => [s.fecha, s]));
  for (const r of dfDia) {
    Object.assign(r, statsPorFecha.get(r.fecha) ?? {});
  }

  // provincia="__usuario__" → rezagos → quitar provincia
  for (const r of dfDia) r.provincia = "__usuario__";
  const conRezagos = agregarRezagosTemporales(dfDia);
  for (const r of conRezagos) delete r.provincia;

  return { dfFeatures: conRezagos, dfHora };
}

// ─────────────────────────────────────────────────────────────────────────────
// Entradas LSTM (weather_fetcher: build_sequence_24h, build_daily_feature_vector,
// get_province_idx, get_ine_features, escalar_para_lstm)
// ─────────────────────────────────────────────────────────────────────────────
export function buildSequence24h(dfHora) {
  // dfHora: filas PROCESADAS (con heat_index_c/wind_chill_c) o crudas (se
  // procesan aquí, como hace el Python).
  if (!dfHora || !dfHora.length) return null;
  let df = procesarHorarioConIndices(dfHora);
  const cols = ["t2m_c", "rh", "wind_speed_kmh", "heat_index_c", "wind_chill_c"];
  if (!df.length) return null;
  df = df.slice().sort((a, b) => (a.datetime < b.datetime ? -1 : 1));
  if (df.length >= 24) df = df.slice(-24);
  const seq = df.map((r) => cols.map((c) => (r[c] == null ? null : r[c])));
  if (seq.length < 24) {
    const last = seq[seq.length - 1];
    while (seq.length < 24) seq.unshift([...last]); // np.tile del último al principio
  }
  return [seq]; // (1, 24, 5)
}

export function buildDailyFeatureVector(dfFeatures, dailyFeatureCols) {
  if (!dfFeatures || !dfFeatures.length) return null;
  const ordenadas = dfFeatures.slice().sort((a, b) => (a.fecha < b.fecha ? -1 : 1));
  const latest = ordenadas[ordenadas.length - 1];
  return dailyFeatureCols.map((c) => (c in latest ? latest[c] : 0.0));
}

export function getProvinceIdx(provincia, mapping) {
  const key = String(provincia).trim().toLowerCase();
  for (const [name, idx] of Object.entries(mapping || {})) {
    if (name.trim().toLowerCase() === key) return idx;
  }
  return 0;
}

export function getIneFeatures(provincia, ineFeatures) {
  const key = String(provincia).trim().toLowerCase();
  const provincias = (ineFeatures && ineFeatures.provincias) || {};
  for (const [name, v] of Object.entries(provincias)) {
    if (name.trim().toLowerCase() === key) {
      return [v.pct_mayores_65, v.pct_mayores_80, v.pct_mujeres, Math.log(v.poblacion_total)];
    }
  }
  return [20.0, 5.0, 50.0, Math.log(1_000_000)];
}

export function escalarParaLstm(seq, ineVec, dailyVec, scalers) {
  // StandardScaler.transform por columna: (x - mean_[i]) / scale_[i]
  const esc = (vec, sc) => vec.map((v, i) => (v == null || !Number.isFinite(v) ? v : (v - sc.mean[i]) / sc.scale[i]));
  // seq: (1,24,5) → escalar por columna (reshape(-1,5).transform)
  const seqFlat = seq[0].map((fila) => esc(fila, scalers.seq));
  const ineS = esc(ineVec, scalers.ine);
  const dailyS = esc(dailyVec, scalers.daily);
  return [seqFlat, ineS, dailyS];
}

// ─────────────────────────────────────────────────────────────────────────────
// process_input (build_features.py:368) — réplica exacta para inferencia
// ─────────────────────────────────────────────────────────────────────────────
// ORDINAL_MAPPINGS y LOGCOLS están vacíos en el proyecto actual; se omiten.
export const COLS_TO_DROP = [
  "provincia", "fecha", "datetime",
  "t2m_min_noche", "horas_wc_severo",
];
export const COLS_TO_DROP_BY_CLASE = {
  calor: ["t2m_min_noche_lag1", "t2m_min_noche_roll7", "dias_consec_wc_severo", "horas_wc_severo_sum14"],
  frio: [
    "grados_dia_calor_roll7", "grados_dia_calor_roll14",
    "wind_chill_mean_roll3", "wind_chill_mean_roll7", "wind_chill_mean_roll14",
    "grados_dia_frio_roll7", "grados_dia_frio_roll14",
    "dias_consec_bajo_umbral",
  ],
};

export function processInput(dfRows, clase, artefactos) {
  // dfRows: filas diarias (df_features) ya con provincia si aplica.
  const featureNames = artefactos[`feature_names_${clase}`];
  const scaler = artefactos[`scaler_${clase}`];
  const aEliminar = [...COLS_TO_DROP, ...(COLS_TO_DROP_BY_CLASE[clase] || [])];
  const filas = dfRows.map((r) => {
    const copia = { ...r };
    for (const c of aEliminar) delete copia[c];
    return copia;
  });

  // fillna(media por columna) sobre columnas numéricas (todas las que quedan)
  const cols = Object.keys(filas[0] || {});
  const medias = {};
  for (const c of cols) {
    const v = filas.map((r) => r[c]).filter((x) => x != null && Number.isFinite(x));
    medias[c] = v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  }
  for (const r of filas) {
    for (const c of cols) {
      if (r[c] == null || !Number.isFinite(r[c])) r[c] = medias[c];
    }
  }

  // scaler.transform en el orden de feature_names (el scaler joblib guarda
  // feature_names_in_ y reordena igual).
  const X = filas.map((r) =>
    featureNames.map((c) => {
      const v = r[c];
      return v == null || !Number.isFinite(v) ? v : (v - scaler.mean[featureNames.indexOf(c)]) / scaler.scale[featureNames.indexOf(c)];
    })
  );
  return X;
}

export function applyClassThresholds(proba2d, t1, t2) {
  // proba2d: (n, 3) → cascada P(2)>=t2 → 2; P(1)+P(2)>=t1 → 1; else 0
  return proba2d.map((p) => (p[2] >= t2 ? 2 : p[1] + p[2] >= t1 ? 1 : 0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Estrato de edad y factor edad (ensemble.py)
// ─────────────────────────────────────────────────────────────────────────────
export const CLASES = ["SEGURO", "PRECAUCION", "PELIGRO"];

const FACTORES_RIESGO_EDAD = {
  calor: { joven: 0.6, adulto: 0.6, mayor: 0.75, anciano: 0.875, viejano: 1.0, todos: 1.0 },
  frio: { joven: 0.75, adulto: 0.75, mayor: 0.875, anciano: 0.95, viejano: 1.0, todos: 1.0 },
};

export function edadAestrato(edad) {
  if (edad == null) return "todos";
  if (edad < 45) return "joven";
  if (edad < 60) return "adulto";
  if (edad < 70) return "mayor";
  if (edad < 80) return "anciano";
  return "viejano";
}

export function aplicarFactorEdad(proba, clase, grupoEdad) {
  const factor = (FACTORES_RIESGO_EDAD[clase] || {})[grupoEdad] ?? 1.0;
  if (factor === 1.0) return proba.slice();
  const p = proba.slice();
  const probRiesgo = 1.0 - p[0];
  const probRiesgoAdj = probRiesgo * factor;
  const pSum = p[1] + p[2];
  if (pSum > 0) {
    const p1Frac = p[1] / pSum;
    const p2Frac = p[2] / pSum;
    p[0] = 1.0 - probRiesgoAdj;
    p[1] = probRiesgoAdj * p1Frac;
    p[2] = probRiesgoAdj * p2Frac;
  }
  return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Escenarios precargados (fallback offline / test de paridad). El JS NO genera
// horas: las lee de scenarios.json (fuente única compartida con Python).
// ─────────────────────────────────────────────────────────────────────────────
export async function cargarEscenarios(url) {
  if (typeof fetch !== "undefined") {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`No se pudo cargar ${url}: HTTP ${r.status}`);
    return r.json();
  }
  // node (test de paridad): lo inyecta el llamante, no llegamos aquí.
  throw new Error("cargarEscenarios requiere fetch");
}

export const round4 = (x) => (x == null || !Number.isFinite(x) ? x : Math.round(x * 10000) / 10000);
