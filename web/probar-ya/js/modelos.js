// modelos.js — port 1:1 a JS de climasafeai/models/ensemble.py + conformal.py
// + calibrate.py, con inferencia ONNX vía onnxruntime-web.
//
// predictEnsemble() replica predict_ensemble(weather=..., provincia=...,
// perfil=..., target_date=...) — misma salida (clase_final, perfil con
// prob_personalizada, perfil_horario, override_fisico).
import {
  heatIndex, windChill, wbgtFromHeatIndex,
  procesarHorarioConIndices, buildSequence24h, buildDailyFeatureVector,
  getProvinceIdx, getIneFeatures, escalarParaLstm, processInput,
  applyClassThresholds, aplicarFactorEdad, edadAestrato, CLASES, round4,
} from "./features.js";
import { personalizarRiesgo } from "./personalizacion.js";
import { getOrt } from "./ort-runtime.js";

// ─────────────────────────────────────────────────────────────────────────────
// Isotónica frío (calibrate.py + iso_calib_frio.json): np.interp con clip y
// re-escala para que las 3 clases sumen 1.
// ─────────────────────────────────────────────────────────────────────────────
function _npInterp(p, x, y) {
  if (p <= x[0]) return y[0];
  if (p >= x[x.length - 1]) return y[y.length - 1];
  for (let i = 0; i < x.length - 1; i++) {
    if (p >= x[i] && p <= x[i + 1]) {
      const t = (p - x[i]) / (x[i + 1] - x[i]);
      return y[i] + t * (y[i + 1] - y[i]);
    }
  }
  return y[y.length - 1];
}

function _calibrarFrio(proba, iso) {
  // proba: (n, 3) → transforma col 2 y renormaliza por fila.
  return proba.map((fila) => {
    const p = fila.slice();
    p[2] = _npInterp(Math.min(Math.max(p[2], iso.x[0]), iso.x[iso.x.length - 1]), iso.x, iso.y);
    const suma = p[0] + p[1] + p[2];
    return [p[0] / suma, p[1] / suma, p[2] / suma];
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Conformal (conformal.py SplitConformalCalibrator.confidence)
// ─────────────────────────────────────────────────────────────────────────────
function _conformalSetSize(probaFila, qhat) {
  let size = 0;
  for (let c = 0; c < 3; c++) {
    if (1.0 - probaFila[c] <= qhat) size++;
  }
  return size;
}

function _confidenceLabel(size) {
  if (size === 0) return "baja";
  if (size === 1) return "alta";
  return "media";
}

// ─────────────────────────────────────────────────────────────────────────────
// _predecir_tabular (ensemble.py:94) — XGBoost_calor / RandomForest_frio
// ─────────────────────────────────────────────────────────────────────────────
async function _predecirTabular(session, clase, dfFeatures, provincia, grupoEdad, artefactos) {
  const dfInput = dfFeatures.map((r) => ({ ...r }));
  if (provincia) {
    for (const r of dfInput) r.provincia = provincia;
  }
  const X = processInput(dfInput, clase, artefactos);
  const n = X.length;
  const nFeat = X[0].length;
  const flat = new Float32Array(n * nFeat);
  X.forEach((fila, i) => fila.forEach((v, j) => { flat[i * nFeat + j] = v ?? NaN; }));

  const ort = await getOrt();
  const out = await session.run({ X: new ort.Tensor("float32", flat, [n, nFeat]) });
  const raw = Array.from(out.probabilities.data);
  let proba = [];
  for (let i = 0; i < n; i++) proba.push([raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]]);

  // Calibración isotónica post-hoc (solo frío)
  if (clase === "frio") {
    proba = _calibrarFrio(proba, artefactos.iso_calib_frio);
  }

  const predArgmax = proba[0].indexOf(Math.max(...proba[0]));

  proba[0] = aplicarFactorEdad(proba[0], clase, grupoEdad);

  const uGlobal = artefactos.class_thresholds.CLASS_THRESHOLDS_RECOMENDADOS[clase] || { t1: 0.5, t2: 0.4 };
  let u = { t1: uGlobal.t1, t2: uGlobal.t2 };
  if (provincia) {
    const umbProv = (artefactos[`umbrales_provincia_${clase}`] || {})[provincia];
    if (umbProv) u = { t1: umbProv.t1, t2: umbProv.t2 };
  }

  const predTh = applyClassThresholds(proba, u.t1, u.t2)[0];
  const probRiesgo = 1.0 - proba[0][0];

  const qhat = artefactos[`conformal_${clase}`].qhat;
  const setSize = _conformalSetSize(proba[0], qhat);

  return {
    clase_argmax: predArgmax,
    clase_threshold: predTh,
    probabilidades: proba[0].map(round4),
    prob_riesgo: round4(probRiesgo),
    thresholds_usados: u,
    conformal_confianza: _confidenceLabel(setSize),
    conformal_set_size: setSize,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// _predecir_lstm (ensemble.py:184) — logits → softmax en JS
// ─────────────────────────────────────────────────────────────────────────────
function _softmax(logits) {
  const m = Math.max(...logits);
  const ex = logits.map((x) => Math.exp(x - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((x) => x / s);
}

async function _predecirLstm(session, dfHora, dfFeatures, provincia, grupoEdad, artefactos) {
  const seq = buildSequence24h(dfHora);
  if (!seq) return { error: "No hay datos horarios para LSTM" };
  const dailyVec = buildDailyFeatureVector(dfFeatures, artefactos.daily_feature_cols.daily_feature_cols);
  if (!dailyVec) return { error: "No se pudieron generar features diarias para LSTM" };

  const provName = provincia || "Madrid";
  const ineVec = getIneFeatures(provName, artefactos.ine_features);
  const pidx = getProvinceIdx(provName, artefactos.provincia_mapping);

  const [seqS, ineS, dailyS] = escalarParaLstm(seq, ineVec, dailyVec, {
    seq: artefactos.scaler_secuencias_lstm,
    ine: artefactos.scaler_provincia_features,
    daily: artefactos.scaler_diarias_lstm,
  });

  const ort = await getOrt();
  const feeds = {
    x_seq: new ort.Tensor("float32", new Float32Array(seqS.flat(2)), [1, 24, 5]),
    provincia_idx: new ort.Tensor("int64", BigInt64Array.from([BigInt(pidx)]), [1]),
    x_ine: new ort.Tensor("float32", new Float32Array(ineS.map((v) => v ?? NaN)), [1, 4]),
    x_diarias: new ort.Tensor("float32", new Float32Array(dailyS.map((v) => v ?? NaN)), [1, 31]),
  };
  const out = await session.run(feeds);
  const logitsC = Array.from(out.logits_calor.data);
  const logitsF = Array.from(out.logits_frio.data);

  let probaC = _softmax(logitsC);
  let probaF = _softmax(logitsF);
  probaC = aplicarFactorEdad(probaC, "calor", grupoEdad);
  probaF = aplicarFactorEdad(probaF, "frio", grupoEdad);

  const uC = artefactos.class_thresholds.CLASS_THRESHOLDS_LSTM.calor || { t1: 0.6, t2: 0.55 };
  const uF = artefactos.class_thresholds.CLASS_THRESHOLDS_LSTM.frio || { t1: 0.4, t2: 0.35 };

  const predC = applyClassThresholds([probaC], uC.t1, uC.t2)[0];
  const predF = applyClassThresholds([probaF], uF.t1, uF.t2)[0];

  return {
    calor: {
      clase_argmax: probaC.indexOf(Math.max(...probaC)),
      clase_threshold: predC,
      probabilidades: probaC.map(round4),
      prob_riesgo: round4(1.0 - probaC[0]),
    },
    frio: {
      clase_argmax: probaF.indexOf(Math.max(...probaF)),
      clase_threshold: predF,
      probabilidades: probaF.map(round4),
      prob_riesgo: round4(1.0 - probaF[0]),
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// _predecir_formulas + _proba_from_formula (ensemble.py:382)
// ─────────────────────────────────────────────────────────────────────────────
function _finito(v, def) {
  const f = Number(v);
  return Number.isFinite(f) ? f : def;
}

export function probaFromFormula(current) {
  const t = _finito(current?.t2m_c, 20.0);
  const rh = _finito(current?.rh, 50.0);
  const ws = _finito(current?.wind_speed_kmh, 10.0);
  const hi = heatIndex(t, rh);
  const wc = windChill(t, ws);

  let probCalor;
  if (hi >= 39) probCalor = 0.95;
  else if (hi >= 32) probCalor = 0.60;
  else if (hi >= 27) probCalor = 0.35;
  else probCalor = 0.05 + (hi / 27.0) * 0.20;

  let probFrio;
  if (wc <= -25) probFrio = 0.95;
  else if (wc <= -10) probFrio = 0.55;
  else if (wc <= 0) probFrio = 0.30;
  else probFrio = 0.05;

  const hiClase = hi >= 39 ? 2 : hi >= 27 ? 1 : 0;
  const wcClase = wc <= -25 ? 2 : wc <= 0 ? 1 : 0;

  return {
    calor: {
      prob_riesgo: round4(Math.min(probCalor, 1.0)),
      clase: hiClase,
      heat_index_c: Math.round(hi * 100) / 100,
      categoria: ["seguro", "precaucion", "peligro", "peligro_extremo"][Math.min(hiClase, 3)],
    },
    frio: {
      prob_riesgo: round4(Math.min(probFrio, 1.0)),
      clase: wcClase,
      wind_chill_c: Math.round(wc * 100) / 100,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// _conformal_weighted_ensemble (ensemble.py:445)
// ─────────────────────────────────────────────────────────────────────────────
function _conformalWeightedEnsemble(modelResults, tipo, classThresholds, persThresholdPeligro) {
  const modelKeys = tipo === "calor"
    ? ["XGBoost_calor", "LSTM", "Formula"]
    : ["RandomForest_frio", "LSTM", "Formula"];

  let probSum = 0.0;
  let weightSum = 0.0;

  for (const key of modelKeys) {
    const res = modelResults[key];
    if (!res || res.error) continue;
    let prob = null;
    let weight = null;
    if (key === "LSTM") {
      const sub = res[tipo];
      if (!sub || sub.error) continue;
      prob = sub.prob_riesgo;
      if (prob == null) continue;
      weight = 1.0 / 2.0;
    } else if (key === "Formula") {
      const sub = res[tipo];
      if (!sub) continue;
      prob = sub.prob_riesgo;
      if (prob == null) continue;
      weight = 1.0 / 2.0;
    } else {
      prob = res.prob_riesgo;
      if (prob == null) continue;
      let setSize = res.conformal_set_size ?? 2;
      if (setSize == null || setSize <= 0) setSize = 2;
      weight = 1.0 / setSize;
    }
    prob = Number(prob);
    if (!Number.isFinite(prob)) continue;
    probSum += prob * weight;
    weightSum += weight;
  }

  if (weightSum <= 0) return { prob_riesgo: 0.0, clase: 0 };

  let probEns = probSum / weightSum;
  probEns = Math.min(Math.max(probEns, 0.0), 1.0);

  const thresholds = classThresholds.CLASS_THRESHOLDS_RECOMENDADOS[tipo] || { t1: 0.25 };
  let clase;
  if (probEns >= persThresholdPeligro) clase = 2;
  else if (probEns >= thresholds.t1) clase = 1;
  else clase = 0;

  return { prob_riesgo: round4(probEns), clase };
}

// ─────────────────────────────────────────────────────────────────────────────
// perfil_horario_desde_df (ensemble.py:291) — solo resolución 60 en la demo
// ─────────────────────────────────────────────────────────────────────────────
export function perfilHorarioDesdeDf(dfHora, targetDate, resMin = 60) {
  if (!dfHora || !dfHora.length) return null;
  const diaObjetivo = targetDate
    ? String(targetDate).slice(0, 10)
    : String(dfHora[dfHora.length - 1].fecha).slice(0, 10);
  const filas = dfHora.filter((r) => String(r.fecha).slice(0, 10) === diaObjetivo);
  if (!filas.length) return null;

  const horas = new Map();   // hora -> HI
  const temps = new Map();   // hora -> temp
  for (const r of filas) {
    const hi = r.heat_index_c;
    if (hi == null || !Number.isFinite(hi)) continue;
    const hora = r.hora;
    if (!horas.has(hora) || hi > horas.get(hora)) {
      horas.set(hora, hi);
      if (r.t2m_c != null && Number.isFinite(r.t2m_c)) {
        temps.set(hora, Math.round(r.t2m_c * 10) / 10);
      }
    }
  }
  if (!horas.size) return null;
  const ordenadas = [...horas.keys()].sort((a, b) => a - b);

  if (resMin === 60) {
    return ordenadas.map((h) => ({ hora: h, HI: horas.get(h), temp: temps.get(h) ?? null }));
  }

  // Interpolación lineal entre máximos horarios (res_min 5/15/30).
  const paso = resMin / 60.0;
  const nIntermedios = Math.trunc(60 / resMin);
  const perfil = [];
  for (let i = 0; i < ordenadas.length; i++) {
    const h = ordenadas[i];
    const hi = horas.get(h);
    const t = temps.get(h) ?? null;
    perfil.push({ hora: h, HI: hi, temp: t });
    let hNext, hiNext, tNext;
    if (i + 1 < ordenadas.length) {
      hNext = ordenadas[i + 1];
      hiNext = horas.get(hNext);
      tNext = temps.get(hNext) ?? null;
    } else {
      hNext = h;
      hiNext = hi;
      tNext = t;
    }
    const span = hNext - h;
    for (let q = 1; q < nIntermedios; q++) {
      let hiQ, tQ;
      if (span > 0) {
        const frac = (q * paso) / span;
        hiQ = hi + frac * (hiNext - hi);
        tQ = t != null && tNext != null ? t + frac * (tNext - t) : null;
      } else {
        hiQ = hi;
        tQ = t;
      }
      perfil.push({
        hora: h + q * paso,
        HI: Math.round(hiQ * 10000) / 10000,
        temp: tQ != null ? Math.round(tQ * 10000) / 10000 : null,
      });
    }
  }
  return perfil;
}

// ─────────────────────────────────────────────────────────────────────────────
// predict_ensemble (ensemble.py:532) — orquesta todo
// ─────────────────────────────────────────────────────────────────────────────
export async function predictEnsemble({ weather, provincia = "Madrid", perfil = {}, targetDate = null }, modelos, artefactos) {
  const dfFeatures = weather.df_features;
  const dfHora = weather.df_hora;

  const estrato = edadAestrato(perfil.edad);

  const [xgbResult, rfResult, lstmResult] = await Promise.all([
    _predecirTabular(modelos.xgb, "calor", dfFeatures, provincia, estrato, artefactos),
    _predecirTabular(modelos.rf, "frio", dfFeatures, provincia, estrato, artefactos),
    _predecirLstm(modelos.lstm, dfHora, dfFeatures, provincia, estrato, artefactos),
  ]);
  const resultados = {
    XGBoost_calor: xgbResult,
    RandomForest_frio: rfResult,
    LSTM: lstmResult,
    Formula: probaFromFormula(weather.current),
  };

  const ct = artefactos.class_thresholds;
  const ensCalor = _conformalWeightedEnsemble(resultados, "calor", ct, ct.PERS_THRESHOLD_PELIGRO);
  const ensFrio = _conformalWeightedEnsemble(resultados, "frio", ct, ct.PERS_THRESHOLD_PELIGRO);

  const claseMlOriginal = Math.max(ensCalor.clase, ensFrio.clase);

  const perfilHorario = perfilHorarioDesdeDf(dfHora, targetDate || weather.target_date, 60);
  if (perfilHorario && perfil) {
    perfil._perfil_horario = perfilHorario;
  }

  let overrideFisico = null;
  const formulaResult = resultados.Formula;
  const HIcurrent = formulaResult.calor.heat_index_c;
  const WC = formulaResult.frio.wind_chill_c;
  const UV = weather.uv_index ?? null;
  let HI = HIcurrent;
  if (perfilHorario) {
    const inicio = perfil.hora_inicio;
    const duracion = perfil.duracion_actividad_h;
    if (inicio != null && duracion != null) {
      const fin = inicio + duracion;
      const ventana = perfilHorario.filter((h) => inicio <= h.hora && h.hora < fin);
      if (ventana.length) HI = Math.max(...ventana.map((h) => h.HI));
      else HI = Math.max(...perfilHorario.map((h) => h.HI));
    } else {
      HI = Math.max(...perfilHorario.map((h) => h.HI));
    }
  }

  const _personalizarSiHay = (probPoblacional, tipo) => {
    if (!Number.isFinite(Number(probPoblacional))) probPoblacional = 0.0;
    const perfilUv = { ...perfil };
    const uv = weather.uv_index;
    if (uv != null) perfilUv._uv_index = uv;
    const current = weather.current || {};
    if (current.wind_speed_kmh != null) perfilUv._wind_speed_kmh = current.wind_speed_kmh;
    if (current.rh != null) perfilUv._rh = current.rh;
    if (Object.values(perfilUv).some((v) => v != null)) {
      return personalizarRiesgo(probPoblacional, perfilUv, tipo, artefactos.factores_riesgo.cap_factores, artefactos.factores_riesgo);
    }
    return {
      indice_personalizado: probPoblacional,
      factor_total: 1.0,
      producto_bruto: 1.0,
      capado: false,
      factores: [],
    };
  };

  const resCalor = _personalizarSiHay(ensCalor.prob_riesgo, "calor");
  const resFrio = _personalizarSiHay(ensFrio.prob_riesgo, "frio");

  const perfilAplicado = {
    calor: {
      prob_poblacional: ensCalor.prob_riesgo,
      factor_total: resCalor.factor_total,
      producto_bruto: resCalor.producto_bruto,
      capado: resCalor.capado,
      prob_personalizada: resCalor.indice_personalizado,
      factores: resCalor.factores,
    },
    frio: {
      prob_poblacional: ensFrio.prob_riesgo,
      factor_total: resFrio.factor_total,
      producto_bruto: resFrio.producto_bruto,
      capado: resFrio.capado,
      prob_personalizada: resFrio.indice_personalizado,
      factores: resFrio.factores,
    },
  };

  const probPers = Math.max(resCalor.indice_personalizado, resFrio.indice_personalizado);
  const umbralPers = ct.CLASS_THRESHOLDS_RECOMENDADOS.calor || { t1: 0.25 };
  let clasePers = 0;
  if (probPers >= ct.PERS_THRESHOLD_PELIGRO) clasePers = 2;
  else if (probPers >= umbralPers.t1) clasePers = 1;

  // Overrides por seguridad física (HI / WC) y downgrade sin calor real.
  if (HI != null && overrideFisico == null) {
    if (HI >= 39 && clasePers < 2) {
      overrideFisico = {
        clase_ml: claseMlOriginal,
        clase_final: 2,
        razon: `ML=${CLASES[claseMlOriginal]}, HI_peak=${HI.toFixed(1)}C>=39 → PELIGRO`,
      };
    } else if (HI >= 27 && clasePers < 1) {
      const edadVulnerable = (perfil.edad ?? 0) >= 60 && !(perfil.entrenado && perfil.aclimatado);
      const vulnerableCalor =
        perfil.comorbilidades?.length ||
        perfil.farmacos?.length ||
        edadVulnerable ||
        perfil.aclimatado === false ||
        resCalor.factor_total > 1.8 ||
        resCalor.capado;
      if (vulnerableCalor && (HI >= 32 || (UV != null && UV > 3))) {
        overrideFisico = {
          clase_ml: claseMlOriginal,
          clase_final: 1,
          razon: `ML=${CLASES[claseMlOriginal]}, HI_peak=${HI.toFixed(1)}C>=${HI >= 32 ? "32" : "27"}+UV>3 → PRECAUCION`,
        };
      }
    }
  }

  if (WC != null && overrideFisico == null && clasePers < 2) {
    if (WC <= -25) {
      overrideFisico = {
        clase_ml: claseMlOriginal,
        clase_final: 2,
        razon: `ML=${CLASES[claseMlOriginal]}, WC=${WC.toFixed(1)}C<=-25 → PELIGRO (riesgo de congelación)`,
      };
    } else if (WC <= -10 && clasePers < 1) {
      const edadVulnerableFrio = (perfil.edad ?? 0) >= 60 && !perfil.entrenado;
      const vulnerables = new Set(perfil.situacion_social || []);
      const vulnerableFrio =
        perfil.comorbilidades?.length ||
        edadVulnerableFrio ||
        [...vulnerables].some((s) => ["vive_solo", "no_sale", "vivienda_fria"].includes(s)) ||
        resFrio.factor_total > 1.8 ||
        resFrio.capado;
      if (vulnerableFrio) {
        overrideFisico = {
          clase_ml: claseMlOriginal,
          clase_final: 1,
          razon: `ML=${CLASES[claseMlOriginal]}, WC=${WC.toFixed(1)}C<=-10 → PRECAUCION`,
        };
      }
    }
  }

  if (HI != null && HI < 27 && WC != null && WC > 0 && (UV == null || UV < 6) && clasePers > 0) {
    if (overrideFisico == null) {
      if (clasePers === 2) {
        overrideFisico = {
          clase_ml: claseMlOriginal,
          clase_final: 1,
          razon: `HI_peak=${HI.toFixed(1)}C<27, WC=${WC.toFixed(1)}C>0, UV<6 → PRECAUCION (ML=${CLASES[claseMlOriginal]}, pero sin calor actual)`,
        };
      } else if (clasePers === 1) {
        overrideFisico = {
          clase_ml: claseMlOriginal,
          clase_final: 1,
          razon: `HI_peak=${HI.toFixed(1)}C<27, WC=${WC.toFixed(1)}C>0, UV<6 (ML detecta tendencia de riesgo aunque sin calor ahora)`,
        };
      }
    }
  }

  const claseFinal = overrideFisico ? overrideFisico.clase_final : clasePers;

  return {
    weather: {
      lat: weather.lat,
      lon: weather.lon,
      current: weather.current,
      uv_index: weather.uv_index ?? null,
      provincia,
      perfil_horario: perfilHorario,
    },
    modelos: resultados,
    perfil: perfilAplicado,
    perfil_usuario: perfil,
    clase_final: claseFinal,
    clase_final_label: CLASES[claseFinal] ?? "DESCONOCIDO",
    override_fisico: overrideFisico,
  };
}
