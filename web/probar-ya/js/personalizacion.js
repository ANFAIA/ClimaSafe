// personalizacion.js — port 1:1 de climasafeai/features/personalizacion.py.
// Modula el índice poblacional 0-1 con los factores del perfil individual.
// Las tablas de coeficientes viven en models/factores_riesgo.json (misma
// estructura que data/factores_riesgo.json de Python).

// Cap del producto de factores (CAP_FACTORES_DEFECTO)
export const CAP_FACTORES_DEFECTO = 3.0;

const ACTIVIDADES_ESFUERZO = new Set(["moderada", "intensa", "muy_intensa"]);

const REF_GRASA_MUJER = [[18, 24.0], [30, 25.5], [40, 27.0], [50, 28.0], [60, 28.0], [70, 27.5], [80, 27.0], [100, 26.0]];
const REF_GRASA_HOMBRE = [[18, 16.0], [30, 18.5], [40, 20.5], [50, 22.5], [60, 23.5], [70, 24.0], [80, 24.0], [100, 23.5]];

function _grasaReferencia(edad, sexo) {
  const tabla = sexo === "mujer" ? REF_GRASA_MUJER : REF_GRASA_HOMBRE;
  if (edad <= tabla[0][0]) return tabla[0][1];
  if (edad >= tabla[tabla.length - 1][0]) return tabla[tabla.length - 1][1];
  for (let i = 0; i < tabla.length - 1; i++) {
    const [x1, y1] = tabla[i];
    const [x2, y2] = tabla[i + 1];
    if (x1 <= edad && edad <= x2) return y1 + ((edad - x1) / (x2 - x1)) * (y2 - y1);
  }
  return tabla[tabla.length - 1][1];
}

function _factorGrasaRelativa(grasa, edad, sexo) {
  if (sexo !== "hombre" && sexo !== "mujer") sexo = "mujer";
  const ref = _grasaReferencia(edad, sexo);
  if (ref <= 0) return 1.0;
  const ratio = grasa / ref;
  return Math.max(0.85, Math.min(1.15, 1.0 + (ratio - 1.0) * 0.3));
}

function _factorEdadCalor(edad) {
  if (edad >= 85) return 2.0;
  if (edad >= 75) return 1.5;
  if (edad >= 65) return 1.2;
  if (edad >= 55) return 1.1;
  if (edad >= 45) return 1.05;
  return 1.0;
}

function _factorEdadFrio(edad) {
  if (edad >= 85) return 1.7;
  if (edad >= 75) return 1.4;
  if (edad >= 65) return 1.2;
  return 1.0;
}

// Tasa metabólica por deporte (MET, Compendium Ainsworth et al. 2024)
export const DEPORTE_MET = {
  pasear: 3.5, caminar: 4.0, senderismo: 6.0, ciclismo_suave: 6.8,
  ciclismo: 7.0, futbol: 7.0, trekking_mochila: 7.8, correr_suave: 7.5,
  tenis_dobles: 6.0, tenis: 8.0, btt: 8.5, futbol_competicion: 9.0,
  ciclismo_fuerte: 9.0, correr: 10.5,
};

export function nivelActividadDesdeMet(met) {
  if (met < 1.6) return "reposo";
  if (met < 3.0) return "ligera";
  if (met < 6.0) return "moderada";
  if (met < 8.0) return "intensa";
  return "muy_intensa";
}

export function nivelActividadDeDeporte(deporte) {
  if (!deporte) return null;
  const met = DEPORTE_MET[String(deporte).trim().toLowerCase()];
  return met != null ? nivelActividadDesdeMet(met) : null;
}

const ACTIVIDAD_CALOR = { reposo: 1.0, ligera: 1.1, moderada: 1.3, intensa: 1.6, muy_intensa: 2.0 };
const ACTIVIDAD_FRIO = { reposo: 1.0, ligera: 0.95, moderada: 0.9, intensa: 1.2, muy_intensa: 1.2 };

const OCUPACION_NIVELES = {
  oficina: [1.0, "Trabajo sedentario (oficina, interior climatizado)"],
  reparto: [1.35, "Reparto / conducción (vehículo, carga ligera, descensos)"],
  mantenimiento: [1.7, "Mantenimiento exterior / jardinería (continua, carga moderada)"],
  construccion: [2.2, "Construcción / albañilería (carga pesada, PPE, sol directo)"],
  campo: [2.7, "Campo / agricultura (máxima exposición, pieza, sin sombra)"],
};

const VIENTO_FRIO_UMBRAL = 10.0;
const VIENTO_FRIO_MAX = 40.0;

function _factorVientoFrio(vientoKmh, actividad) {
  if (vientoKmh == null) return 1.0;
  if (!(actividad === "intensa" || actividad === "muy_intensa")) return 1.0;
  if (vientoKmh <= VIENTO_FRIO_UMBRAL) return 1.0;
  const fraccion = Math.min((vientoKmh - VIENTO_FRIO_UMBRAL) / (VIENTO_FRIO_MAX - VIENTO_FRIO_UMBRAL), 1.0);
  return 1.0 + 0.5 * fraccion;
}

function _factorDuracionCalor(horas) {
  if (horas > 4) return 1.4;
  if (horas > 2) return 1.25;
  if (horas > 1) return 1.1;
  return 1.0;
}

function _solapamientoHoras(inicio, duracion, ventanaInicio, ventanaFin) {
  const fin = inicio + duracion;
  const overlapStart = Math.max(inicio, ventanaInicio);
  const overlapEnd = Math.min(fin, ventanaFin);
  const overlapH = Math.max(0.0, overlapEnd - overlapStart);
  return duracion > 0 ? overlapH / duracion : 0.0;
}

function _factorHoraCalor(horaInicio, duracion) {
  if (horaInicio == null || duracion == null) return 1.0;
  const overlap = _solapamientoHoras(horaInicio, duracion, 12, 18);
  if (overlap >= 0.75) return 1.3;
  if (overlap >= 0.5) return 1.2;
  if (overlap > 0.0) return 1.1;
  return 1.0;
}

function _factorHoraFrio(horaInicio, duracion) {
  if (horaInicio == null || duracion == null) return 1.0;
  const overlap = _solapamientoHoras(horaInicio, duracion, 4, 8);
  if (overlap >= 0.75) return 1.3;
  if (overlap >= 0.5) return 1.2;
  if (overlap > 0.0) return 1.1;
  return 1.0;
}

const UMBRAL_DURACION_FATIGA = { muy_intensa: 1, intensa: 2, moderada: 3 };
const FACTOR_FATIGA_BASE = { muy_intensa: 1.35, intensa: 1.2, moderada: 1.1 };

function _factorFatigaAcumulada(perfil) {
  const horaInicio = perfil.hora_inicio;
  const duracion = perfil.duracion_actividad_h;
  const actividad = perfil.nivel_actividad || "reposo";
  const perfilHorario = perfil._perfil_horario;
  if (horaInicio == null || duracion == null || !perfilHorario) return null;
  if (!ACTIVIDADES_ESFUERZO.has(actividad)) return null;
  const umbral = UMBRAL_DURACION_FATIGA[actividad] ?? 4;
  if (duracion < umbral) return null;
  const fin = horaInicio + duracion;
  const window = perfilHorario.filter((h) => h.HI != null && horaInicio <= h.hora && h.hora < fin);
  if (!window.length) return null;
  const peak = window.reduce((a, b) => (a.HI > b.HI ? a : b));
  const peakHI = peak.HI;
  if (peakHI < 27) return null;
  const horasHastaPico = peak.hora - horaInicio;
  const factorBase = FACTOR_FATIGA_BASE[actividad] ?? 1.1;
  const fraccionPico = duracion > 0 ? horasHastaPico / duracion : 0;
  const bonus = 0.05 * Math.max(0, fraccionPico - 0.3);
  const factor = Math.min(factorBase + bonus, 1.5);
  const label = `fatiga acumulada (${horasHastaPico.toFixed(0)}/${duracion.toFixed(0)}h de ${actividad}, HI pico ${peakHI.toFixed(1)}C)`;
  return [label, factor];
}

// ─────────────────────────────────────────────────────────────────────────────
// Factores implementados desde models/factores_riesgo.json
// ─────────────────────────────────────────────────────────────────────────────
function _factoresImplementados(factoresJson, tipo, categoria) {
  const raw = ((factoresJson[tipo] || {})[categoria]) || {};
  const out = {};
  for (const [k, v] of Object.entries(raw)) {
    if (v.implementado) out[k] = { coef: v.coef, nombre: v.nombre };
  }
  return out;
}

function _factoresCalor(perfil, fr) {
  const f = [];
  const fisioCal = _factoresImplementados(fr, "calor", "fisiologico");
  const comorbCal = _factoresImplementados(fr, "calor", "comorbilidades");
  const farmacosCal = _factoresImplementados(fr, "calor", "farmacos");
  const socialCal = _factoresImplementados(fr, "calor", "situacional");

  if (perfil.sexo === "hombre" || perfil.sexo === "mujer") {
    const v = perfil.sexo === "hombre" ? 0.96 : 1.04;
    f.push(["sexo " + perfil.sexo, "fisiologico", v]);
  }

  if (perfil.edad != null) {
    const v = _factorEdadCalor(perfil.edad);
    if (v !== 1.0) f.push([`edad ${perfil.edad} años`, "fisiologico", v]);
  }

  const actividad = perfil.nivel_actividad || "reposo";
  let vAct = ACTIVIDAD_CALOR[actividad] ?? 1.0;
  if (vAct !== 1.0) {
    if (perfil.entrenado && ACTIVIDADES_ESFUERZO.has(actividad)) {
      vAct = 1.0 + (vAct - 1.0) * 0.5;
    }
    let label = `actividad ${actividad}`;
    if (perfil.entrenado && ACTIVIDADES_ESFUERZO.has(actividad)) label += " (entrenado)";
    if (perfil.deporte) label = `${perfil.deporte} (${label})`;
    f.push([label, "fisiologico", Math.round(vAct * 1000) / 1000]);
  }

  if (perfil.duracion_actividad_h != null) {
    const v = _factorDuracionCalor(perfil.duracion_actividad_h);
    if (v !== 1.0) f.push([`duración ${perfil.duracion_actividad_h} h`, "fisiologico", v]);
  }

  const vHora = _factorHoraCalor(perfil.hora_inicio, perfil.duracion_actividad_h);
  if (vHora !== 1.0 && perfil.duracion_actividad_h != null) {
    f.push([`hora inicio ${perfil.hora_inicio}:00 (solapa pico calor)`, "fisiologico", vHora]);
  }

  if (perfil.porcentaje_grasa != null && perfil.edad != null) {
    const v = _factorGrasaRelativa(perfil.porcentaje_grasa, perfil.edad, perfil.sexo);
    if (v !== 1.0) {
      const ref = _grasaReferencia(perfil.edad, perfil.sexo || "mujer");
      const direccion = perfil.porcentaje_grasa > ref ? "superior" : "inferior";
      f.push([`grasa corporal ${perfil.porcentaje_grasa}% (ref. ${ref.toFixed(0)}% para ${perfil.edad}a/${perfil.sexo || '?'}: ${direccion})`, "fisiologico", Math.round(v * 1000) / 1000]);
    }
  }

  if (perfil.aclimatado === false) {
    const ac = fisioCal.no_aclimatado;
    if (ac) f.push(["no aclimatado", "fisiologico", ac.coef]);
  }
  if (perfil.falta_sueno) {
    const fs = fisioCal.falta_sueno;
    if (fs) f.push([fs.nombre, "fisiologico", fs.coef]);
  }
  if (perfil.enfermedad_reciente) {
    const er = fisioCal.enfermedad_reciente;
    if (er) f.push([er.nombre, "fisiologico", er.coef]);
  }

  const comorb = new Set(perfil.comorbilidades || []);
  for (const k of comorb) {
    if (comorbCal[k] && k !== "mental") f.push([comorbCal[k].nombre, "medico", comorbCal[k].coef]);
  }

  const farmacos = new Set(perfil.farmacos || []);
  if (comorb.has("mental") || farmacos.has("antipsicoticos")) {
    const ap = comorbCal.mental || farmacosCal.antipsicoticos;
    if (ap) f.push([ap.nombre, "medico", ap.coef]);
  }
  for (const k of farmacos) {
    if (farmacosCal[k] && k !== "antipsicoticos") f.push([farmacosCal[k].nombre, "medico", farmacosCal[k].coef]);
  }

  const fatiga = _factorFatigaAcumulada(perfil);
  if (fatiga) f.push([fatiga[0], "fisiologico", fatiga[1]]);

  const sociales = new Set(perfil.situacion_social || []);
  if (perfil.fiesta) f.push(["fiesta / consumo de alcohol reciente", "situacional", 1.8]);
  else if (sociales.has("fiesta")) f.push(["fiesta / consumo de alcohol reciente", "situacional", 1.8]);
  const presentes = [...sociales].filter((k) => socialCal[k] && k !== "fiesta").map((k) => [k, socialCal[k]]);
  if (presentes.length) {
    const mejor = presentes.reduce((a, b) => (b[1].coef > a[1].coef ? b : a));
    f.push([`aislamiento/dependencia (${mejor[1].nombre})`, "situacional", mejor[1].coef]);
  }

  // Factor UV según fototipo (solo si hay índice UV disponible)
  const uvIndex = perfil._uv_index;
  const fototipo = perfil.fototipo;
  if (uvIndex != null && fototipo != null && uvIndex > 3) {
    const fotoMap = { "1": 1.3, "2": 1.2, "3": 1.1, "4": 1.0, "5": 0.9, "6": 0.85 };
    const uvFactor = fotoMap[String(fototipo)] ?? 1.0;
    const intensidad = uvIndex / 11.0;
    const factor = 1.0 + (uvFactor - 1.0) * intensidad;
    if (factor > 1.0) f.push([`UV ${uvIndex.toFixed(1)} + fototipo ${fototipo}`, "fisiologico", Math.round(factor * 100) / 100]);
  }

  if (perfil.ocupacion && OCUPACION_NIVELES[perfil.ocupacion]) {
    const [coef, label] = OCUPACION_NIVELES[perfil.ocupacion];
    if (coef !== 1.0) f.push([`trabajo ${label}`, "ocupacional", coef]);
  }

  return f;
}

function _factoresFrio(perfil, fr) {
  const f = [];
  const comorbFrio = _factoresImplementados(fr, "frio", "comorbilidades");
  const socialFrio = _factoresImplementados(fr, "frio", "situacional");

  if (perfil.sexo === "hombre" || perfil.sexo === "mujer") {
    const v = perfil.sexo === "hombre" ? 1.15 : 0.87;
    f.push(["sexo " + perfil.sexo, "fisiologico", v]);
  }

  if (perfil.edad != null) {
    const v = _factorEdadFrio(perfil.edad);
    if (v !== 1.0) f.push([`edad ${perfil.edad} años`, "fisiologico", v]);
  }

  const actividad = perfil.nivel_actividad || "reposo";
  let vAct = ACTIVIDAD_FRIO[actividad] ?? 1.0;
  const vViento = _factorVientoFrio(perfil._wind_speed_kmh, actividad);
  let v = vAct * vViento;
  if (v !== 1.0) {
    if (perfil.entrenado && ACTIVIDADES_ESFUERZO.has(actividad)) {
      vAct = 1.0 + (vAct - 1.0) * 0.5;
      v = vAct * vViento;
    }
    let label = `actividad ${actividad}`;
    if (perfil.entrenado && ACTIVIDADES_ESFUERZO.has(actividad)) label += " (entrenado)";
    if (perfil.deporte) label = `${perfil.deporte} (${label})`;
    if (vViento > 1.0) label += ` + viento ${perfil._wind_speed_kmh ?? '?'} km/h`;
    f.push([label, "fisiologico", Math.round(v * 100) / 100]);
  }

  const vHora = _factorHoraFrio(perfil.hora_inicio, perfil.duracion_actividad_h);
  if (vHora !== 1.0) {
    f.push([`hora inicio ${perfil.hora_inicio}:00 (solapa amanecer)`, "fisiologico", vHora]);
  }

  if (perfil.porcentaje_grasa != null && perfil.edad != null) {
    const vg = _factorGrasaRelativa(perfil.porcentaje_grasa, perfil.edad, perfil.sexo);
    if (vg !== 1.0) {
      const ref = _grasaReferencia(perfil.edad, perfil.sexo || "mujer");
      const direccion = perfil.porcentaje_grasa > ref ? "superior" : "inferior";
      f.push([`grasa corporal ${perfil.porcentaje_grasa}% (ref. ${ref.toFixed(0)}% para ${perfil.edad}a/${perfil.sexo || '?'}: ${direccion})`, "fisiologico", Math.round(vg * 1000) / 1000]);
    }
  }

  const comorb = new Set(perfil.comorbilidades || []);
  for (const k of comorb) {
    if (comorbFrio[k]) f.push([comorbFrio[k].nombre, "medico", comorbFrio[k].coef]);
  }

  const sociales = new Set(perfil.situacion_social || []);
  if (perfil.fiesta) f.push(["fiesta / consumo de alcohol reciente", "situacional", 1.8]);
  else if (sociales.has("fiesta")) f.push(["fiesta / consumo de alcohol reciente", "situacional", 1.8]);
  const presentes = [...sociales].filter((k) => socialFrio[k] && k !== "fiesta").map((k) => [k, socialFrio[k]]);
  if (presentes.length) {
    const mejor = presentes.reduce((a, b) => (b[1].coef > a[1].coef ? b : a));
    f.push([`aislamiento/vivienda fría (${mejor[1].nombre})`, "situacional", mejor[1].coef]);
  }

  return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// personalizar_riesgo (personalizacion.py:484) — composición en ODDS.
// ─────────────────────────────────────────────────────────────────────────────
export function personalizarRiesgo(indice, perfil, tipo, capFactores = CAP_FACTORES_DEFECTO, factoresJson) {
  if (!(indice >= 0.0 && indice <= 1.0)) {
    throw new Error(`indice debe estar en [0, 1], no ${indice}`);
  }
  if (tipo !== "calor" && tipo !== "frio") {
    throw new Error(`tipo debe ser 'calor' o 'frio', no ${tipo}`);
  }
  const p = { ...perfil };
  for (const alias of ["grasa_corporal", "grasa"]) {
    if (alias in p && !("porcentaje_grasa" in p)) {
      p.porcentaje_grasa = p[alias];
      delete p[alias];
    }
  }

  const factores = tipo === "calor" ? _factoresCalor(p, factoresJson) : _factoresFrio(p, factoresJson);

  let producto = 1.0;
  for (const [, , valor] of factores) producto *= valor;
  const factorTotal = Math.min(producto, capFactores);
  const capado = producto > capFactores;

  let personalizado;
  if (indice === 0.0 || indice === 1.0) {
    personalizado = indice;
  } else {
    const odds = indice / (1.0 - indice);
    const oddsInd = odds * factorTotal;
    personalizado = oddsInd / (1.0 + oddsInd);
  }

  return {
    indice_original: indice,
    factor_total: Math.round(factorTotal * 1000) / 1000,
    producto_bruto: Math.round(producto * 1000) / 1000,
    capado,
    indice_personalizado: Math.round(personalizado * 10000) / 10000,
    factores: factores.map(([nombre, categoria, factor]) => ({ nombre, categoria, factor })),
  };
}
