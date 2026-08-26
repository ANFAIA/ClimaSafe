// main.js — orquesta la demo probar-ya (WEB-012):
//   1. Carga artefactos JSON + sesiones ONNX (XGB/RF/LSTM).
//   2. Al pulsar «Predecir»: lee el formulario → perfil; intenta Open-Meteo
//      (weather.js) y si falla la red usa un escenario precargado
//      (scenarios.json — la misma fuente que el test de paridad).
//   3. predictEnsemble → render: clase, % riesgo, perfil horario por barras,
//      recomendaciones y detalle.
import { cargarArtefactos, cargarModelosOrt, leerJson } from "./artefactos.js";
import { getOrt } from "./ort-runtime.js";
import { fetchWeatherData, provinciaMasCercana, getProvinceCoords } from "./weather.js?v=20260814";
import { predictEnsemble } from "./modelos.js?v=20260814";
import { generarRecomendaciones } from "./recomendaciones.js";
import { nivelActividadDeDeporte } from "./personalizacion.js";
// WEB-016: redacción local del parte con un LLM (transformers.js, opt-in).
import { initParteIA } from "./llm.js?v=20260826";

// i18n (WEB-014): textos dinámicos vía el diccionario de js/i18n.js. Si el
// mecanismo no está cargado (p.ej. en tests), t() devuelve la clave tal cual.
const { t } = window.ClimaSafeI18n || { t: (k) => k };

const $ = (id) => document.getElementById(id);

// WEB-015: localStorage key for demo profile
const PROFILE_KEY = "climasafe_demo_profile";

function guardarPerfil(perfil) {
  try {
    localStorage.setItem(PROFILE_KEY, JSON.stringify(perfil));
  } catch (e) {
    console.warn("No se pudo guardar el perfil en localStorage:", e);
  }
}

function cargarPerfil() {
  try {
    const raw = localStorage.getItem(PROFILE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (e) {
    console.warn("No se pudo cargar el perfil de localStorage:", e);
    return null;
  }
}

function borrarPerfil() {
  try {
    localStorage.removeItem(PROFILE_KEY);
  } catch (e) {
    console.warn("No se pudo borrar el perfil de localStorage:", e);
  }
}

function restaurarFormularioDesdePerfil(perfil) {
  if (!perfil) return;
  $("edad").value = perfil.edad ?? 45;
  $("sexo").value = perfil.sexo ?? "hombre";
  $("grasa").value = perfil.porcentaje_grasa ?? "";
  $("aclimatado").checked = !!perfil.aclimatado;
  $("entrenado").checked = !!perfil.entrenado;
  if (perfil.nivel_actividad) $("nivel_actividad").value = perfil.nivel_actividad;
  if (perfil.deporte) {
    $("deporte").value = perfil.deporte;
  }
  $("hora_inicio").value = perfil.hora_inicio ?? 10;
  $("duracion").value = perfil.duracion_actividad_h ?? 2;
  // Comorbilidades
  document.querySelectorAll(".comorb").forEach((c) => (c.checked = false));
  (perfil.comorbilidades || []).forEach((v) => {
    const el = document.querySelector(`.comorb[value="${v}"]`);
    if (el) el.checked = true;
  });
  // Farmacos
  document.querySelectorAll(".farmaco").forEach((c) => (c.checked = false));
  (perfil.farmacos || []).forEach((v) => {
    const el = document.querySelector(`.farmaco[value="${v}"]`);
    if (el) el.checked = true;
  });
  // Social
  document.querySelectorAll(".social").forEach((c) => (c.checked = false));
  (perfil.situacion_social || []).forEach((v) => {
    const el = document.querySelector(`.social[value="${v}"]`);
    if (el) el.checked = true;
  });
  $("fototipo").value = perfil.fototipo ?? "";
  $("falta_sueno").checked = !!perfil.falta_sueno;
  $("enfermedad_reciente").checked = !!perfil.enfermedad_reciente;
  $("fiesta").checked = !!perfil.fiesta;
  $("alcohol_reciente").checked = !!perfil.alcohol_reciente;
  if (perfil.ocupacion) $("ocupacion").value = perfil.ocupacion;
  // Tipo de actividad (para mostrar/ocultar filas correctas)
  const tipo = perfil.deporte ? (perfil.deporte === "competicion" ? "competicion" : "deporte") : "reposo";
  // Actually, we need to infer from the original tipo_actividad. Let's check if we stored it.
  // For simplicity, we'll derive from deporte/ocupacion.
  if (perfil.ocupacion) {
    $("tipo_actividad").value = "trabajo";
  } else if (perfil.deporte) {
    $("tipo_actividad").value = "deporte"; // could be competicion too, but good enough
  } else {
    $("tipo_actividad").value = "reposo";
  }
  onCambiarTipoActividad();
  // Location fields (optional)
  if (perfil.lat) $("lat").value = perfil.lat;
  if (perfil.lon) $("lon").value = perfil.lon;
  if (perfil.provincia) $("provincia").value = perfil.provincia;
  if (perfil.fecha) $("fecha").value = perfil.fecha;
}

// Aviso médico-legal: ventana emergente al cargar. Se recuerda la aceptación
// en localStorage (mismo navegador), pero el disclaimer queda visible de forma
// permanente en los resultados.
const AVISO_KEY = "climasafe-aviso-medico-aceptado";
function initAvisoMedico() {
  const overlay = $("aviso-medico");
  const aceptar = $("aviso-aceptar");
  if (!overlay || !aceptar) return;
  const mostrar = () => overlay.classList.add("abierto");
  aceptar.addEventListener("click", () => {
    try { localStorage.setItem(AVISO_KEY, "1"); } catch { /* sin almacenamiento */ }
    overlay.classList.remove("abierto");
  });
  try {
    if (!localStorage.getItem(AVISO_KEY)) mostrar();
  } catch {
    mostrar(); // sin acceso a localStorage → mostrar siempre
  }
}

const CLASE_COLOR = { 0: "clase-0", 1: "clase-1", 2: "clase-2" };

let artefactos = null;
let modelos = null;
let catalogoRec = null;
let escenarios = null;
// WEB-016: última salida de predictEnsemble; el LLM local la usa como contexto.
let ultimaSalida = null;

// ─────────────────────────────────────────────────────────────────────────────
// Carga inicial
// ─────────────────────────────────────────────────────────────────────────────
async function init() {
  initAvisoMedico();
  initMapa();
  rellenarProvincias();
  rellenarHoras();
  $("tipo_actividad").addEventListener("change", onCambiarTipoActividad);
  onCambiarTipoActividad();

  // WEB-015: restore profile from localStorage on load
  const perfilGuardado = cargarPerfil();
  if (perfilGuardado) {
    restaurarFormularioDesdePerfil(perfilGuardado);
    // Update map marker if lat/lon restored
    if (perfilGuardado.lat && perfilGuardado.lon) {
      ponerMarcador(perfilGuardado.lat, perfilGuardado.lon);
    }
    // Show brief toast-like status
    const estado = $("estado");
    estado.className = "status";
    estado.textContent = t("perfil_restaurado");
    setTimeout(() => { estado.className = "status oculto"; }, 3000);
  }

  // WEB-015: delete button handler
  const btnBorrar = $("borrar-datos");
  if (btnBorrar) {
    btnBorrar.addEventListener("click", () => {
      borrarPerfil();
      // Reset form to defaults
      $("edad").value = 45;
      $("sexo").value = "hombre";
      $("grasa").value = "";
      $("aclimatado").checked = false;
      $("entrenado").checked = false;
      $("nivel_actividad").value = "moderada";
      $("deporte").value = "";
      $("hora_inicio").value = "10";
      $("duracion").value = "2";
      document.querySelectorAll(".comorb, .farmaco, .social").forEach((c) => (c.checked = false));
      $("fototipo").value = "";
      $("falta_sueno").checked = false;
      $("enfermedad_reciente").checked = false;
      $("fiesta").checked = false;
      $("alcohol_reciente").checked = false;
      $("ocupacion").value = "";
      $("tipo_actividad").value = "reposo";
      onCambiarTipoActividad();
      // Reset location fields
      $("lat").value = "";
      $("lon").value = "";
      $("provincia").value = "Madrid";
      $("fecha").value = "hoy";
      // Reset map marker
      ponerMarcador(40.4168, -3.7038);
      // Show confirmation
      const estado = $("estado");
      estado.className = "status";
      estado.textContent = t("datos_borrados");
      setTimeout(() => { estado.className = "status oculto"; }, 3000);
    });
  }

  try {
    const ort = await getOrt();
    // Los .wasm/.mjs de onnxruntime-web viven en vendor/, junto a ort.min.js.
    // Sin wasmPaths, onnxruntime resuelve los ficheros relativos al directorio
    // del propio script (vendor/) — que es donde están. Poner "./vendor/" aquí
    // duplicaba el prefijo (vendor/vendor/...) y rompía la carga (bug 2026-08-14).
    artefactos = await cargarArtefactos("./models");
    modelos = await cargarModelosOrt("./models", ort);
    catalogoRec = await leerJson("./models/recomendaciones.json");
    try {
      escenarios = await leerJson("./scenarios.json");
    } catch { escenarios = null; }
    $("carga-modelos").textContent = t("modelos_listos");
    $("predecir").disabled = false;
  } catch (e) {
    $("carga-modelos").className = "status err";
    $("carga-modelos").textContent = t("error_carga", e.message);
    console.error(e);
  }
}

function rellenarProvincias() {
  const sel = $("provincia");
  // Mismo orden alfabético que provincia_mapping.json (orden del LSTM).
  const nombres = [
    "Albacete", "Almería", "Araba/Álava", "Asturias", "Badajoz", "Barcelona",
    "Bizkaia", "Burgos", "Cantabria", "Ceuta", "Ciudad Real", "Cuenca",
    "Cáceres", "Cádiz", "Córdoba", "Gipuzkoa", "Girona", "Granada",
    "Guadalajara", "Huelva", "Huesca", "Jaén", "León", "Lleida", "Lugo",
    "Madrid", "Melilla", "Murcia", "Málaga", "Navarra", "Ourense", "Palencia",
    "Pontevedra", "Salamanca", "Santa Cruz de Tenerife", "Segovia", "Sevilla",
    "Soria", "Tarragona", "Teruel", "Toledo", "Valladolid", "Zamora",
    "Zaragoza", "Ávila",
  ];
  for (const n of nombres) {
    const opt = document.createElement("option");
    opt.value = n;
    opt.textContent = n;
    sel.appendChild(opt);
  }
  sel.value = "Madrid";
}

function rellenarHoras() {
  const sel = $("hora_inicio");
  for (let h = 0; h < 24; h++) {
    const opt = document.createElement("option");
    opt.value = String(h);
    opt.textContent = `${String(h).padStart(2, "0")}:00`;
    sel.appendChild(opt);
  }
  sel.value = "10";
}

// ─────────────────────────────────────────────────────────────────────────────
// Mapa interactivo (Leaflet + tiles oscuros): clic → lat/lon + provincia
// ─────────────────────────────────────────────────────────────────────────────
let mapa = null;
let marcador = null;

function _normOpcion(s) {
  return String(s).toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[/\s]+/g, "").trim();
}

function _opcionProvinciaMasCercana(clave) {
  const sel = $("provincia");
  for (const opt of sel.options) {
    if (_normOpcion(opt.value) === _normOpcion(clave) ||
        _normOpcion(opt.textContent) === _normOpcion(clave)) {
      return opt.value;
    }
  }
  return null;
}

function ponerMarcador(lat, lon) {
  if (!mapa) return;
  if (marcador) marcador.setLatLng([lat, lon]);
  else {
    marcador = L.circleMarker([lat, lon], {
      radius: 8, fillColor: "#4f8ef7", color: "#6ba3ff", weight: 2, fillOpacity: 0.85,
    }).addTo(mapa);
  }
  mapa.setView([lat, lon], Math.max(mapa.getZoom(), 6));
}

function initMapa() {
  const el = $("mapa");
  if (!el || typeof L === "undefined") return;
  mapa = L.map(el, { zoomControl: true, scrollWheelZoom: false }).setView([40.4168, -3.7038], 6);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; CARTO',
    maxZoom: 18,
  }).addTo(mapa);
  mapa.on("click", (e) => {
    const lat = +e.latlng.lat.toFixed(4);
    const lon = +e.latlng.lng.toFixed(4);
    $("lat").value = lat;
    $("lon").value = lon;
    const clave = provinciaMasCercana(lat, lon);
    if (clave) {
      const val = _opcionProvinciaMasCercana(clave);
      if (val) $("provincia").value = val;
    }
    ponerMarcador(lat, lon);
  });
  $("provincia").addEventListener("change", () => {
    const [plat, plon] = getProvinceCoords($("provincia").value);
    ponerMarcador(plat, plon);
  });
  ponerMarcador(40.4168, -3.7038);
}

// ─────────────────────────────────────────────────────────────────────────────
// Formulario → perfil
// ─────────────────────────────────────────────────────────────────────────────
function perfilDesdeFormulario() {
  const checks = (cls) =>
    [...document.querySelectorAll(`.${cls}:checked`)].map((c) => c.value);
  const deporte = $("deporte").value || null;
  // El MET del deporte fija la intensidad (igual que la web: _aplicar_deporte_a_nivel).
  const nivel = deporte ? (nivelActividadDeDeporte(deporte) || $("nivel_actividad").value)
                        : $("nivel_actividad").value;
  return {
    edad: Number($("edad").value) || 45,
    sexo: $("sexo").value,
    porcentaje_grasa: $("grasa").value ? Number($("grasa").value) : null,
    aclimatado: $("aclimatado").checked,
    entrenado: $("entrenado").checked,
    nivel_actividad: nivel,
    deporte,
    hora_inicio: Number($("hora_inicio").value),
    duracion_actividad_h: Number($("duracion").value) || 2,
    comorbilidades: checks("comorb"),
    farmacos: checks("farmaco"),
    situacion_social: checks("social"),
    fototipo: $("fototipo").value || null,
    falta_sueno: $("falta_sueno").checked,
    enfermedad_reciente: $("enfermedad_reciente").checked,
    fiesta: $("fiesta").checked || $("alcohol_reciente").checked,
    ocupacion: $("ocupacion").value || null,
    // WEB-015: location fields for persistence
    lat: $("lat").value ? Number($("lat").value) : null,
    lon: $("lon").value ? Number($("lon").value) : null,
    provincia: $("provincia").value,
    fecha: $("fecha").value,
  };
}

// Tipo de salida → muestra deporte (deporte/competición) u ocupación (trabajo).
function onCambiarTipoActividad() {
  const tipo = $("tipo_actividad").value;
  $("ocupacion-row").classList.toggle("oculto", tipo !== "trabajo");
  $("deporte-row").classList.toggle("oculto", !(tipo === "deporte" || tipo === "competicion"));
}

function ubicacionDesdeFormulario() {
  const lat = $("lat").value ? Number($("lat").value) : null;
  const lon = $("lon").value ? Number($("lon").value) : null;
  return { lat, lon, provincia: $("provincia").value };
}

// ─────────────────────────────────────────────────────────────────────────────
// Datos meteorológicos: Open-Meteo con fallback a escenario precargado
// ─────────────────────────────────────────────────────────────────────────────
async function obtenerWeather() {
  const { lat, lon, provincia } = ubicacionDesdeFormulario();
  const fechaSel = $("fecha").value;
  const targetDate = fechaSel === "manana" ? sumarDiasLocal(1) : hoyLocal();

  try {
    const weather = await fetchWeatherData({ lat, lon, provincia, targetDate });
    return { weather, offline: false, aviso: null };
  } catch (e) {
    // Fallback: escenario precargado (scenarios.json) — misma fuente del test.
    if (!escenarios || !escenarios.escenarios.length) {
      throw new Error(`Sin red y sin escenario precargado: ${e.message}`);
    }
    const esc = escenarios.escenarios.find(
      (x) => x.provincia.toLowerCase() === provincia.toLowerCase()
    ) || escenarios.escenarios[0];
    const targetRows = esc.horas.filter((r) => r.datetime.startsWith(esc.target_date));
    const histRows = esc.horas.filter((r) => !r.datetime.startsWith(esc.target_date));
    const { generarFeaturesCompletas } = await import("./features.js");
    const { dfFeatures, dfHora } = generarFeaturesCompletas(targetRows, histRows);
    const weather = {
      lat: esc.lat,
      lon: esc.lon,
      current: esc.current,
      df_hora: dfHora,
      df_features: dfFeatures,
      uv_index: null,
      target_date: esc.target_date,
    };
    return {
      weather,
      offline: true,
      aviso: `Open-Meteo no respondió (${e.message}). Usando escenario precargado: ${esc.nombre} (${esc.provincia}, ${esc.target_date}).`,
    };
  }
}

function hoyLocal() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
function sumarDiasLocal(n) {
  const d = new Date();
  d.setDate(d.getDate() + n);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Render
// ─────────────────────────────────────────────────────────────────────────────
function renderResultado(res, info) {
  const clase = res.clase_final;
  const badge = $("clase-badge");
  badge.textContent = t("clase_" + clase);
  badge.className = `clase-badge ${CLASE_COLOR[clase]}`;

  const probPers = Math.max(
    res.perfil.calor.prob_personalizada,
    res.perfil.frio.prob_personalizada
  );
  $("riesgo-big").textContent = (probPers * 100).toFixed(1);
  const p = res.perfil_usuario;
  $("info-salida").textContent =
    `${res.weather.provincia} · ${res.weather.target_date} · ` +
    `salida ${p.hora_inicio}:00 (${p.duracion_actividad_h} h) · ` +
    `HI pico ${res.weather.perfil_horario?.length ? Math.max(...res.weather.perfil_horario.map((x) => x.HI)).toFixed(1) : "—"} °C`;

  $("override-box").className = "oculto";
  if (res.override_fisico) {
    $("override-box").className = "override";
    $("override-box").textContent = res.override_fisico.razon;
  }

  $("modo-datos").textContent = info.offline
    ? `ℹ ${info.aviso}`
    : t("datos_tiempo_real");

  renderPerfil(res);
  renderRecomendaciones(res);
  renderDetalle(res);
}

function renderPerfil(res) {
  const perfil = res.weather.perfil_horario || [];
  const barras = $("perfil-barras");
  const eje = $("perfil-eje");
  barras.innerHTML = "";
  eje.innerHTML = "";
  if (!perfil.length) {
    $("perfil-leyenda").textContent = t("sin_perfil");
    return;
  }
  const hiMax = Math.max(...perfil.map((p) => p.HI));
  const inicio = res.perfil_usuario.hora_inicio;
  const fin = inicio + res.perfil_usuario.duracion_actividad_h;
  const etiquetas = [0, 6, 12, 18, 23];
  for (const p of perfil) {
    const h = Math.round(p.hora);
    const div = document.createElement("div");
    const hiClase = p.HI >= 39 ? 2 : p.HI >= 27 ? 1 : 0;
    div.className = "barra rango-" + hiClase + (h >= inicio && h < fin ? " ventana" : "");
    div.style.height = `${Math.max(4, (p.HI / hiMax) * 100)}%`;
    div.title = `${h}:00 · HI ${p.HI.toFixed(1)} °C${p.temp != null ? ` · ${p.temp.toFixed(1)} °C` : ""}`;
    barras.appendChild(div);
  }
  for (const h of etiquetas) {
    const s = document.createElement("span");
    s.textContent = `${h}:00`;
    eje.appendChild(s);
  }
  $("perfil-leyenda").textContent =
    `Barras: Heat Index por hora (máx. ${hiMax.toFixed(1)} °C). ` +
    `Verde: seguro · Ámbar: precaución · Rojo: peligro. ` +
    `Contorno blanco: ventana de actividad (${inicio}:00–${fin}:00).`;
}

function renderRecomendaciones(res) {
  const ul = $("recomendaciones");
  ul.innerHTML = "";
  const recs = generarRecomendaciones(res.perfil_usuario, res, catalogoRec);
  for (const r of recs) {
    const li = document.createElement("li");
    li.textContent = r;
    ul.appendChild(li);
  }
  if (!recs.length) {
    const li = document.createElement("li");
    li.textContent = t("sin_reco");
    ul.appendChild(li);
  }
}

function renderDetalle(res) {
  const d = $("detalle");
  d.innerHTML = "";

  const fila = (modelo, probRiesgo, clase, extra = "") => `
    <tr><td>${modelo}</td><td>${probRiesgo != null ? (probRiesgo * 100).toFixed(1) + " %" : "—"}</td>
    <td>${clase != null ? t("clase_" + clase) : "—"}</td><td>${extra}</td></tr>`;

  const m = res.modelos;
  let html = `<table class="tabla-modelos">
    <tr><th>${t("th_modelo")}</th><th>${t("th_prob")}</th><th>${t("th_clase")}</th><th>${t("th_nota")}</th></tr>
    ${fila("XGBoost calor", m.XGBoost_calor?.prob_riesgo, m.XGBoost_calor?.clase_threshold, `set_size ${m.XGBoost_calor?.conformal_set_size}`)}
    ${fila("RandomForest frío", m.RandomForest_frio?.prob_riesgo, m.RandomForest_frio?.clase_threshold, `set_size ${m.RandomForest_frio?.conformal_set_size}`)}
    ${fila("LSTM calor", m.LSTM?.calor?.prob_riesgo, m.LSTM?.calor?.clase_threshold)}
    ${fila("LSTM frío", m.LSTM?.frio?.prob_riesgo, m.LSTM?.frio?.clase_threshold)}
    ${fila("Fórmula (HI/WC)", m.Formula?.calor?.prob_riesgo, m.Formula?.calor?.clase, `HI ${m.Formula?.calor?.heat_index_c} °C`)}
    ${fila("Fórmula frío", m.Formula?.frio?.prob_riesgo, m.Formula?.frio?.clase, `WC ${m.Formula?.frio?.wind_chill_c} °C`)}
  </table>`;
  d.insertAdjacentHTML("beforeend", html);

  for (const canal of ["calor", "frio"]) {
    const p = res.perfil[canal];
    const factores = p.factores || [];
    let fhtml = "";
    for (const f of factores) {
      fhtml += `<div class="factor-item"><span class="f-nombre">${f.nombre}</span><span class="f-valor">×${f.factor}</span></div>`;
    }
    if (!fhtml) fhtml = `<p class="meta">Sin factores activos.</p>`;
    d.insertAdjacentHTML("beforeend", `
      <h4 style="margin:16px 0 8px;font-size:0.85rem;">Canal ${canal}</h4>
      <p class="meta">Poblacional ${(p.prob_poblacional * 100).toFixed(1)} % → personalizada ${(p.prob_personalizada * 100).toFixed(1)} %
      (factor total ×${p.factor_total}${p.capado ? ", CAP 3.0 aplicado" : ""})</p>
      ${fhtml}`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Acción
// ─────────────────────────────────────────────────────────────────────────────
async function predecir() {
  const estado = $("estado");
  estado.className = "status";
  estado.innerHTML = '<span class="spinner"></span>' + t("descargando");

  try {
    const perfil = perfilDesdeFormulario();
    // WEB-015: save profile to localStorage
    guardarPerfil(perfil);

    const { weather, offline, aviso } = await obtenerWeather();
    const res = await predictEnsemble(
      {
        weather,
        provincia: ubicacionDesdeFormulario().provincia,
        perfil,
        targetDate: weather.target_date,
      },
      modelos,
      artefactos
    );

    $("resultado-vacio").classList.add("oculto");
    $("resultado-contenido").classList.remove("oculto");
    renderResultado(res, {
      offline,
      aviso,
    });
    // WEB-016: disponible como contexto para la redacción local con el LLM.
    ultimaSalida = res;
    estado.className = "status oculto";
    $("resultado").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    estado.className = "status err";
    estado.textContent = `Error: ${e.message}`;
    console.error(e);
  }
}

$("predecir").addEventListener("click", predecir);
// WEB-016: tarjeta «Redactar el parte con IA local». Nada se descarga hasta
// que el usuario pulsa; si algo falla, el parte de plantilla sigue intacto.
initParteIA(() => ultimaSalida);
init();
