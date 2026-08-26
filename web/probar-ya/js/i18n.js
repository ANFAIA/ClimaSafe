/* i18n.js — detección de idioma es/en para la demo probar-ya (WEB-014).
 *
 * Mecanismo (idéntico al i18n.js de cacelass.github.io):
 *   1. localStorage['climasafe_lang'] (es|en) tiene prioridad sobre el navegador.
 *   2. Sin preferencia guardada → navigator.language: 'en*' → inglés; resto → español.
 *   3. El selector #lang-select cambia el idioma al instante y lo guarda.
 *
 * Uso:
 *   <script src="js/i18n.js"></script> antes del módulo main.js.
 *   Textos estáticos: <span data-i18n="clave">…</span> (+ data-i18n-html si hay HTML).
 *   Textos dinámicos (main.js/weather.js): ClimaSafeI18n.t('clave', arg0, …).
 *
 * Decisión de alcance (WEB-014): se traduce toda la UI del formulario (etiquetas,
 * opciones, botones, títulos) y los mensajes de estado principales. El contenido
 * profundo generado por JS (recomendaciones largas, razones de override, detalle
 * del ensemble) permanece en español por defecto.
 */
(function () {
  "use strict";

  var KEY = "climasafe_lang";

  var DICT = {
    es: {
      nav_form: "formulario",
      nav_resultado: "resultado",
      nav_volver: "volver",
      hero_h1_1: "Prueba el modelo de riesgo",
      hero_h1_2: "sin servidor, en tu navegador.",
      hero_sub: "El pipeline completo de <code>predict_ensemble</code> (3 modelos ONNX + fórmula + ensemble conformal + personalización) portado a JavaScript. Los datos vienen de Open-Meteo; si no hay red, se usa un escenario precargado.",
      cta_back: "← Volver",
      sec01_tag: "perfil y salida",
      sec01_title: "¿Dónde, cuándo y quién?",
      carga_models: "Cargando modelos ONNX (~26 MB)…",
      card_salida: "Salida y ubicación",
      lbl_provincia: "Provincia",
      lbl_fecha: "Fecha",
      opt_hoy: "Hoy",
      opt_manana: "Mañana",
      lbl_lat: "Latitud (opcional; por defecto la de la provincia)",
      lbl_lon: "Longitud (opcional)",
      lbl_mapa: "Mapa — haz clic para fijar la ubicación",
      meta_mapa: "Al hacer clic se rellenan latitud/longitud y se elige la provincia más cercana.",
      card_persona: "Persona",
      lbl_edad: "Edad",
      lbl_sexo: "Sexo",
      opt_hombre: "Hombre",
      opt_mujer: "Mujer",
      lbl_grasa: "% grasa corporal (opcional)",
      lbl_nivel: "Nivel de actividad",
      opt_reposo: "Reposo",
      opt_ligera: "Ligera",
      opt_moderada: "Moderada",
      opt_intensa: "Intensa",
      opt_muy_intensa: "Muy intensa",
      lbl_hora: "Hora de inicio",
      lbl_duracion: "Duración (horas)",
      check_aclimatado: "Aclimatado al calor",
      check_entrenado: "Entrenado para la actividad",
      card_que: "¿Qué vas a hacer?",
      lbl_tipo: "Tipo de salida",
      opt_sin_esfuerzo: "Reposo / sin esfuerzo",
      opt_trabajo: "Trabajo",
      opt_deporte: "Deporte / ejercicio",
      opt_competicion: "Competición",
      lbl_ocupacion: "Ocupación en la salida",
      opt_sel: "— selecciona —",
      opt_oficina: "Oficina / bajo techo",
      opt_reparto: "Reparto / conducción",
      opt_mantenimiento: "Mantenimiento exterior",
      opt_construccion: "Construcción",
      opt_campo: "Campo / agricultura",
      lbl_deporte: "Deporte (fija la intensidad por su MET)",
      opt_ninguno: "— ninguno —",
      opt_pasear: "Pasear (3,5 MET)",
      opt_caminar: "Caminar (4 MET)",
      opt_senderismo: "Senderismo (6 MET)",
      opt_tenis_dobles: "Tenis dobles (6 MET)",
      opt_ciclismo_suave: "Bici tranquila (6,8 MET)",
      opt_ciclismo: "Bici (7 MET)",
      opt_futbol: "Fútbol (7 MET)",
      opt_correr_suave: "Trotar (7,5 MET)",
      opt_trekking: "Trekking con mochila (7,8 MET)",
      opt_tenis: "Tenis individual (8 MET)",
      opt_btt: "BTT / montaña (8,5 MET)",
      opt_futbol_comp: "Fútbol competición (9 MET)",
      opt_ciclismo_fuerte: "Bici fuerte (9 MET)",
      opt_correr: "Correr (10,5 MET)",
      meta_deporte: "Si eliges deporte, la intensidad se fija sola según su MET (puedes ajustarla en «Persona»).",
      card_salud: "Salud y entorno",
      lbl_comorb: "Comorbilidades",
      check_cardiovascular: "Cardiovascular / HTA",
      check_diabetes: "Diabetes",
      check_respiratoria: "Respiratoria",
      check_mental: "Salud mental",
      lbl_medicacion: "Medicación",
      check_diureticos: "Diuréticos de asa",
      check_antipsicoticos: "Antipsicóticos",
      lbl_social: "Situación social",
      check_vive_solo: "Vive solo",
      check_no_sale: "No sale de casa",
      check_sin_aire: "Sin aire acondicionado",
      check_vivienda_fria: "Vivienda fría",
      lbl_fototipo: "Fototipo (Fitzpatrick)",
      opt_desconocido: "— desconocido —",
      opt_foto_1: "I — siempre se quema",
      opt_foto_2: "II — se quema fácil",
      opt_foto_3: "III — se quema a veces",
      opt_foto_4: "IV — se quema poco",
      opt_foto_5: "V — rara vez se quema",
      opt_foto_6: "VI — nunca se quema",
      check_dormi: "Dormí mal / menos de 6 h",
      check_enfermo: "Estoy resfriado / enfermo",
      check_fiesta: "Fiesta / celebración (alcohol + baile)",
      check_alcohol: "Consumí alcohol recientemente",
      btn_predecir: "Predecir riesgo →",
      sec02_tag: "resultado",
      sec02_title: "Riesgo para la salida elegida",
      resultado_vacio: "Completa el formulario y pulsa «Predecir riesgo». El resultado (clase, % de riesgo, perfil horario y recomendaciones) aparecerá aquí.",
      meta_clase: "Clase final",
      meta_riesgo: "Riesgo personalizado",
      card_perfil: "Perfil horario — Heat Index del día",
      card_reco: "Recomendaciones",
      summary_detalle: "Detalle del ensemble (probabilidades poblacionales, factores, override)",
      cta_entender: "Entender mejor el resultado →",
      disclaimer: "ClimaSafe no es un dispositivo médico y no sustituye el criterio clínico. Si presentas síntomas, contacta con emergencias (112) o con tu médico.",
      footer_extra: "· probar-ya · ONNX en el navegador",
      aviso_title: "Aviso médico-legal",
      aviso_p1: "<strong>ClimaSafe no es un dispositivo médico</strong> ni sustituye el criterio clínico de un profesional sanitario.",
      aviso_li1: "La estimación de riesgo se basa en modelos estadísticos e índices de la literatura epidemiológica; puede contener errores y no debe usarse para diagnosticar, decidir tratamientos ni planificar actividades de riesgo.",
      aviso_li2: "Es una <strong>demo educativa y de investigación</strong>: en la UE, los sistemas de salud pueden clasificarse como de alto riesgo según el Reglamento de Inteligencia Artificial.",
      aviso_li3: "Si presentas síntomas (mareo, confusión, dificultad para respirar, dolor en el pecho…), <strong>contacta con emergencias (112) o con tu médico</strong>.",
      btn_entendido: "Entendido",
      // Dinámicos (main.js / weather.js vía t())
      clase_0: "SEGURO",
      clase_1: "PRECAUCIÓN",
      clase_2: "PELIGRO",
      modelos_listos: "Modelos listos. Configura el formulario y pulsa Predecir.",
      error_carga: "Error cargando modelos: {0}",
      descargando: "Descargando datos y ejecutando los 3 modelos ONNX…",
      sin_reco: "Sin recomendaciones específicas para este perfil.",
      datos_tiempo_real: "Datos meteorológicos de Open-Meteo (tiempo real).",
      sin_perfil: "Sin perfil horario disponible.",
      th_modelo: "Modelo",
      th_prob: "P(riesgo)",
      th_clase: "Clase",
      th_nota: "Nota",
      err_fecha_pasada: "La fecha {0} ya pasó. Solo se predice hoy o el futuro cubierto por el forecast meteorológico.",
      err_forecast: "El forecast meteorológico no cubre {0}. No se puede predecir sin datos.",
      // WEB-015: localStorage profile + privacy
      privacy_notice: "Tus datos se guardan solo en tu navegador (localStorage). No salen de tu dispositivo ni se envían a ningún servidor. Puedes borrarlos en cualquier momento con el botón «Borrar mis datos».",
      btn_borrar_datos: "Borrar mis datos",
      btn_borrar_datos_aria: "Borrar los datos del perfil guardados en el navegador",
      datos_borrados: "Datos borrados. El formulario se ha restablecido.",
      perfil_restaurado: "Perfil restaurado desde el navegador.",
      // WEB-016: redacción local del parte con LLM (transformers.js)
      card_ia: "Redactar el parte con IA local",
      ia_desc: "Opcional: un LLM pequeño (IBM Granite 4.0 1B) se descarga una sola vez desde HuggingFace y redacta el parte en prosa usando los resultados ya calculados arriba.",
      ia_btn_activar: "Activar IA local (descarga única ≈ 1–2 GB)",
      ia_btn_repetir: "Volver a generar el parte",
      ia_cargando_lib: "Cargando transformers.js desde CDN…",
      ia_descargando: "Descargando modelo ({0})… {1}%",
      ia_generando: "Generando el parte en tu dispositivo (sin GPU puede tardar 1–3 min)…",
      ia_listo: "Parte redactado localmente con {0}.",
      ia_error: "IA local no disponible ({0}). El parte con la plantilla clásica sigue funcionando sin cambios.",
      ia_privacy: "Privacidad: el modelo corre íntegramente en tu navegador (WebGPU o WASM) y queda cacheado para usos siguientes. Ni tus datos ni el texto generado se envían a ningún servidor; solo la descarga inicial viene de HuggingFace.",
      ia_disclaimer: "Texto redactado por un LLM local a partir de los resultados deterministas del modelo. Puede contener errores; las cifras válidas son las mostradas arriba."
    },
    en: {
      nav_form: "form",
      nav_resultado: "results",
      nav_volver: "back",
      hero_h1_1: "Try the risk model",
      hero_h1_2: "no server, in your browser.",
      hero_sub: "The full <code>predict_ensemble</code> pipeline (3 ONNX models + formula + conformal ensemble + personalization) ported to JavaScript. Data comes from Open-Meteo; if offline, a preloaded scenario is used.",
      cta_back: "← Back",
      sec01_tag: "profile & outing",
      sec01_title: "Where, when and who?",
      carga_models: "Loading ONNX models (~26 MB)…",
      card_salida: "Outing & location",
      lbl_provincia: "Province",
      lbl_fecha: "Date",
      opt_hoy: "Today",
      opt_manana: "Tomorrow",
      lbl_lat: "Latitude (optional; defaults to the province)",
      lbl_lon: "Longitude (optional)",
      lbl_mapa: "Map — click to set the location",
      meta_mapa: "Clicking fills latitude/longitude and picks the nearest province.",
      card_persona: "Person",
      lbl_edad: "Age",
      lbl_sexo: "Sex",
      opt_hombre: "Man",
      opt_mujer: "Woman",
      lbl_grasa: "% body fat (optional)",
      lbl_nivel: "Activity level",
      opt_reposo: "Rest",
      opt_ligera: "Light",
      opt_moderada: "Moderate",
      opt_intensa: "Intense",
      opt_muy_intensa: "Very intense",
      lbl_hora: "Start time",
      lbl_duracion: "Duration (hours)",
      check_aclimatado: "Heat-acclimatised",
      check_entrenado: "Trained for the activity",
      card_que: "What will you do?",
      lbl_tipo: "Type of outing",
      opt_sin_esfuerzo: "Rest / no effort",
      opt_trabajo: "Work",
      opt_deporte: "Sport / exercise",
      opt_competicion: "Competition",
      lbl_ocupacion: "Occupation during the outing",
      opt_sel: "— select —",
      opt_oficina: "Office / indoors",
      opt_reparto: "Delivery / driving",
      opt_mantenimiento: "Outdoor maintenance",
      opt_construccion: "Construction",
      opt_campo: "Field / agriculture",
      lbl_deporte: "Sport (sets intensity by its MET)",
      opt_ninguno: "— none —",
      opt_pasear: "Walking (3.5 MET)",
      opt_caminar: "Walking (4 MET)",
      opt_senderismo: "Hiking (6 MET)",
      opt_tenis_dobles: "Doubles tennis (6 MET)",
      opt_ciclismo_suave: "Easy cycling (6.8 MET)",
      opt_ciclismo: "Cycling (7 MET)",
      opt_futbol: "Football (7 MET)",
      opt_correr_suave: "Jogging (7.5 MET)",
      opt_trekking: "Trekking with backpack (7.8 MET)",
      opt_tenis: "Singles tennis (8 MET)",
      opt_btt: "MTB / mountain (8.5 MET)",
      opt_futbol_comp: "Competitive football (9 MET)",
      opt_ciclismo_fuerte: "Hard cycling (9 MET)",
      opt_correr: "Running (10.5 MET)",
      meta_deporte: "If you pick a sport, intensity is set automatically by its MET (you can adjust it under «Person»).",
      card_salud: "Health & environment",
      lbl_comorb: "Comorbidities",
      check_cardiovascular: "Cardiovascular / high blood pressure",
      check_diabetes: "Diabetes",
      check_respiratoria: "Respiratory",
      check_mental: "Mental health",
      lbl_medicacion: "Medication",
      check_diureticos: "Loop diuretics",
      check_antipsicoticos: "Antipsychotics",
      lbl_social: "Social situation",
      check_vive_solo: "Lives alone",
      check_no_sale: "Doesn't go out",
      check_sin_aire: "No air conditioning",
      check_vivienda_fria: "Cold housing",
      lbl_fototipo: "Phototype (Fitzpatrick)",
      opt_desconocido: "— unknown —",
      opt_foto_1: "I — always burns",
      opt_foto_2: "II — burns easily",
      opt_foto_3: "III — burns sometimes",
      opt_foto_4: "IV — burns rarely",
      opt_foto_5: "V — rarely burns",
      opt_foto_6: "VI — never burns",
      check_dormi: "Slept poorly / less than 6 h",
      check_enfermo: "I have a cold / am ill",
      check_fiesta: "Party / celebration (alcohol + dancing)",
      check_alcohol: "I drank alcohol recently",
      btn_predecir: "Predict risk →",
      sec02_tag: "result",
      sec02_title: "Risk for the chosen outing",
      resultado_vacio: "Fill in the form and press «Predict risk». The result (class, risk %, hourly profile and recommendations) will appear here.",
      meta_clase: "Final class",
      meta_riesgo: "Personalised risk",
      card_perfil: "Hourly profile — day Heat Index",
      card_reco: "Recommendations",
      summary_detalle: "Ensemble details (population probabilities, factors, override)",
      cta_entender: "Understand the result better →",
      disclaimer: "ClimaSafe is not a medical device and does not replace clinical judgement. If you have symptoms, contact emergency services (112) or your doctor.",
      footer_extra: "· probar-ya · ONNX in the browser",
      aviso_title: "Medical-legal notice",
      aviso_p1: "<strong>ClimaSafe is not a medical device</strong> and does not replace the clinical judgement of a healthcare professional.",
      aviso_li1: "The risk estimate is based on statistical models and epidemiological literature indices; it may contain errors and must not be used to diagnose, decide treatments or plan risky activities.",
      aviso_li2: "It is an <strong>educational and research demo</strong>: in the EU, health systems may be classified as high-risk under the Artificial Intelligence Regulation.",
      aviso_li3: "If you have symptoms (dizziness, confusion, trouble breathing, chest pain…), <strong>contact emergency services (112) or your doctor</strong>.",
      btn_entendido: "Got it",
      // Dinámicos (main.js / weather.js vía t())
      clase_0: "SAFE",
      clase_1: "CAUTION",
      clase_2: "DANGER",
      modelos_listos: "Models ready. Fill in the form and press Predict.",
      error_carga: "Error loading models: {0}",
      descargando: "Downloading data and running the 3 ONNX models…",
      sin_reco: "No specific recommendations for this profile.",
      datos_tiempo_real: "Weather data from Open-Meteo (real time).",
      sin_perfil: "No hourly profile available.",
      th_modelo: "Model",
      th_prob: "P(risk)",
      th_clase: "Class",
      th_nota: "Note",
      err_fecha_pasada: "Date {0} has already passed. Only today or the future covered by the weather forecast can be predicted.",
      err_forecast: "The weather forecast does not cover {0}. Cannot predict without data.",
      // WEB-015: localStorage profile + privacy
      privacy_notice: "Your data is stored only in your browser (localStorage). It never leaves your device and is not sent to any server. You can delete it at any time using the «Delete my data» button.",
      btn_borrar_datos: "Delete my data",
      btn_borrar_datos_aria: "Delete the profile data stored in the browser",
      datos_borrados: "Data deleted. The form has been reset.",
      perfil_restaurado: "Profile restored from the browser.",
      // WEB-016: local briefing drafting with LLM (transformers.js)
      card_ia: "Draft the briefing with local AI",
      ia_desc: "Optional: a small LLM (IBM Granite 4.0 1B) is downloaded once from HuggingFace and writes the briefing in prose using the results already computed above.",
      ia_btn_activar: "Enable local AI (one-time download ≈ 1–2 GB)",
      ia_btn_repetir: "Regenerate the briefing",
      ia_cargando_lib: "Loading transformers.js from CDN…",
      ia_descargando: "Downloading model ({0})… {1}%",
      ia_generando: "Writing the briefing on your device (without GPU it may take 1–3 min)…",
      ia_listo: "Briefing drafted locally with {0}.",
      ia_error: "Local AI unavailable ({0}). The classic template briefing keeps working unchanged.",
      ia_privacy: "Privacy: the model runs entirely in your browser (WebGPU or WASM) and is cached for subsequent uses. Neither your data nor the generated text is sent to any server; only the initial download comes from HuggingFace.",
      ia_disclaimer: "Text drafted by a local LLM from the model's deterministic results. It may contain errors; the valid figures are the ones shown above."
    }
  };

  var current = "es";

  function getSaved() {
    try {
      var v = localStorage.getItem(KEY);
      if (v === "es" || v === "en") return v;
    } catch (e) { /* sin almacenamiento: se usa el navegador */ }
    return null;
  }

  function detect() {
    var saved = getSaved();
    if (saved) return saved;
    var nav = (navigator.language || navigator.userLanguage || "es").toLowerCase();
    return nav.indexOf("en") === 0 ? "en" : "es";
  }

  function translate(lang) {
    var dict = DICT[lang] || DICT.es;
    document.documentElement.lang = lang;
    var nodes = document.querySelectorAll("[data-i18n]");
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      var key = el.getAttribute("data-i18n");
      if (!(key in dict)) continue;
      if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
        el.placeholder = dict[key];
      } else if (el.getAttribute("data-i18n-html") !== null) {
        el.innerHTML = dict[key];
      } else {
        el.textContent = dict[key];
      }
    }
    var sel = document.getElementById("lang-select");
    if (sel) sel.value = lang;
  }

  function setLang(lang) {
    lang = lang === "en" ? "en" : "es";
    try { localStorage.setItem(KEY, lang); } catch (e) { /* sin almacenamiento */ }
    translate(lang);
  }

  function t(key) {
    var v = (DICT[current] || DICT.es)[key];
    if (v == null) return key;
    var args = Array.prototype.slice.call(arguments, 1);
    return String(v).replace(/\{(\d+)\}/g, function (m, i) {
      return args[i] != null ? String(args[i]) : m;
    });
  }

  function ready() {
    current = detect();
    var sel = document.getElementById("lang-select");
    if (sel) {
      sel.addEventListener("change", function () { setLang(sel.value); });
    }
    translate(current);
  }

  window.ClimaSafeI18n = {
    t: t,
    get: function () { return current; },
    set: setLang
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ready);
  } else {
    ready();
  }
})();