// llm.js — redacción local del parte con un LLM en el navegador (WEB-016).
//
// Modelo: IBM Granite 4.0 1B instruct vía la conversión ONNX para web de
// HuggingFace (onnx-community/granite-4.0-1b-ONNX-web, Apache 2.0). Se descarga
// en runtime desde HuggingFace SOLO cuando el usuario lo activa (nada
// empaquetado en el repo), queda cacheado por el navegador (Cache API de
// transformers.js) y ejecuta 100 % local: WebGPU (q4f16, ≈1,25 GB) o WASM
// (q4, ≈1,8 GB). Requiere transformers.js v4: Granite 4.0 usa la arquitectura
// GraniteMoeHybridForCausalLM, soportada desde la v4 (la v3.x no la carga).
//
// El LLM NO sustituye la predicción: recibe como contexto los resultados YA
// calculados por predictEnsemble y solo redacta el parte en prosa. Cualquier
// fallo (sin red, descarga interrumpida, WebGPU/WASM no disponible, generación)
// lanza y main.js mantiene el parte con plantilla clásica sin cambios.
//
// Piezas puras (elegirDispositivoYDtype, contextoDesdeResultado,
// mensajesParaParte, limpiarSalidaLlm) son testables en node sin red ni DOM:
// ver test/llm_unit.mjs. La ejecución real del modelo requiere navegador:
// verificación manual en documentacion/wasm/llm_navegador.md.

export const MODELO_LLM = "onnx-community/granite-4.0-1b-ONNX-web";
export const VERSION_TRANSFORMERS = "4.2.0";
// jsdelivr sirve el build ESM del paquete con solo el specifiers raíz.
export const URL_CDN_TRANSFORMERS =
  `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${VERSION_TRANSFORMERS}`;
const MAX_NUEVOS_TOKENS = 256;

// Etiquetas de clase para el prompt (i18n.js es una IIFE de navegador; aquí va
// el mínimo necesario para que llm.js sea autónomo en node).
const CLASE_ETIQUETA = {
  0: { es: "SEGURO", en: "SAFE" },
  1: { es: "PRECAUCIÓN", en: "CAUTION" },
  2: { es: "PELIGRO", en: "DANGER" },
};

// ─────────────────────────────────────────────────────────────────────────────
// Piezas puras (testeadas en node)
// ─────────────────────────────────────────────────────────────────────────────

// WebGPU permite q4f16 (más rápido y ligero); sin él, WASM exige q4 entero.
export function elegirDispositivoYDtype(hayWebGpu) {
  return hayWebGpu ? { device: "webgpu", dtype: "q4f16" } : { device: "wasm", dtype: "q4" };
}

// Contexto mínimo para el LLM: SOLO campos ya calculados por el pipeline ML
// (misma forma que la `res` de predictEnsemble que consume renderResultado).
export function contextoDesdeResultado(res) {
  const p = res.perfil_usuario || {};
  const canal =
    (res.perfil?.calor?.prob_personalizada ?? 0) >=
    (res.perfil?.frio?.prob_personalizada ?? 0)
      ? "calor"
      : "frio";
  const activo = res.perfil?.[canal] || {};
  const perfilHorario = res.weather?.perfil_horario || [];
  const inicio = Number(p.hora_inicio);
  const fin = inicio + Number(p.duracion_actividad_h || 0);
  return {
    provincia: res.weather?.provincia ?? "",
    fecha: res.weather?.target_date ?? "",
    clase: res.clase_final,
    riesgo_pct: ((activo.prob_personalizada ?? 0) * 100).toFixed(1),
    hi_pico: perfilHorario.length ? Math.max(...perfilHorario.map((x) => x.HI)) : null,
    ventana:
      Number.isFinite(inicio) && Number.isFinite(p.duracion_actividad_h)
        ? `${String(Math.round(inicio)).padStart(2, "0")}:00–${String(Math.round(fin)).padStart(2, "0")}:00`
        : null,
    canal_activo: canal,
    factores: (activo.factores || []).map((f) => `${f.nombre} ${f.factor}`),
    override_razon: res.override_fisico?.razon ?? null,
  };
}

// Mensajes de chat para el pipeline text-generation: el system acota al modelo
// a redactar (no calcular) y le prohíbe inventar cifras nuevas.
export function mensajesParaParte(ctx, lang) {
  const es = lang !== "en";
  const etiqueta = (CLASE_ETIQUETA[ctx.clase] || CLASE_ETIQUETA[1])[lang] ?? "?";
  const sistema = es
    ? "Eres un redactor de partes meteorológicos-sanitarios en español. " +
      "Recibes datos YA calculados por un modelo determinista de riesgo térmico. " +
      "Redacta un parte claro de 3-5 frases usando SOLO esos datos: no inventes " +
      "ni cambies cifras, clases ni recomendaciones, y no añadas consejos médicos nuevos. " +
      "Responde únicamente con el texto del parte, en prosa plana, sin listas ni markdown."
    : "You are a writer of weather-health briefings in English. " +
      "You receive figures ALREADY computed by a deterministic heat-risk model. " +
      "Write a clear briefing of 3-5 sentences using ONLY those figures: do not invent " +
      "or alter numbers, classes or recommendations, and add no new medical advice. " +
      "Reply with the briefing text only, plain prose, no lists or markdown.";
  const lineas = [
    es ? "Datos del parte:" : "Briefing data:",
    `- ${es ? "Provincia" : "Province"}: ${ctx.provincia}`,
    `- ${es ? "Fecha" : "Date"}: ${ctx.fecha}`,
    `- ${es ? "Clase final" : "Final class"}: ${etiqueta}`,
    `- ${es ? "Riesgo personalizado" : "Personalised risk"}: ${ctx.riesgo_pct}%`,
    `- ${es ? "Heat Index pico" : "Peak Heat Index"}: ${ctx.hi_pico != null ? ctx.hi_pico.toFixed(1) + " °C" : "—"}`,
    `- ${es ? "Ventana de actividad" : "Activity window"}: ${ctx.ventana ?? "—"}`,
    `- ${es ? "Factores personales activos" : "Active personal factors"}: ${ctx.factores.length ? ctx.factores.join("; ") : es ? "ninguno" : "none"}`,
  ];
  const usuario =
    lineas.join("\n") +
    "\n\n" +
    (es ? "Redacta el parte." : "Write the briefing.");
  return [
    { role: "system", content: sistema },
    { role: "user", content: usuario },
  ];
}

// La salida a veces trae comillas envolventes o restos del rol assistant.
export function limpiarSalidaLlm(texto) {
  let s = String(texto).trim();
  s = s.replace(/^(assistant|asistente)\s*\n?/i, "").trim();
  while ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("«") && s.endsWith("»"))) {
    s = s.slice(1, -1).trim();
  }
  return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Integración con transformers.js (solo navegador)
// ─────────────────────────────────────────────────────────────────────────────

function hayWebGpu() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

// Porcentaje global de descarga ponderado por bytes entre todos los ficheros
// del repo HF (onnx + tokenizer). Devuelve 0–100.
function rastreadorProgreso() {
  const ficheros = new Map();
  return (evento) => {
    if (evento.status === "progress" && evento.total) {
      ficheros.set(evento.file, { loaded: evento.loaded || 0, total: evento.total });
    } else if (evento.status === "done" && ficheros.has(evento.file)) {
      ficheros.get(evento.file).loaded = ficheros.get(evento.file).total;
    }
    let loaded = 0;
    let total = 0;
    for (const f of ficheros.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    return total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : 0;
  };
}

// Fábrica del redactor. `importar` existe para tests (inyección del módulo);
// en producción importa transformers.js desde el CDN en el primer uso.
export async function crearRedactorLocal({ onProgreso, importar } = {}) {
  const cargar = importar || (() => import(/* @vite-ignore */ URL_CDN_TRANSFORMERS));
  const progreso = rastreadorProgreso();

  const crearPipeline = async (cfg) => {
    const mod = await cargar();
    if (mod.env) {
      // Los pesos vienen del Hub de HF; nada que resolver bajo /models local.
      mod.env.allowLocalModels = false;
      // Cache del navegador (Cache API) activada por defecto; explícito por contrato.
      mod.env.useBrowserCache = true;
    }
    return mod.pipeline("text-generation", MODELO_LLM, {
      device: cfg.device,
      dtype: cfg.dtype,
      progress_callback: (e) => {
        if (onProgreso) onProgreso(progreso(e), e.file || "");
      },
    });
  };

  // Cadena de intentos: WebGPU→WASM. Si ambos fallan, propaga el último error
  // y la demo sigue con la plantilla clásica.
  const intentos = hayWebGpu()
    ? [{ device: "webgpu", dtype: "q4f16" }, { device: "wasm", dtype: "q4" }]
    : [{ device: "wasm", dtype: "q4" }];
  let ultimoError = null;
  let generador = null;
  let cfgUsada = null;
  for (const cfg of intentos) {
    try {
      generador = await crearPipeline(cfg);
      cfgUsada = cfg;
      break;
    } catch (e) {
      ultimoError = e;
      console.warn(`IA local: fallo cargando ${cfg.device}/${cfg.dtype}:`, e);
    }
  }
  if (!generador) throw ultimoError || new Error("No se pudo crear el pipeline LLM");

  return {
    modelo: MODELO_LLM,
    device: cfgUsada.device,
    dtype: cfgUsada.dtype,
    // Redacta el parte a partir del contexto ya calculado. Nunca calcula riesgo.
    async generar(ctx, lang) {
      const mensajes = mensajesParaParte(ctx, lang);
      const salida = await generador(mensajes, {
        max_new_tokens: MAX_NUEVOS_TOKENS,
        do_sample: false,
      });
      const generado = salida?.[0]?.generated_text;
      const ultimo = Array.isArray(generado) ? generado.at(-1)?.content : generado;
      return limpiarSalidaLlm(ultimo ?? "");
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Cableado UI (invocado una vez desde main.js)
// ─────────────────────────────────────────────────────────────────────────────
export function initParteIA(getResultado) {
  if (typeof document === "undefined") return;
  const { t, get } = window.ClimaSafeI18n || { t: (k) => k, get: () => "es" };
  const $ = (id) => document.getElementById(id);
  const btn = $("parte-ia-activar");
  const estadoEl = $("parte-ia-estado");
  const salidaEl = $("parte-ia-salida");
  if (!btn || !estadoEl || !salidaEl) return;

  let redactor = null; // cacheado: la descarga es una sola vez
  let enMarcha = false;

  function pintarEstado(texto, modo) {
    estadoEl.className = "status" + (modo === "err" ? " err" : "");
    estadoEl.textContent = "";
    if (modo === "spin") {
      const sp = document.createElement("span");
      sp.className = "spinner";
      estadoEl.appendChild(sp);
    }
    estadoEl.appendChild(document.createTextNode(texto));
  }

  btn.addEventListener("click", async () => {
    const res = typeof getResultado === "function" ? getResultado() : null;
    if (!res) {
      pintarEstado(t("ia_error", "aún no hay un resultado del pipeline"), "err");
      return;
    }
    if (enMarcha) return;
    enMarcha = true;
    btn.disabled = true;
    try {
      if (!redactor) {
        pintarEstado(t("ia_cargando_lib"), "spin");
        redactor = await crearRedactorLocal({
          onProgreso: (pct, fichero) => pintarEstado(t("ia_descargando", fichero, pct), "spin"),
        });
      }
      salidaEl.textContent = "";
      salidaEl.classList.add("oculto");
      pintarEstado(t("ia_generando"), "spin");
      const ctx = contextoDesdeResultado(res);
      const texto = await redactor.generar(ctx, get());
      salidaEl.textContent = texto;
      salidaEl.classList.remove("oculto");
      pintarEstado(
        t("ia_listo", `${MODELO_LLM.split("/")[1]} (${redactor.device}/${redactor.dtype})`),
        "ok"
      );
      btn.textContent = t("ia_btn_repetir");
    } catch (e) {
      console.error("IA local falló:", e);
      pintarEstado(t("ia_error", e.message), "err");
    } finally {
      enMarcha = false;
      btn.disabled = false;
    }
  });
}
