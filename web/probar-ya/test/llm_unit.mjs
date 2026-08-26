// llm_unit.mjs — unidades puras del modo LLM local (WEB-016): sin red, sin DOM,
// sin transformers.js. Verifica las piezas testables de js/llm.js:
//   - elegirDispositivoYDtype: webgpu→q4f16, wasm→q4
//   - contextoDesdeResultado: solo datos YA calculados por el pipeline ML
//   - mensajesParaParte: chat messages con prohibición de inventar cifras
//   - limpiarSalidaLlm: recorte de comillas/espacios de la salida del LLM
//
// Uso: node llm_unit.mjs   (invocado por tests/test_demo_llm_units.py)
import assert from "node:assert/strict";
import {
  MODELO_LLM,
  VERSION_TRANSFORMERS,
  elegirDispositivoYDtype,
  contextoDesdeResultado,
  mensajesParaParte,
  limpiarSalidaLlm,
} from "../js/llm.js";

let ok = 0;
function prueba(nombre, fn) {
  fn();
  ok += 1;
  console.log(`  OK  ${nombre}`);
}

// Fixture con la misma forma que la salida de predictEnsemble (ver main.js).
const res = {
  clase_final: 1,
  perfil: {
    calor: {
      prob_personalizada: 0.342,
      prob_poblacional: 0.21,
      factor_total: 1.6,
      factores: [
        { nombre: "Edad ≥65", factor: "×1.3" },
        { nombre: "Diuréticos de asa", factor: "×1.15" },
      ],
    },
    frio: { prob_personalizada: 0.05, prob_poblacional: 0.04, factor_total: 1.0, factores: [] },
  },
  weather: {
    provincia: "Madrid",
    target_date: "2026-08-26",
    perfil_horario: [
      { hora: 10, HI: 24.0 },
      { hora: 14, HI: 31.8 },
    ],
  },
  perfil_usuario: { hora_inicio: 10, duracion_actividad_h: 2 },
  override_fisico: null,
};

prueba("MODELO_LLM es Granite 4.0 1B web de onnx-community", () => {
  assert.equal(MODELO_LLM, "onnx-community/granite-4.0-1b-ONNX-web");
});

prueba("VERSION_TRANSFORMERS es v4 (GraniteMoeHybrid requiere v4)", () => {
  assert.match(VERSION_TRANSFORMERS, /^4\./);
});

prueba("elegirDispositivoYDtype: WebGPU → q4f16", () => {
  assert.deepEqual(elegirDispositivoYDtype(true), { device: "webgpu", dtype: "q4f16" });
});

prueba("elegirDispositivoYDtype: sin WebGPU → wasm q4", () => {
  assert.deepEqual(elegirDispositivoYDtype(false), { device: "wasm", dtype: "q4" });
});

prueba("contextoDesdeResultado extrae solo datos ya calculados", () => {
  const ctx = contextoDesdeResultado(res);
  assert.equal(ctx.provincia, "Madrid");
  assert.equal(ctx.fecha, "2026-08-26");
  assert.equal(ctx.clase, 1);
  assert.ok(Math.abs(ctx.riesgo_pct - 34.2) < 1e-9);
  assert.ok(Math.abs(ctx.hi_pico - 31.8) < 1e-9);
  assert.equal(ctx.ventana, "10:00–12:00");
  assert.equal(ctx.canal_activo, "calor");
  assert.equal(ctx.factores.length, 2);
  assert.equal(ctx.factores[0], "Edad ≥65 ×1.3");
});

prueba("contextoDesdeResultado tolera perfil_horario vacío y override nulo", () => {
  const ctx = contextoDesdeResultado({
    ...res,
    weather: { ...res.weather, perfil_horario: [] },
    override_fisico: { razon: "HI ≥ 40 °C" },
  });
  assert.equal(ctx.hi_pico, null);
  assert.equal(ctx.override_razon, "HI ≥ 40 °C");
});

prueba("mensajesParaParte es un chat system+user que prohíbe inventar cifras", () => {
  const ctx = contextoDesdeResultado(res);
  const msgs = mensajesParaParte(ctx, "es");
  assert.equal(msgs.length, 2);
  assert.equal(msgs[0].role, "system");
  assert.equal(msgs[1].role, "user");
  assert.match(msgs[0].content.toLowerCase(), /no inventes|no cambies/);
  assert.match(msgs[0].content, /3-5 frases|tres a cinco frases/i);
  // El user lleva los números reales del pipeline como único contexto.
  assert.match(msgs[1].content, /34\.2/);
  assert.match(msgs[1].content, /31\.8/);
  assert.match(msgs[1].content, /PRECAUCIÓN/);
  assert.match(msgs[1].content, /Edad ≥65 ×1\.3/);
});

prueba("mensajesParaParte respeta el idioma en", () => {
  const msgs = mensajesParaParte(contextoDesdeResultado(res), "en");
  assert.match(msgs[0].content, /\bEnglish\b/i);
  assert.match(msgs[1].content, /CAUTION/);
});

prueba("limpiarSalidaLlm recorta comillas, espacios y prefijos de rol", () => {
  assert.equal(limpiarSalidaLlm('  "El parte.\n" '), "El parte.");
  assert.equal(limpiarSalidaLlm("assistant\nEl parte real."), "El parte real.");
  assert.equal(limpiarSalidaLlm("Ya"), "Ya");
});

console.log(`\n${ok} pruebas de unidades LLM en verde.`);
