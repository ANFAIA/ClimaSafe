// paridad.mjs — test de paridad node (WEB-012): ejecuta el pipeline JS de
// web/probar-ya (mismos datos que el Python) y escribe la salida JS para que
// tests/test_demo_paridad.py la compare con predict_ensemble.
//
// Uso (desde web/probar-ya/test/):
//   node paridad.mjs <escenarios.json> <salida_python.json> <salida_js.json>
import { readFile, writeFile } from "node:fs/promises";
import * as ort from "onnxruntime-web";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { generarFeaturesCompletas } from "../js/features.js";
import { cargarArtefactos, cargarModelosOrt } from "../js/artefactos.js";
import { predictEnsemble } from "../js/modelos.js";
import { setOrt } from "../js/ort-runtime.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const raiz = path.resolve(__dirname, "..");
ort.env.wasm.numThreads = 1;
setOrt(ort);

const [escenariosPath, pythonOutPath, jsOutPath] = process.argv.slice(2);
if (!escenariosPath || !pythonOutPath || !jsOutPath) {
  console.error("Uso: node paridad.mjs <escenarios.json> <python_out.json> <js_out.json>");
  process.exit(2);
}

const escenarios = JSON.parse(await readFile(escenariosPath, "utf8"));

const artefactos = await cargarArtefactos(path.join(raiz, "models"));
const modelos = await cargarModelosOrt(path.join(raiz, "models"), ort);

const salidas = [];
for (const esc of escenarios.escenarios) {
  const targetRows = esc.horas.filter((r) => r.datetime.startsWith(esc.target_date));
  const histRows = esc.horas.filter((r) => !r.datetime.startsWith(esc.target_date));
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

  const res = await predictEnsemble(
    { weather, provincia: esc.provincia, perfil: esc.perfil, targetDate: esc.target_date },
    modelos,
    artefactos
  );

  const probPers = Math.max(
    res.perfil.calor.prob_personalizada,
    res.perfil.frio.prob_personalizada
  );

  salidas.push({
    nombre: esc.nombre,
    clase_final: res.clase_final,
    prob_pers: probPers,
    perfil: {
      calor: {
        prob_poblacional: res.perfil.calor.prob_poblacional,
        prob_personalizada: res.perfil.calor.prob_personalizada,
        factor_total: res.perfil.calor.factor_total,
      },
      frio: {
        prob_poblacional: res.perfil.frio.prob_poblacional,
        prob_personalizada: res.perfil.frio.prob_personalizada,
        factor_total: res.perfil.frio.factor_total,
      },
    },
    override_fisico: res.override_fisico,
  });
}

await writeFile(jsOutPath, JSON.stringify(salidas, null, 2));
console.log(`Paridad JS OK: ${salidas.length} escenarios → ${jsOutPath}`);
