// ort-runtime.js — devuelve el objeto `ort` de onnxruntime-web.
// Navegador: script clásico vendor/ort.min.js define el global `ort`.
// Node (test de paridad): el llamante inyecta el módulo con setOrt(ort)
// (paridad.mjs lo importa de su node_modules local); como último recurso se
// intenta un import dinámico.
let cache = null;

export function setOrt(m) {
  cache = m;
  return m;
}

export async function getOrt() {
  if (cache) return cache;
  if (typeof ort !== "undefined") {
    cache = ort;
    return cache;
  }
  const m = await import("onnxruntime-web");
  cache = m;
  return cache;
}
