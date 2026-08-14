// recomendaciones.js — port 1:1 de climasafeai/models/recomendaciones.py.
// El catálogo (models/recomendaciones.json, copia de
// climasafeai/data/recomendaciones.json) lo carga el llamante y se pasa aquí,
// para que este módulo sea puro (funciona igual en navegador y en node).

function _riesgoDominante(resultado) {
  const modelos = resultado.modelos || {};
  const heatClases = [];
  const coldClases = [];
  for (const [nombre, res] of Object.entries(modelos)) {
    if (!res || res.error) continue;
    const lower = nombre.toLowerCase();
    if (nombre === "LSTM") {
      const c1 = res.calor?.clase_threshold || 0;
      if (c1) heatClases.push(c1);
      const c2 = res.frio?.clase_threshold || 0;
      if (c2) coldClases.push(c2);
    } else if (nombre === "Formula") {
      const c1 = res.calor?.clase || 0;
      if (c1) heatClases.push(c1);
      const c2 = res.frio?.clase || 0;
      if (c2) coldClases.push(c2);
    } else if (lower.includes("calor")) {
      const c = res.clase_threshold || 0;
      if (c) heatClases.push(c);
    } else if (lower.includes("frio")) {
      const c = res.clase_threshold || 0;
      if (c) coldClases.push(c);
    }
  }
  const maxHeat = heatClases.length ? Math.max(...heatClases) : 0;
  const maxCold = coldClases.length ? Math.max(...coldClases) : 0;
  if (maxHeat > maxCold) return "calor";
  if (maxCold > maxHeat) return "frio";
  return "ambos";
}

function _clasificarClima(current, resultado, riesgoDominante = "ambos") {
  const etiquetas = [];
  const t = current?.t2m_c;
  const wc = resultado?.modelos?.Formula?.frio?.wind_chill_c;
  const hi = resultado?.modelos?.Formula?.calor?.heat_index_c;
  const uv = current?.uv_index;

  if (riesgoDominante !== "frio") {
    if (t != null && t >= 35) etiquetas.push("calor_extremo");
    else if (t != null && t >= 30) etiquetas.push("calor_moderado");
  }
  if (riesgoDominante !== "calor") {
    if (wc != null && wc <= -25) etiquetas.push("frio_extremo");
    else if (wc != null && wc <= 0) etiquetas.push("frio_moderado");
  }
  if (uv != null && uv >= 8) etiquetas.push("uv_alto");
  else if (uv != null && uv >= 6) etiquetas.push("uv_alto");

  return etiquetas;
}

function _nivelActividadSegura(claseFinal) {
  if (claseFinal >= 2) return "reposo";
  if (claseFinal === 1) return "ligera";
  return "";
}

function _ventanaActividad(perfil) {
  const inicio = perfil.hora_inicio;
  const duracion = perfil.duracion_actividad_h;
  if (inicio != null && duracion != null) return [inicio, inicio + duracion];
  if (inicio != null) return [inicio, inicio + 1];
  return null;
}

function _enHorasCentrales(ventana) {
  if (!ventana) return true;
  return ventana[1] > 12 && ventana[0] < 18;
}

function _actividadLabel(perfil) {
  if (perfil.deporte) return perfil.deporte;
  if (perfil.nivel_actividad) return `actividad ${perfil.nivel_actividad}`;
  return "actividad";
}

export function generarRecomendaciones(perfil, resultado, catalogo) {
  if (!catalogo || Object.keys(catalogo).length === 0) {
    return ["No hay catalogo de recomendaciones disponible."];
  }
  if (!perfil) return [];

  const current = resultado?.weather?.current || {};
  const claseFinal = resultado?.clase_final ?? 0;
  const riesgoDom = _riesgoDominante(resultado);
  const recomendaciones = [];

  const ventana = _ventanaActividad(perfil);
  const enHorasCentrales = _enHorasCentrales(ventana);

  const climaTags = _clasificarClima(current, resultado, riesgoDom);
  for (const tag of climaTags) {
    const seccion = catalogo.clima?.[tag];
    if (seccion && "texto" in seccion) recomendaciones.push(seccion.texto);
  }

  const fototipo = perfil.fototipo ?? "";
  if (fototipo) {
    const seccion = catalogo.fototipo?.[fototipo];
    if (seccion && "texto" in seccion) {
      let texto = seccion.texto;
      if (!enHorasCentrales && ventana) {
        texto = texto.replace(
          "Busca sombra en horas centrales del dia.",
          `Tu actividad es a partir de las ${ventana[0].toFixed(0)}:00, fuera del pico UV. Aun asi, proteccion solar recomendada.`
        );
        texto = texto.replace(
          "Evita la exposicion directa entre las 12:00 y las 18:00.",
          `Tu actividad empieza a las ${ventana[0].toFixed(0)}:00, fuera del horario de maximo UV, pero lleva proteccion.`
        );
      }
      recomendaciones.push(texto);
    }
  } else {
    const seccion = catalogo.fototipo?.desconocido;
    if (seccion && "texto" in seccion) recomendaciones.push(seccion.texto);
  }

  const actividad = (perfil.nivel_actividad ?? "").toLowerCase();
  const nivelSeguro = _nivelActividadSegura(claseFinal);
  if (nivelSeguro === "reposo") {
    recomendaciones.push(
      "El nivel de riesgo es PELIGRO. No se recomienda realizar actividad fisica al aire libre. Busca un lugar fresco y permanece en reposo."
    );
  } else if (actividad) {
    const seccion = catalogo.actividad?.[actividad];
    if (seccion && "texto" in seccion) {
      let rec = seccion.texto;
      if (nivelSeguro === "ligera" && ["moderada", "intensa", "muy_intensa"].includes(actividad)) {
        rec += " Dado el nivel de riesgo actual, considera reducir la intensidad de tu actividad.";
      }
      recomendaciones.push(rec);
    }
  }

  for (const comorb of perfil.comorbilidades || []) {
    const seccion = catalogo.comorbilidades?.[comorb.toLowerCase()];
    if (seccion && "texto" in seccion) recomendaciones.push(seccion.texto);
  }
  for (const farmaco of perfil.farmacos || []) {
    const seccion = catalogo.farmacos?.[farmaco.toLowerCase()];
    if (seccion && "texto" in seccion) recomendaciones.push(seccion.texto);
  }
  for (const sit of perfil.situacion_social || []) {
    const seccion = catalogo.situacion_social?.[sit.toLowerCase()];
    if (seccion && "texto" in seccion) recomendaciones.push(seccion.texto);
  }

  const generales = catalogo.generales || {};
  for (const key of ["hidratacion", "ropa", "comidas"]) {
    if (generales[key]) recomendaciones.push(generales[key]);
  }

  if (ventana) {
    const inicioLabel = `${ventana[0].toFixed(0)}:00`;
    const finLabel = `${ventana[1].toFixed(0)}:00`;
    if (enHorasCentrales) {
      recomendaciones.push(
        `Tu actividad (${inicioLabel}-${finLabel}) coincide con las horas de mayor riesgo. Toma precauciones extra.`
      );
    } else {
      recomendaciones.push(
        `Tu actividad es en horario seguro (${inicioLabel}-${finLabel}), fuera del pico de calor (12:00-18:00).`
      );
    }
  } else if (generales.horas_peligro) {
    recomendaciones.push(generales.horas_peligro);
  }

  if (perfil.fiesta && generales.hidratacion) {
    recomendaciones.push(
      "Has indicado que tienes planes de ocio/fiesta. Si consumes alcohol, hazlo con moderacion: el alcohol acelera la deshidratacion y altera la percepcion del riesgo termico. Alterna con agua."
    );
  }

  if (!perfil.aclimatado) {
    recomendaciones.push(
      "No estas aclimatado al clima local. Tu riesgo de golpe de calor o hipotermia es significativamente mayor. Limita la exposicion los primeros 3-5 dias y aumentala gradualmente."
    );
  }

  if (perfil.falta_sueno) {
    recomendaciones.push(
      "Has indicado falta de sueno o mala noche. La fatiga empeora la tolerancia al calor y la capacidad de tomar decisiones. Extremar precauciones."
    );
  }

  if (claseFinal >= 1) {
    if (riesgoDom !== "frio" && generales.senal_alarma_calor) recomendaciones.push(generales.senal_alarma_calor);
    if (riesgoDom !== "calor" && generales.senal_alarma_frio) recomendaciones.push(generales.senal_alarma_frio);
  }

  if (perfil.duracion_actividad_h != null && perfil.duracion_actividad_h > 2) {
    recomendaciones.push(
      `Tu actividad esta prevista para ${perfil.duracion_actividad_h.toFixed(0)} horas. Planifica pausas regulares y lleva suficiente agua (minimo 1 litro cada 2 horas).`
    );
  }

  const vistos = new Set();
  const unicos = [];
  for (const r of recomendaciones) {
    if (!vistos.has(r)) {
      vistos.add(r);
      unicos.push(r);
    }
  }
  return unicos;
}
