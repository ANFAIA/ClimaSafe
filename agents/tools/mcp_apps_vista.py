"""Vista MCP App del parte de riesgo (MCP-APPS-001).

Plantilla HTML autocontenida que sirve el recurso ``ui://prediccion-riesgo``
del servidor MCP: un host con soporte de MCP Apps (extensión
``io.modelcontextprotocol/ui``) renderiza este HTML en el chat tras llamar a
``predict_risk_mcp``; un host sin soporte ignora el meta y recibe el JSON
normal.

El HTML vive aquí, en el backend Python, con el CSS y el JS inline: no hay un
fichero ``.js`` suelto en el repositorio. Las funciones de visualización
(``mostrarFinal`` y ``mostrarGraficaRiesgo``) son las mismas firmas que usa la
web (``chat/static/index.html``) para que el parte pinte igual en ambos sitios.

Estado: el recurso pinta el ÚLTIMO resultado de ``predict_risk_mcp``. Es el
patrón mínimo de la spec (la tool deja el dato y el recurso lo renderiza);
con varias llamadas seguidas se vería el último, igual que el ejemplo del
reloj de la doc oficial de MCP Apps.
"""

from __future__ import annotations

import json
from typing import Any

# MIME type del recurso y meta tag pre-GA: los hosts actuales negocian por el
# MIME (`text/html;profile=mcp-app`), los pre-GA miraban el meta tag.
UI_MIME_TYPE = "text/html;profile=mcp-app"
_UI_PROFILE = "mcp-app"

# ── CSS de la vista (inline, igual escala visual que la web) ───────────────
_VISTA_CSS = """
:root{--seguro:#27ae60;--precaucion:#f39c12;--peligro:#e74c3c;--bg:#f0f2f5;
--card:#ffffff;--text:#2c3e50;--text-light:#7f8c8d;--accent:#3498db;
--border:#dee2e6;--radius:12px;--shadow:0 2px 8px rgba(0,0,0,0.08);}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--text);line-height:1.5;padding:16px;margin:0;}
.card{background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);
padding:20px 24px;margin-bottom:16px;}
.card-title{font-size:0.9rem;font-weight:700;text-transform:uppercase;
letter-spacing:0.5px;color:var(--text-light);margin-bottom:12px;
border-bottom:1px solid var(--border);padding-bottom:8px;}
.class-SEGURO{color:var(--seguro);}.class-PRECAUCION{color:var(--precaucion);}
.class-PELIGRO{color:var(--peligro);}
.bg-SEGURO{background:#d5f5e3;border-color:var(--seguro);}
.bg-PRECAUCION{background:#fef5e7;border-color:var(--precaucion);}
.bg-PELIGRO{background:#fdedec;border-color:var(--peligro);}
.final-result{padding:16px;border-radius:10px;border-left:6px solid;}
.final-result .class{font-size:2.2rem;font-weight:800;}
.final-result .sub{font-size:0.9rem;opacity:0.8;margin-top:4px;}
#risk-chart{width:100% !important;height:280px !important;}
"""

# ── JS de la vista (inline, mismas firmas que la web) ──────────────────────
# Las funciones mostrarFinal (el parte) y mostrarGraficaRiesgo (la curva por
# hora) reproducen la lógica de chat/static/index.html para que el MCP App
# pinte exactamente igual que la web.
_VISTA_JS = r"""
// Confianza conformal del resultado (misma lógica que la web).
const CONF_ICONOS = { alta: '&#9679;', media: '&#9675;', baja: '&#9888;' };
const CONF_COLOR = { alta: 'var(--seguro)', media: 'var(--precaucion)', baja: 'var(--peligro)' };
const CONF_LABEL = { alta: 'confianza alta', media: 'confianza media', baja: 'confianza baja' };

function _confianzaHtml(conf) {
  if (!conf || conf === 'desconocida') return '';
  const icono = CONF_ICONOS[conf] || '&#9679;';
  const color = CONF_COLOR[conf] || 'var(--text-light)';
  return ' <span style="font-size:0.75rem;color:' + color + '" title="' + (CONF_LABEL[conf] || conf) + '">' + icono + '</span>';
}

function _confianzaGlobal(d) {
  const confs = [];
  for (const mod of ['XGBoost_calor', 'RandomForest_frio', 'LSTM']) {
    const c = d.modelos?.[mod]?.conformal_confianza;
    if (c && c !== 'desconocida') confs.push(c);
  }
  if (!confs.length) return null;
  if (confs.includes('baja')) return 'baja';
  if (confs.includes('media')) return 'media';
  return 'alta';
}

// Parte: clase final + confianza conformal + modelo determinante.
function mostrarFinal(d) {
  document.getElementById('final-card').style.display = 'block';
  const label = d.clase_final_label ?? '?';
  const det = d.explicacion?.modelo_determinante;
  let sub = 'Escenario mas restrictivo';
  if (det) {
    if (det.startsWith('Override')) {
      sub = det.replace('Override — ', '');
    } else {
      sub = 'Determinado por: ' + det;
    }
  }
  const conf = _confianzaGlobal(d);
  const confHtml = _confianzaHtml(conf);
  document.getElementById('final-result').className = 'final-result bg-' + label;
  document.getElementById('final-result').innerHTML = '<div class="class class-' + label + '">' + label + confHtml + '</div>'
    + '<div class="sub">' + sub + '</div>';
}

let _riskChart = null;

// Heat Index (°C) → peligrosidad ambiental 1-10 (misma escala tramo a tramo
// que la web y que _hi_a_nivel del backend, para que todo se lea igual).
function hiToNivel(hi) {
  if (hi <= 15) return 1;
  if (hi <= 27) return 1 + (hi - 15) / 12;
  if (hi <= 32) return 2 + (hi - 27) / 5 * 2;
  if (hi <= 39) return 4 + (hi - 32) / 7 * 3;
  if (hi <= 46) return 7 + (hi - 39) / 7 * 2;
  return Math.min(10, 9 + (hi - 46) / 9);
}

function nivelLabel(n) {
  if (n >= 9) return 'Peligro extremo';
  if (n >= 7) return 'Peligro alto';
  if (n >= 4) return 'Peligro';
  if (n >= 2) return 'Precaución';
  return 'Seguro';
}

// Curva de riesgo por hora: peligrosidad ambiental (eje izquierdo) y riesgo
// personal en % (eje derecho), con la ventana de actividad sombreada, la
// franja recomendada y el pico marcado. Misma gráfica que la web.
function mostrarGraficaRiesgo(d, perfilesExtra, umbrales) {
  const ph = d.weather?.perfil_horario;
  if (!ph || !ph.length) return;
  document.getElementById('graph-card').style.display = 'block';
  const ctx = document.getElementById('risk-chart');

  const horas = ph.map(e => e.hora + ':00');
  const hiVals = ph.map(e => e.HI);
  const nivelVals = ph.map(e => hiToNivel(e.HI));

  const zonas = nivelVals.map(n => {
    if (n >= 9) return 'rgba(139,0,0,0.10)';
    if (n >= 7) return 'rgba(192,57,43,0.12)';
    if (n >= 4) return 'rgba(231,76,60,0.10)';
    if (n >= 2) return 'rgba(243,156,18,0.10)';
    return 'rgba(39,174,96,0.10)';
  });

  const datasets = [{
    label: 'Peligrosidad',
    data: nivelVals,
    borderColor: '#2c3e50',
    backgroundColor: 'rgba(44,62,80,0.1)',
    fill: false,
    tension: 0.3,
    pointRadius: 2,
    pointHitRadius: 8,
    yAxisID: 'y',
  }];

  const colores = ['#e74c3c', '#2980b9', '#27ae60', '#8e44ad', '#d35400', '#16a085'];
  const legHtml = [];

  // Curva de riesgo por hora (no una recta): sube y baja con el calor de cada
  // hora y con la carga térmica acumulada del día. La calcula el backend.
  const curvaHoraria = (rh) => {
    if (!rh || !rh.length) return null;
    const m = {};
    rh.forEach(e => { m[e.hora] = e.riesgo; });
    return ph.map(e => (m[e.hora] != null ? m[e.hora] : null));
  };

  const probUser = d.perfil?.calor?.prob_personalizada ?? d.modelos?.XGBoost_calor?.prob_riesgo;
  const dataUser = curvaHoraria(d.riesgo_horario) || (probUser != null ? Array(ph.length).fill(probUser) : null);
  if (dataUser) {
    const picoUser = Math.round(Math.max(...dataUser.filter(v => v != null)) * 100);
    const labelUser = 'Tu riesgo (' + (d.perfil_usuario?.edad ?? '?') + ' años)';
    datasets.push({
      label: labelUser,
      data: dataUser,
      borderColor: colores[0],
      borderWidth: 2.5,
      pointRadius: 0,
      tension: 0.35,
      spanGaps: true,
      fill: false,
      yAxisID: 'y1',
    });
    legHtml.push('<span style="color:' + colores[0] + '">▬▬ ' + labelUser + ' · pico ' + picoUser + '%</span>');
  }

  if (perfilesExtra && perfilesExtra.length) {
    for (let i = 0; i < perfilesExtra.length; i++) {
      const pe = perfilesExtra[i];
      const dataPe = curvaHoraria(pe.riesgo_horario ?? pe.d?.riesgo_horario);
      if (!dataPe) continue;
      const edad = pe.edad;
      const color = colores[i + 1] || '#666';
      const picoPe = Math.round(Math.max(...dataPe.filter(v => v != null)) * 100);
      datasets.push({
        label: edad + ' años',
        data: dataPe,
        borderColor: color,
        borderDash: [5, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.35,
        spanGaps: true,
        fill: false,
        yAxisID: 'y1',
      });
      legHtml.push('<span style="color:' + color + '">- - ' + edad + ' años · pico ' + picoPe + '%</span>');
    }
  }

  if (_riskChart) _riskChart.destroy();

  _riskChart = new Chart(ctx, {
    type: 'line',
    data: { labels: horas, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(ctx) {
              const ds = ctx.dataset;
              if (ds.label === 'Peligrosidad') {
                const n = ctx.parsed.y;
                return 'Nivel ' + n.toFixed(1) + '/10 — ' + nivelLabel(n);
              }
              return ds.label + ': ' + (ctx.parsed.y * 100).toFixed(0) + '%';
            },
            afterLabel: function(ctx) {
              if (ctx.dataset.label === 'Peligrosidad') {
                const hi = hiVals[ctx.dataIndex];
                return hi != null ? 'HI: ' + hi.toFixed(1) + 'C' : '';
              }
              return '';
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: 'Hora', font: { size: 11 } }, grid: { display: false } },
        y: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'Nivel de peligro (1-10)', font: { size: 11 } },
          min: 0.5,
          max: 10.5,
          grid: { color: 'rgba(0,0,0,0.06)' },
        },
        y1: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: 'Riesgo personal (%)', font: { size: 11 } },
          min: 0,
          max: 1,
          ticks: { callback: v => Math.round(v * 100) + '%' },
          grid: { display: false },
        },
      },
    },
    plugins: [{
      afterDraw: function(chart) {
        const ctx2 = chart.ctx;
        const xAxis = chart.scales.x;
        const yAxis = chart.scales.y;
        const left = xAxis.left;
        const w = xAxis.width;
        const h = yAxis.height;
        const bottom = yAxis.bottom;

        for (let i = 0; i < 24; i++) {
          const x = xAxis.getPixelForValue(i, 0);
          const x1 = i < 23 ? xAxis.getPixelForValue(i + 1, 0) : left + w;
          const x0 = i > 0 ? x : left;
          const zona = zonas[i];
          if (zona) {
            ctx2.fillStyle = zona;
            ctx2.fillRect(x0, yAxis.top, x1 - x0, h);
          }
        }

        const h_ini = d.perfil_usuario?.hora_inicio;
        const dur = d.perfil_usuario?.duracion_actividad_h;
        if (h_ini != null && dur != null) {
          const xIni = xAxis.getPixelForValue(h_ini, 0);
          const xFin = xAxis.getPixelForValue(Math.min(h_ini + dur, 24), 0);
          ctx2.save();
          ctx2.strokeStyle = 'rgba(231,76,60,0.6)';
          ctx2.lineWidth = 2;
          ctx2.setLineDash([4, 4]);
          ctx2.beginPath();
          ctx2.moveTo(xIni, yAxis.top);
          ctx2.lineTo(xIni, bottom);
          ctx2.stroke();
          ctx2.beginPath();
          ctx2.moveTo(xFin, yAxis.top);
          ctx2.lineTo(xFin, bottom);
          ctx2.stroke();
          ctx2.restore();
          ctx2.fillStyle = 'rgba(231,76,60,0.08)';
          ctx2.fillRect(xIni, yAxis.top, xFin - xIni, h);
          ctx2.fillStyle = '#e74c3c';
          ctx2.font = '10px sans-serif';
          ctx2.textAlign = 'center';
          ctx2.fillText('actividad', (xIni + xFin) / 2, bottom + 14);
        }

        // Umbrales de PRECAUCION y PELIGRO sobre el eje de riesgo personal:
        // son los mismos cortes que usa el backend para decidir la clase.
        const y1Axis = chart.scales.y1;
        if (umbrales && y1Axis) {
          ctx2.save();
          ctx2.setLineDash([2, 3]);
          ctx2.lineWidth = 1;
          ctx2.font = '9px sans-serif';
          ctx2.textAlign = 'left';
          for (const l of [
            { v: umbrales.precaucion, color: '#f39c12', txt: 'precaución' },
            { v: umbrales.peligro, color: '#c0392b', txt: 'peligro' },
          ]) {
            if (l.v == null) continue;
            const y = y1Axis.getPixelForValue(l.v);
            ctx2.strokeStyle = l.color;
            ctx2.beginPath();
            ctx2.moveTo(left, y);
            ctx2.lineTo(left + w, y);
            ctx2.stroke();
            ctx2.fillStyle = l.color;
            ctx2.fillText(l.txt, left + 4, y - 3);
          }
          ctx2.restore();
        }
      },
    }],
  });

  document.getElementById('graph-age-legend').innerHTML = legHtml.join('');
}
"""


def html_vista_predict_risk(result: dict) -> str:
    """HTML autocontenido del recurso ui://: parte + gráfica de riesgo.

    El JSON del resultado se inyecta como ``window.RIESGO_DATA`` y el JS de la
    vista se embebe a continuación. ``profile=mcp-app`` en el MIME type y en
    el HTML cubre tanto los hosts actuales como los pre-GA que miraban el meta
    tag.
    """
    datos_json = json.dumps(result, ensure_ascii=False, default=str)
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="profile" content="{_UI_PROFILE}">
<title>ClimaSafeAI — Riesgo por hora</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>{_VISTA_CSS}</style>
</head>
<body>
  <div class="card" id="final-card" style="display:none">
    <div class="card-title">Riesgo individual</div>
    <div class="final-result" id="final-result"></div>
  </div>
  <div class="card" id="graph-card" style="display:none">
    <div class="card-title">Riesgo por hora</div>
    <canvas id="risk-chart"></canvas>
    <div id="graph-age-legend" style="margin-top:6px;display:flex;flex-wrap:wrap;gap:8px;font-size:0.8rem"></div>
  </div>
<script>
window.RIESGO_DATA = {datos_json};
</script>
<script>
{_VISTA_JS}
</script>
<script>
// MCP Apps: puente postMessage JSON-RPC mínimo. El host habla con la app
// dentro del iframe por postMessage; respondemos a los pings de vida y
// avisamos de que la vista está lista.
function _enviar(objeto) {{
  if (window.parent && window.parent !== window) {{
    window.parent.postMessage(JSON.stringify(objeto), '*');
  }}
}}
window.addEventListener('message', (event) => {{
  let msg;
  try {{ msg = JSON.parse(event.data); }} catch (_) {{ return; }}
  if (msg && msg.jsonrpc === '2.0' && msg.method === 'ping' && msg.id != null) {{
    _enviar({{ jsonrpc: '2.0', id: msg.id, result: {{ pong: true }} }});
  }}
}});
_enviar({{ jsonrpc: '2.0', method: 'notifications/initialized' }});
mostrarFinal(RIESGO_DATA);
mostrarGraficaRiesgo(RIESGO_DATA);
</script>
</body>
</html>
"""


def html_vista_sin_resultado() -> str:
    """HTML del recurso antes de la primera llamada a predict_risk_mcp."""
    return "<!DOCTYPE html><html><body><p>Aún no hay predicción: llama a predict_risk_mcp primero.</p></body></html>"
