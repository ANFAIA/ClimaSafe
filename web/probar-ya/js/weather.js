// weather.js — port de climasafeai/data/weather_fetcher.py (fetch_weather_data)
// a JS, usando Open-Meteo (CORS OK, sin API key). El índice UV de OpenUV queda
// fuera (requiere key) → uv_index siempre null; los overrides que dependen de
// UV se comportan como en Python con uv_index=None.
import { generarFeaturesCompletas } from "./features.js";

const OPENMETEO_BASE = "https://api.open-meteo.com/v1/forecast";
const OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive";
const FORECAST_HORIZON_DAYS = 7;

async function _openmeteoJson(url, params) {
  const qs = new URLSearchParams(params).toString();
  const r = await fetch(`${url}?${qs}`);
  if (!r.ok) throw new Error(`Open-Meteo HTTP ${r.status}`);
  return r.json();
}

export async function fetchCurrentWeather(lat, lon) {
  const data = await _openmeteoJson(OPENMETEO_BASE, {
    latitude: lat,
    longitude: lon,
    current: "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
    timezone: "auto",
  });
  const c = data.current || {};
  const out = {};
  if (c.temperature_2m != null) out.t2m_c = c.temperature_2m;
  if (c.relative_humidity_2m != null) out.rh = c.relative_humidity_2m;
  if (c.wind_speed_10m != null) out.wind_speed_kmh = c.wind_speed_10m;
  if (c.surface_pressure != null) out.sp = c.surface_pressure;
  return out;
}

export async function fetchHourlyForecast(lat, lon, days = FORECAST_HORIZON_DAYS) {
  const data = await _openmeteoJson(OPENMETEO_BASE, {
    latitude: lat,
    longitude: lon,
    hourly: "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
    timezone: "auto",
    forecast_days: days,
  });
  return _dfDesdeOm(data);
}

export async function fetchHistoricalHourly(lat, lon, days = 14) {
  const hoy = fechaLocalHoy();
  const start = sumarDias(hoy, -days);
  const data = await _openmeteoJson(OPENMETEO_ARCHIVE, {
    latitude: lat,
    longitude: lon,
    start_date: start,
    end_date: hoy,
    hourly: "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
    timezone: "auto",
  });
  return _dfDesdeOm(data);
}

function _dfDesdeOm(data) {
  const hourly = data.hourly;
  if (!hourly || !hourly.time) return [];
  const times = hourly.time;
  const filas = [];
  for (let i = 0; i < times.length; i++) {
    filas.push({
      datetime: times[i],
      t2m_c: hourly.temperature_2m?.[i] ?? null,
      rh: hourly.relative_humidity_2m?.[i] ?? null,
      wind_speed_kmh: hourly.wind_speed_10m?.[i] ?? null,
      sp: hourly.surface_pressure?.[i] ?? null,
    });
  }
  return filas;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fechas locales (YYYY-MM-DD)
// ─────────────────────────────────────────────────────────────────────────────
export function fechaLocalHoy() {
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${m}-${day}`;
}

export function sumarDias(iso, dias) {
  const d = new Date(`${iso}T12:00:00`); // mediodía local: evita saltos de DST
  d.setDate(d.getDate() + dias);
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${m}-${day}`;
}

const PROVINCIA_COORDS = {
  albacete: [38.9942, -1.8585], almeria: [36.8381, -2.4597],
  alava: [42.8465, -2.6719], asturias: [43.3619, -5.8494],
  badajoz: [38.8794, -6.9706], barcelona: [41.3874, 2.1686],
  bizkaia: [43.2569, -2.9234], burgos: [42.3439, -3.6969],
  cantabria: [43.4623, -3.8099], ceuta: [35.8893, -5.3198],
  "ciudad real": [38.9862, -3.929], cuenca: [40.0718, -2.1341],
  caceres: [39.4765, -6.3722], cadiz: [36.5345, -6.2939],
  cordoba: [37.8882, -4.7794], girona: [41.9794, 2.8214],
  gipuzkoa: [43.305, -1.9793], granada: [37.1773, -3.5986],
  guadalajara: [40.6283, -3.1636], huelva: [37.2614, -6.9447],
  huesca: [42.1398, -0.4089], jaen: [37.7796, -3.7849],
  leon: [42.5987, -5.5665], lleida: [41.6148, 0.6266],
  lugo: [43.0121, -7.5558], madrid: [40.4168, -3.7038],
  melilla: [35.2937, -2.9383], murcia: [37.9922, -1.1307],
  malaga: [36.7213, -4.4214], navarra: [42.8184, -1.6455],
  ourense: [42.3358, -7.8641], palencia: [42.0096, -4.5285],
  pontevedra: [42.431, -8.6444], salamanca: [40.9701, -5.6633],
  "santa cruz de tenerife": [28.4682, -16.2546],
  segovia: [40.9429, -4.1088], sevilla: [37.3891, -5.9845],
  soria: [41.7636, -2.465], tarragona: [41.1189, 1.2445],
  teruel: [40.3457, -1.1065], toledo: [39.8628, -4.0273],
  valladolid: [41.6523, -4.7245], zamora: [41.5034, -5.7443],
  zaragoza: [41.6488, -0.8891], avila: [40.6564, -4.6993],
};

// Normaliza un nombre de provincia para buscar en PROVINCIA_COORDS: minúsculas,
// sin acentos y sin barras (p. ej. "Ávila" → "avila", "Araba/Álava" → "alava").
const _normKey = (s) =>
  String(s).toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[/\s]+/g, "").trim();

const PROVINCIA_COORDS_NORM = {};
for (const [k, v] of Object.entries(PROVINCIA_COORDS)) {
  PROVINCIA_COORDS_NORM[_normKey(k)] = v;
}

export function getProvinceCoords(provincia) {
  const key = _normKey(provincia);
  return PROVINCIA_COORDS_NORM[key] || [40.4168, -3.7038];
}

// Provincia (clave canónica de PROVINCIA_COORDS) más cercana a un punto.
// Lo usa el mapa de la demo para rellenar el selector al hacer clic.
export function provinciaMasCercana(lat, lon) {
  let mejor = null;
  let mejorD = Infinity;
  for (const [k, [plat, plon]] of Object.entries(PROVINCIA_COORDS)) {
    const d = (plat - lat) ** 2 + (plon - lon) ** 2;
    if (d < mejorD) {
      mejorD = d;
      mejor = k;
    }
  }
  return mejor;
}

export { PROVINCIA_COORDS, PROVINCIA_COORDS_NORM, _normKey };

// ─────────────────────────────────────────────────────────────────────────────
// fetch_weather_data (weather_fetcher.py:230)
// ─────────────────────────────────────────────────────────────────────────────
export async function fetchWeatherData({ lat, lon, provincia, targetDate } = {}) {
  let cLat, cLon;
  if (lat != null && lon != null) {
    cLat = lat; cLon = lon;
  } else if (provincia) {
    [cLat, cLon] = getProvinceCoords(provincia);
  } else {
    cLat = 40.4168; cLon = -3.7038;
  }

  const hoy = fechaLocalHoy();
  const target = targetDate || hoy;
  const isToday = target === hoy;

  let dfHoraHist = [];
  try {
    dfHoraHist = await fetchHistoricalHourly(cLat, cLon, 14);
  } catch { dfHoraHist = []; }
  const lastHistDate = dfHoraHist.length
    ? dfHoraHist.reduce((a, r) => (r.datetime.slice(0, 10) > a ? r.datetime.slice(0, 10) : a), "")
    : null;

  let current = {};
  if (isToday) {
    try { current = await fetchCurrentWeather(cLat, cLon); } catch { current = {}; }
  }

  let dfForecast = [];
  try {
    dfForecast = await fetchHourlyForecast(cLat, cLon, FORECAST_HORIZON_DAYS);
  } catch { dfForecast = []; }

  let targetRows = dfForecast.filter((r) => r.datetime.slice(0, 10) === target);

  if (!isToday && targetRows.length > 0) {
    const midday = targetRows[Math.floor(targetRows.length / 2)];
    current = {
      t2m_c: midday.t2m_c,
      rh: midday.rh,
      wind_speed_kmh: midday.wind_speed_kmh,
      sp: midday.sp,
    };
  }

  if (isToday) {
    if (targetRows.length < 24 && lastHistDate && lastHistDate >= hoy) {
      const histToday = dfHoraHist.filter((r) => r.datetime.slice(0, 10) === hoy);
      if (histToday.length) {
        const vistos = new Set(targetRows.map((r) => r.datetime));
        targetRows = [...targetRows, ...histToday.filter((r) => !vistos.has(r.datetime))];
      }
    } else if (targetRows.length < 24) {
      const manana = sumarDias(hoy, 1);
      targetRows = dfForecast.filter((r) => {
        const f = r.datetime.slice(0, 10);
        return f === hoy || f === manana;
      });
    }
  }

  if (!targetRows.length) {
    if (target < hoy) {
      throw new Error(`La fecha ${target} ya pasó. Solo se predice hoy o el futuro cubierto por el forecast meteorológico.`);
    }
    throw new Error(`El forecast meteorológico no cubre ${target}. No se puede predecir sin datos.`);
  }

  // UV opcional: OpenUV requiere API key → se omite (uv_index = null).
  const { dfFeatures, dfHora } = generarFeaturesCompletas(targetRows, dfHoraHist);

  return {
    lat: cLat,
    lon: cLon,
    current,
    df_hora: dfHora,
    df_features: dfFeatures,
    uv_index: null,
    target_date: target,
  };
}
