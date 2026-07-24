import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, date, timedelta

# ---------------------------------------------------------------------------
# Open-Meteo API client (inlined from openmeteo_client.py)
# ---------------------------------------------------------------------------
OPENMETEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

_CURRENT_PARAMS = {
    "temperature_2m": "t2m_c",
    "relative_humidity_2m": "rh",
    "wind_speed_10m": "wind_speed_kmh",
    "surface_pressure": "sp",
}
_HOURLY_PARAMS = [
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "surface_pressure",
]

def _openmeteo_request(url, params, timeout=30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_current_weather(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(_CURRENT_PARAMS),
        "timezone": "auto",
    }
    data = _openmeteo_request(OPENMETEO_BASE, params)
    current = data.get("current", {})
    out = {}
    for om_key, proj_key in _CURRENT_PARAMS.items():
        val = current.get(om_key)
        if val is not None:
            out[proj_key] = val
    return out

def fetch_hourly_forecast(lat: float, lon: float, hours: int = 48) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(_HOURLY_PARAMS),
        "timezone": "auto",
        "forecast_hours": hours,
    }
    data = _openmeteo_request(OPENMETEO_BASE, params)
    hourly = data.get("hourly", {})
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "t2m_c": hourly.get("temperature_2m", [np.nan] * len(hourly["time"])),
        "rh": hourly.get("relative_humidity_2m", [np.nan] * len(hourly["time"])),
        "wind_speed_kmh": hourly.get("wind_speed_10m", [np.nan] * len(hourly["time"])),
        "sp": hourly.get("surface_pressure", [np.nan] * len(hourly["time"])),
    })
    return df

def fetch_historical_hourly(lat: float, lon: float, days: int = 14) -> pd.DataFrame:
    today = date.today()
    start = today - timedelta(days=days)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": today.isoformat(),
        "hourly": ",".join(_HOURLY_PARAMS),
        "timezone": "auto",
    }
    data = _openmeteo_request(OPENMETEO_ARCHIVE, params)
    hourly = data.get("hourly", {})
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "t2m_c": hourly.get("temperature_2m", [np.nan] * len(hourly["time"])),
        "rh": hourly.get("relative_humidity_2m", [np.nan] * len(hourly["time"])),
        "wind_speed_kmh": hourly.get("wind_speed_10m", [np.nan] * len(hourly["time"])),
        "sp": hourly.get("surface_pressure", [np.nan] * len(hourly["time"])),
    })
    return df
from climasafeai.data.make_dataset import (
    download_openuv,
    _agregar_estadisticas_diarias,
    _agregar_rezagos_temporales,
    HEAT_INDEX_UMBRAL_C,
    WIND_CHILL_UMBRAL_C,
)
from climasafeai.features.weather_indices import (
    add_weather_index_columns,
    heat_index,
    wind_chill,
    wbgt_from_heat_index,
    categorize_heat_index,
)
from climasafeai.features.external_features import (
    crear_mapping_provincias,
    alinear_features_provincia,
    escalar_features_provincia,
    DEMOGRAPHIC_FEATURES,
    _EMBEDDED_DEMOGRAPHICS,
)
from climasafeai.models.lstm_province_hybrid import (
    DAILY_FEATURE_COLS,
    escalar_diarias,
    LSTM_HYBRID_SCALER_DIARIAS_PATH,
)
from climasafeai.utils.paths import ARTIFACTS_DIR
from climasafeai.data.sequences import FEATURE_COLS_SEQ


def _procesar_horario_con_indices(df_hora: pd.DataFrame) -> pd.DataFrame:
    df = df_hora.copy()
    df["fecha"] = pd.to_datetime(df["datetime"]).dt.date
    df = add_weather_index_columns(df, temp_col="t2m_c", rh_col="rh", wind_col="wind_speed_kmh")
    return df


def _generar_features_completas(
    df_hora_actual: pd.DataFrame,
    df_hora_hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if df_hora_hist is not None and len(df_hora_hist) > 0:
        df_hora = pd.concat([df_hora_hist, df_hora_actual], ignore_index=True)
    else:
        df_hora = df_hora_actual
    df_hora = _procesar_horario_con_indices(df_hora)
    df_hora = df_hora.drop_duplicates(subset=["fecha", "datetime"])
    df_hora = df_hora.sort_values(["fecha", "datetime"])

    df_stats = _agregar_estadisticas_diarias(
        df_hora, group_cols=["fecha"],
        heat_col="heat_index_c", cold_col="wind_chill_c",
        umbral_calor=HEAT_INDEX_UMBRAL_C, umbral_frio=WIND_CHILL_UMBRAL_C,
    )

    df_dia = df_hora.groupby("fecha", as_index=False).first()
    df_dia = df_dia.merge(df_stats, on="fecha", how="left")

    df_dia["provincia"] = "__usuario__"
    df_dia = _agregar_rezagos_temporales(
        df_dia, group_col="provincia", date_col="fecha",
        umbral_calor=HEAT_INDEX_UMBRAL_C, umbral_frio=WIND_CHILL_UMBRAL_C,
    )
    df_dia = df_dia.drop(columns=["provincia"])
    return df_dia, df_hora


def _clase_desde_heat_index(hi_c: float) -> int:
    if pd.isna(hi_c):
        return 0
    if hi_c >= 39.0:
        return 2
    if hi_c >= 27.0:
        return 1
    return 0


def _clase_desde_windchill(wc_c: float) -> int:
    if pd.isna(wc_c):
        return 0
    if wc_c <= -25.0:
        return 2
    if wc_c <= 0.0:
        return 1
    return 0


def _get_province_coords(provincia: str) -> tuple[float, float]:
    coords = {
        "albacete": (38.9942, -1.8585), "almeria": (36.8381, -2.4597),
        "alava": (42.8465, -2.6719), "asturias": (43.3619, -5.8494),
        "badajoz": (38.8794, -6.9706), "barcelona": (41.3874, 2.1686),
        "bizkaia": (43.2569, -2.9234), "burgos": (42.3439, -3.6969),
        "cantabria": (43.4623, -3.8099), "ceuta": (35.8893, -5.3198),
        "ciudad real": (38.9862, -3.9290), "cuenca": (40.0718, -2.1341),
        "caceres": (39.4765, -6.3722), "cadiz": (36.5345, -6.2939),
        "cordoba": (37.8882, -4.7794), "girona": (41.9794, 2.8214),
        "gipuzkoa": (43.3050, -1.9793), "granada": (37.1773, -3.5986),
        "guadalajara": (40.6283, -3.1636), "huelva": (37.2614, -6.9447),
        "huesca": (42.1398, -0.4089), "jaen": (37.7796, -3.7849),
        "leon": (42.5987, -5.5665), "lleida": (41.6148, 0.6266),
        "lugo": (43.0121, -7.5558), "madrid": (40.4168, -3.7038),
        "melilla": (35.2937, -2.9383), "murcia": (37.9922, -1.1307),
        "malaga": (36.7213, -4.4214), "navarra": (42.8184, -1.6455),
        "ourense": (42.3358, -7.8641), "palencia": (42.0096, -4.5285),
        "pontevedra": (42.4310, -8.6444), "salamanca": (40.9701, -5.6633),
        "santa cruz de tenerife": (28.4682, -16.2546),
        "segovia": (40.9429, -4.1088), "sevilla": (37.3891, -5.9845),
        "soria": (41.7636, -2.4650), "tarragona": (41.1189, 1.2445),
        "teruel": (40.3457, -1.1065), "toledo": (39.8628, -4.0273),
        "valladolid": (41.6523, -4.7245), "zamora": (41.5034, -5.7443),
        "zaragoza": (41.6488, -0.8891), "avila": (40.6564, -4.6993),
    }
    key = provincia.strip().lower()
    if key in coords:
        return coords[key]
    return (40.4168, -3.7038)


def fetch_weather_data(
    lat: float | None = None,
    lon: float | None = None,
    provincia: str | None = None,
    target_date: date | None = None,
) -> dict:
    if lat is None or lon is None:
        if provincia:
            lat, lon = _get_province_coords(provincia)
        else:
            lat, lon = 40.4168, -3.7038

    today = date.today()
    if target_date is None:
        target_date = today

    is_today = target_date == today

    try:
        df_hora_hist = fetch_historical_hourly(lat, lon, days=14)
    except Exception:
        df_hora_hist = pd.DataFrame()
    if df_hora_hist is not None and len(df_hora_hist) > 0:
        last_hist_date = pd.to_datetime(df_hora_hist["datetime"]).dt.date.max()
    else:
        last_hist_date = None

    if is_today:
        try:
            current = fetch_current_weather(lat, lon)
        except Exception:
            current = {}
    else:
        current = {}

    try:
        df_hora_forecast = fetch_hourly_forecast(lat, lon, hours=48)
    except Exception:
        df_hora_forecast = pd.DataFrame()

    if df_hora_forecast is not None and len(df_hora_forecast) > 0:
        df_hora_forecast["fecha"] = pd.to_datetime(df_hora_forecast["datetime"]).dt.date
        df_hora_target = df_hora_forecast[df_hora_forecast["fecha"] == target_date].copy()

        if not is_today and len(df_hora_target) > 0:
            midday = df_hora_target.iloc[len(df_hora_target) // 2]
            current = {
                "t2m_c": float(midday["t2m_c"]),
                "rh": float(midday["rh"]),
                "wind_speed_kmh": float(midday["wind_speed_kmh"]),
                "sp": float(midday["sp"]),
            }

        if is_today:
            if len(df_hora_target) < 24 and last_hist_date is not None and last_hist_date >= today:
                if df_hora_hist is not None and len(df_hora_hist) > 0:
                    hist_today_mask = pd.to_datetime(df_hora_hist["datetime"]).dt.date == today
                    hist_today = df_hora_hist[hist_today_mask]
                    if len(hist_today) > 0:
                        df_hora_target = pd.concat([df_hora_target, hist_today]).drop_duplicates(subset="datetime")
            elif len(df_hora_target) < 24:
                tomorrow = today + timedelta(days=1)
                df_hora_target = df_hora_forecast[
                    (df_hora_forecast["fecha"] == today) | (df_hora_forecast["fecha"] == tomorrow)
                ].copy()
    else:
        df_hora_target = pd.DataFrame()

    if len(df_hora_target) == 0:
        if current:
            now = datetime.now()
            df_hora_target = pd.DataFrame([{
                "datetime": now,
                "t2m_c": current.get("t2m_c", 20.0),
                "rh": current.get("rh", 50.0),
                "wind_speed_kmh": current.get("wind_speed_kmh", 10.0),
                "sp": current.get("sp", 1013.0),
            }])

    if is_today:
        try:
            uv_data = download_openuv(lat, lon, filename=f"openuv_{lat}_{lon}.csv")
            if isinstance(uv_data, pd.DataFrame) and not uv_data.empty:
                uv_index = float(uv_data["uv_max"].iloc[0]) if "uv_max" in uv_data.columns else None
            else:
                uv_index = None
        except Exception:
            uv_index = None
    else:
        uv_index = None

    df_features, df_hora_proc = _generar_features_completas(df_hora_target, df_hora_hist)

    return {
        "lat": lat,
        "lon": lon,
        "current": current,
        "df_hora": df_hora_proc,
        "df_features": df_features,
        "uv_index": uv_index,
        "target_date": target_date.isoformat(),
    }


def build_sequence_24h(df_hora: pd.DataFrame) -> np.ndarray | None:
    if df_hora is None or len(df_hora) == 0:
        return None
    df = df_hora.copy()
    df = _procesar_horario_con_indices(df)
    available = [c for c in FEATURE_COLS_SEQ if c in df.columns]
    if not available:
        return None
    if len(df) >= 24:
        df = df.sort_values("datetime").tail(24)
    seq = df[available].to_numpy(dtype=np.float32)
    if len(seq) < 24:
        pad = np.tile(seq[-1:], (24 - len(seq), 1))
        seq = np.vstack([pad, seq])
    return seq[np.newaxis, ...]


def build_daily_feature_vector(df_features: pd.DataFrame) -> np.ndarray | None:
    if df_features is None or df_features.empty:
        return None
    latest = df_features.sort_values("fecha").iloc[-1:]
    available = [c for c in DAILY_FEATURE_COLS if c in latest.columns]
    missing = [c for c in DAILY_FEATURE_COLS if c not in latest.columns]
    for col in missing:
        latest[col] = 0.0
    vec = latest[DAILY_FEATURE_COLS].to_numpy(dtype=np.float32)
    return vec


def get_province_idx(
    provincia: str,
    mapping: dict[str, int] | None = None,
) -> int:
    if mapping is None:
        all_provs = list(_EMBEDDED_DEMOGRAPHICS.keys())
        mapping = {p: i for i, p in enumerate(sorted(all_provs))}
    key = provincia.strip().lower()
    for name, idx in mapping.items():
        if name.strip().lower() == key:
            return idx
    return 0


def get_ine_features(provincia: str) -> np.ndarray:
    key = provincia.strip().lower()
    for name, (p65, p80, pmuj, pop) in _EMBEDDED_DEMOGRAPHICS.items():
        if name.strip().lower() == key:
            return np.array([p65, p80, pmuj, np.log(pop)], dtype=np.float32)
    return np.array([20.0, 5.0, 50.0, np.log(1_000_000)], dtype=np.float32)


def escalar_para_lstm(
    seq: np.ndarray,
    ine_vec: np.ndarray,
    daily_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        scaler_seq = joblib.load(ARTIFACTS_DIR / "scaler_secuencias_lstm.joblib")
        nf = seq.shape[2]
        s_flat = scaler_seq.transform(seq.reshape(-1, nf))
        seq_s = s_flat.reshape(seq.shape)
    except Exception:
        seq_s = seq

    try:
        scaler_prov = joblib.load(ARTIFACTS_DIR / "scaler_provincia_features.joblib")
        ine_s = scaler_prov.transform(ine_vec.reshape(1, -1)).flatten()
    except Exception:
        ine_s = ine_vec

    try:
        scaler_dia = joblib.load(LSTM_HYBRID_SCALER_DIARIAS_PATH)
        daily_s = scaler_dia.transform(daily_vec.reshape(1, -1)).flatten()
    except Exception:
        daily_s = daily_vec

    return seq_s, ine_s, daily_s
