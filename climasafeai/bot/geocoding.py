"""
climasafeai.bot.geocoding — resolver nombres de lugar a coordenadas.

Contra Nominatim (OpenStreetMap), que es gratis y no pide clave. Se hace aquí y
no con un LLM a propósito: un modelo devuelve coordenadas plausibles inventadas y
la predicción saldría con el tiempo de otro sitio sin que salte ninguna alarma.
Un geocodificador o encuentra el sitio o falla, que es lo que queremos.

Nominatim pide un User-Agent identificable y como mucho una petición por segundo:
https://operations.osmfoundation.org/policies/nominatim/
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_BASE = "https://nominatim.openstreetmap.org"
_USER_AGENT = "ClimaSafeAI/1.0 (riesgo cardiovascular por calor; github.com/cacelass)"
_TIMEOUT = 10

# Nominatim no siempre llama igual al nivel provincial: en España la provincia
# suele venir en `province`, pero en comunidades uniprovinciales (Madrid, Murcia,
# Asturias...) llega como `state`, y en algunas zonas solo hay `county`.
_CLAVES_PROVINCIA = ("province", "state", "county", "region")


def _extraer_provincia(address: dict) -> str | None:
    """Saca el nombre de provincia de un `address` de Nominatim.

    Función pura: se puede testear sin red.
    """
    for clave in _CLAVES_PROVINCIA:
        valor = address.get(clave)
        if valor:
            # "Provincia de Pontevedra" → "Pontevedra"
            for prefijo in ("Provincia de ", "Província de ", "Province of "):
                if valor.startswith(prefijo):
                    valor = valor[len(prefijo):]
            return valor.strip()
    return None


def _extraer_nombre(item: dict) -> str:
    """Nombre corto y legible del resultado: 'Aldán, Pontevedra'.

    Se prefiere el `name` del propio resultado —lo que el usuario escribió— al
    municipio que lo contiene: quien busca "Aldán" quiere ver "Aldán", no "Cangas
    de Morrazo", aunque las coordenadas sean las mismas.
    """
    address = item.get("address") or {}
    provincia = _extraer_provincia(address)

    nombre = item.get("name")
    if not nombre:
        for clave in ("village", "town", "city", "municipality", "hamlet", "suburb"):
            if address.get(clave):
                nombre = address[clave]
                break
    if not nombre:
        nombre = (item.get("display_name") or "").split(",")[0].strip()

    return f"{nombre}, {provincia}" if provincia and nombre else (nombre or "")


def _pedir(ruta: str, **params) -> list | dict | None:
    params.setdefault("format", "jsonv2")
    url = f"{_BASE}/{ruta}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Nominatim falló en %s: %s", ruta, exc)
        return None


def buscar_lugar(nombre: str, pais: str = "es") -> dict | None:
    """Busca un lugar por nombre. Devuelve {lat, lon, provincia, nombre} o None.

    None significa "no lo he encontrado", y el bot debe decirlo y volver a
    preguntar — nunca inventarse unas coordenadas.
    """
    if not nombre or not nombre.strip():
        return None

    datos = _pedir("search", q=nombre.strip(), countrycodes=pais,
                   addressdetails=1, limit=1)
    if not datos:
        return None

    item = datos[0]
    try:
        return {
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "provincia": _extraer_provincia(item.get("address") or {}),
            "nombre": _extraer_nombre(item),
        }
    except (KeyError, TypeError, ValueError):
        # SEC-001: `item` trae la dirección completa (calle, ciudad, país);
        # no se escribe al log.
        logger.warning("Respuesta de Nominatim sin lat/lon utilizables")
        return None


def provincia_desde_coords(lat: float, lon: float) -> dict | None:
    """Geocodificación inversa: de unas coordenadas al nombre del sitio.

    Se usa cuando el usuario pulsa "enviar ubicación": Telegram da lat/lon
    exactas, pero el modelo necesita además el nombre de la provincia.
    """
    datos = _pedir("reverse", lat=lat, lon=lon, addressdetails=1, zoom=10)
    if not isinstance(datos, dict):
        return None
    address = datos.get("address") or {}
    return {
        "lat": lat,
        "lon": lon,
        "provincia": _extraer_provincia(address),
        "nombre": _extraer_nombre(datos),
    }
