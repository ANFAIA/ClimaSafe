"""Guardia WEB-002: la demo WASM no llama al backend.

Criterio 2 de WEB-002: "La predicción individual se ejecuta entera en el
navegador, sin llamar al backend". La demo web/probar-ya/ es estática; estos
tests verifican de forma determinista (sin navegador) que ningún módulo JS ni
el HTML de la demo hace llamadas de red a rutas/hosts del backend, y que toda
URL absoluta que aparece está en la lista blanca (Open-Meteo, CDN del runtime,
Leaflet, fuentes, OSM). El único `fetch` de datos es a api.open-meteo.com
(meteorología, CORS, con fallback offline a scenarios.json).

Demostración en vivo equivalente: servir la demo con
`python3 -m http.server 8091 --directory web/probar-ya`, pulsar «Predecir» y
abrir la pestaña Red de DevTools: solo aparecen ficheros estáticos locales y
Open-Meteo; ninguna petición a /api/*.
"""

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEMO = _ROOT / "web" / "probar-ya"

# Rutas/hosts que delatan una llamada al backend del proyecto (en argumentos
# de fetch o URLs). Nota: 'climasafeai' no está aquí porque aparece en
# comentarios que citan ficheros Python; la URL absoluta se valida con la
# lista blanca de abajo.
_PATRONES_BACKEND = [
    "/api/",
    "/predict",
    "localhost:",
    "127.0.0.1",
    "0.0.0.0",
    ":8000",
    ":8090",
]

# URLs absolutas legítimas de la demo (meteorología, CDN respaldo, mapa, fuentes).
_URLS_PERMITIDAS = (
    "api.open-meteo.com",
    "archive-api.open-meteo.com",
    "unpkg.com",               # Leaflet (mapa)
    "cdn.jsdelivr.net",        # respaldo CDN del runtime onnxruntime-web
    "fonts.googleapis.com",    # tipografías del sitio
    "fonts.gstatic.com",
    "github.com",              # enlaces (attribution / repo)
    "openstreetmap.org",       # attribution del mapa
    "basemaps.cartocdn.com",   # tiles del mapa
)

_FICHEROS = sorted(
    list((_DEMO / "js").glob("*.js"))
    + [_DEMO / "index.html"]
)


def _texto() -> str:
    return "\n".join(f.read_text(encoding="utf-8") for f in _FICHEROS)


def test_no_hay_llamadas_al_backend():
    """Ningún fetch/URL de la demo apunta a rutas del backend."""
    texto = _texto()
    for patron in _PATRONES_BACKEND:
        assert patron not in texto, (
            f"patrón de backend '{patron}' encontrado en la demo (WEB-002 criterio 2)"
        )
    # La demo tampoco usa otros mecanismos de red que no sean fetch a ficheros.
    for token in ("XMLHttpRequest", "WebSocket", "EventSource"):
        assert token not in texto, f"mecanismo de red '{token}' no esperado en la demo"


def test_urls_absolutas_en_lista_blanca():
    urls = set(re.findall(r"https?://([a-zA-Z0-9._-]+)", _texto()))
    fuera = sorted(u for u in urls if not any(u == p or u.endswith("." + p) for p in _URLS_PERMITIDAS))
    assert not fuera, f"URLs fuera de la lista blanca en la demo: {fuera}"


def test_fetch_no_escapa_de_la_demo():
    """Los fetch de la demo usan rutas relativas dentro de web/probar-ya/:
    ni rutas absolutas de servidor (/modelos/...) ni '..' que salga de la demo.
    """
    for f in _FICHEROS:
        texto = f.read_text(encoding="utf-8")
        for m in re.finditer(r"fetch\(\s*([`'\"][^`'\"]+[`'\"])", texto):
            ruta = m.group(1).strip("`'\"")
            if ruta.startswith("http"):
                continue
            assert not ruta.startswith("/"), f"{f.name}: fetch absoluto '{ruta}'"
            assert ".." not in ruta, f"{f.name}: fetch que escapa de la demo '{ruta}'"
