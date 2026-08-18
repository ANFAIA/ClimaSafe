# Claves de API y datos externos

Guía de las claves que necesita el proyecto y de la configuración de ERA5.
Todo lo operativo (arrancar el bot, comandos del pipeline) está en
[`inicio_rapido.md`](inicio_rapido.md).

---

## 1. ERA5 (Copernicus CDS): `~/.cdsapirc`

El cliente oficial de Copernicus (`cdsapi`) requiere un archivo de configuración
llamado `.cdsapirc`, ubicado en el directorio personal del usuario
(`~/.cdsapirc` en Linux/macOS o `C:\Users\<usuario>\.cdsapirc` en Windows).
Cada usuario debe generar su propio **Personal Access Token** desde su cuenta del
**Copernicus Climate Data Store (CDS)** y crear este archivo siguiendo la
documentación oficial. Este archivo es personal, **no debe incluirse en el
repositorio ni compartirse con otros usuarios**.

**Documentación oficial:** https://cds.climate.copernicus.eu/how-to-api

Crear el archivo:

```bash
nano ~/.cdsapirc
```

Contenido del archivo:

```yaml
url: https://cds.climate.copernicus.eu/api
key: TU_PERSONAL_ACCESS_TOKEN
```

Guardar el archivo y, opcionalmente, restringir sus permisos:

```bash
chmod 600 ~/.cdsapirc
```

> **Shapefile de límites (CNIG):** para los límites municipales, provinciales y
> autonómicos, descarga el shapefile de
> https://centrodedescargas.cnig.es/CentroDescargas/limites-municipales-provinciales-autonomicos
> y añádelo a `data/raw`.

## 2. Claves en `.env`

Todas van en **`.env`**, en la raíz del repo. Está en `.gitignore`, así que no
se sube. Es el único sitio: `make bot-start` lo carga, y el resto de comandos lo
leen desde ahí.

```bash
# Datos climáticos
ERA5S_API_KEY=...          # https://cds.climate.copernicus.eu  (además de ~/.cdsapirc)
AEMET_API_KEY=...          # https://opendata.aemet.es/centrodedescargas/altaUsuario
OpenUV_API_KEY=...         # https://www.openuv.io

# Bot de Telegram
TELEGRAM_BOT_TOKEN=...     # te lo da @BotFather — formato 1234567890:AA...
GEMINI_API_KEY=...         # https://aistudio.google.com/api-keys
GROQ_API_KEY=...           # https://console.groq.com/keys — formato gsk_...
```

> **No las exportes también en `~/.bashrc`.** Una copia vieja ahí pisa la de `.env`
> y el bot falla con un 401 que no dice de dónde viene. Si ya la tienes puesta,
> bórrala de `.bashrc` y déjala solo en `.env`.

### Comprobar que las claves del bot valen

No el formato, sino que el proveedor las acepta:

```bash
curl -s https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" | head
curl -s https://generativelanguage.googleapis.com/v1beta/openai/models \
  -H "Authorization: Bearer $GEMINI_API_KEY" | head
```

### Formatos que despistan

- **Google emite dos formatos de clave**: el clásico `AIzaSy…` de 39 caracteres y
  el nuevo `AQ.…` de 53. Los dos son válidos. Que una clave tenga buena pinta no
  significa que sirva: una revocada tiene el formato perfecto, por eso se prueba
  contra el proveedor.
- Si Google responde `401 ACCESS_TOKEN_TYPE_UNSUPPORTED`, la clave no vale —
  cópiala otra vez desde AI Studio con el botón de copiar de su fila.
