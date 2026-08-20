# Seguridad de la base de datos de perfiles (SEC-001)

> Decisión y registro del acceso a `data/climasafe.db` (SQLite de perfiles con
> edad, sexo, porcentaje de grasa, comorbilidades, medicación, situación
> social, ubicación y chat_id de personas reales). Fecha: 2026-08-20.

## Qué protege esta medida

La BD de perfiles es el único sitio del proyecto donde se cruzan datos de
salud con identificadores: un perfil puede tener edad, comorbilidades,
medicación, coordenadas exactas y el chat_id de Telegram de la misma persona.
Una filtración en el fichero, en un backup o en un log convierte esa fila en
una ficha médica personal. SEC-001 trata el fichero como lo que es: un dato
sensible en reposo, no un artefacto de desarrollo más.

## Quién escribe en la BD

| Escritor | Dónde | Qué escribe | Validación |
|----------|-------|-------------|------------|
| Bot de Telegram | `climasafeai/bot/telegram_bot.py`, `climasafeai/bot/chat_flow.py` | Perfiles completos (formulario de 10 campos + ubicación), rutinas semanales, horas de aviso, última salida | `DBManager._validar_campos_perfil` (rechaza campos que no son columnas con `CampoDesconocidoError`), conversión de booleanos y de `entrenado`, arrays M2M (`comorbilidades`, `farmacos`, ...) |
| Web (chat y predicción) | `chat/app.py` (`/api/predict`, `/api/perfil`, `/api/perfil/{id}/tags`) | Perfiles y consultas al historial | La misma `_validar_campos_perfil`; el error se devuelve al cliente (400), no se esconde |
| MCP | `agents/tools/factors_mcp_tool.py`, `agents/tools/prediction_mcp_tool.py` | Factores de riesgo (sugerir/aprobar/actualizar) y lectura/escritura de perfiles vía tokens | `DBManager` exige campos válidos; los tokens MCP se guardan solo como hash (`mcp_token_hash`) |

Los tres escritores comparten el mismo objeto `DBManager` y por tanto la misma
BD y el mismo esquema. No hay una copia por canal: el perfil creado por el bot
lo ve la web y lo lee MCP.

## Qué pasa si dos escriben a la vez

SQLite con `PRAGMA journal_mode=WAL` (activado en cada conexión por
`DBManager.conn()`): varios lectores concurrentes sin bloqueo y **un solo
escritor a la vez**. Si dos procesos escriben simultáneamente, el segundo espera
el *lock* de escritura hasta 5 segundos (el `timeout` por defecto de
`sqlite3.connect`) y entonces lanza `database is locked`. Ninguno de los tres
escritores (bot, web, MCP) hace reintentos de escritura: en la práctica la
ventana de conflicto es mínima (todas las escrituras son transacciones cortas),
pero un pico de escrituras simultáneas puede fallar con 500. Si eso empieza a
ocurrir, la solución documentada es subir el `timeout` en `DBManager.conn()` o
centralizar las escrituras en un único proceso.

## Permisos y control de versiones

- El fichero `data/climasafe.db` (y sus auxiliares WAL `-wal`/`-shm`) se crean
  con permisos `600` (solo el propietario) en cada conexión
  (`DBManager._asegurar_permisos`), en vez del `644` que deja el umask del
  proceso.
- La ruta está fuera del control de versiones: regla `data/**/*.db` en
  `.gitignore`. Se comprueba con `git check-ignore -v data/climasafe.db`.
- Los backups (`make backup-bd`) se crean también con `600`.

## Backup y restauración

- Comando: `make backup-bd` → `scripts/backup_bd.py backup` (por defecto
  `data/backups/climasafe_AAAAMMDD_HHMMSS.db`).
- Restauración: `make restore-bd ORIGEN=ruta/al/backup.db` →
  `scripts/backup_bd.py restore`.
- Técnica: API de backup de sqlite3 (equivalente a `VACUUM INTO`), consistente
  incluso con WAL. La restauración elimina los auxiliares `-wal`/`-shm` antes de
  sobrescribir y debe hacerse con bot y web detenidos.

## Decisión: ¿cifrado en reposo? No por ahora

**Decisión (2026-08-20): los datos de salud NO se cifran en reposo.**

Motivos:

1. **El cifrado de SQLite a nivel de fichero no encaja con la arquitectura.**
   Tres procesos (bot, web, MCP) abren la misma BD directamente con sqlite3.
   Cifrar la BD (SQLCipher u otro) exigiría una capa de descifrado en cada
   lector, claves compartidas entre procesos y perder el acceso nativo de
   sqlite-vec (el RAG hace búsquedas dentro de SQLite) y de las herramientas de
   mantenimiento.
2. **El riesgo actual ya está cubierto por medidas más simples:** permisos 600,
   fuera de git, backups con 600 y logs sin datos identificables (este mismo
   ticket). El fichero vive en una máquina del propietario, no en un bucket
   compartido ni en un servicio multi-tenant.
3. **La protección correcta para "en reposo" es el cifrado de disco** (LUKS /
   dm-crypt) en la máquina que aloja el servicio: protege la BD, los logs, los
   backups y todo lo demás con una sola llave a nivel de sistema, sin tocar
   una línea del código de aplicación.

**Cuándo se revisaría:** si la BD pasa a un hosting compartido/cloud, si se
abre el acceso a terceros o si cambia el modelo de amenazas. En ese caso la
opción natural es mover la BD a PostgreSQL con cifrado de columna para los
campos de salud, o SQLCipher si se mantiene SQLite, y documentar la gestión de
claves.

## Protección de los logs

Desde SEC-001, el bot aplica dos filtros a todos sus handlers:

- `_OcultarToken` (preexistente): tapa el token de Telegram y las claves
  `*_API_KEY` en cualquier línea.
- `_OcultarChatId` (nuevo): tapa los identificadores numéricos de 6+ dígitos
  (chat_id de Telegram) en cualquier línea, aunque una línea futura meta uno
  sin querer.

Además se han eliminado de los mensajes de log los datos que ningún filtro
puede adivinar: el texto libre del usuario (puede contener edad, medicación o
comorbilidades en lenguaje natural), las coordenadas exactas de ubicación y los
nombres de los campos de salud de los callbacks. En `logs/` no debe aparecer
ningún dato identificable; la demostración está en
`scripts/demo_seguridad_logs.py` y en `tests/test_sec_001.py`.

## Verificación

```bash
ls -l data/climasafe.db                 # -rw-------  → permisos 600
git check-ignore -v data/climasafe.db   # .gitignore:data/**/*.db
make backup-bd                          # backup de ida
make restore-bd ORIGEN=...              # restauración de vuelta
python scripts/demo_seguridad_logs.py   # grep sobre logs/ sin PII
make test                               # suite completa
```