# Control de acceso por identidad en el MCP (MCP-003)

**Estado:** implementado. El punto único es `_requiere_identidad` en
`agents/tools/prediction_mcp_tool.py`; los tests están en
`tests/test_mcp_control_acceso.py`.
**Fecha:** 2026-08-07.
**Alcance:** `agents/tools/prediction_mcp_tool.py`, `climasafeai/db/manager.py`,
`data/schema.sql`, tests del MCP y `skills/climasafeai/SKILL.md`.

Este documento fija las decisiones de diseño **antes** de escribir código, porque
tres de ellas cambian el esquema de la base de datos y romper una migración a
medias sale caro. El inventario que lo sustenta está en
`progress/explorer-MCP-003.md`.

---

## 1. El agujero, con evidencia

Hay 12 tools registradas con `@_mcp.tool()` (`prediction_mcp_tool.py:956-1371`).
Ninguna sabe quién llama: **no existe hoy ningún mecanismo de identidad** — ni
token, ni cabecera leída, ni sesión, ni variable de entorno. Las únicas env vars
que lee el servidor son `SSL_KEYFILE`, `SSL_CERTFILE` e `INSECURE`
(`prediction_mcp_tool.py:1452-1454`).

Cuatro tools leen o escriben perfiles ajenos sin ninguna comprobación:

| Línea | Tool | Qué expone |
|---|---|---|
| 1013 | `listar_usuarios_mcp` | alias, id, edad, sexo, provincia y tags de **todos** |
| 1031 | `cargar_perfil_mcp` | `json.dumps(perfil)` **completo** con solo saber el alias |
| 1045 | `cargar_perfil_por_chat_id_mcp` | ídem, indexado por chat de Telegram |
| 1059 | `vincular_chat_id_mcp` | reasigna un perfil ajeno a tu chat |

El dict de `obtener_perfil` (`manager.py:235-244`) arrastra `comorbilidades`,
`farmacos`, `situacion_social`, `porcentaje_grasa`, `fototipo`, `lat`, `lon`,
`fecha_nacimiento` y `telegram_chat_id`. `cargar_perfil_mcp:1042` lo serializa
entero, sin filtro de campos.

`vincular_chat_id_mcp` es escalada de privilegios en dos pasos: quien sepa un
alias reasigna ese perfil a su chat y luego `cargar_perfil_por_chat_id_mcp` se
lo devuelve entero, legítimamente. **Cerrar solo la lectura no basta.**

### El «patrón de MCP-001» no existe como pieza reutilizable

El criterio 6 dice «reutilizando el patrón de MCP-001». Ese patrón **no es un
decorador ni una función**: son dos helpers de *resolución* —`_resolver_chat`
(806) y `_resolver_perfil` (843)— que traducen alias/perfil_id → chat/perfil y
**se fían de lo que les pasen**, más una comprobación de propiedad escrita a
mano dentro de `borrar_rutina_mcp:1184`. Reutilizar el patrón significa
reutilizar la *forma* (resolver sujeto → comprobar propiedad → error JSON
uniforme), no una pieza existente. **El punto único hay que crearlo.**

### Escala real del problema (lo que abarata la migración)

```
perfiles: 22 · con telegram_chat_id: 1 · rutinas: 1
```

22 perfiles pero **un solo usuario real conectado**. El «más usuarios» de la
nota del humano es prácticamente greenfield: la migración no tiene que preservar
un parque de usuarios en producción, y el coste operativo de emitir credenciales
es de una orden.

---

## 2. Las tres decisiones abiertas

### 2.1 De dónde sale la identidad del llamante

**El hallazgo que manda sobre el diseño:** los dos clientes configurados
(`.mcp.json` y `opencode.json:43-46`) arrancan el servidor con **`--stdio`**.
En stdio no hay cabeceras HTTP ni sesión. El bearer/OAuth nativo de
`mcp>=1.28.1` (`FastMCP(auth=…, token_verifier=…)` + `get_access_token()`) y el
`Request` de Starlette vía `ctx.request_context.request` **solo existen en
HTTP**. Si la identidad se implementa únicamente sobre ellos, en el modo que
realmente se usa hoy no habría identidad nunca: o todo devuelve error, o queda
un bypass abierto.

**Decisión: identidad por proceso en stdio, bearer en HTTP, un solo punto de
lectura.**

- **stdio** → `CLIMASAFE_MCP_TOKEN` (env var) o `--identidad <token>`, leído al
  arrancar. **Un proceso = un llamante.** Es el único mecanismo que funciona
  donde el proyecto corre hoy, y encaja con el modelo real de uso: cada host
  (Claude Desktop, opencode) lanza su propio proceso.
- **HTTP** → `Authorization: Bearer <token>`.
- Las dos vías desembocan en la **misma** función `_identidad_actual()`. El
  transporte se resuelve ahí dentro y en ningún otro sitio.

Descartado: pasar `identidad: str` como parámetro de cada tool. Funciona en los
dos transportes y es trivial de testear, pero mete la credencial en el
`inputSchema` de las 12 tools — o sea, en el contexto del LLM y en los logs de
cualquier host. Es exactamente la lección de BOT-004 que el criterio 4 de
MCP-002 obliga a no repetir.

### 2.2 Migración de `alias` / `chat_id` como llave de acceso

Hoy no hay ninguna columna que sirva de ID opaco: `perfiles.id` es
`INTEGER PRIMARY KEY AUTOINCREMENT` (secuencial y adivinable — el mismo defecto
que MCP-001 encontró en los ids de rutinas), y `alias` y `telegram_chat_id` son
`TEXT UNIQUE` que funcionan **como llave de acceso**.

**Decisión: separar identificador público de credencial.** Conflacionar los dos
obligaría a que el identificador no apareciera nunca en una respuesta, y varias
tools necesitan devolverlo.

Tres columnas nuevas en `perfiles`, todas vía `_migrate()`:

| Columna | Qué es | ¿Sale en respuestas? |
|---|---|---|
| `uid TEXT UNIQUE` | identificador público opaco (`usr_` + 16 bytes base32) | **Sí** — sustituye a alias/chat_id como llave |
| `mcp_token_hash TEXT` | `sha256` del secreto del llamante | **Nunca** |
| `rol TEXT DEFAULT 'usuario'` | `usuario` \| `admin` | Sí |

El patrón de migración ya existe y es trivial: `PRAGMA table_info(perfiles)` → si
falta la columna, `ALTER TABLE … ADD COLUMN` + `CREATE INDEX IF NOT EXISTS`
(`manager.py:121-131`). `_validar_campos_perfil` (138) lee las columnas reales,
así que las nuevas se aceptan solas en `crear_perfil`/`actualizar_perfil` sin
tocar validaciones.

**`_migrate` no hace backfill hoy** — ninguna de sus tres ramas rellena nada.
Hay que añadirlo explícitamente o los 22 perfiles existentes quedan con `uid`
NULL, y `NULL` en SQLite no colisiona con `UNIQUE`: veintidós perfiles
indistinguibles por su llave. **El backfill de `uid` no es opcional.**

`mcp_token_hash` sí se queda NULL a propósito: significa «este perfil todavía no
tiene acceso por MCP». Se emite bajo demanda con un comando (`make mcp-token
ALIAS=…`) que imprime el secreto una sola vez. Con un único perfil vinculado a
un chat, el coste operativo del cambio es una orden.

**Lo que NO se toca:** `rutinas.chat_id` (`schema.sql:124`) y
`avisos_config.chat_id` (137) siguen siendo la clave de *almacenamiento*. El
`uid` sustituye al chat_id como llave de *acceso*, no como clave de fila.
Cambiar el almacenamiento en la misma feature dejaría rutinas huérfanas sin
ganar nada: la propiedad ya se comprueba antes de llegar a la consulta.

### 2.3 Qué pasa con `listar_usuarios_mcp`

El criterio 3 admite dos salidas: que desaparezca o que quede tras rol de
administración.

**Decisión: se queda, tras `rol == 'admin'`.** Motivo concreto:
`tests/test_prediction_mcp_grafica.py:27-32` congela los 11 nombres de tool en
`TOOLS_PREVIAS` y `test_las_11_tools_previas_siguen_registradas` (153) exige que
sigan registradas y con `outputSchema is not None`. Borrarla rompe ese test por
una razón ajena a la feature. Además un llamante normal invocándola debe recibir
**error** (criterio 3), no una lista vacía: una lista vacía es indistinguible de
«no hay usuarios» y no demuestra nada.

---

## 3. El punto único (criterio 6)

El criterio no se aguanta con un `if` repetido: exige que **una tool que se lo
salte falle un test**.

```python
@_mcp.tool()
@_requiere_identidad("perfil_propio")
def cargar_perfil_mcp(uid: str) -> str:
    ...
```

Orden de decoradores: `_requiere_identidad` envuelve primero, `_mcp.tool()`
registra el envoltorio. `FastMCP.tool()` devuelve `fn` **sin envolver**
(`mcp/server/fastmcp/server.py:504`), así que las tools siguen siendo funciones
normales del módulo y los tests las pueden llamar directas, como hacen hoy. El
envoltorio usa `functools.wraps`, e `inspect.signature` sigue `__wrapped__`, así
que el `inputSchema` que genera FastMCP no cambia.

Tres piezas, ninguna más:

1. **`_identidad_actual() -> tuple[dict | None, dict | None]`** — resuelve el
   token del transporte (§2.1), lo compara por hash contra `mcp_token_hash` y
   devuelve `(perfil_del_llamante, error)`. Es el **único** sitio del proyecto
   que sabe de dónde sale una identidad.
2. **`_requiere_identidad(nivel)`** — decorador. Llama a `_identidad_actual()`,
   corta con el error JSON uniforme si no hay identidad válida, y comprueba
   propiedad (`perfil["id"] == solicitante["id"]`) o rol según el nivel. Marca
   la función con `__climasafe_acceso__ = nivel`.
3. **`_acceso_publico`** — para las dos tools que no tocan la BD
   (`predict_risk_mcp:957` y `grafica_riesgo_horario_mcp:1309`, ambas reciben
   todo por parámetro). Pone la **misma marca** con valor `"publico"`. La
   exención es explícita y auditable; nunca implícita por omisión.

**El test que cierra el criterio 6:**

```python
def test_toda_tool_registrada_pasa_por_el_control_de_acceso():
    nombres = {t.name for t in asyncio.run(_mcp._mcp.list_tools())}
    for nombre in nombres:
        fn = getattr(prediction_mcp_tool, nombre)
        assert hasattr(fn, "__climasafe_acceso__"), (
            f"{nombre} no pasa por _requiere_identidad ni se declaró pública"
        )
```

`asyncio.run(mcp._mcp.list_tools())` ya se usa en
`tests/test_prediction_mcp_grafica.py:156`, así que no es maquinaria nueva.
Añadir una tool nueva sin decorador **rompe la suite**, que es justo lo que pide
el criterio: «no que se confíe en que el próximo se acuerde».

`_requiere_identidad` respeta la convención de error del fichero —
`json.dumps({"error": …})` o `{"success": False, "error": …}` como string, nunca
una excepción. Romperla tumbaría los tests existentes que hacen
`_json(...)["error"]`.

**Ojo con el `try/except ImportError`:** las tools viven dentro del bloque
`try:` de `prediction_mcp_tool.py:951-1375` (`_HAS_MCP` en 1373-1375). Los tres
helpers se definen **fuera** del try, antes de la línea 951, para que existan
aunque `mcp` no esté instalado y para poder testearlos sin el servidor.

---

## 4. Minimización de campos (criterio 5)

Política escrita, que es lo que el criterio pide:

**Un perfil ajeno no devuelve ni un campo.** No hay vista reducida del perfil de
otro: el acceso cruzado es un error, no una respuesta recortada (criterio 2:
«NO devuelve ni un campo»).

**De un perfil propio** el llamante ve todo salvo lo que no le aporta nada y sí
sirve para pivotar: `mcp_token_hash` no sale nunca, y `telegram_chat_id` solo
sale en las tools de rutinas, donde es la clave de almacenamiento.

**Nunca salen de un perfil que no sea el propio**, ni siquiera agregados:
`farmacos`, `comorbilidades`, `situacion_social`, `porcentaje_grasa`,
`fototipo`, `fecha_nacimiento`, `lat`, `lon`.

> **Trampa de nombres:** la medicación se llama **`farmacos`** en BD
> (`perfil_farmacos`, `manager.py:237`). Escribir `medicacion` como clave de
> perfil ya ha costado dos veces en este proyecto — está documentado en el
> comentario de `_validar_campos_perfil` (`manager.py:138-145`). El criterio 5
> dice «medicación»; en código es `farmacos`.

`riesgo_rutinas_dia_mcp:1296-1298` devuelve hoy `alias`, `perfil_id` y `chat_id`
en la respuesta: pasan a ser `uid`. `listar_rutinas_mcp:1121` ya hace
`r.pop("chat_id")` — es el único precedente de minimización en el fichero y la
forma a seguir.

---

## 5. Impacto en los tests existentes

| Fichero | Impacto |
|---|---|
| `tests/test_prediction_mcp_rutinas.py` | **El que más se rompe.** 19 tests que llaman directo con `alias=`/`chat_id=`/`perfil_id=` (88-240). Su fixture `db` (29-41) monkeypatchea `DBManager` sobre un SQLite temporal y se reutiliza tal cual; hace falta una fixture nueva que fije la identidad monkeypatcheando `_identidad_actual` |
| `tests/test_prediction_mcp_grafica.py` | `TOOLS_PREVIAS` (27-32) sigue verde: no se borra ninguna tool. El `outputSchema` no cambia (es la salida); el `inputSchema` sí |
| `tests/test_riesgo_colectivo_bug002.py:145` | Usa `predict_group_risk`, que **no** es tool registrada. Intacto |

Las 6 tools de perfil **no tienen ningún test hoy**. El criterio 8 (acceso
propio permitido y ajeno denegado para *cada* tool de lectura) los crea desde
cero.

`predict_group_risk` (400) lee perfiles vía `db.buscar_por_tag` (426) pero **no
está decorada como tool** — solo la usa la web. Queda fuera del alcance de
MCP-003; el acceso desde la web es otra frontera y hoy tiene el mismo defecto
(`chat/app.py:475` `_chat_id_de_perfil` + comprobación inline en 1359: resuelve,
no autentica).

---

## 6. Fronteras

- **MCP-002** decide *qué operaciones* se permiten (interruptor
  lectura/escritura). MCP-003 decide *de quién son los datos* que devuelve cada
  operación. Las dos envuelven el mismo punto de entrada, y por eso **no se
  hacen en paralelo**: MCP-002 se apoyará en `_requiere_identidad` en vez de
  crear otro envoltorio. Va antes MCP-003, porque el agujero de lectura es el
  grave y es justo el que MCP-002 deja abierto por diseño.
- **SEC-001** cubre el dato en reposo (permisos de fichero, PII en logs,
  backup). MCP-003 cubre el acceso a través del MCP. No se solapan. Ojo: las dos
  escriben en `climasafeai/db/manager.py`.
- **MCP-001** es el precedente: comprobó recurso-contra-sujeto. Lo que falta
  aquí es la capa de arriba, **llamante-contra-sujeto**.

### Sin solape con DATA-003 (opencode)

MCP-003 **no obliga a tocar** `climasafeai/models/ensemble.py`,
`climasafeai/data/weather_fetcher.py` ni `tests/test_ensemble.py`.
`prediction_mcp_tool.py` solo *importa y llama* `predict_ensemble` (202-203,
468, 526), `perfil_horario_desde_df` (598) y `PERS_THRESHOLD_PELIGRO` (636,
654). El control de acceso se aplica antes de llegar ahí; ninguna firma cambia.

---

## 7. Riesgos abiertos

1. **`skills/climasafeai/SKILL.md:101-106`** documenta las tools de perfil,
   incluida «`listar_usuarios_mcp` | Lista todos los perfiles». Si no se
   actualiza, el SKILL miente sobre el comportamiento.
2. **El único usuario real pierde acceso hasta que se le emita un token.** Es el
   cambio pretendido, pero conviene emitirlo en la misma sesión del despliegue.
3. **No verificado en ejecución:** que `ctx.request_context.request` llegue
   poblado en streamable-HTTP (deducido de `mcp/server/streamable_http.py:268-274`)
   y que `FastMCP(auth=…)` arranque en 1.28.1 (solo se leyó la validación de
   `server.py:217-231`). Ninguna de las dos bloquea el diseño: la vía stdio, que
   es la que se usa hoy, no depende de ellas.
4. **`uid` con `UNIQUE` y backfill en la misma migración.** Si el backfill falla
   a medias quedan filas con NULL que no violan `UNIQUE`. El `ALTER` y el
   `UPDATE` van en la misma transacción de `_migrate`.

---

## 8. Decisiones tomadas al implementar (no cubiertas arriba)

1. **`uid` no lleva `UNIQUE` en la columna, sino un índice único.** SQLite no
   admite `ALTER TABLE … ADD COLUMN … UNIQUE`, así que la unicidad la da
   `idx_perfiles_uid`, creado siempre (con `IF NOT EXISTS`) para que una BD
   nueva y una migrada acaben iguales.
2. **`cargar_perfil_mcp(uid)` es opcional.** El llamante tiene un token, no un
   uid: obligarle a nombrarse le dejaba sin forma de averiguar el suyo. Sin
   `uid` carga el perfil del token. La misma regla vale para las tools de
   rutinas: si no se nombra sujeto, `_requiere_identidad` inyecta el propio.
3. **`crear_perfil_mcp` es nivel `"identidad"`, no `"admin"`.** Cualquier
   llamante autenticado puede crear perfiles (un cuidador, por ejemplo); el
   perfil nace con `mcp_token_hash` NULL, o sea sin acceso por MCP hasta que se
   le emita uno. No hay onboarding anónimo.
4. **`vincular_chat_id_mcp` cambió de firma:** `(chat_id, uid=None)` en vez de
   `(alias, chat_id)`. El `chat_id` de esta tool es el chat que se vincula, no
   el sujeto, así que el decorador se limita a `sujeto=("uid",)`. Además
   rechaza un `chat_id` que ya sea de otro perfil: las rutinas y los avisos
   cuelgan del chat, y apropiárselo era el segundo paso de la escalada.
   `crear_perfil_mcp` lleva la misma comprobación.
5. **`obtener_perfil` borra `mcp_token_hash` del dict**, no solo las tools. Ese
   getter alimenta también la web y el bot; dejar la credencial dentro era
   confiar en que los tres consumidores se acordaran de quitarla.
6. **El error de «perfil ajeno» y el de «perfil inexistente» son idénticos**, y
   no repiten el identificador probado. Distinguirlos convertía el mensaje en
   un oráculo de enumeración, que es justo lo que prohíbe el criterio 4.
7. **`crear_perfil_mcp` devuelve `uid` en vez de `id`**, en línea con §4.
8. **Un solo sujeto por llamada, y se validan todos los que lleguen.** La
   primera versión cortaba en el primer parámetro de sujeto no vacío, y eso era
   un bypass: el bucle miraba `uid` primero y `_resolver_chat` mira `chat_id`
   primero, así que `alias=<propio>, chat_id=<ajeno>` hacía que el guardián
   aprobara un sujeto y la tool usara otro. Afectaba a `listar_rutinas_mcp`,
   `crear_rutina_mcp`, `borrar_rutina_mcp` y `configurar_hora_aviso_mcp`;
   `riesgo_rutinas_dia_mcp` se libraba solo porque `_resolver_perfil` ordena
   igual que el bucle — una coincidencia, no una garantía.

   Ahora se hacen las dos cosas, en este orden: primero se comprueban **todos**
   los sujetos nombrados (cualquiera que no sea del llamante → `ERROR_AJENO`), y
   después se rechaza nombrar más de uno (`ERROR_SUJETO_AMBIGUO`). El orden
   importa: la comprobación de propiedad es la que sostiene la seguridad y tiene
   que ser incondicional, de modo que siga cerrando el agujero aunque alguien
   toque o quite la regla de ambigüedad. Y así el ataque mixto recibe el mismo
   error que un acceso cruzado normal, sin convertir el mensaje en una señal.

   El invariante lo congela `TestSujetoMixto::test_cada_tool_con_dos_sujetos_
   rechaza_propio_mas_ajeno`, que recorre las tools registradas, se queda con
   las que declaran más de un sujeto (`__climasafe_sujeto__`) y prueba cada par
   propio+ajeno. Una tool nueva con dos sujetos entra sola en el test.

---

## 9. MCP-002: solo lectura por defecto, escritura por token

**Estado:** implementado. El check es `_requiere_token_escritura` en
`agents/tools/prediction_mcp_tool.py`; los tests están en
`tests/test_mcp_escritura.py`. **Fecha:** 2026-08-10.

MCP-003 decidió *quién* llama; MCP-002 decide *qué* operaciones están abiertas.
Por defecto el servidor es de **solo lectura**: las cinco tools que escriben en
la BD de perfiles — `crear_perfil_mcp`, `crear_rutina_mcp`,
`borrar_rutina_mcp`, `vincular_chat_id_mcp`, `configurar_hora_aviso_mcp` —
responden `{"error": …}` y **no tocan nada** si el proceso no arrancó con el
token de escritura. Las siete de lectura (predicciones, perfil propio, rutinas,
riesgo por rutinas) no cambian y no piden nada nuevo.

### Cómo se habilita la escritura

```bash
CLIMASAFE_MCP_WRITE_TOKEN=<secreto> uv run python -m agents.tools.prediction_mcp_tool --stdio --identidad <token-identidad>
# equivalente, como parámetro de arranque:
uv run python -m agents.tools.prediction_mcp_tool --stdio --identidad <token-identidad> --token-escritura <secreto>
```

El token es una credencial de **arranque** del proceso, igual que la identidad
en stdio (un proceso = un llamante, §2.1). El operador del host decide quién
escribe: el host cuyo comando lleva la variable puede escribir; el que no, solo
lee. En HTTP la variable vive en el entorno del servidor y decide globalmente.

### Decisiones de diseño

1. **El token NO entra en la firma de ninguna tool.** §2.1 descartó pasar la
   credencial como parámetro porque aparece en el `inputSchema`, en el contexto
   del LLM y en los logs del host (lección BOT-004). El token de escritura se
   comporta igual: solo existe en el entorno del proceso, nunca como argumento.
2. **La capa va DEBAJO de `_requiere_identidad`.** Orden de decoradores:
   `@_mcp.tool()` → `@_requiere_identidad(...)` → `@_requiere_token_escritura`
   → la función. Identidad y propiedad se comprueban antes: un llamante anónimo
   o que nombra un sujeto ajeno recibe exactamente el mismo error que antes de
   MCP-002; solo después se decide si la operación puede escribir. Por eso los
   tests de MCP-003 no cambian de semántica: el fixture del llamante legítimo
   pone también el token de escritura.
3. **Clasificación explícita con `__climasafe_escritura__`**, igual que
   `__climasafe_acceso__` para la identidad. La lista congelada está en
   `TestClasificacion` de `tests/test_mcp_escritura.py`; una tool de escritura
   nueva sin marcar (o mal clasificada) rompe la suite.
4. **Sin token no hay ninguna ruta que escriba.** El check es el punto único:
   toda tool de escritura pasa por `_requiere_token_escritura` o no está
   marcada como tal.
5. **El token nunca se devuelve ni se loguea.** Ni siquiera se lee como dato:
   el check solo comprueba su presencia. El error de solo lectura nombra la
   variable (`CLIMASAFE_MCP_WRITE_TOKEN`), nunca un valor real.
6. **`configurar_hora_aviso_mcp` se capa entera**, incluida su consulta con
   `hora=None`: es una tool que *puede* escribir, y distinguir dentro de la
   misma tool añadiría una excepción por caso sin ganancia de seguridad.
