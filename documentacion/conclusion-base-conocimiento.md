# Conclusión — Base de conocimiento de usuarios

**Fecha:** 2026-07-17

## Decisión técnica

| Aspecto | Elección | Por qué |
|---------|----------|---------|
| Motor de BBDD | **SQLite** | OLTP, transaccional, sin servidor, embebible |
| Búsqueda semántica | **sqlite-vec** | Extensión nativa de SQLite, `pip install`, sin dependencias externas |
| Interfaz LLM | **MCP server propio** (Python FastMCP) | Control total sobre esquema, validaciones y tools de dominio |
| Formato de contexto | **Markdown generado desde SQL** | Legible para el LLM, inyectable en el prompt |

## Arquitectura

```
Usuario → LLM cloud (OpenAI/Anthropic) → MCP tools → MCP server (Python)
                                                          ↓
                                                    SQLite + sqlite-vec
                                                          ↓
                                              Markdown "vista" para contexto
```

## Patrón de escritura

1. **LLM sugiere** cambios de perfil vía tool MCP (`suggest_profile_update`).
2. **Middleware valida** la sugerencia (tipos, rangos, consistencia).
3. **Sistema pregunta** al usuario si confirma el cambio.
4. **Usuario confirma** → se escribe en SQLite. Usuario rechaza → se descarta o replanifica.

Esto evita escrituras inconsistentes del LLM y da control al usuario.

## Stack resultante

- Python (FastAPI existente + FastMCP)
- SQLite (single file, dockerizado con volumen persistente)
- sqlite-vec (`pip install sqlite-vec`)
- MCP tools: ~30 líneas con FastMCP SDK

## Próximos pasos inmediatos

| # | Tarea | Prioridad |
|---|-------|-----------|
| 1 | Definir esquema SQLite de perfiles (campos, tipos, relaciones) | Alta |
| 2 | Implementar MCP server con tools: `get_profile`, `suggest_update`, `confirm_update` | Alta |
| 3 | Añadir sqlite-vec para búsqueda semántica de perfiles | Media |
| 4 | Integrar lógica temporal de aclimatación (≥14 días, ver `papers/aclimatacion/aclimatacion-calor-tiempos.md`) | Media |
| 5 | Inyectar markdown de perfil como contexto inicial del LLM | Media |

## Documentación relacionada

- `proximos_pasos.md` — hoja de ruta general
- `papers/aclimatacion/aclimatacion-calor-tiempos.md` — evidencia de aclimatación
- `arquitectura/agentes_ia.md` — arquitectura de agentes existente
