# Schedule Agent — Gestión de Cron

Valida, describe y calcula próximas ejecuciones de expresiones cron.

## Acciones

### `validate` — Validar expresión cron
```bash
uv run python -m agents run schedule validate --expression "0 6 * * 1-5"
```

### `to_human` — Traducir cron a texto legible
```bash
uv run python -m agents run schedule to_human --expression "@daily"
# → "una vez al día (medianoche)"
```

### `next_runs` — Próximas ejecuciones
```bash
uv run python -m agents run schedule next_runs --expression "0 */2 * * *"
# → "Próximas 5 ejecuciones: ..."
```

## Formato soportado
- 5 campos estándar: minuto hora día-del-mes mes día-de-la-semana
- Alias: `@hourly`, `@daily`, `@weekly`, `@monthly`, `@yearly`, `@annually`, `@midnight`
- Rangos: `1-5`, `*/15`, listas: `0,30`
