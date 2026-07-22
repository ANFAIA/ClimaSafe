# Explicaciones contrafactuales

Genera cambios realistas en el perfil del usuario y muestra cuánto bajaría su riesgo con cada uno.

## Endpoint

`POST /api/contrafactuales`

Mismo body que `/api/predict`: `{ provincia, perfil, lat?, lon? }`.

Respuesta:

```json
{
  "clase_final_sistema": "PELIGRO",
  "clase_personalizada": "PELIGRO",
  "probabilidad_personalizada": 0.78,
  "contrafactuales": [
    {
      "id": "aclimatado",
      "descripcion": "Aclimatarse al calor (14 días de exposición gradual)",
      "probabilidad_sin_cap": { "actual": 0.876, "tras_cambio": 0.815 },
      "probabilidad_con_cap": { "actual": 0.777, "tras_cambio": 0.777 },
      "clase_actual": "PELIGRO",
      "clase_tras_cambio": "PELIGRO",
      "mejora_porcentual_puntos": 6.1,
      "enmascarado_por_cap": true,
      "factor_total_actual": 6.082,
      "factor_total_tras_cambio": 3.0,
      "producto_bruto_tras_cambio": 3.801
    }
  ]
}
```

## Cambios evaluados

| id | Condición | Cambio |
|----|-----------|--------|
| `nivel_actividad` | `intensa`/`muy_intensa`/`moderada`/`ligera` | reduce un nivel |
| `aclimatado` | `false` | → `true` |
| `falta_sueno` | `true` | → `false` |
| `enfermedad_reciente` | `true` | → `false` |
| `hora_inicio` | entre 12:00-17:00 | → 08:00 o 19:00 |
| `duracion_actividad` | >1h | → 1h |

## Cap de factores

El producto de factores individuales se capa a 3.0 (`CAP_FACTORES_DEFECTO`). Cuando hay múltiples factores activos, el cap oculta el efecto marginal de cada cambio individual. Por eso cada contrafactual devuelve dos probabilidades:

- **`probabilidad_sin_cap`** — impacto marginal real (cap=10.0 para evitar saturación)
- **`probabilidad_con_cap`** — impacto realista (cap=3.0, igual que producción)

El campo `enmascarado_por_cap` es `true` cuando el cap de 3.0 impide ver la diferencia.

## Implementación

- `climasafeai/models/explicabilidad.py`: `generar_contrafactuales()` — función principal. Re-ejecuta `personalizar_riesgo` para cada cambio con dos caps distintos.
- `chat/app.py:598`: endpoint `POST /api/contrafactuales`
- `chat/static/index.html`: tarjeta "Como reducir tu riesgo" se auto-muestra tras `/api/predict`
