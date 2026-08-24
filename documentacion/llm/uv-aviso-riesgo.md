# UV en el aviso de riesgo — Análisis UV-001

## Estado actual

El UV ya está integrado en el pipeline pero con limitaciones:

| Capa | Cómo usa UV | Limitación |
|------|-------------|------------|
| Personalización | Factor UV+fototipo (si UV>3) | Solo si UV disponible |
| Overrides | UV>3 + vulnerable → PRECAUCIÓN | Solo si UV disponible |
| Web demo | **Siempre `uv_index: null`** | Sin API key de OpenUV |
| Bot Telegram | UV del forecast (si OpenUV responde) | 50 req/día, sin histórico |

### Fuente actual: OpenUV

- Endpoint: `api.openuv.io/api/v1/uv`
- Requiere API key (plan gratuito: 50 req/día)
- **Sin histórico** — solo UV en tiempo real para coordenadas
- Si falla o falta la key, `uv_index=None` y los 3 puntos de integración se desactivan silenciosamente

### Fuente alternativa: Open-Meteo

- Open-Meteo **sí sirve UV** como parámetro en su forecast horario
- Sin API key, sin cuota
- Ya se usa para temperatura, humedad, viento — sería coherente usarlo también para UV
- El código actual no lo usa para UV en producción (solo `generar_dataset_frio.py` lo usa para estimar histórico)

## Alcance propuesto

### Dato de entrada

**Open-Meteo UV Index** (parámetro `uv` en el forecast horario):
- Sin API key
- Sin cuota
- Cobertura global
- Resolución horaria (compatible con el perfil horario del proyecto)

### Rangos peligrosos

| UV Index | Riesgo | Para quién |
|----------|--------|------------|
| 0-2 | Bajo | Todos |
| 3-5 | Moderado | Vulnerables (edad >65, comorbilidades, medicación fotosensible) |
| 6-7 | Alto | Todos, especialmente piel clara |
| 8-10 | Muy alto | Evitar exposición prolongada |
| 11+ | Extremo | Evitar exposición |

### Integración propuesta

1. **weather_fetcher.py**: Añadir UV de Open-Meteo al forecast (junto a t2m, rh, wind)
2. **overrides**: Mantener la lógica actual (UV>3 + vulnerable → PRECAUCIÓN)
3. **personalización**: Mantener factor UV+fototipo
4. **recomendaciones**: Añadir texto específico de UV en el parte

## Decisión

**Entra en v2** como parte de la mejora de recomendaciones. Razones:

1. Open-Meteo ya está integrado — es un cambio quirúrgico
2. La lógica de UV ya existe en personalización y overrides — solo falta alimentarla con datos reales
3. La demo web hoy no tiene UV — con Open-Meteopassaría a tenerlo sin coste
4. El PRD lo declara "línea futura" — es el momento de implementarlo

### Ticket propuesto

**UV-002**: Integrar UV de Open-Meteo en weather_fetcher y demo web
- Criterios: UV del forecast en el perfil horario, factor UV+fototipo activo, demo web muestra UV, make test pasa

---

*Análisis UV-001 · 2026-08-24*
