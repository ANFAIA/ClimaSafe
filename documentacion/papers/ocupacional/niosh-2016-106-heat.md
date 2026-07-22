# NIOSH — Occupational Exposure to Heat and Hot Environments [2016-106]

> **Dominio público** (obra del gobierno de EE. UU., DHHS/NIOSH — no sujeta a
> copyright). Fuente: NIOSH. *Criteria for a Recommended Standard: Occupational
> Exposure to Heat and Hot Environments.* DHHS (NIOSH) Publication No. 2016-106,
> 2016. Revised Criteria. https://www.cdc.gov/niosh/docs/2016-106/
>
> Nota: el PDF completo no fue accesible vía descarga automática; este es un
> extracto de los datos clave usados en el proyecto (hechos no protegibles),
> con la cita para consulta directa. Las tablas completas por categoría de
> carga y ciclos trabajo/descanso están en el documento original, págs. 76-77.

## Límites de exposición (WBGT)

NIOSH no usa el Heat Index para exposición laboral, sino el **WBGT** (Wet Bulb
Globe Temperature), que incorpora radiación solar y viento. Define dos umbrales
según la carga de trabajo:

- **REL** (Recommended Exposure Limit) — trabajadores **aclimatados**.
- **RAL** (Recommended Alert Limit) — trabajadores **no aclimatados**, más
  restrictivo. La diferencia RAL/REL sustenta el coeficiente **×1.6 de "sin
  aclimatación"** que el proyecto ya aplica.

Valores de referencia parciales (trabajo continuo):
- Trabajo pesado, aclimatado → REL ≈ 25 °C WBGT
- Trabajo moderado, aclimatado → REL ≈ 27.8 °C WBGT

## Categorías de carga metabólica (MET)

NIOSH gradúa la exposición por carga de trabajo: **ligera / moderada / pesada /
muy pesada** — equivalente a las categorías MET del proyecto. El umbral WBGT
**baja** al subir la carga: a más actividad, menos calor ambiental hace falta
para el mismo riesgo. Base fisiológica del factor `nivel_actividad` de
`personalizacion_individual.md`.

## Uso en el proyecto

- Coeficiente ×1.6 (aclimatación) y factores de nivel/duración de actividad de
  la tabla de personalización.
- Aproximación WBGT desde Heat Index (Bernard & Iheanacho 2015) en
  `formulas_deterministas.md` §1.4.
