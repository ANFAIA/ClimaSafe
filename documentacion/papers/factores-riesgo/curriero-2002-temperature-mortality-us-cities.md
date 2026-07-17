# Temperature and Mortality in 11 Cities of the Eastern United States

> Curriero FC, Heiner KS, Samet JM, Zeger SL, Strug L, Patz JA.
> *American Journal of Epidemiology.* 2002;155(1):80–87.
> DOI: 10.1093/aje/155.1.80.

**Autores:** Frank C. Curriero, Karlyn S. Heiner, Jonathan M. Samet,
Scott L. Zeger, Lisa Strug, Jonathan A. Patz
**Revista / año:** American Journal of Epidemiology, 2002
**DOI:** 10.1093/aje/155.1.80

## Resumen

Estudio clásico que analiza la relación entre temperatura y mortalidad
en 11 ciudades del este de EE.UU. usando modelos de series temporales.
Encuentra una relación J-shaped (no lineal) consistente entre ciudades,
e identifica diferencias regionales clave en la pendiente.

## Hallazgos clave

| Hallazgo | Valor | Nota |
|----------|-------|------|
| Forma funcional | J-shaped | Mortalidad mínima en temperatura moderada, sube en extremos |
| Norte EE.UU. vs sur | Mayor efecto calor en norte | Menos acceso a AC en norte (infraestructura) |
| Sur EE.UU. vs norte | Mayor efecto frío en sur | Menos calefacción / vivienda peor aislada |
| Desplazamiento temporal | Efecto calor: días 0–3 | El efecto del calor es agudo (lag corto) |
| Desplazamiento temporal | Efecto frío: días 3–10 | El efecto del frío es más retardado |

## Relevancia para ClimaSafeAI

Confirma que la relación temperatura-mortalidad es no lineal (J-shaped),
lo que justifica el uso de modelos flexibles como Random Forest / LSTM
en lugar de modelos lineales. También apoya la necesidad de adaptar
umbrales de alerta por región (norte/sur tienen distinta tolerancia).

## Datos numéricos extraídos

- Relación J-shaped temperatura-mortalidad (confirmación gráfica)
- Lag calor: 0–3 días
- Lag frío: 3–10 días

---

_Clasificado automáticamente como **factor-riesgo**._
