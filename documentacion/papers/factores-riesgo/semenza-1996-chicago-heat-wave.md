# Heat-Related Deaths during the July 1995 Heat Wave in Chicago

> Semenza JC, Rubin CH, Falter KH, Selanikio JD, Flanders WD, Howe HL,
> Wilhelm JL. *New England Journal of Medicine.* 1996;335(2):84–90.
> DOI: 10.1056/NEJM199607113350203.

**Autores:** Jan C. Semenza, Carol H. Rubin, Kenneth H. Falter, et al.
**Revista / año:** New England Journal of Medicine, 1996
**DOI:** 10.1056/NEJM199607113350203

## Resumen

Estudio de casos y controles de las muertes relacionadas con el calor
durante la ola de calor de Chicago en julio de 1995 (739 muertes excesivas
en una semana). Se entrevistó a familiares de 339 casos y 339 controles
vecinos para identificar factores de riesgo individuales. Es el estudio de
referencia para factores de vulnerabilidad social en calor.

## Odds Ratios de factores individuales

| Factor | OR | IC 95% | Nota |
|--------|----|--------|------|
| Encamado / no sale de casa | 5.5 | (2.5–12.0) | Factor situacional más potente |
| No salir durante la ola de calor | 6.7 | (3.1–14.5) | Similar al anterior |
| Vive solo | 2.3 | (1.2–4.2) | Aislamiento social |
| Sin aire acondicionado en casa | 2.5 | (1.0–6.6) | En coche o vecino = protector |
| No visitar centros con AC | 2.6 | (1.4–4.8) | Refugio climático |
| Enfermedad cardiovascular preexistente | 2.0 | (1.1–3.6) | Comorbilidad clave |
| Enfermedad pulmonar preexistente | 1.8 | (0.9–3.4) | No significativo estadísticamente |
| Diabetes | 1.5 | (0.9–2.5) | Elevado pero IC cruza 1 |
| Alcohol | 3.7 | (1.5–9.0) | Consumo reciente |
| IMC alto (obesidad) | — | — | **Sin asociación significativa** (hallazgo clave) |
| IMC bajo (<18.5) | elevado | — | Fragilidad como factor de riesgo |

## Relevancia para ClimaSafeAI

Fuente primaria de los factores de vulnerabilidad social y aislamiento
usados en `personalizar_riesgo()`. Los OR de encamado (5.5), no salir
(6.7) y vive solo (2.3) son la base de los coeficientes de la categoría
`situacional`. El hallazgo de que la obesidad NO fue factor de riesgo
en reposo fundamenta la decisión de aplicar el factor de obesidad solo
en esfuerzo (ver coeficientes de personalización).

## Datos numéricos extraídos

- OR encamado = 5.5 (IC 95%: 2.5–12.0)
- OR no sale = 6.7 (IC 95%: 3.1–14.5)
- OR vive solo = 2.3 (IC 95%: 1.2–4.2)
- OR sin AC = 2.5 (IC 95%: 1.0–6.6)
- OR cardiopatía = 2.0 (IC 95%: 1.1–3.6)
- OR alcohol = 3.7 (IC 95%: 1.5–9.0)
- Obesidad: sin señal (no significativo)

---

_Clasificado automáticamente como **factor-riesgo**. Fuente primaria vía
PubMed / NEJM._
