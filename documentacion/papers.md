# Papers

La base científica del sistema. El catálogo completo con citas está en el
repositorio, en [`documentacion/papers/`](https://github.com/ANFAIA/ClimaSafe/tree/main/documentacion/papers).

## Índices y sensación térmica

| Documento | Uso en el sistema |
|---|---|
| **Rothfusz Heat Index (1990)** — NWS Technical Attachment SR/SSD 90-23 | Ecuación de referencia del Heat Index |
| **NWS Wind Chill Advisory** | Ecuación de referencia del Wind Chill |

## Estrés térmico y exposición laboral

| Documento | Uso en el sistema |
|---|---|
| **NIOSH 2016-106** — *Criteria for a Recommended Standard: Occupational Exposure to Heat and Hot Environments* | Umbrales de exposición (WBGT), ciclos trabajo/descanso, factor de no aclimatación ×1.6 |
| **INSST NTP-322** — *Valoración del riesgo de estrés térmico* | Estrés térmico y normativa española (WBGT) |
| **OIT 2024** — *Seguridad Climática* | Mortalidad laboral por calor |

## Salud pública y planes de acción

| Documento | Uso en el sistema |
|---|---|
| **WHO Heat Health Action Plans** | Recomendaciones a ciudades y sistemas de salud |
| **Ministerio de Sanidad — Plan Calor** | Episodios extremos y mortalidad en España |

## Mortalidad y target

| Documento | Uso en el sistema |
|---|---|
| **MoMo / ISCIII** — Monitorización de la Mortalidad Diaria | **Fuente del target**: muertes atribuibles a calor (X30) y frío (X31), por provincia y día |

## Factores individuales (personalización)

Los coeficientes de los factores individuales salen de estudios epidemiológicos
publicados; la tabla completa con RR/OR, intervalos y calidad por factor vive en
el repositorio:

- [`documentacion/riesgo/personalizacion_individual.md`](https://github.com/ANFAIA/ClimaSafe/blob/main/documentacion/riesgo/personalizacion_individual.md)
- `data/factores_riesgo.json` (coef, DOI, calidad, implementado)

Fuentes destacadas: Semenza 1996 (NEJM, ola de calor de Chicago), Sci. Reports
2025 (antipsicóticos y calor extremo), eBioMedicine 2016 (Bunker, mortalidad por
calor/frío), PLOS One 2020 (Medicare, diuréticos), Flouris *Lancet Public
Health* 2018 (fatiga térmica).
