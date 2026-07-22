# Coeficientes extraídos de la literatura

Compilación de todos los datos numéricos (RR, OR, HR, %, thresholds) extraídos
de los papers en `papers/`. Organizado por factor de riesgo. Cada coeficiente
incluye su cita al paper original.

> **Propósito:** servir como fuente de verdad para la tabla de factores de
> personalización (`personalizacion_individual.md`) y para calibrar o
> contrastar las predicciones del modelo ML.

---

## 1. Temperatura ambiente y mortalidad general

| Coeficiente | Valor | IC 95% | Población | Fuente |
|-------------|-------|--------|-----------|--------|
| Mortalidad atribuible a temperatura no óptima | 7.71% | (7.43–7.91) | 13 países, 74M muertes | Gasparrini 2015 (`mortality-risk-attributable...`) |
| Atribuible a frío | 7.29% | (7.02–7.49) | ídem | ídem |
| Atribuible a calor | 0.42% | (0.39–0.44) | ídem | ídem |
| Atribuible a temperatura extrema | 0.86% | (0.84–0.87) | ídem | ídem |
| MMT (temperatura mínima mortalidad) | ~P60 (tropical) / ~P80-90 (templado) | — | 13 países | ídem |
| Mortalidad CV por 1°C (calor) | +3.44% | (3.10–3.78) | ≥65 años | Bunker 2016 (`effects-of-air-temperature...`) |
| Mortalidad respiratoria por 1°C (calor) | +3.60% | (3.18–4.02) | ≥65 años | ídem |
| Mortalidad cerebrovascular por 1°C (calor) | +1.40% | (0.06–2.75) | ≥65 años | ídem |
| Mortalidad CV por 1°C (frío) | +1.66% | (1.19–2.14) | ≥65 años | ídem |
| Mortalidad respiratoria por 1°C (frío) | +2.90% | (1.84–3.97) | ≥65 años | ídem |
| Neumonía por frío | +6.89% | (0.20–12.99) | ≥65 años | ídem |
| Morbilidad respiratoria por frío | +4.93% | (1.54–8.44) | ≥65 años | ídem |
| Mortalidad por 1°C sobre P95 (heat threshold) | RR 1.03 | (1.02–1.03) | Inglaterra y Gales | Hajat 2006 (`heat-related-and-cold-related-deaths...`) |
| Mortalidad por 1°C bajo P5 (cold threshold) | RR 1.06 | (1.05–1.06) | Inglaterra y Gales | ídem |
| Incremento mortalidad por 1°C (global, ~50% locations) | +1–3% | — | múltiples | Hajat 2009 (`heat-related-mortality-a-review...`) |

## 2. Frío — mortalidad y morbilidad CV

| Coeficiente | Valor | IC 95% | Fuente |
|-------------|-------|--------|--------|
| Mortalidad CV por 1°C menos | RR 1.016 | (1.015–1.018) | Fan 2023 (`a-systematic-review-and-meta-analysis-of-cold-exposure...`) |
| Morbilidad CV por 1°C menos | RR 1.012 | (1.010–1.014) | ídem |
| Mortalidad coronaria por 1°C menos | RR 1.015 | (1.011–1.019) | ídem |
| Aneurisma aórtico por 1°C menos | RR 1.026 | (1.021–1.031) | ídem |
| Mortalidad por ola de frío | RR 1.324 | (1.234–1.421) | ídem |
| Morbilidad por ola de frío | RR 1.138 | (1.015–1.276) | ídem |
| Exceso mortalidad invernal — Portugal | +28% | (25–31%) | Healy 2003 (`excess-winter-mortality-in-europe...`) |
| Exceso mortalidad invernal — España | +21% | (19–23%) | ídem |
| Exceso mortalidad invernal — Irlanda | +21% | (18–24%) | ídem |

## 3. Medicación y calor

| Coeficiente | Valor | IC 95% | Contexto | Fuente |
|-------------|-------|--------|----------|--------|
| Diuréticos de asa (global, fuera de ola) | RR 1.37 | (1.14–1.66) | ≥65 años, Medicare | Layton 2020 (`layton-2020-medications-heat`) |
| Antipsicóticos (global, fuera de ola) | RR 1.37 | (1.14–1.66) | ≥65 años | ídem |
| Anticolinérgicos (global, fuera de ola) | RR 1.16 | (1.00–1.35) | ≥65 años | ídem |
| Diuréticos + demencia | RR 1.54 | (1.06–2.24) | ≥65 años | ídem |
| Diuréticos + insuficiencia cardíaca | RR 1.64 | (1.20–2.23) | ≥65 años | ídem |
| Diuréticos + infarto miocardio | RR 1.84 | (1.13–3.00) | ≥65 años | ídem |
| Enf. mental grave + antipsicóticos (calor) | IRR 1.48 | (1.40–1.56) | Población UK | Wong 2024 (`wong-2024-psychotropics-heat`) |
| Enf. mental grave + antipsicóticos (IM en calor) | IRR 1.21 | (1.06–1.39) | ídem | ídem |
| Enf. mental grave sin antipsicóticos (calor) | IRR 1.45 | (1.35–1.56) | ídem | ídem |
| Depresión + antidepresivos (calor) | IRR 1.54 | (1.46–1.64) | ídem | ídem |
| Depresión sin antidepresivos (calor) | IRR 1.64 | (1.57–1.71) | ídem | ídem |

> **Nota:** Wong et al. muestra que el riesgo lo lleva sobre todo la condición
> de salud mental, no el fármaco. Las IRR con y sin medicación son muy similares.

## 4. Exposición ocupacional al calor

| Coeficiente | Valor | IC 95% | Contexto | Fuente |
|-------------|-------|--------|----------|--------|
| Lesiones laborales (hombres, calor) | OR 2.12 | (1.87–2.42) | 58,495 trabajadores Tailandia | Tawatsupa 2013 (`association-between-heat-stress-and-occupational-injury...`) |
| Lesiones laborales (mujeres, calor) | OR 1.89 | (1.64–2.18) | ídem | ídem |
| Heat strain (estrés térmico vs termoneutral) | OR 4.01 | (2.45–6.58) | 111 estudios, 30 países | Flouris 2018 (`workers-health-and-productivity...`) |
| Umbral WBGT para estrés térmico | >22.0 / >24.8°C | — | según intensidad | ídem |
| Trabajo pesado aclimatado (REL WBGT) | ~25°C WBGT | — | NIOSH | NIOSH 2016 (`niosh-2016-106-heat`) |
| Trabajo moderado aclimatado (REL WBGT) | ~27.8°C WBGT | — | NIOSH | ídem |
| Factor no aclimatado (RAL/REL) | ×1.6 | — | NIOSH | ídem |
| Screening HI para riesgo laboral | ≥29.4°C (85°F) | — | OSHA | Tustin 2018 (`evaluation-of-occupational-exposure-limits...`) |
| Reducción capacidad laboral por WBGT | 26–30°C | — | países cálidos | Kjellstrom 2009 (`workplace-heat-stress-health-and-productivity...`) |
| Mujeres expuestas sobre umbral WBGT | 71% | — | India (Avg WBGT 30±2.3°C) | Venugopal 2016 (`heat-stress-and-inadequate-sanitary-facilities...`) |

## 5. Planes de acción y sistemas de alerta

| Dato | Valor | Contexto | Fuente |
|------|-------|----------|--------|
| Muertes por calor en Europa (verano 2022) | 61,672 (37,643–86,807) | 35 países, 823 regiones | Ballester 2023 (`heat-related-mortality-in-europe-during-the-summer-of-2022`) |
| Italia (2022) | 18,010 muertes (295/millón) | ídem | ídem |
| España (2022) | 11,324 muertes (237/millón) | ídem | ídem |
| Grecia (2022) | 280/millón | ídem | ídem |
| Mujeres vs hombres (2022) | +56% muertes en mujeres | ídem | ídem |
| Hombres 0-64 (2022) | +41% vs mujeres misma edad | ídem | ídem |
| Vidas salvadas por aviso (Filadelfia) | ~2.6/aviso, 117 en 3 años | 1995–1998 | Ebi 2004 (`heat-watch-warning-systems-save-lives...`) |
| Muertes atribuibles calor España (MoMo 2025) | 3,832 (+87.6% vs 2024) | >50% ≥85 años | Plan Nacional Calor-Sanidad (`plan-nacional-calor-sanidad`) |
| Países europeos con HEWS | 12 (2011) | — | (`heatwave-early-warning-systems-and-adaptation-advice...`) |

## 6. Definiciones de ola de calor

| Definición | Criterio | Fuente |
|------------|----------|--------|
| 3+ días consecutivos sobre P90 Tmax | EHF (Excess Heat Factor) | Perkins 2012 (`on-the-measurement-of-heat-waves`) |
| 3+ días consecutivos sobre P90 Tmin | ídem | ídem |
| Umbral de Heat Index ≥ 37.8°C (100°F) | NYC estudio vulnerabilidad | Rosenthal 2014 (`intra-urban-vulnerability-to-heat-related-mortality...`) |
| Percentil 95 de temperatura regional (heat threshold) | Hajat 2006 | Hajat 2006 |
| Percentil 5 de temperatura regional (cold threshold) | ídem | ídem |

## 7. Índices biometeorológicos — fórmulas

### Heat Index (Rothfusz 1990)
```
HI = -42.379 + 2.04901523·T + 10.14333127·RH - 0.22475541·T·RH
     - 0.00683783·T² - 0.05481717·RH² + 0.00122874·T²·RH
     + 0.00085282·T·RH² - 0.00000199·T²·RH²
```
Ajustes: RH <13% (T 27–44°C) y RH >85% (T 27–31°C).
Fórmula simplificada (T < ~27°C):
```
HI = 0.5 · {T + 61.0 + [(T-68.0)·1.2] + (RH·0.094)}
```

### Wind Chill (NWS 2001)
```
WC(°C) = 13.12 + 0.6215·T - 11.37·V^0.16 + 0.3965·T·V^0.16
```
Válido: T ≤ 10°C, V > 4.8 km/h.
Congelación: a WC -28°C, piel expuesta se congela en ~30 min.

### WBGT (INSST NTP 322)
- Exterior con sol: WBGT = 0.7·THN + 0.2·TG + 0.1·TA
- Interior/sin sol: WBGT = 0.7·THN + 0.3·TG
- THN = temperatura húmeda natural, TG = temperatura de globo, TA = temperatura ambiente

### UTCI
- Evaluado frente a índices tradicionales → superior en rango completo de condiciones
- RR mortalidad al percentil 1 de UTCI: 1.19–1.22 (vs 1.29–1.30 para temperatura sola)
- El viento en UTCI explica diferencias frente a modelos solo-temperatura

## 8. Vulnerabilidad por grupos

| Factor | Efecto | Fuente |
|--------|--------|--------|
| Edad ≥65 años | Mayor riesgo (RR 1.03/°C sobre P95) | Hajat 2006 |
| Edad ≥80 años | RR más alto en Lazio, Italia | Ceccarelli 2025 |
| Mujeres | +56% muertes calor Europa 2022 | Ballester 2023 |
| Residencias / nursing homes | Mayor vulnerabilidad | Hajat 2006 |
| Pobreza / falta AC | Mayor MRR en NYC | Rosenthal 2014 |
| Población densidad | Incrementa pendiente calor-mortalidad | Hajat 2009 |
| Bajo PIB | Incrementa pendiente calor-mortalidad | Hajat 2009 |
| Norte EE.UU. (vs sur) | Mayor efecto calor (menos AC) | Curriero 2002 |
| Sur EE.UU. (vs norte) | Mayor efecto frío (menos calefacción) | Curriero 2002 |

## 9. Incertidumbre y límites

- Alta heterogeneidad entre estudios (I²~83.5% en meta-análisis CV)
- MMT varía por clima: ~P60 tropicales, ~P80-90 templados (Gasparrini 2015)
- Multiplicar factores asume independencia — los mecanismos se solapan
- Límite recomendado para producto acumulado de factores: 3.0×
- Los thresholds de calor se sitúan más altos en comunidades más cercanas al ecuador (aclimatación poblacional)
- Mayoría de estudios en países de ingresos altos — transferibilidad limitada

---

*Compilado automáticamente desde papers en `papers/` · Última actualización: 2026-07-17*
