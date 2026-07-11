# Coeficientes de personalización individual del riesgo

**Fecha:** 2026-07-10 · **Estado:** propuesta para revisión (no implementado)

## Qué es esto y cómo encaja

La LSTM (y los modelos ML) dan un **índice de peligrosidad poblacional 0-1**
por provincia/día (`indice_riesgo_softmax`, ver `diseño_modelo.md` §6). Ese
índice no distingue individuos: dos personas en la misma provincia el mismo
día reciben el mismo número. En producción se quiere **modular ese índice
según el perfil de la persona** — el ejemplo del que partió esto: "índice 0.8,
pero con mucha grasa corporal retienes más calor → tu riesgo individual sube,
p. ej. ×1.1".

Este documento propone la tabla de factores multiplicativos, con su fuente en
literatura epidemiológica. **No sale del entrenamiento** (MoMo no tiene
atributos individuales): son coeficientes documentados, exactamente como el
×1.6 de "sin aclimatación" (NIOSH RAL/REL) que el proyecto ya usa en
`formulas_riesgo_deterministico.md` §1.4.

## Cómo se compone (la matemática importa)

El índice es una probabilidad 0-1. Multiplicarlo directo (`0.95 × 1.2 = 1.14`)
se sale de escala. Se compone **en odds**, que es lo estándar en epidemiología
(los riesgos relativos publicados son razones de odds/hazard):

```
odds      = p / (1 - p)
odds_ind  = odds × f1 × f2 × ... × fn      # factores del perfil
p_individual = odds_ind / (1 + odds_ind)
```

Así el factor empuja más donde hay margen y nunca se sale de [0, 1]:
`p=0.50, f=1.2 → 0.545`; `p=0.95, f=1.2 → 0.958`.

**Caveat honesto (documentar siempre):** multiplicar factores asume que son
**independientes**, y no lo son — una persona mayor, obesa y con cardiopatía
no es exactamente `1.5 × 1.2 × 1.4`, porque los mecanismos se solapan. Para un
sistema de **aviso preventivo** esto es aceptable (peca de precavido, el lado
correcto), pero conviene **capar el producto total de factores** (p. ej. a 3.0)
para no generar riesgos individuales absurdos por acumulación.

---

## Campos del perfil individual

La función `personalizar_riesgo` recibe un `perfil` con estos campos (todos
opcionales — un campo ausente = factor neutro ×1.0, no penaliza por falta de
dato):

| Campo | Tipo | Valores | Confirmado por el usuario |
|---|---|---|---|
| `edad` | int | años | ✓ |
| `sexo` | str | "hombre" \| "mujer" | ✓ |
| `imc` | float | kg/m² | ✓ |
| `nivel_actividad` | str | "reposo" \| "ligera" \| "moderada" \| "intensa" \| "muy_intensa" | ✓ |
| `duracion_actividad_h` | float | horas de exposición activa | ✓ |
| `aclimatado` | bool | — | añadido (ya en el proyecto) |
| `comorbilidades` | set | {"cardiovascular", "diabetes", "respiratoria", "mental"} | añadido |
| `farmacos` | set | {"antipsicoticos", "diureticos_asa"} | añadido |
| `situacion_social` | set | {"vive_solo", "encamado", "no_sale", "vivienda_fria"} | añadido (capa aparte, ver §Vulnerabilidad social) |

## Tabla de coeficientes propuesta — CALOR

Factores multiplicativos sobre las odds del índice. Redondeados desde los
riesgos relativos / odds ratios publicados (columna RR/OR con su intervalo).

| Factor | Condición | Coef. propuesto | RR/OR publicado | Fuente | Confianza |
|---|---|---|---|---|---|
| **Edad** | 65–74 | ×1.2 | RR 1.25 (1.20–1.30) | Meta-análisis vulnerabilidad calor (Bunker 2016; Benmarhnia 2015) | Alta |
| | 75–84 | ×1.5 | OR crece con edad; >75 notablemente mayor | íd. | Alta |
| | ≥85 | ×2.0 | "riesgo aumenta sustancialmente >85" | íd. | Media |
| **Sexo** | mujer | ×1.1 | OR 1.45 mujeres vs 1.34 global | Meta-análisis (Benmarhnia 2015) | Media |
| **Nivel de actividad** | ligera | ×1.1 | calor metabólico ∝ MET; umbral WBGT (RAL) baja al subir la carga | NIOSH [2016-106] §carga metabólica | Alta (base fisiológica) |
| | moderada | ×1.3 | íd. | íd. | Alta |
| | intensa | ×1.6 | íd. | íd. | Alta |
| | muy intensa | ×2.0 | íd.; donde la obesidad dispara el OR 3.5–4.3 | NIOSH; Military Medicine 1996 | Alta |
| **Duración actividad** | 1–2 h / 2–4 h / >4 h | ×1.1 / ×1.25 / ×1.4 | ciclos trabajo/descanso NIOSH: sin recuperación el riesgo se acumula | NIOSH [2016-106] (modificador de exposición, no RR de mortalidad) | Media |
| **Aclimatación** | no aclimatado | ×1.6 | RAL vs REL NIOSH | NIOSH [2016-106] (ya en el proyecto) | Alta |
| **Enf. cardiovascular** | cardiopatía/HTA previa | ×1.4 | exceso CVD por calor ~1.15; OR cardiopatía elevado (Chicago) | Semenza 1996 (NEJM); Circ. Res. 2024 | Alta |
| **Diabetes** | diabetes mellitus | ×1.2 | más ingresos/muertes por calor | Rev. sistemática LMIC 2025 | Media |
| **Enf. mental / psicofármacos** | esquizofrenia, antipsicóticos | ×1.8 | OR 2.43 (1.52–4.01) dispensación antipsicótico; psiquiátricos 2× | Sci. Reports 2025; Psychiatric Services 1998 | Alta |
| **Diuréticos** | de asa (deshidratación) | ×1.3 | RR 1.54 (dementia + loop) | PLOS One 2020 (Medicare) | Media |
| **Aislamiento / dependencia** | vive solo / no sale de casa / encamado | ×2.0 | OR 2.3 vive solo; 5.5 encamado; 6.7 no sale | Semenza 1996 (Chicago, NEJM) | Alta (pero situacional) |
| **Obesidad** | IMC ≥30 | ×1.2 † | ver nota  abajo | mixta | **Baja** |
| **Fragilidad** | IMC <18.5 | ×1.3 | IMC bajo en ancianos = fragilidad/desnutrición | Semenza 1996 (Chicago) | Media |

† El ×1.2 de obesidad **solo se aplica si `nivel_actividad` ≥ moderada** (ver nota).

### Nivel y duración de la actividad — base fisiológica

El calor metabólico que genera el cuerpo escala con la tasa metabólica (MET):
reposo ~1 MET, trabajo ligero ~2, moderado ~3–4, pesado ~5–6, muy pesado ≥7.
NIOSH define sus umbrales de exposición (WBGT) **bajando** conforme sube la
carga de trabajo — es decir, a más actividad, menos calor ambiental hace falta
para el mismo riesgo. Por eso el nivel de actividad es un multiplicador con
base fisiológica sólida (a diferencia de un RR de mortalidad poblacional).

La **duración** entra por los ciclos trabajo/descanso de NIOSH: la exposición
continua sin recuperación acumula estrés térmico. Es un modificador de
exposición pragmático, no un riesgo relativo publicado — marcado como tal.

**Conexión clave con la obesidad:** el OR alto de la obesidad (3.5–4.3)
aparece precisamente **en esfuerzo** (estudios militares/deportivos), no en
reposo. Por eso el factor de obesidad se activa solo con actividad ≥ moderada:
es donde la fisiología del aislamiento graso realmente se traduce en riesgo.

###  Nota sobre la obesidad (el ejemplo motivador)

La intuición "más grasa → retiene más calor → más riesgo" es fisiológicamente
real (la grasa aísla y reduce la disipación de calor core→piel), pero la
evidencia de mortalidad **está dividida según el contexto**:

- **En esfuerzo físico** (militares, deportistas): asociación fuerte — OR de
  golpe de calor **3.5–4.3** para obesos (Military Medicine 1996; estudios de
  golpe de calor por esfuerzo). Aquí la grasa importa mucho.
- **En reposo / población general** (que es tu caso — mortalidad poblacional
  MoMo): el estudio de referencia (Chicago 1995, Semenza NEJM) **NO encontró**
  mayor mortalidad para IMC alto; de hecho el IMC **bajo** salió más peligroso
  (fragilidad/desnutrición en ancianos).

Por eso propongo **×1.2 modesto para IMC≥30 solo si el perfil indica actividad
física / exposición al aire libre**, y añadir un factor de **fragilidad
(IMC<18.5 → ×1.3)** que la literatura sí respalda en reposo. Multiplicar por
la grasa "a lo bruto" para todo el mundo sobreestimaría el riesgo de la mayoría.

---

## Tabla de coeficientes propuesta — FRÍO

La evidencia individual del frío es **más escasa** que la del calor (menos
estudios caso-control con RR por factor), y los mecanismos se solapan más
(casi todo pasa por el sistema cardiovascular). Coeficientes más conservadores
y marcados como provisionales.

| Factor | Condición | Coef. propuesto | Base | Confianza |
|---|---|---|---|---|
| **Edad** | 65–74 / 75–84 / ≥85 | ×1.2 / ×1.4 / ×1.7 | termorregulación deteriorada con la edad | Media |
| **Sexo** | mujer | ×1.05 | señal menor que en calor | Baja |
| **Nivel de actividad** | ligera / moderada | ×0.95 / ×0.9 | la actividad genera calor → protectora frente al frío moderado | Media (fisiológica) |
| | intensa con sudor + viento | ×1.2 | sudor + ropa húmeda + viento acelera la pérdida (hipotermia) | Media |
| **Enf. cardiovascular** | cardiopatía/HTA | ×1.5 | el frío sube TA, colesterol, fibrinógeno; CVD = hasta 70% del exceso invernal | Alta |
| **Enf. respiratoria** | EPOC, asma | ×1.4 | 2ª causa del exceso invernal; lag largo tras ola de frío | Media |
| **Aislamiento / vivienda fría** | vive solo / pobreza energética | ×1.5 | exceso invernal mayor en climas templados con vivienda mal aislada | Media (situacional) |
| **Obesidad** | IMC ≥30 | ×1.0 (neutro) | la grasa aísla → protectora frente al frío; sin señal de mayor mortalidad | — |
| **Fragilidad** | IMC <18.5 | ×1.3 | poca masa/aislamiento → menor tolerancia al frío | Media |

Notas:
- La obesidad que en calor es (débil) factor de riesgo, en frío es **neutra o
  levemente protectora** — el mismo aislamiento graso que perjudica en calor
  ayuda en frío. Buen ejemplo de por qué los factores son **específicos de
  calor/frío**, no globales.
- El **nivel de actividad se invierte**: en frío moderado la actividad genera
  calor y protege (coef. <1), pero la actividad **intensa con sudoración y
  viento** empapa la ropa y acelera la pérdida de calor (coef. >1). Por eso el
  factor de frío depende también de si hay viento/humedad, no solo del MET.

---

## Fuentes

- Semenza JC et al. *Heat-Related Deaths during the July 1995 Heat Wave in Chicago.* NEJM 1996. (OR de aislamiento, dependencia, cardiopatía; ausencia de señal de IMC alto)
- Benmarhnia T et al. *Vulnerability to Heat-related Mortality: A Systematic Review, Meta-analysis and Meta-regression.* Epidemiology 2015. (RR edad/sexo)
- Bunker A et al. *Effects of Air Temperature on Climate-Sensitive Mortality and Morbidity in the Elderly.* eBioMedicine 2016.
- *Heat and Cardiovascular Mortality: An Epidemiological Perspective.* Circulation Research 2024. (exceso CVD ~1.15)
- *Antipsychotics and mortality among people with schizophrenia during an extreme heat event.* Scientific Reports 2025. (OR 2.43)
- *Heatwaves, medications, and heat-related hospitalization in older Medicare beneficiaries.* PLOS One 2020. (diuréticos RR 1.54)
- Chung HY et al. *Obesity and the Occurrence of Heat Disorders.* Military Medicine 1996. (OR 3.5–4.3 en esfuerzo)
- NIOSH [2016-106] *Occupational Exposure to Heat and Hot Environments.* (RAL/REL, ×1.6)
- *Preventing cold-related morbidity and mortality in a changing climate.* (factores de frío, CVD/respiratorio)

## Implementación

Función: `climasafeai/features/personalizacion.py` → `personalizar_riesgo(indice, perfil, tipo="calor"|"frio")`.
Tests: `tests/test_personalizacion.py`. Decisiones ya tomadas:

- **Cap del producto de factores = 3.0** (los factores no son independientes).
- **Composición en odds** (no multiplicación directa) para no salirse de [0,1].
- **Tablas separadas calor/frío** (`_factores_calor` / `_factores_frio`): la
  obesidad y la actividad se comportan al revés.
- **Obesidad solo en esfuerzo** (actividad ≥ moderada) — donde la literatura la
  respalda.
- **Salud mental y antipsicóticos = un único factor** (no doble conteo; el
  riesgo lo marca la condición, no el fármaco — ver `papers/wong-2024-...`).
- **Factores sociales por MÁXIMO, no producto** (los OR de Chicago se solapan).
- La función devuelve un **desglose por factor** (nombre/categoría/valor):
  un sistema de salud necesita explicar el porqué, no dar un número opaco.
  Las categorías (`fisiologico` / `medico` / `situacional`) permiten a quien
  consuma separar la vulnerabilidad social si lo prefiere.

## Pendiente (mejoras futuras)

1. Contrastar los coeficientes con literatura **española** si existe (Plan
   Calor del Ministerio de Sanidad, MoMo desagregado) para ajustar al contexto.
2. Afinar el factor de **duración+viento en frío** (hoy la actividad intensa en
   frío es un ×1.2 fijo; idealmente dependería de viento/humedad reales del día).
3. Descargar el resto de fuentes de dominio público (NIOSH 2016-106, CDC MMWR)
   a markdown cuando el acceso al PDF lo permita — hoy solo citadas (ver `papers/README.md`).
