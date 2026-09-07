# ClimaSafe como plataforma de bienestar laboral: expansión multi-riesgo

Documento marco que describe la evolución de ClimaSafe de "app de calor" a
**plataforma de bienestar laboral** que evalúa seis factores ambientales con
criterios técnicos reconocidos, sobre un mismo motor horario personalizado.

> **Propósito:** fijar la arquitectura conceptual del producto ampliado y
> servir de índice cruzado de los documentos de riesgo.

---

## 1. El problema que resuelve (reformulado)

- **Antes:** "¿tengo calor en el trabajo?" (una variable, un riesgo).
- **Ahora:** "¿es seguro para ESTA persona hacer ESTA actividad en ESTE lugar
  y momento?" combinando temperatura, humedad, viento, radiación UV, calidad
  del aire y frío — con criterios oficiales, no inventados.

El trabajador exterior no se enfrenta a un único peligro: la exposición
laboral a la intemperie es **simultánea** (sol + calor + ozono + humedad) o
**estacional inversa** (frío en invierno). Una plataforma de bienestar
laboral debe cubrir todo el arco.

## 2. Los seis factores y su base técnica

| # | Factor | Indicador reconocido | Base normativa/científica | Datos |
|---|--------|----------------------|---------------------------|-------|
| 1 | Calor | Heat Index / WBGT | ISO 7243, ACGIH, NTP 922 | Temperatura y humedad AEMET |
| 2 | Humedad | HR % (conf. 30-70%), Heat Index | RD 486/97, ISO 7730 | Humedad relativa horaria |
| 3 | Radiación UV | Índice UV (OMS) | OMS, ICNIRP, RD 1076/2021 | UVI AEMET |
| 4 | Calidad del aire | ICA 1-6 + contaminante | RD 102/2011, Orden TEC/351/2019, OMS 2021, Dir. 2024/2881 | MITECO/CAMS |
| 5 | Frío | tWC / IREQ (ISO 11079) | NTP 462/1037, RD 486/97 | Temp. mínima y viento AEMET |
| 6 | Esfuerzo | Ventana de actividad, pausas | ACGIH, NIOSH, INVASSAT | Perfil de usuario |

**Documentos de detalle:** `formulas_deterministas.md` (calor, Wind Chill,
quemadura solar), `indice_uv_laboral.md`, `calidad_aire_laboral.md`,
`frio_extremo_laboral.md`, `humedad_y_hidratacion.md`,
`pausas_y_horas_optimas.md`.

## 3. Pipeline unificado (mismo motor, seis entradas)

```
Entrada meteorológica (por hora/municipio):
  T, HR, viento, UVI, ICA (AEMET + MITECO/CAMS)
        │
        ▼
Indicadores reconocidos por factor:
  Heat Index · Wind Chill/tWC · UVI sin nubes · ICA 1-6
        │
        ▼
Personalización (perfil):
  edad, sexo, grasa, comorbilidades, medicación,
  nivel_actividad, aclimatado, fototipo, vestimenta
        │
        ▼
Riesgo por hora (0-100) por factor y compuesto
        │
        ▼
Medidas preventivas por factor:
  pausas, hidratación, horas óptimas, EPI (crema, mascarilla, ropa),
  relevos, aclimatación, reprogramación
        │
        ▼
Registro de acciones (cumplimiento / evidencia)
```

## 4. Soldadura con el modelo existente

- El motor `predict_risk` ya consume temperatura/humedad por hora y produce
  riesgo por ventana de actividad (`formulas_deterministas.md`,
  `personalizacion_individual.md`).
- Los **parámetros de perfil ya existentes** alimentan los nuevos factores
  sin inventar nada:
  - `fototipo` (Fitzpatrick) → dosis UV admisible (ya usado para quemadura solar).
  - `comorbilidades` (respiratorias, cardiovasculares) → multiplicadores de
    riesgo por ICA.
  - `medicacion` (diuréticos, fotosensibilizadores) → riesgo por
    deshidratación y UV.
  - `aclimatado` → velocidad de aclimatación y umbrales de pausas.
  - `nivel_actividad`, `duracion_h`, `hora_inicio` → ciclos trabajo-descanso
    y franja óptima.
- La salida recomendada pasa de "riesgo de calor" a **"riesgo ambiental
  compuesto"** con plan de acción por factor y franja horaria verde.

## 5. Justificación económica ampliada

Además de los costes de absentismo y accidentes por calor
(`absentismo_calor_costes.md`, `accidentes_calor_laboral.md`,
`justificacion_economica_climasafe.md`):

| Factor | Evidencia de coste evitable |
|--------|------------------------------|
| UV | 1,5M cánceres de piel/año (OMS); RR 1,60 en trabajadores exteriores; melanomas y carcinomas como enfermedad profesional reconocida |
| Aire | +12% mortalidad cardiovascular en exteriores (Toren 2007); sin umbral de seguridad (OMS 2021); nueva Directiva 2024/2881 con derecho a indemnización |
| Frío | 0,3% accidentes atribuibles en España; +4% riesgo en días extremos (Martinez-Solanas 2018); jornadas máximas reguladas (RD 1561/1995) |
| Pausas/horas | La productividad cae −2-3%/°C sobre 20°C y hasta −50% a 33-34°C; planificar horas y pausas evita tanto la lesión como la pérdida de rendimiento |

**Tesis:** una plataforma que anticipa el riesgo de **seis** factores y
sugiere horas/pausas/EPI protege la salud y mantiene la actividad productiva
— con retornos del orden de los ya calculados por escenario (constructora,
agrícola, hostelería).

## 6. Límites: lo que ClimaSafe NO hace

- **No sustituye** a la evaluación de riesgos del artículo 16 de la LPRL ni a
  los servicios de prevención.
- **No fija** límites legales: usa criterios reconocidos (ACGIH, ISO, INSST,
  OMS) como referencia técnica, no como norma.
- **No recomienda** suspender la actividad salvo que exista aviso oficial
  (RD 486/1997 disp. adic.) o criterio técnico claro; la decisión final es
  de la empresa y sus delegados de prevención.
- **No mide** contaminantes in situ: usa el ICA oficial de la estación más
  cercana / modelo CAMS.
- La ultravioleta y el ozono tienen dinámica por altura y orientación:
  ClimaSafe usa valores oficiales de referencia por municipio.

## 7. Referencias cruzadas

- Calor y normativa: `marco_normativo_estrés_térmico.md`,
  `formulas_deterministas.md`, `coeficientes_literatura.md`.
- Personalización: `personalizacion_individual.md`.
- Costes: `absentismo_calor_costes.md`, `accidentes_calor_laboral.md`,
  `justificacion_economica_climasafe.md`.
- Factores nuevos: `indice_uv_laboral.md`, `calidad_aire_laboral.md`,
  `frio_extremo_laboral.md`, `humedad_y_hidratacion.md`,
  `pausas_y_horas_optimas.md`.