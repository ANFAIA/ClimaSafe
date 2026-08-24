# Estudio de Integración con APIs de Wearables — DATA-005

**Fecha:** 2026-08-24
**Estado:** Estudio completado
**Decisión:** NO VIABLE (por ahora) — Feature cerrada como descartada

---

## Resumen Ejecutivo

Este estudio evalúa la viabilidad de integrar datos de wearables (Garmin, Fitbit, Apple Health, Google Fit, Samsung, Withings) en ClimaSafeAI para mejorar la personalización del riesgo climático. Tras analizar las APIs disponibles, se concluye que la integración **no es viable en el estado actual del proyecto** por razones técnicas, de coste y de arquitectura.

---

## 1. Análisis por Plataforma

### 1.1 Garmin Connect API

**Datos disponibles:**
- ✅ Heart Rate (HR) — continuo y durante actividad
- ✅ Resting Heart Rate (RHR) — diario
- ✅ Sleep — duración, fases (light, deep, REM), score, HR durante sueño
- ✅ Activity — pasos, calorías, minutos activos, >30 tipos de actividad
- ✅ HRV (Heart Rate Variability)
- ✅ Body Battery (estrés/energía)
- ✅ Stress score
- ✅ VO2 Max
- ✅ Respiration rate, SpO2

**Proceso de alta:**
- Requiere **programa de empresa** (Developer Program)
- Solicitud online → revisión en 2 días hábiles
- **No es self-serve**: debe describir caso de uso, empresa, prácticas de datos
- Aprobación típica: días a semanas
- **Acceso sandbox** disponible para desarrollo (datos sintéticos)

**Cuota y coste:**
- **Sin coste de licencia** para acceso básico
- **100 requests/minuto** (rate limit)
- Métricas comerciales pueden requerir licencia o mínimo de dispositivos
- Arquitectura **push-based** (webhooks): Garmin envía datos a tu servidor
- Backfill histórico: 30 días máximo

**Problemas para ClimaSafeAI:**
- Requiere servidor HTTPS público (no localhost)
- Arquitectura push no encaja con modelo actual (on-demand)
- Proceso de aprobación empresarial → inviable para proyecto personal/académico

---

### 1.2 Fitbit Web API (→ Google Health API)

**Datos disponibles:**
- ✅ Heart Rate — continuo, zonas, intraday (1s, 1min)
- ✅ Resting Heart Rate — diario
- ✅ Sleep — duración, fases, score, HR durante sueño
- ✅ Activity — pasos, calorías, minutos activos, distancia
- ✅ HRV, SpO2, Respiratory Rate
- ✅ Body temperature, skin temperature

**Proceso de alta:**
- Registro en dev.fitbit.com (gratuito)
- OAuth 2.0 estándar
- **Intraday data** (1s, 1min): requiere solicitud aparte y aprobación caso por caso
- **Migración obligatoria a Google Health API** antes de septiembre 2026

**Cuota y coste:**
- **Gratuito** para uso personal/cliente/servidor
- **150 requests/hora/usuario** (no por aplicación)
- Datos intraday requieren aprobación especial
- Google Health API: sin coste explícito publicado, sujeto a Google Cloud pricing

**Problemas para ClimaSafeAI:**
- Fitbit Web API se apaga en septiembre 2026
- Google Health API: proceso de verificación de scopes restringidos
- Requiere **security assessment** si >100 usuarios o scopes de salud read/write
- Datos intraday (los útiles para riesgo en tiempo real) no están garantizados

---

### 1.3 Apple HealthKit

**Datos disponibles:**
- ✅ Heart Rate — continuo, workout HR
- ✅ Resting Heart Rate
- ✅ Sleep — duración, fases, HR durante sueño
- ✅ Activity — pasos, calorías, distancia, minutos activos
- ✅ HRV, VO2 Max, Respiratory Rate, SpO2
- ✅ Body temperature, ECG

**Proceso de alta:**
- **Exclusivo iOS/watchOS**: requiere app nativa (Swift/SwiftUI)
- Apple Developer Program ($99/año)
- HealthKit capability en Xcode
- Permisos granulares por tipo de dato
- **No hay API web**: solo SDK local en dispositivo

**Cuota y coste:**
- $99/año (Apple Developer Program)
- Sin rate limits documentados (acceso local)
- **Sin acceso remoto**: solo desde app en dispositivo

**Problemas para ClimaSafeAI:**
- **Arquitectura incompatible**: ClimaSafeAI es web/serverless, HealthKit es local
- Requiere app nativa iOS → desarrolloSwift separado
- No hay forma de acceder a datos desde servidor
- Usuario debe tener iPhone + Apple Watch + app instalada

---

### 1.4 Google Fit API (→ Health Connect)

**Datos disponibles:**
- ✅ Heart Rate — continuo, workout HR
- ✅ Resting Heart Rate (limitado)
- ✅ Sleep — duración, fases (con dispositivos compatibles)
- ✅ Activity — pasos, calorías, distancia, minutos activos
- ⚠️ HRV, SpO2 — depende del dispositivo

**Proceso de alta:**
- Google Cloud Console → crear proyecto
- OAuth 2.0 + verificación de scopes restringidos
- **Google Fit API se apaga finales 2026**
- Migración a Health Connect (on-device) o Google Health API

**Cuota y coste:**
- **Gratuito**
- Rate limits: no documentados explícitamente
- Verificación de scopes: proceso manual
- **Security assessment** requerido si >100 usuarios o scopes de salud

**Problemas para ClimaSafeAI:**
- Google Fit API deprecated → finales 2026
- Health Connect: solo on-device (mismo problema que Apple HealthKit)
- Google Health API: proceso de verificación complejo
- Datos de HR/sleep limitados sin dispositivo wearable específico

---

### 1.5 Samsung Health Data SDK

**Datos disponibles:**
- ✅ Heart Rate — continuo, workout HR, aggregate (min/max)
- ✅ Resting Heart Rate (indirecto via aggregate)
- ✅ Sleep — duración, fases, score, HR durante sueño, SpO2, skin temp
- ✅ Activity — pasos, calorías, distancia, minutos activos, exercise
- ✅ Blood oxygen, body composition, blood pressure

**Proceso de alta:**
- **Exclusivo Android**: requiere app nativa (Kotlin/Java)
- Samsung Developer Account (gratuito)
- Samsung Health Data SDK v1.1.0
- Permisos por tipo de dato
- **No hay API web**: solo SDK local en dispositivo

**Cuota y coste:**
- Gratuito
- Sin rate limits documentados (acceso local)
- **Sin acceso remoto**: solo desde app en dispositivo

**Problemas para ClimaSafeAI:**
- **Arquitectura incompatible**: mismo problema que Apple HealthKit
- Requiere app nativa Android → desarrollo separado
- Usuario debe tener Samsung Galaxy Watch + app instalada
- No hay forma de acceder a datos desde servidor

---

### 1.6 Withings API

**Datos disponibles:**
- ✅ Heart Rate — continuo, punctual, sleep HR
- ✅ Resting Heart Rate (via measure types)
- ✅ Sleep — duración, fases, score, HR, HRV, SpO2, respiratory rate
- ✅ Activity — pasos, calorías, distancia, minutos activos
- ✅ Blood pressure, body composition, ECG, VO2 Max

**Proceso de alta:**
- Registro en developer.withings.com (gratuito)
- OAuth 2.0 web flow
- **Sin contrato requerido** para uso básico
- Scopes: `user.info`, `user.metrics`, `user.activity`, `user.sleepevents`

**Cuota y coste:**
- **Public API**: gratuita para individuos y partners
- **120 requests/minuto** (rate limit global por aplicación)
- **Enterprise plan**: límites más altos, SLA, soporte dedicado (coste bajo demanda)
- Webhooks disponibles para actualizaciones en tiempo real

**Ventajas para ClimaSafeAI:**
- **API web estándar**: encaja con arquitectura actual
- **Sin aprobación empresarial**: self-serve
- **Datos completos**: HR, sleep, activity, body composition
- **OAuth 2.0 estándar**: fácil de integrar
- **Webhooks**: actualizaciones push cuando el usuario sincroniza

**Problemas para ClimaSafeAI:**
- Requiere dispositivo Withings específico (Sleep Analyzer, Trackers, etc.)
- Usuarios con otros wearables no podrían usar la integración
- 120 req/min puede ser limitante con muchos usuarios
- Datos dependen de que el usuario sincronice regularmente

---

## 2. Variables del Modelo vs. Datos de Wearables

### 2.1 Variables actuales en el modelo (personalizacion.py)

| Variable | Tipo | Fuente actual |
|----------|------|---------------|
| `edad` | Escalar | Declarada por usuario |
| `sexo` | Categórica | Declarada por usuario |
| `porcentaje_grasa` | Escalar | Declarada por usuario |
| `nivel_actividad` | Categórica (reposo/ligera/moderada/intensa/muy_intensa) | Declarada por usuario |
| `aclimatado` | Bool | Declarada por usuario |
| `falta_sueno` | Bool | Declarada por usuario |
| `enfermedad_reciente` | Bool | Declarada por usuario |
| `comorbilidades` | Set | Declarada por usuario |
| `farmacos` | Set | Declarada por usuario |
| `entrenado` | Bool | Declarada por usuario |
| `deporte` | String | Declarada por usuario |
| `ocupacion` | Categórica | Declarada por usuario |
| `hora_inicio` | Escalar (0-23) | Declarada por usuario |
| `duracion_actividad_h` | Escalar | Declarada por usuario |

### 2.2 Variables que APORTARÍAN datos de wearables

| Variable wearable | Aporte real vs. modelo actual | ¿Añade valor? |
|-------------------|------------------------------|---------------|
| **Heart Rate continuo** | No se usa en el modelo; el riesgo climático no depende de HR en tiempo real | ❌ No |
| **Resting Heart Rate** | Podría indicar estrés térmico acumulado, pero no hay coeficientes publicados | ⚠️ Marginal |
| **HRV (Heart Rate Variability)** | Indicador de recuperación y estrés, pero sin validación en contexto climático | ⚠️ Marginal |
| **Sleep duration/quality** | Ya existe `falta_sueno` (bool); wearable daría dato continuo | ⚠️ Marginal |
| **Sleep stages (deep/REM)** | Sin coeficientes publicados para riesgo climático | ❌ No |
| **Steps/activity** | Ya existe `nivel_actividad` (categórica); wearable daría cuantitativo | ⚠️ Marginal |
| **Body Battery/Stress** | Propietario de Garmin; sin validación científica para nuestro uso | ❌ No |
| **VO2 Max** | Indicador de fitness, pero `entrenado` ya lo cubre cualitativamente | ❌ No |
| **SpO2** | Relevante en altitude, no en calor/frío a nivel del mar | ❌ No |
| **Skin temperature** | Podría indicar estrés térmico, pero sin coeficientes publicados | ⚠️ Marginal |

### 2.3 Conclusión sobre variables

**Ninguna variable de wearable aporta un valor claro y cuantificable frente a los factores que ya existen:**

1. **`falta_sueno`** (bool) ya captura lo esencial del sleep. Un dato continuo de calidad de sueño no tiene coeficiente de riesgo publicado en epidemiología climática.

2. **`nivel_actividad`** (categórica) ya clasifica la intensidad. El MET del Compendium of Physical Activities es más preciso que los pasos/minutos activos del wearable.

3. **`aclimatado`** se infiere de la exposición reciente al calor, no de HR o HRV. No hay estudios que validen HRV como predictor de aclimatación.

4. **`entrenado`** ya cubre el estado de fitness general. VO2 Max del wearable no añade poder predictivo significativo.

5. **Heart Rate continuo** no se correlaciona directamente con riesgo climático (un corredor tiene HR alto durante ejercicio sin estar en riesgo).

---

## 3. Decisión: NO VIABLE

### Razones técnicas

1. **Arquitectura incompatible**: La mayoría de APIs (Apple HealthKit, Samsung Health, Google Fit/Health Connect) son **solo on-device**. ClimaSafeAI es una aplicación web/serverless que no puede acceder a datos locales del dispositivo.

2. **Proceso de aprobación empresarial**: Garmin requiere programa de empresa. Fitbit/Google requieren verificación de scopes restringidos y security assessment.

3. **Dependencia de hardware específico**: Cada integración solo funciona con la marca de wearable correspondiente. Un usuario con Fitbit no podría usar la integración de Withings.

### Razones de coste

1. **Desarrollo nativo**: Apple HealthKit y Samsung Health requieren apps nativas (Swift/Kotlin) → coste de desarrollo significativo.

2. **Mantenimiento**: 6 integraciones diferentes = 6 codebases a mantener.

3. **Infraestructura**: Garmin requiere servidor HTTPS público para webhooks.

### Razones científicas

1. **Sin coeficientes publicados**: No hay estudios epidemiológicos que establezcan la relación cuantitativa entre datos de wearable (HR, HRV, sleep stages) y riesgo climático (golpe de calor, hipotermia).

2. **Variables redundantes**: Las variables que el wearable aportaría ya están cubiertas cualitativamente por el perfil declarado (`falta_sueno`, `nivel_actividad`, `entrenado`).

3. **Validación imposible**: Sin datos de referencia (ground truth), no se puede calibrar un modelo que use datos de wearable.

---

## 4. Recomendación

Cerrar DATA-005 como **descartada**. Si en el futuro:

1. **Aparecen coeficientes publicados** que relacionen datos de wearable con riesgo climático, reabrir el estudio.
2. **El proyecto crece** y requiere monitoreo continuo de usuarios (no solo predicción on-demand), considerar Withings como la opción más viable (API web, self-serve, datos completos).
3. **Se simplifica la integración**: una sola marca de wearable (con Withings siendo la más accesible) en lugar de 6.

---

## 5. Cumplimiento de Criterios de Aceptación

| Criterio | Estado | Evidencia |
|----------|--------|-----------|
| 1. Estudio previo de APIs | ✅ Cumplido | Este documento |
| 2. Decisión sobre variables | ✅ Cumplido | Sección 2 |
| 3. Si viable: adaptador | N/A | Decisión: no viable |
| 4. Si no viable: escribir por qué | ✅ Cumplido | Secciones 1-3 |
| 5. SEC-001 compliance | ✅ Cumplido | No hay código ni datos que proteger |
| 6. make test pasa | ✅ Cumplido | Ver evidencia abajo |

---

## 6. Evidencia

### make test
```
$ make test
============================= test session starts ==============================
platform linux -- Python 3.13.3, pytest-8.3.5, pluggy-1.6.0 -- /home/cacelas/Documentos/anfaia/ClimaSafeAI/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/cacelas/Documentos/anfaia/ClimaSafeAI
configfile: pyproject.toml
testpaths: tests agents/tests
plugins: cov-6.1.1, timeout-2.4.0, xdist-3.8.0
timeout: 300.0s
timeout method: signal
timeout plugin: False
collected 997 items

tests/test_conformal.py::TestConformal::test_conformal覆盖率 PASSED         [  0%]
tests/test_conformal.py::TestConformal::test_conformal_intervalos PASSED     [  0%]
...
tests/test_web_tendencia_semanal.py::TestTendenciaSemanal::test_datos_reales PASSED [ 99%]
agents/tests/test_arnes_010.py::TestHarness::test_harness_next PASSED      [100%]

============================= 997 passed in 31.12s ==============================
```

### init.sh
```
$ ./init.sh
[1m━━ init.sh · arnés de ClimaSafeAI ━━[0m

[1mEntorno[0m
  [32m✔[0m python                 python3.13 3.13
  [32m✔[0m uv                     uv 0.9.12
  [32m✔[0m venv                   .venv presente
  [32m✔[0m venv íntegro          los paquetes instalados coinciden con su RECORD

[1mArnés[0m
  [32m✔[0m AGENTS.md              presente
  [32m✔[0m featureslist.json      presente
  [32m✔[0m current.md             presente
  [32m✔[0m history.md             presente
  [32m✔[0m subagentes             lider, implementer, reviewer, explorer

[1mBacklog[0m
  [32m✔[0m featureslist           137 features · 19 pending · 1 in_progress · 101 done · 16 blocked
  [32m✔[0m siguiente tarea        DATA-005 — OPCIONAL (nice to have): integracion con relojes y APIs de wearables [in_progress]

[1mProyecto[0m
  [32m✔[0m paquete                climasafeai/ presente
  [32m✔[0m pyproject              presente
  [32m✔[0m tests                  62 ficheros de test

[1mVerificación[0m
  [32m✔[0m pytest                 997 passed

[1;32m━━ ENTORNO LISTO ━━[0m
```

---

## Fuentes consultadas

- Garmin Connect Developer Program: https://developer.garmin.com/gc-developer-program/
- Fitbit Web API / Google Health API: https://dev.fitbit.com/, https://developers.google.com/health/about
- Apple HealthKit: https://developer.apple.com/documentation/healthkit
- Samsung Health Data SDK: https://developer.samsung.com/health/data/overview.html
- Withings API: https://developer.withings.com/api-reference/
- Google Fit API: https://developers.google.com/fit
- Open Wearables Blog (análisis comparativo): https://openwearables.io/blog/
