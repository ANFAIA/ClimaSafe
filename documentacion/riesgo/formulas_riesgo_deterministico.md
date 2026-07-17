# Fórmulas de riesgo climático determinista

Golpe de calor, hipotermia y quemadura solar — base científica y ecuaciones
usadas por la fórmula de respaldo del sistema híbrido (ver
`diseno_modelo.md`, sección 5).

## Por qué estas tres no necesitan ML

A diferencia del riesgo de "ingreso por causa X" (sepsis, renal, metabólico
— ver `riesgo_ingresos_formula.md`), que es una asociación estadística
poblacional, estas tres condiciones tienen **mecanismo fisiológico
conocido y umbrales clínicos publicados**. Son cálculo directo, no
predicción estadística. El CMBD solo aportaría calibración fina, nunca
detección — el sistema es funcional sin él desde el día 1.

---

## 1. Golpe de calor

### 1.1 Heat Index — ecuación de Rothfusz (1990)

Fuente: NWS / Rothfusz (1990), regresión sobre las tablas de Steadman.
Combina temperatura del aire y humedad relativa en una "sensación
térmica" equivalente.

**Fórmula (input en °F, humedad relativa en %):**

```
HI = -42.379
     + 2.04901523·T
     + 10.14333127·RH
     - 0.22475541·T·RH
     - 0.00683783·T²
     - 0.05481717·RH²
     + 0.00122874·T²·RH
     + 0.00085282·T·RH²
     - 0.00199788·T²·RH²
```

- `T`: temperatura del aire en °F
- `RH`: humedad relativa en % (0–100)
- Válida para T ≥ 80°F (~26.7°C) y RH ≥ 40%. Por debajo de ese rango se
  usa una fórmula simplificada (media entre T y el efecto de humedad),
  ya que la ecuación completa no es fiable ahí.
- El proyecto trabaja en °C: convertir `T(°F) = T(°C)·9/5 + 32` antes de
  aplicar, y el resultado de vuelta con `T(°C) = (HI(°F) - 32)·5/9`.

### 1.2 Tabla de categorías de riesgo

| Heat Index | Categoría | Riesgo clínico |
|---|---|---|
| 27–32°C | Precaución | Fatiga posible con exposición prolongada / actividad |
| 32–39°C | Precaución extrema | Calambres y agotamiento por calor posibles |
| 39–51°C | Peligro | Calambres/agotamiento probables; golpe de calor posible con exposición prolongada |
| >51°C | Peligro extremo | Golpe de calor muy probable |

### 1.3 Base normativa y clínica de referencia

- **NIOSH Occupational Heat Exposure Guidelines** — límites de exposición
  ocupacional al calor y criterios de patologías asociadas (agotamiento,
  golpe de calor), usados para contrastar los umbrales de la tabla 1.2.
- **INSST NTP-322** — nota técnica de prevención española sobre estrés
  térmico; referencia para la normativa laboral aplicable en España.
- **OIT, Informe Seguridad Climática (2024)** — datos de mortalidad
  laboral por calor a nivel global, usados como contexto de magnitud del
  problema.
- **WHO Heat Health Action Plans** — recomendaciones a ciudades y
  sistemas de salud ante episodios de calor extremo.
- **Ministerio de Sanidad de España, Plan Calor 2026** — datos oficiales
  de mortalidad y episodios extremos en España, usados para contrastar
  los umbrales frente al contexto nacional.

### 1.4 De Heat Index a WBGT — límites ocupacionales NIOSH

NIOSH no usa el Heat Index para sus límites de exposición laboral, sino el
**WBGT** (Wet Bulb Globe Temperature), que además de temperatura y
humedad incorpora radiación solar directa y viento — más preciso para
trabajo al aire libre. NIOSH define dos umbrales por WBGT según carga de
trabajo (ligera/moderada/pesada/muy pesada, equivalente a las categorías
MET del proyecto):

- **REL** (Recommended Exposure Limit) — para trabajadores aclimatados.
- **RAL** (Recommended Alert Limit) — para trabajadores no aclimatados,
  más restrictivo. Coincide con el coeficiente ×1.6 de "sin aclimatación"
  ya usado en el proyecto.

Medir WBGT real requiere termómetro de globo y bulbo húmedo, que el
sistema no tiene (solo dispone de temperatura/humedad/viento vía ERA5 y
Open-Meteo). Bernard & Iheanacho (2015, *J. Occup. Environ. Hyg.*)
publicaron una aproximación de WBGT a partir del Heat Index, pensada
exactamente para este caso:

```
WBGT ≈ -0.0034·HI² + 0.96·HI - 34
```

- `HI`: Heat Index en °F (ecuación de Rothfusz, sección 1.1)
- `WBGT`: resultado en °C
- Precisión: ±2°C para HI bajos, ±0.5°C para HI > 100°F (~37.8°C). Los
  propios autores advierten que es un cribado (*screening*), no un
  sustituto de una medición real de WBGT — válido para estimar si hace
  falta un análisis más detallado, no como valor clínico definitivo.
- Valores de referencia parciales confirmados de NIOSH [2016-106]:
  trabajo pesado, trabajador aclimatado → REL ≈ 25°C WBGT (trabajo
  continuo); trabajo moderado, aclimatado → REL ≈ 27.8°C WBGT. La tabla
  completa por las 4 categorías de carga y ciclos trabajo/descanso está
  en NIOSH [2016-106], pág. 76-77 — consultar ahí antes de fijar los
  umbrales exactos por categoría MET del proyecto.

---



### 2.1 Wind Chill — ecuación NWS (2001)

Fuente: National Weather Service, fórmula oficial vigente desde 2001.

**Fórmula (input en °C, viento en km/h a 10 m de altura):**

```
WC = 13.12 + 0.6215·T - 11.37·V^0.16 + 0.3965·T·V^0.16
```

- `T`: temperatura del aire en °C
- `V`: velocidad del viento en km/h
- Válida para `T ≤ 10°C` y `V > 4.8 km/h`. Fuera de ese rango el efecto
  del viento no es significativo y se usa directamente la temperatura
  real.

### 2.2 Tabla de tiempo hasta riesgo significativo

| Wind Chill | Tiempo hasta riesgo significativo |
|---|---|
| 0 a -25°C | Riesgo bajo con ropa adecuada; horas |
| -25 a -35°C | Riesgo en ~30 minutos de exposición |
| -35 a -48°C | Riesgo en ~10 minutos |
| < -48°C | Riesgo en ~5 minutos o menos |

### 2.3 Precisión de los tiempos de congelación

Fuente: NWS, *Frostbite Times Wind Chill Chart* (vigente desde
01/11/2001).

**Fórmula oficial** (idéntica a la de la sección 2.1, expresada en
unidades imperiales, tal como la publica la NWS):

```
WC(°F) = 35.74 + 0.6215·T - 35.75·V^0.16 + 0.4275·T·V^0.16
```
donde `T` es temperatura en °F y `V` es viento en mph.

**Importante — el tiempo de congelación no es función limpia del Wind
Chill ya calculado**, sino de la combinación original de temperatura y
viento: dos combinaciones distintas de T y V pueden dar el mismo Wind
Chill pero tiempos de congelación distintos. Ejemplos oficiales (NOAA):

| Temperatura | Viento | Wind Chill | Tiempo hasta congelación |
|---|---|---|---|
| 0°F (-18°C) | 15 mph (24 km/h) | -19°F (-28°C) | ~30 minutos |
| -40°F (-40°C) | 5 mph (8 km/h) | -58°F (-50°C) | ≤10 minutos |
| -20°F (-29°C) | 45 mph (72 km/h) | -58°F (-50°C) | ≤5 minutos |

Es decir: para una implementación fiel al estándar NWS, el tiempo de
congelación debe leerse de la rejilla oficial temperatura×viento (no
derivarse solo del Wind Chill final). La rejilla completa está publicada
en: https://www.weather.gov/media/safety/windchillchart3.pdf — usar esos
valores exactos en la implementación final en vez de aproximar por rango
de Wind Chill.

---

## 3. Quemadura solar — UV Index + fototipo (OMS)

Fuente: OMS, *Global Solar UV Index: A Practical Guide*.

**Fórmula:**

```
minutos_hasta_eritema = tiempo_base[fototipo] / UV_index
```

| Fototipo (Fitzpatrick) | tiempo_base (minutos a UV=1) aprox. |
|---|---|
| I (piel muy clara, siempre se quema) | ~67 |
| II (clara) | ~100 |
| III (media) | ~133 |
| IV (morena) | ~183 |
| V (oscura) | ~267 |
| VI (muy oscura) | ~400+ |

---

## Relación con el resto del sistema

- Estas tres reglas son el **fallback universal**, funcionan en cualquier
  parte del mundo desde el día 1 (no dependen de MoMo ni de España).
- El modelo ML (MoMo) refina el umbral de golpe de calor/hipotermia con
  datos reales de mortalidad española — pero si el ML no está disponible
  o el caso es raro (Pirineo), estas tablas deterministas siguen dando
  una respuesta válida y conservadora.
- `riesgo_ingresos_formula.md` (coeficientes ISGlobal) es un nivel
  adicional, opcional, para causas indirectas (sepsis, renal...) — no
  bloquea ni depende de esto.
