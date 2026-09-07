# Pausas de trabajo y horas óptimas para trabajo al aire libre

Criterios técnicos reconocidos (ACGIH, NIOSH, INSST/INVASSAT) para
estructurar las pausas trabajo-descanso en ambiente caluroso, las pautas de
aclimatación y la selección de las horas óptimas de trabajo exterior.

> **Propósito:** fundamentar las dos recomendaciones operativas centrales de
> ClimaSafe: **cuándo trabajar** (horas óptimas) y **cómo trabajar**
> (pausas/relevos).

---

## 1. Ciclos trabajo-descanso (Work-Rest Cycles)

### Base científica

- Cuando el calor corporal aumenta, el rendimiento físico cae y el riesgo de
  daño sube de forma no lineal. Los ciclos de reposo en ambiente fresco
  **recuperan la capacidad de termorregulación** (bajan el ritmo cardíaco y
  la temperatura profunda).
- Los descansos deben ser **frecuentes y cortos** (mejor que largos y
  escasos) en ambientes calurosos: cada pausa evita que la temperatura
  corporal supere los límites de seguridad.
- Las pausas deben hacerse **a la sombra** y, en episodios extremos, en
  locales climatizados/refrigerados.

### Pauta general (Ficha Calor Extremo INSST)

| Situación | Pauta |
|-----------|-------|
| Jornada normal en calor moderado | Pausas de 1-2 min cada 30 min de trabajo + pausa de comida |
| Ola de calor / aviso rojo | Relevos frecuentes, pausas más largas, reducir carga |
| Trabajo pesado con temperatura alta | Ciclos trabajo/descanso según ACGIH (ver tabla) |

### Tabla ACGIH TLV para ciclos trabajo-descanso

En ambiente caluroso, el **% de trabajo por hora** depende de la carga
metabólica (ligera, moderada, pesada, muy pesada) y del WBGT del puesto:

| Carga metabólica (M) | WBGT criterio (aclimatado) | Ciclo de trabajo por hora recomendado |
|----------------------|----------------------------|----------------------------------------|
| Ligera (M<200 W/m²) | Variable con M y calor húmedo | Pausas proporcionales cuando se supera TLV |
| Moderada (200-320 W/m²) | 28°C (aclimatado) | Reducir % de trabajo según tabla ACGIH |
| Pesada (320-400 W/m²) | 26°C (aclimatado) | Ej. 75% / 50% / 25% trabajo → resto recuperación |
| Muy pesada (>400 W/m²) | 25°C (aclimatado) | Ej. 50% trabajo en zona límite; 15 min por hora en reposo |

(En interior sin aclimatar, los TLVs bajan ~1,5-2°C. Ver
`marco_normativo_estrés_térmico.md` para la tabla WBGT completa.)

### Principio INVASSAT

- **Pausas frecuentes y cortas**: 1-2 minutos de descanso cada 30 minutos.
- En **temperaturas extremas** o **trabajo intenso**: ampliar la duración de
  las pausas y reducir el tiempo de exposición continua. Priorizar
  **relevos** de trabajadores en turnos rotatorios.

## 2. Aclimatación al calor

| Momento | Recomendación INSST/OSHA |
|---------|--------------------------|
| Día 1 | Jornada de trabajo al calor de ~50% del normal; en trabajadores muy expuestos, no superar 20% de la exposición habitual |
| Días 2-7 | Incremento gradual del tiempo de exposición (20% diario aprox.) a jornada completa |
| Días 5-7 | Aclimatación sustancial alcanzada (tras ~1-2 semanas completa) |
| Nueva ola de calor tras reposo >1 semana | Re-aclimatación parcial: tratar como inicio |

- La aclimatación **reduce el riesgo de golpe de calor** y mejora la
  sudoración (más precoz, más diluida).
- En trabajadores **no aclimatados**, el riesgo de lesión por calor se
  multiplica (ver `personalizacion_individual.md`: `aclimatado` es ya un
  parámetro del perfil ClimaSafe).

## 3. Horas óptimas para el trabajo exterior

### Observación general (INSST / guías sectoriales, p. ej. Valencia, Sevilla)

- Las **primeras horas de la mañana** (hasta ~11:00 en verano) y las del
  final de la tarde/noche (19:00 en adelante) son las idóneas para tareas
  con esfuerzo físico o exposición directa al sol.
- Evitar el tramo central **12:00-16:00** (máximo: temperatura, UVI y
  ozono).
- Planificar los trabajos más pesados en la franja más fresca y los ligeros
  en las horas de más calor.
- Cuando el aviso naranja/rojo AEMET esté activo: evaluar **adelantar o
  retrasar la jornada** y reducir la exposición al mediodía. La empresa debe
  documentar las medidas organizativas adoptadas (disposición adicional del
  RD 486/1997, ver `marco_normativo_estrés_térmico.md`).
- Para trabajos nocturnos o en doble turno: asegurar recuperación y
  ventilación adecuada.

### Regla práctica por perfil (ClimaSafe)

```
Riesgo horario = f(T, HR, viento, UVI, ICA) en cada hora
                 × f(perfil: edad, comorbilidades, actividad, aclimatación)
Franja recomendada = horas consecutivas con riesgo < umbral
                     dentro de la ventana de actividad
```

- La **ventana de actividad** (hora_inicio → hora_inicio + duración) ya es el
  eje temporal del modelo: ClimaSafe la desplaza o la subdivide en función
  del riesgo horario (ver `formulas_deterministas.md`).
- Si no es posible desplazar toda la ventana: **dividirla** con pausas de
  sombra/climatización en las horas críticas y **reprogramar** las tareas
  más pesadas a las horas de menor riesgo.

## 4. Buenas prácticas adicionales

| Medida | Detalle |
|--------|---------|
| Sombras | Telas de media sombra, toldos, zonas refrigeradas |
| Relevos | Rotación de puestos frente a la fuente de calor |
| Trabajo en solitario | Evitarlo en condiciones extremas (temperaturas o tWC extremas) |
| Control del agua | Ver `humedad_y_hidratacion.md`: agua a 10-15°C, cada 20-30 min |
| Vigilancia | Observar signos de golpe de calor en compañeros (confusión, piel seca, cefalea) |
| Mecanización | Reducir carga física con maquinaria/ayudas |
| Autogestión del ritmo | Permitir al trabajador autorregular el ritmo en condiciones adversas |

## 5. ClimaSafe y las pausas/horas óptimas

- **Salida operativa del producto:** dado un perfil y una ventana de
  actividad, ClimaSafe entrega:
  1. Franja horaria recomendada de trabajo exterior (verde, según riesgo).
  2. Plan de pausas trabajo-descanso por hora (duración y periodicidad).
  3. Calendario de aclimatación personalizado (día 1: 50%, +20%/día).
  4. Alertas cuando la franja de actividad entra en zona de riesgo.
- Todo ello combinando los seis factores del modelo (calor, UV, aire, frío,
  humedad, esfuerzo) sobre el mismo motor horario (`predict_risk`).

## 6. Referencias

- **ACGIH (2023).** TLVs and BEIs: Heat Stress and Strain (tablas de ciclos
  trabajo-descanso en función de WBGT y carga metabólica).
- **NIOSH (2016).** Criteria for a Recommended Standard: Occupational
  Exposure to Heat and Hot Environments.
- **INSST.** Ficha Técnica: Trabajos en condiciones de calor extremo (2023)
  — campañas y recomendaciones operativas.
- **INVASSAT.** Guía estrés térmico por exposición a calor: pautas de pausas
  y aclimatación.
- **Junta de Andalucía / Conselleria de Educación (modelo de jornada
  continua en verano)** — referencia de práctica sectorial.
- **OMS (2023).** Heat-health action plans (pausas, horas frescas, sombra).