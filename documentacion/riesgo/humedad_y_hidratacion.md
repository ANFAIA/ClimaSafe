# Humedad, hidratación y confort hídrico en el trabajo

Datos sobre el papel de la humedad relativa en el bienestar y el riesgo
térmico laboral, los rangos confortables establecidos, la sinergia de la
humedad alta con el calor, y las pautas oficiales de hidratación.

> **Propósito:** fundamentar la incorporación de la humedad como factor de
> riesgo propio y como modulador del riesgo por calor (Heat Index), además
> de las recomendaciones de hidratación de ClimaSafe.

---

## 1. Rangos de humedad en el lugar de trabajo

| Norma | Rango de humedad relativa |
|-------|---------------------------|
| RD 486/1997 (Anexo III) | 30% a **70%** (excepto locales con exigencias técnicas) |
| Guía Técnica INSST | Recomendaciones específicas por actividad |
| ISO 7730 (bienestar térmico) | Confort: 30-70% |
| ASHRAE 55 | Zona de confort: límites 30-60% |

**Por qué importa:**
- Humedad **alta** (>70-80%): dificulta la evaporación del sudor → el cuerpo
  no puede disipar calor → incremento drástico del riesgo por calor.
- Humedad **baja** (<30%): sequedad de mucosas, irritación ocular y
  respiratoria, electricidad estática (riesgo en atmósferas con polvo o
  productos inflamables), incremento de la sensación de frío en invierno.

## 2. La humedad como modulador del riesgo por calor

### Índice de calor (Heat Index, NWS)

El Heat Index (índice de temperatura aparente) combina **temperatura y
humedad relativa**: a mayor humedad, el cuerpo percibe y sufre más calor.

Ejemplos con temperatura de aire 32°C (sombra):

| Humedad relativa | Heat Index | Percepción | Riesgo |
|------------------|------------|------------|--------|
| 40% | 32°C | Cautela | Bajo |
| 60% | 39°C | Peligro | Medio |
| 80% | 47°C | Peligro extremo | Muy alto |
| 100% | 55°C | Peligro extremo | Extremo |

La fórmula está calibrada para condiciones de sombra; al sol se puede sumar
hasta 8°C adicionales (aunque el algoritmo estándar añade +4,3°C clásico).

### Humidex (Canadá, usado también en España por AEMET en boletines)

```
Humidex = ta + 5/9 × (e − 10)
e = 6,112 × 10^(7,5×td/(237,7+td))   (presión de vapor, hPa)
```

| Humidex | Sensación |
|---------|-----------|
| 20-29 | Sin incomodidad |
| 30-39 | Alguna incomodidad |
| 40-45 | Mucha incomodidad; evitar esfuerzos |
| 46-54 | Cierta incomodidad al trabajar; riesgo de golpe de calor |
| >54 | Insolación posible |

### Regla práctica INVASSAT

- Con **HR > 70%** y temperaturas altas, el enfriamiento por sudoración se
  ve seriamente comprometido: el riesgo real es mucho mayor que el que
  sugiere la temperatura de bulbo seco.
- En ambientes húmedos y calurosos la evaluación debe usar **WBGT en
  interior** (que incorpora bulbo húmedo) o el Heat Index como
  aproximación práctica (ISO 7243).

## 3. Pérdidas de agua y rendimiento

| Situación | Pérdida de agua | Efecto en rendimiento |
|-----------|------------------|------------------------|
| Trabajo ligero en calor (8h) | 3-5 litros | Deshidratación progresiva |
| Trabajo intenso en calor (8h) | 10-12 litros (casos extremos) | Caída del rendimiento físico y mental |
| Deshidratación leve (1-2% peso corporal) | ~1-2% | Reducción de capacidad de trabajo |
| Deshidratación 3-5% | — | Riesgo de golpe de calor, fatiga extrema |
| Deshidratación >6% | — | Riesgo vital, mortalidad posible |

**Clave:** el cuerpo no siente sed de forma fiable durante el ejercicio en
calor: la sed aparece cuando ya se ha perdido 2% del peso. Por eso la
hidratación debe ser **programada**, no a demanda.

## 4. Pautas oficiales de hidratación (INSST/INVASSAT)

### Insistencia en el agua

| Momento | Cantidad |
|---------|----------|
| Antes de la jornada | 500 ml |
| Durante la jornada | 1 vaso (200-250 ml) cada 20-30 minutos |
| Después de la jornada | 500-750 ml |
| En jornadas de mucho calor | Hasta 6-8 litros/día (con control de electrolitos) |

### Buenas prácticas

- El agua debe estar fresca (10-15°C), no helada.
- Bebidas isotónicas recomendables si la sudoración es abundante o la
  jornada supera 2 horas de esfuerzo en calor.
- La **sal (electrolitos)** debe reponerse con la comida; no tomar
  comprimidos de sal sin indicación médica.
- Evitar bebidas con cafeína y alcohol (diuréticas; el café además
  vasodilata en frío, véase `frio_extremo_laboral.md`).
- Vigilar el color de la orina como señal simple de hidratación.
- Vídeo/guía del INSST: "El agua, bebida recomendable en el trabajo".

## 5. Humedad exterior y meteorología

- La humedad relativa exterior es un dato estándar de AEMET (observatorios y
  previsiones por hora/municipio).
- Días con **calima + humedad** o **chubascos + calor** en costa pueden
  combinar HR > 80-90% con temperaturas altas: Heat Index muy elevado.
- En **frío**, humedad alta y viento maximizan la sensación térmica de frío
  (mayor conductividad del aire húmedo); la ropa mojada multiplica la pérdida
  de calor.
- La humedad interviene en la **tWC** (wind chill) y en el cálculo de WBGT
  (bulbo húmedo).

## 6. ClimaSafe y la humedad

- La humedad es **dato de entrada** del Heat Index (riesgo por calor) y
  modula el estrés por frío: un solo valor meteorológico (HR por hora) sirve
  para re-parametrizar ambos extremos.
- Fuente: previsiones y observaciones AEMET (humedad relativa horaria).
- La recomendación de hidratación se puede **personalizar** con:
  nivel_actividad, duración_h, duración de la jornada y HR ambiente.
- Umbrales propuestos: HR > 70% + Heat Index en zona de peligro → aviso de
  pausas hidratación intensivas; HR < 30% (interiores) → recomendaciones de
  humectación/protocolo de estática.

## 7. Referencias

- **RD 486/1997.** Lugares de trabajo (Anexo III): humedad 30-70%.
- **Guía Técnica INSST RD 486/1997.**
- **INSST.** Pautas de hidratación y agua en el trabajo (campañas).
- **INVASSAT.** Guía estrés térmico por exposición a calor (humedad alta).
- **ISO 7243 / ISO 7730.** WBGT y bienestar térmico.
- **NWS (NOAA).** Heat Index chart and formula.
- **AEMET.** Humidex y boletines de calor.
- **ISO 8996.** Determinación de la producción de calor metabólico.