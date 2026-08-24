# Guía de usuario — ClimaSafe

> **Para ti si no eres desarrollador.** Esta guía explica cómo usar ClimaSafe
> sin necesidad de entender código ni modelos.

## Qué es ClimaSafe

ClimaSafe es un sistema de **aviso de riesgo por calor y frío** para España.
Te dice si el día de mañana (o cualquier día que elijas) hay riesgo
cardiovascular por temperatura en tu zona, y te avisa con antelación.

No es un tiempo meteorológico: es un **sistema de salud pública** que combina
datos del tiempo con tu perfil personal para estimar tu riesgo real.

## Cómo funciona en tu día a día

### 1. Consulta tu riesgo en el navegador

Abre la **[demo en el navegador](https://cacelass.github.io/climasafe/probar-ya/)**.
No necesitas instalar nada.

1. Escribe tu **latitud** y **longitud** (o busca tu ciudad en Google Maps
   y copia las coordenadas).
2. Introduce tu **edad**.
3. Selecciona la **fecha** que te interese.
4. Pulsa **Calcular**.

El sistema te devolverá:

- **SEGURO** — el día no presenta riesgo significativo.
- **PRECAUCIÓN** — hay condiciones adversas; toma precauciones si eres
  vulnerable (edad avanzada, problemas cardíacos, medicación...).
- **PELIGRO** — riesgo alto; se recomienda evitar esfuerzos al aire libre
  y mantenerse en ambientes climatizados.

### 2. Recibe alertas en Telegram

Si usas Telegram, puedes añadir el **bot de ClimaSafe**:

1. Busca `@ClimaSafeBot` en Telegram.
2. Envía el comando `/start`.
3. Rellena tu perfil cuando te lo pida (edad, provincia, si tomas
   medicación, si tienes comorbilidades...).
4. El bot te enviará un **aviso diario** antes de los días peligrosos.

El bot también responde a preguntas como:

- "¿Qué riesgo tengo mañana en Madrid?"
- "¿Cómo está el calor en Valencia hoy?"

### 3. En tu asistente de IA (MCP)

Si usas Claude Desktop, Cline u otro asistente con soporte MCP, puedes
añadir el servidor de ClimaSafe para que tu asistente pueda consultar
el riesgo directamente. Pide a tu asistente que añada el servidor MCP
con la configuración que encontrarás en la sección técnica.

## Qué significa cada nivel de riesgo

| Nivel | Significado | Qué hacer |
|-------|-------------|-----------|
| **SEGURO** | Sin riesgo significativo | Actividad normal |
| **PRECAUCIÓN** | Condiciones adversas | Evita esfuerzos prolongados al aire libre si eres vulnerable |
| **PELIGRO** | Riesgo alto | Permanece en ambientes climatizados, hidrátate, sigue las indicaciones de sanitarios |

## ¿Cómo se calcula?

El sistema combina:

- **Datos meteorológicos** de la zona (temperatura, humedad, viento,
  radiación solar) del día que seleccionas.
- **Tus datos personales** (edad, sexo, comorbilidades, medicación,
  nivel de aclimatación, actividad que vas a realizar).

Los datos del tiempo se comparan con modelos entrenados con datos reales
de mortalidad en España. Tu perfil personal ajusta ese riesgo poblacional
a tu situación concreta.

## ¿Es fiable?

- Los modelos se entrenaron con **datos reales de mortalidad** (MoMo,
  registros del Ministerio de Sanidad).
- Se usan **tres modelos de ML** diferentes y se combinan para mayor
  robustez.
- La predicción se **calibra** para que las probabilidades sean reales,
  no solo clasificaciones.
- El sistema se **evalúa honestamente**: publicamos las métricas reales,
  incluyendo dónde falla.

## Preguntas frecuentes

**¿Necesito internet?**
Sí, la demo y el bot necesitan conexión para obtener los datos
meteorológicos en tiempo real.

**¿Mis datos están seguros?**
El bot almacena tu perfil en el servidor. La demo no almacena nada:
todo se ejecuta en tu navegador.

**¿Funciona en todo España?**
Sí, para cualquier punto con coordenadas válidas.

**¿Puedo usarlo para mi empresa o comunidad?**
Sí. ClimaSafe está publicado bajo la **licencia Apache 2.0**, que permite
usarlo, modificarlo y distribuirlo libremente (incluso comercialmente),
siempre que se conserve el aviso de licencia y se indique si se hicieron
cambios. El código está en [ANFAIA/ClimaSafe](https://github.com/ANFAIA/ClimaSafe).

---

*Para la documentación técnica (arquitectura, modelos, pesos, integración),
consulta la [documentación para desarrolladores](index.md).*
