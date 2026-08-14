# Retos técnicos — estudio de viabilidad

> Feature RESEARCH-001. Documenta la viabilidad de seis familias de modelos
> (HMM, Redes Bayesianas, Procesos Gaussianos, GNNs, TFT/N-BEATS, RL) en el
> contexto **real** de ClimaSafeAI, respetando las decisiones ya tomadas en
> `../arquitectura/diseño_modelo.md` §7, `alternativas_descartadas.md` y
> `../modelos/README.md` (GP, RL, causalidad). Este documento **no reinventa**
> decisiones cerradas: las confirma, las matiza o las revisa solo si hay
> evidencia nueva.
>
> Contexto del sistema: clasificación de riesgo 3 clases
> (`0=seguro, 1=precaución, 2=peligro`) por `(provincia, día)`, con
> XGBoost (calor) / RandomForest (frío) / LSTM híbrida / fórmula
> determinista, ensamble por criterio más restrictivo, conformal prediction
> (α=0.1) + calibración isotónica como incertidumbre, y personalización
> post-ensemble por perfil (`personalizacion.py`). Datos: MoMo (mortalidad
> diaria por provincia) + ERA5 (histórico horario) + Open-Meteo (predicción).
> Restricciones de proyecto: CPU, coste cero, código abierto.

---

## Resumen ejecutivo

| # | Técnica | Veredicto | En una frase |
|---|---------|-----------|--------------|
| 1 | HMM / Cadenas de Markov | **Viable con condiciones** | Suavizado temporal barato de la trayectoria de riesgo, pero los lags ya capturan la persistencia y el estado latente no se alinea con las 3 clases. |
| 2 | Redes Bayesianas (grafo causal) | **No viable ahora** | El razonamiento contrafactual que aportarían ya está implementado (`contrafactuales.md`); el grafo de factores es conocido a priori y pequeño. |
| 3 | Procesos Gaussianos | **No viable ahora** | O(n³) con ~20k filas; conformal + isotónica ya cubren la incertidumbre. Confirmar `../modelos/README.md`. |
| 4 | GNN para embeddings de factores | **No viable** | Ya probado (GCN/GAT, ~20 nodos): no supera a los features planos. La variante espaciotemporal entre provincias choca con el techo de señal/etiqueta documentado. |
| 5 | TFT / N-BEATS | **No viable** | Confirmar `diseño_modelo.md` §7.2 y `alternativas_descartadas.md`: el caso de uso no necesita forecast a 7 días; la LSTM híbrida cubre la dimensión temporal. |
| 6 | RL (predecir → reducir riesgo) | **No viable ahora** | Sin bucle de feedback real no hay señal de recompensa; los contrafactuales deterministas ya reducen riesgo sin entrenar un agente. |

Decisión de ubicación: `documentacion/ml/` porque toda la documentación de
decisiones ML del proyecto vive aquí (`alternativas_descartadas.md`,
`conclusiones_modelos.md`, `lstm_hibrida.md`). Una carpeta nueva
`documentacion/retos/` fragmentaría el conocimiento que este README ya
indexa.

---

## 1. Cadenas de Markov / HMM — trayectoria de riesgo

**Qué aportaría.** Modelar el riesgo como un proceso temporal explícito: una
cadena de Markov sobre los estados de riesgo (SEGURO → PRECAUCIÓN → PELIGRO)
capturaría la probabilidad de transición entre días consecutivos — la
estructura de *ola* (una ola de calor encadena varios días de PELIGRO, y el
riesgo sube de forma progresiva, no aleatoria). Un HMM añadiría además una
capa de observación: las clases observadas son una emisión ruidosa de un
estado latente, y el filtrado (forward-backward) o Viterbi suavizaría la
trayectoria día a día.

**Cómo encajaría en el pipeline actual.** Como capa post-hoc sobre las
probabilidades diarias ya existentes (XGBoost/RF/LSTM): estimar la matriz de
transición por provincia o estación y aplicar Viterbi a la secuencia de
clases para eliminar parpadeo (días aislados de PELIGRO rodeados de SEGURO,
que hoy se avisan sin contexto). Coste de implementación bajo: `hmmlearn` o
una matriz de transición empírica, CPU, sin dependencias nuevas pesadas.

**Riesgos.**
- **Desalineación estado latente ↔ clases.** Las 3 clases se definen por
  percentiles de mortalidad MoMo por provincia. Un HMM aprenderá sus propios
  estados latentes, que no tienen por qué coincidir con `0/1/2`. Habría que
  fijar los estados a las clases observadas (cadena de Markov observada, no
  HMM) o aceptar una semántica distinta.
- **Redundancia con los lags.** El mayor salto de métrica del proyecto lo
  dieron las features de persistencia (`dias_consec_sobre_umbral`,
  `grados_dia_*_roll7/14`, ver `conclusiones_modelos.md` §2): el modelo ya
  "sabe" en qué día de la ola está. Un suavizado Markoviano añadiría coherencia
  temporal que los lags ya aportan a nivel de feature.
- **Métrica.** La selección es por `Rec_riesgo` por día; suavizar la
  trayectoria puede bajar el recall de días de riesgo aislados (que son
  precisamente los que el sistema debe avisar).

**Veredicto: viable con condiciones.** Es la técnica más barata de las seis y
la única con un hueco real (coherencia temporal de la salida, no de la
entrada). Pero es **experimental y de prioridad baja**: exige primero validar
que suavizar la salida no degrada `Rec_riesgo`, y compite con el hecho de que
la persistencia ya está en las features. Si se hace, empezar por una cadena de
Markov observada (transiciones entre clases) por estación, no por un HMM
latente.

---

## 2. Redes Bayesianas — grafo causal

**Qué aportaría.** Un modelo generativo `P(riesgo | meteorología, perfil)` con
estructura causal explícita (temperatura → Heat Index → riesgo;
comorbilidad → susceptibilidad; viento → Wind Chill). Permitiría razonar con
`do-calculus` (intervenciones, no condicionamientos) y responder "¿qué pasaría
si cambio X?" con una distribución, no con un punto.

**Cómo encajaría.** El proyecto ya usa la causalidad como **marco mental**
(`../modelos/README.md`: "No hay nada que implementar... cambia cómo se
diseñan features y se interpretan resultados"). El razonamiento contrafactual
que sería el principal beneficio de una BN **ya está implementado**:
`contrafactuales.md` / `explicabilidad.py` re-ejecuta `personalizar_riesgo`
con cambios de perfil (aclimatado, sombra, duración) y devuelve la bajada de
probabilidad — sin entrenar ningún grafo.

**Riesgos.**
- **Datos.** MoMo es mortalidad poblacional agregada por provincia; no existe
  joint sobre los factores individuales (fototipo, comorbilidades,
  medicación). Una BN completa necesitaría inventar esa distribución o
  quedarse en el subgrafo meteorológico, donde el beneficio sobre el
  clasificador actual es marginal.
- **Estructura conocida a priori.** El grafo de factores (~20 nodos) tiene
  relaciones mayoritariamente conocidas por la literatura; aprender la
  estructura de datos de mortalidad arriesga aristas espurias (confounding con
  eventos de exposición poblacional, ver `diseño_modelo.md` §5).
- **Coste.** `pgmpy` + ajuste de estructura + inferencia exacta/aproximada es
  semanas de trabajo para un beneficio que el pipeline de contrafactuales ya
  entrega.

**Veredicto: no viable ahora.** Coincide con la decisión de
`alternativas_descartadas.md` (Bayes Jerárquico + EVT descartado por coste de
migración probabilística) y con la de `../modelos/README.md` (causalidad como
marco mental, no como librería). Si algún día se necesita razonamiento causal
más profundo, el siguiente paso no es una BN completa sino formalizar los
contrafactuales existentes con la notación de intervenciones.

---

## 3. Procesos Gaussianos — incertidumbre nativa

**Qué aportaría.** Una distribución predictiva completa `P(y | x)` con
incertidumbre calibrada de fábrica — sin conformal ni calibración post-hoc.
Para un sistema de aviso, la incertidumbre es tan importante como la clase.

**Cómo encajaría.** Hoy la incertidumbre ya está resuelta con **conformal
prediction (α=0.1)** — prediction sets con cobertura garantizada — más
**calibración isotónica** para las probabilidades de PELIGRO
(`../conformal_prediction.md`). Un GP sustituiría ese mecanismo por
incertidumbre analítica, pero el problema ya tiene respuesta.

**Riesgos.**
- **Escalado.** El GP clásico es O(n³): inviable con el dataset real (~20k
  filas por modelo, `../modelos/README.md` ya lo documenta). Una aproximación
  sparse (SVGP, ~1000 puntos inductivos) sí correría en CPU, pero añade una
  dependencia (GPyTorch/GPflow) y una curva de ajuste de hiperparámetros.
- **Regresión colapsada.** El experimento de regresión del notebook 0-3 ya
  mostró que predecir el percentil continuo de mortalidad colapsa como sistema
  de aviso (`Rec_riesgo` 0.08 calor / 0.00 frío, `diseño_modelo.md` §6). Un GP
  es intrínsecamente regresor: habría que adaptarlo a clasificación (GP
  multiclase) o usarlo sobre el índice softmax, con ese antecedente en contra.
- **Redundancia.** Conformal ya garantiza cobertura empíricamente; el GP daría
  una incertidumbre "más bonita" sin una métrica de producto que la pida.

**Veredicto: no viable ahora.** Confirma `../modelos/README.md`: "El GP
clásico escala O(n³)... Un SVGP sería viable para el índice personalizado,
pero conformal prediction cubre la misma necesidad sin añadir dependencias".
Re-evaluar solo si el sistema necesitara una **función de incertidumbre
suave y diferenciable** (p. ej. para optimizar la hora de actividad como
función del riesgo), que hoy no tiene.

---

## 4. GNNs — embeddings de factores

Hay que separar dos usos distintos, porque el proyecto ya tiene evidencia
sobre uno y papers scouteados sobre el otro (`../modelos/gnn/`).

**Uso A — embeddings del grafo de factores (el que pide el criterio).**
- **Qué aportaría:** aprender representaciones de los ~20 factores de riesgo
  (comorbilidades, medicación, situación social) y sus interacciones, en vez
  de codificarlos a mano.
- **Evidencia previa:** ya se probaron **GCN y GAT** sobre ese grafo y "las
  representaciones aprendidas no mejoraban la clasificación frente a los
  features planos... el grafo es pequeño (~20 nodos) y las relaciones son
  mayoritariamente conocidas a priori — una GNN no aporta valor sobre la
  ingeniería manual" (`alternativas_descartadas.md`). Existen checkpoints de
  ese estudio (`models/gnn_*.pt`).
- **Veredicto: no viable.** Decisión ya cerrada y con evidencia empírica del
  propio proyecto. El criterio se documenta **confirmándola**: los embeddings
  de factores solo tendrían sentido con un grafo grande o con relaciones
  desconocidas — ni una cosa ni la otra se dan aquí.

**Uso B — correlación espaciotemporal entre provincias.**
- **Qué aportaría:** los papers de `../modelos/gnn/` (STGCN, GraphWaveNet,
  GNNs espaciotemporales) modelan la dependencia entre series vecinas. Una GNN
  sobre el grafo de provincias (adyacencia geográfica o climática) compartiría
  señal entre provincias con clima similar.
- **Riesgos:** el techo documentado del sistema **no es de capacidad de
  modelo**: "el techo es la precisión de las clases de riesgo (~0.08–0.27)...
  Es limitación de señal/etiqueta, no de modelo" (`conclusiones_modelos.md`
  §6). Añadir un modelo más complejo no levanta ese techo. Además MoMo es una
  serie independiente por provincia; la correlación espacial está sobre todo
  en las features meteorológicas (que ya comparten los mismos modelos) y en
  los eventos de ola a escala peninsular.
- **Veredicto: viable con condiciones (baja prioridad).** Solo tiene sentido
  después de agotar las palancas de señal que el propio proyecto señala (más
  histórico, variables de exposición/población), y como experimento de
  notebook, no como reemplazo del pipeline.

---

## 5. TFT / N-BEATS — series temporales

**Qué aportaría.** TFT: forecast multi-horizonte interpretable con atención y
variables estáticas/conocidas/futuras. N-BEATS: forecast univariante por
descomposición residual en bloques.

**Decisión previa (a respetar):** descartados en dos sitios —
`../arquitectura/diseño_modelo.md` §7.2 y `alternativas_descartadas.md`:
"**TFT/N-BEATS resolverían un problema que no tenemos: forecast a 5-7 días.**
El usuario necesita el riesgo *hoy* o *mañana*, con precisión e
incertidumbre, no horizontes lejanos" y "la arquitectura seq2seq de TFT y
N-BEATS está diseñada para horizonte múltiple con decodificación
autorregresiva, que aquí sobra. El LSTM híbrido cumple mejor".

**Revisión (2026-08-14):** la decisión se mantiene, con tres motivos que
siguen vigentes:
1. **Producto.** El usuario consulta el riesgo actual/próximo (la app usa
   Open-Meteo con `FORECAST_HORIZON_DAYS` de 1-2 días). No existe ningún
   flujo que pida "riesgo del domingo" en el producto.
2. **Fit técnico.** El problema es clasificación del día siguiente con
   features tabulares + secuencia de 24 h, no generación de secuencias. La
   LSTM híbrida (`lstm_hibrida.md`) ya combina ambas dimensiones y queda a
   0.011 de `Rec_riesgo` del RandomForest en frío — la dimensión temporal
   está cubierta. N-BEATS además es univariante puro, y el argumento que
   descartó ARIMA/SARIMA (problema multivariante, `alternativas_descartadas.md`)
   aplica igual.
3. **Coste-beneficio.** Los papers scouteados (`../modelos/nbeats/`,
   `../modelos/transformers/`) son forecast de demanda/meteorología, no
   clasificación de riesgo sanitario. Integrar TFT exige una librería nueva
   (PyTorch Forecasting) y un pipeline de secuencias multi-horizonte, para
   un problema que no existe.

**Veredicto: no viable.** Confirmar la decisión de §7.2. **Condición de
re-apertura única:** si el producto añade planificación multi-día ("¿qué
riesgo habrá en mi ruta del domingo?"), re-evaluar TFT como **generador de
features de forecast** para el clasificador, no como sustituto — y aun así
primero comparar contra el forecast gratuito de Open-Meteo que ya se usa.

---

## 6. RL — de predecir a reducir riesgo

**Qué aportaría.** Pasar de *avisar* a *intervenir*: un agente que recomienda
acciones (hidratarse, buscar sombra, cambiar la hora, acortar la duración)
optimizando una recompensa acumulada (riesgo reducido a lo largo del tiempo).

**Cómo encajaría.** Hoy la "reducción de riesgo" ya existe por vía
determinista: `recomendaciones.py` genera recomendaciones y
`contrafactuales.md` cuantifica cuánto baja el riesgo con cada cambio de
comportamiento. No hay un agente aprendido, pero **sí hay una política**.

**Decisión previa (a respetar):** `alternativas_descartadas.md` descartó el
RL híbrido porque "el tiempo de entrenamiento para converger a una política
estable era demasiado alto, y el espacio de estados/acciones requería un
simulador realista de la interacción usuario-entorno que no existía".
`../modelos/README.md` añade el argumento decisivo: "sin bucle de feedback
real, RL es una solución en busca de problema". El prototipo de PPO usaba un
entorno sintético con reglas escritas a mano — la política aprendida no puede
ser mejor que esas reglas, que son las que ya generan las recomendaciones.

**Riesgos.**
- **Sin señal de recompensa.** No hay feedback de usuarios (¿siguió la
  recomendación? ¿mejoró el resultado?). Un RL entrenado en simulador sintético
  hereda las suposiciones del simulador y no añade información al mundo real.
- **Coste y mantenimiento.** Entorno + entrenamiento + despliegue de una
  política es el cambio más caro de los seis, para un beneficio que la
  heurística actual ya entrega.

**Veredicto: no viable ahora.** Confirmar la decisión previa.
**Condición de re-apertura:** cuando el sistema acumule histórico real de
(recomendación → acción del usuario → resultado), el paso natural no es RL
completo sino **contextual bandits** sobre un espacio pequeño de acciones
(cambiar hora, acortar duración) con un proxy de recompensa explícito — y aun
así, validar primero contra la política determinista actual como baseline.

---

## Decisiones previas respetadas

| Decisión previa | Dónde | Este documento |
|-----------------|-------|----------------|
| TFT/N-BEATS descartados (no se necesita forecast 7 días) | `../arquitectura/diseño_modelo.md` §7.2, `alternativas_descartadas.md` | Confirma (sección 5), con condición de re-apertura |
| GNN sobre grafo de factores no mejora (~20 nodos, relaciones conocidas) | `alternativas_descartadas.md` | Confirma (sección 4, uso A) |
| GP clásico O(n³) inviable; SVGP viable pero conformal ya cubre | `../modelos/README.md` | Confirma (sección 3) |
| Bayes jerárquico + EVT descartado por coste de migración probabilística | `alternativas_descartadas.md` | Coherente (sección 2) |
| RL sin bucle de feedback real = solución en busca de problema | `alternativas_descartadas.md`, `../modelos/README.md` | Confirma (sección 6) |
| Causalidad como marco mental, no como librería | `../modelos/README.md` | Coherente (sección 2) |
| Contrafactuales ya implementados (reducir riesgo) | `contrafactuales.md` | Usado como evidencia (secciones 2 y 6) |
| Techo de señal/etiqueta, no de modelo | `conclusiones_modelos.md` §6 | Usado como criterio (secciones 4) |

## Qué NO se ha evaluado aquí

- La viabilidad **técnica de integración** (código) de cada técnica: este
  estudio es de viabilidad de producto/modelo. Si una técnica se aprueba, el
  siguiente paso es un experimento en notebook siguiendo el flujo de
  `../modelos/README.md`.
- Priorización temporal: la única con hueco real y coste bajo (HMM) queda a
  la cola de las palancas de señal ya identificadas en `conclusiones_modelos.md` §6.
