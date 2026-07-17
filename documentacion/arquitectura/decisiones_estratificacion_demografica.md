# Decisiones de estratificación demográfica

## 1. El problema

El modelo predice **riesgo poblacional**: probabilidad de que un día tenga
mortalidad excesiva para el CONJUNTO de la población. Pero este riesgo no
es uniforme — un anciano de 85 años tiene un riesgo muy superior al de un
joven de 25 ante las mismas condiciones meteorológicas.

El sistema necesita **personalizar el riesgo por edad** sin tener datos
individuales de mortalidad (solo agregados por provincia/día).

---

## 2. Arquitectura actual (Julio 2026)

### 2.1 Modelo único "todos"

Se entrena **un solo modelo** XGBoost (calor) y RandomForest (frío) sobre
el conjunto completo de la población ("todos"). La salida son 3 clases:
SEGURO/PRECAUCION/PELIGRO basadas en percentiles de mortalidad agregada.

### 2.2 Ajuste post-hoc por edad

En inferencia, la probabilidad del modelo se ajusta con un **ratio
demográfico estático**:

```
ratio_estrato = muertes_atribuibles_del_estrato / muertes_atribuibles_totales
               (por provincia, sobre TODO el histórico)
```

El ratio se aplica en espacio de odds:

```python
odds_adj = odds_todos * ratio_estrato
prob_ratio = odds_adj / (1 + odds_adj)
```

### 2.3 Personalización individual

Además del ratio demográfico, el usuario puede tener factores de
personalización (actividad física, comorbilidades, etc). Se aplican
sobre la probabilidad BASE (sin ratio):

```python
prob_final = prob_ratio                  # sin factores
           = max(prob_ratio, prob_pers)  # con factores
```

### 2.4 Umbrales y distribución por estrato

Cada estrato tiene su propia:
- **p2_frac** = n_peligro / (n_precaucion + n_peligro) — qué fracción
  de los días de riesgo son PELIGRO vs PRECAUCION
- **t1, t2** = umbrales de decisión calibrados por recall en test del
  estrato

Guardados en `models/artifacts/params_estrato.joblib`.

### 2.5 Valores actuales

| Estrato | p2_frac (calor) | t1 | t2 | ratio Madrid |
|---------|-----------------|----|----|-------------|
| joven   | 0.000000 | 0.95 | 0.90 | 0.0019 |
| adulto  | 0.000055 | 0.95 | 0.90 | ~0.05 |
| mayor   | 0.003634 | 0.55 | 0.50 | ~0.10 |
| anciano | 0.014459 | 0.40 | 0.20 | ~0.12 |
| viejano | 0.064953 | 0.35 | 0.025 | 0.111 |
| todos   | 0.300000 | 0.50 | 0.40 | 1.0 |

---

## 3. Decisiones tomadas y su racional

### Decisión 1: Modelo único vs modelos por estrato

**Opción evaluada**: Entrenar un modelo separado para cada grupo de edad.

**Problema**: La señal de mortalidad en jóvenes y adultos es tan escasa
que los modelos no aprenden nada (siempre SEGURO). Solo viejano tiene
señal suficiente, y anciano/mayor tienen señal marginal.

**Decisión**: Modelo único "todos" + ajuste post-hoc con ratio. Mejor
relación señal/ruido.

**Coste**: El ratio es una constante estática. No captura que en días
EXTREMOS la proporción de ancianos en las muertes sube muy por encima
de su media histórica.

### Decisión 2: Ratio en espacio de odds vs probabilidad lineal

**Racional**: La mortalidad no responde linealmente a la temperatura.
Aplicar el ratio en odds-space (logit) mantiene la forma sigmoidea de
la relación riesgo-temperatura, mientras que aplicarlo linealmente
comprime las probabilidades de forma incorrecta para valores extremos.

**Decisión**: `odds_adj = odds_todos * ratio`.

### Decisión 3: max(ratio, personalización) condicional

**Problema original**: max() incondicional hacía que un joven SIN
factores recibiera la misma probabilidad que la población general
(prob_ratio → 0.0006, prob_todos → 0.3, max = 0.3).

**Decisión**: Sin factores → prob_ratio (ajuste demográfico puro).
Con factores → max(ratio, personalizada) para no anular casos como
maratón.

### Decisión 4: Umbrales por estrato en vez de globales

**Racional**: La distribución PELIGRO/PRECAUCION es muy distinta por
edad. Con el split global 70/30, PELIGRO era imposible incluso para
viejano en días extremos (necesitaba prob > 1.33).

**Decisión**: p2_frac, t1, t2 calculados por estrato desde las labels
de MoMo (con min_mortalidad_peligro=2.0). Guardados en
`params_estrato.joblib`.

### Decisión 5: Incluir clase raw del modelo en el ensemble

**Racional**: El modelo "todos" predice 3 clases directamente.
El _prob_a_clase con umbrales de estrato puede ser más restrictivo
(ej: joven t1=0.95 → SEGURO incluso tras personalización). Incluir
ambas señales permite que el ensemble elija la más alta.

---

## 4. Alternativas investigadas (con papers)

### 4.1 Multicalibración

**Paper**: Hebert-Johnson et al. 2018 ("Multicalibration"), Hansen et al.
2024 NeurIPS ("When is Multicalibration Post-Processing Necessary?")

**Qué es**: Algoritmo post-hoc que ajusta las predicciones para que
estén calibradas en CADA subgrupo definido (edad, provincia, etc).
Disponible como `pip install multicalibration`.

**Ventaja sobre lo actual**: Soluciona la calibración por edad sin
necesidad de ratio ni umbrales manuales. El modelo
"todos" + multicalibración produce probabilidades calibradas para
cada estrato automáticamente.

**Desventaja**: Es binario (riesgo sí/no). Habría que adaptarlo a 3
clases o binarizar el problema.

**Estado**: No implementado. Pendiente de evaluación.

### 4.2 Transfer Learning Estructurado

**Paper**: CORE-Cox (Yu et al. 2026, arXiv:2605.15633)

**Qué es**: Modelo fuente entrenado con muchos datos ("todos") + 
adaptación residual para cada subgrupo objetivo (cada estrato).
Aprende una estructura compartida de bajo rango y permite
especialización por grupo con pocos datos.

**Ventaja**: Los estratos con poca señal (joven, adulto) heredan
conocimiento del modelo fuente sin overfitting.

**Estado**: No implementado. Requeriría reescribir la arquitectura.

### 4.3 Bayes Jerárquico + Extreme Value Theory

**Papers**: 
- Mortality-Duration-Frequency (Ouarda & Charron 2024)
- Hierarchical Bayesian small-area mortality (Dwyer-Lindgren 2018)

**Qué es**: Modelo bayesiano donde cada estrato tiene su propio riesgo
que se contrae hacia la media poblacional (shrinkage). Extreme Value
Theory modela la cola de la distribución (los días extremos).

**Ventaja**: Maneja incertidumbre explícitamente. Captura que el ratio
cambia en eventos extremos.

**Estado**: No implementado. Requeriría migrar a PyMC/Stan.

### 4.4 Edad como feature (próximo paso)

Ver sección 5. Es la implementación en curso.

### 4.5 Ratio dinámico

**Qué es**: Calcular el ratio demográfico SOLO sobre los días extremos
(top 10% de mortalidad), no sobre todo el histórico.

**Ventaja**: Captura que en olas de calor la proporción de ancianos
en las muertes es mucho mayor que la media histórica. Mínimo cambio
de código sobre lo actual.

**Estado**: No implementado. Pendiente de evaluación tras edad-como-feature.

---

## 5. Próxima implementación: edad como feature

### 5.1 Motivación

El ratio estático no captura interacciones no lineales entre edad y
clima. Un modelo que vea `grupo_edad` como feature puede aprender
directamente que 35°C con humedad = PELIGRO para viejano pero SEGURO
para joven.

### 5.2 Diseño

1. **Datos de entrenamiento**: Se crea una fila por (provincia, día,
   grupo_edad) en lugar de una por (provincia, día). El tiempo
   meteorológico es el mismo para todos los grupos en una provincia/día;
   la mortalidad y la etiqueta de riesgo varían por grupo.

2. **Feature**: `grupo_edad` como variable categórica (one-hot o
   ordinal). Interacciones explícitas: `t2m_c * es_viejano`,
   `heat_index_c * es_viejano`, etc.

3. **Modelo**: XGBoost entrena sobre este dataset enriquecido. Aprende
   splits del tipo "si grupo_edad=joven → SEGURO, si grupo_edad=viejano
   Y heat_index>35 → PELIGRO".

4. **Inferencia**: Se pasa el `grupo_edad` del usuario como feature.
   El modelo predice el riesgo específico para ese grupo sin necesidad
   de ratio post-hoc.

### 5.3 Cambios en el código

| Archivo | Cambio |
|---------|--------|
| `make_dataset.py` | Nueva función `dataset_calor_edad()` / `dataset_frio_edad()` |
| `build_features.py` | Añadir `grupo_edad` a las features permitidas, codificación one-hot |
| `train_model.py` | Usar dataset edad-estratificado para entrenar |
| `ensemble.py` | Pasar `grupo_edad` como feature durante inferencia |

### 5.4 Riesgos

- **Multiplicación de filas**: 4 estratos × filas actuales = ~4M filas.
  XGBoost escala bien, pero el tiempo de entrenamiento aumenta.
- **Dominancia de grupos mayoritarios**: El modelo podría ignorar
  los grupos pequeños si no se pondera. Usar `scale_pos_weight`
  o sample weights por estrato.
- **Co-linealidad**: `grupo_edad` correlaciona con `provincia`
  (distinta composición demográfica). Posible pero manejable.

---

## 6. Línea temporal de decisiones

| Fecha | Decisión | Estado |
|-------|----------|--------|
| 2026-07-14 | Modelos por estrato (4 modelos) | ✅ probado, descartado |
| 2026-07-14 | Modelo único + ratio estático | ✅ implementado |
| 2026-07-15 | max(ratio, personalización) sin condición | ❌ bug detectado |
| 2026-07-16 | Umbrales por estrato (params_estrato.joblib) | ✅ implementado |
| 2026-07-17 | max() condicional (factores → max, no factores → ratio) | ✅ implementado |
| 2026-07-17 | Incluir clase raw del modelo en ensemble | ✅ implementado |
| 2026-07-17 | **Edad como feature** | 🚧 en curso |
