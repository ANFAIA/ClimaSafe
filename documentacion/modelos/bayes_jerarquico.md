# Modelo bayesiano jerárquico por provincia — BAYES-001

Implementación: `climasafeai/models/bayes_jerarquico.py` (nuevo; no toca
`bayes.py`, que es la red bayesiana discreta con pgmpy para diagnóstico
inverso y no comparte propósito).

## Qué es

Regresión logística **ordinal acumulativa** de 3 clases
(0=SEGURO, 1=PRECAUCIÓN, 2=PELIGRO) con **efectos aleatorios por provincia**
(partial pooling):

```
score_i   = X_i·β + u_{prov[i]}            u_j ~ Normal(0, τ), τ ~ HalfNormal(0,1)
P(y_i≤k)  = σ(c_k − score_i)   k = 1, 2    c_1 < c_2
β_m       ~ Normal(0, 2)
```

`u_j` deja que cada provincia tenga su propia curva **informada por la
distribución nacional**: las provincias con pocos episodios se encogen hacia
la media en vez de sobreajustar (el problema que motivó la feature: hoy las
provincias pequeñas comparten los mismos parámetros que Madrid o Sevilla).

## Sampler: Metropolis-Hastings propio (sin pymc)

**Decisión documentada:** no se instaló pymc. Para este modelo (~50
parámetros) numpy/scipy bastan; pymc arrastra pytensor como dependencia y el
proyecto valora cero dependencias innecesarias. El muestreo es MH por
bloques (β por coordenada, c1, δ=log(c2−c1), u, log τ) con escalas de
propuesta adaptativas (objetivo 0.234) e **inicialización frecuentista**
(LogisticRegression multinomial sobre features + dummies de provincia) para
arrancar las cadenas cerca del modo.

**Divergencias:** las divergencias son un diagnóstico específico de
NUTS/HMC; **no existen en MH**. La métrica equivalente de salud de
transiciones es la **tasa de aceptación por bloque** (se reporta); las de
convergencia global son **r_hat** (Gelman-Rubin, entre cadenas) y **ESS**
(Geyer, autocorrelación), ambas reportadas en `resumen_muestreo()`.

## Datos de provincia

Los CSVs `X_train_*.csv` **no** llevan provincia (drop deliberado en
`COLS_TO_DROP`, sesgo geográfico — build_features.py:45). La provincia se
reconstruye desde los **parquets labelizados**
`data/processed/dataset_{calor,frio}_labeled.parquet` (172395×39 con
fecha+provincia+clase_riesgo), única fuente que la conserva por fila:
`cargar_datos_entrenamiento(clase)`. La etiqueta ya es relativa a cada
provincia (labels.py `por_provincia=True`), coherente con modelar un efecto
por provincia.

## Partición temporal y comparativa con el ensemble

- Split **por fechas** (test = último 20% de fechas distintas), el mismo
  criterio que `preprocess_data(split_by_date=True)` — mismas particiones
  para el jerárquico y el baseline.
- Baseline = componente tabular del ensemble (**XGBoost** para calor,
  **RandomForest** para frío; los mismos modelos e hiperparámetros de
  `temporal_cv._build_models_cv`) **reentrenado en la misma partición** con
  las **mismas features**. No se comparan los joblib de `models/` porque
  vieron todo el histórico: compararlos sobre un test temporal sería fuga.
- La demo entrena sobre una **submuestra estratificada por provincia de
  15000 filas** del train temporal: la posterior con 137k filas es tan
  picuda que MH no converge con r_hat razonable. El baseline se entrena en
  la misma submuestra — comparación justa.
- La LSTM y la Fórmula no se reentrenan en la comparativa (no se pueden
  reentrenar sobre particiones arbitrarias); el criterio 2 se cumple contra
  el componente tabular dominante del ensemble, documentado como limitación.

## Resultados (demo sobre datos completos, 2026-08-18)

Reproducible con:

```bash
.venv/bin/python -m climasafeai.models.bayes_jerarquico --clase calor
.venv/bin/python -m climasafeai.models.bayes_jerarquico --clase frio
```

Tablas guardadas en `reports/comparativa_bayes_vs_ensemble_{calor,frio}.csv`
y `reports/comparativa_pocos_muchos_{calor,frio}.csv`.

### Calor (vs XGBoost)

| Métrica (test temporal, 34470 filas) | XGBoost | Jerárquico |
|---|---|---|
| F1 macro | 0.4850 | **0.5484** (+0.063) |
| Brier | 0.0571 | **0.0454** |
| Accuracy | 0.8815 | 0.8616 |

Provincias con pocos episodios (22 provincias): ganancia F1 **+0.0919**.
Provincias con muchos (23): +0.0549. **El beneficio es mayor donde menos
datos hay**, que es la razón de ser del modelo.

### Frío (vs RandomForest)

| Métrica (test temporal, 34470 filas) | RandomForest | Jerárquico |
|---|---|---|
| F1 macro | 0.4562 | **0.4779** (+0.022) |
| Brier | 0.0944 | **0.0476** |
| Accuracy | 0.7758 | **0.8998** |

En frío el beneficio de F1 **no** aparece en pocos_datos (−0.035 vs +0.065
en muchos_datos), aunque el Brier mejora mucho en ambos grupos. La ventaja
del frío es de **calibración**, no de ordenación de clases.

### Cobertura del intervalo (limitación honesta)

La cobertura del intervalo 90% de `prob_riesgo` sobre la clase binaria real
es 0.37 (calor) y 0.29 (frío): **el intervalo de credibilidad de la
probabilidad NO es un intervalo predictivo calibrado de la clase**. El
intervalo expresa incertidumbre sobre la probabilidad posterior, no sobre
el valor observado. Si se quisiera usar como banda predictiva habría que
recalibrarlo (p.ej. ensancharlo por conformal), como ya hace el ensemble con
su banda semanal.

## Traducción a SEGURO/PRECAUCIÓN/PELIGRO

La clase puntual sale de la **misma cascada que el ensemble**
(`apply_class_thresholds` en predict_model.py con
`CLASS_THRESHOLDS_RECOMENDADOS`): `P(2) ≥ t2 → PELIGRO`;
`P(1)+P(2) ≥ t1 → PRECAUCIÓN`; si no → `SEGURO`. Aplicada a las medianas
posteriores de cada clase. `predecir()` además marca `clase_incierta` cuando
los intervalos de dos clases se solapan (la mediana sola no decide).

## Decisión: entra en el ensemble, lo sustituye, o contraste — **CONTRATE**

**Queda decidido: el modelo jerárquico NO entra en el ensemble ni lo
sustituye. Se queda como modelo de contraste.** Motivos, con los números de
la demo:

1. **No mejora consistentemente**: en calor gana F1 (+0.063) y en frío
   apenas (+0.022), y en frío pierde en provincias con pocos episodios.
2. **Cobertura del intervalo pobre** (0.29-0.37 en vez de 0.90): no se puede
   prometer la incertidumbre como intervalo predictivo sin recalibrar.
3. **predict_ensemble está desplegado** y funciona; integrar el jerárquico
   (que compone probabilidades puntuales) cambiaría el contrato de salida
   sin ganancia clara.
4. El ensemble actual tampoco necesita el "pooling": su baseline ya se
   beneficia de las 27/19 features del pipeline completo, que el jerárquico
   no usa (5 features para que MH converja).

**Vía de entrada documentada** si en el futuro se quiere su incertidumbre:
(a) recalibrar el intervalo (ensanchado conformal) y usarlo en la banda de
`prediccion_semanal`, o (b) añadir su `prob_riesgo` como miembro del
ensemble con peso `1/conformal_set_size`, igual que los demás. Requisito
previo para (b): resolver la convergencia con el train completo (más
warmup, HMC/NUTS o variacional) o aceptar la submuestra como base de
entrenamiento.

## Limitaciones

- La comparativa es contra el componente tabular del ensemble, no contra la
  LSTM ni la Fórmula (no reentrenables sobre particiones arbitrarias).
- La demo usa submuestra de train (15000 filas) por convergencia del MH; los
  joblib de `models/` no se usan en la comparativa (fuga temporal).
- r_hat mayoritario < 1.1 en la demo, pero algunos efectos provinciales
  llegan a ~2.5 (Barcelona/calor, Cádiz/frío): conviene más warmup o NUTS
  antes de usar los intervalos en producción.
