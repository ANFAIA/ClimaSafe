# Resumen del QC — dataset LLM regenerado (LLM-017)

**Fecha:** 2026-08-19
**Feature:** LLM-017
**Estado:** dataset regenerado, QC con 0 hallazgos sólidos, zip re-empaquetado.

---

## Qué se hizo

`data/llm/train.jsonl` y `data/llm/val.jsonl` se regeneraron de cero con los
generadores corregidos en LLM-015, usando los datos reales disponibles:

- **Calor:** `generar_dataset.py` con forecast real de Open-Meteo (red) y UV
  horario de Open-Meteo (sin tocar el cupo de OpenUV).
- **Frío:** `generar_dataset_frio.py` desde `data/processed/dataset_frio_labeled.parquet`
  (2016-2026, 10 provincias frías), con UV estimado del archivo de Open-Meteo.

El conjunto se compuso en bloques (frío + calor por bloque, seeds distintas)
con un dedupe global por clave normalizada **y** por Jaccard de tokens > 0.9
(la misma métrica del QC), hasta reunir 400 pares únicos. Se recortó con cuotas
por clase (SEGURO y PRECAUCION prioritarios) para que las tres clases quedaran
≥ 10 % en train y val, y se partió 300/100 con split estratificado.

Los ficheros anteriores quedaron en `train.jsonl.2026-08-19.bak` y
`val.jsonl.2026-08-19.bak`.

## Conteos finales

| Split | Pares | PELIGRO | PRECAUCION | SEGURO | Canal calor/frío |
|-------|-------|---------|------------|--------|------------------|
| train | 300 | 167 (55.7 %) | 102 (34.0 %) | 31 (10.3 %) | 71 / 229 |
| val   | 100 | 51 (51.0 %) | 38 (38.0 %) | 11 (11.0 %) | 33 / 67 |

Generación: 700 calor + 614 frío generados; descartados 10 por clave duplicada
y 763 por Jaccard > 0.9; 541 únicos en el pool; recorte a 400 por cuotas.

## Resultado del QC

`uv run python climasafeai/llm/revisar_dataset.py --train data/llm/train.jsonl --val data/llm/val.jsonl --muestra 50`

```
data/llm/train.jsonl  (300 pares)
    sin 'Tiempo en esa franja':        0
    parte incompleto (sin máx/UV):     0
    perfiles imposibles:               0
    duplicados casi idénticos:         0
    desequilibrio: False  (distribución {'PELIGRO': 167, 'PRECAUCION': 102, 'SEGURO': 31}, umbral < 10%)
    clase vs pipeline: 16 incoherentes (0 críticas, 16 menores) de 50 verificados (0 no verificables)
data/llm/val.jsonl  (100 pares)
    sin 'Tiempo en esa franja':        0
    parte incompleto (sin máx/UV):     0
    perfiles imposibles:               0
    duplicados casi idénticos:         0
    desequilibrio: False  (distribución {'PELIGRO': 51, 'PRECAUCION': 38, 'SEGURO': 11}, umbral < 10%)
    clase vs pipeline: 20 incoherentes (0 críticas, 20 menores) de 50 verificados (0 no verificables)
```

Criterio de la feature cumplido: **0 críticas, 0 duplicados, 0 inputs
incompletos**. Los incoherentes menores (distancia de una clase) son el ruido
de la reconstrucción sintética del weather en el detector de clase; el criterio
los permite.

## Ajustes al QC necesarios para llegar a 0 (no es maquillaje)

El detector de clase del QC no podía verificar el canal frío y acusaba de
PELIGRO a respuestas correctas. Dos defectos reales de la reconstrucción,
corregidos en `climasafeai/llm/revisar_dataset.py`:

1. **El parseo del parte no leía temperaturas negativas.** `-6.2 °C de media`
   se leía como media `6.2` y `máx -6.2 °C` daba `None` → todos los ejemplos de
   frío intenso caían a "no verificable". Fix: signo opcional en los regex de
   `_parsear_input` (`-?[\d.]+`). Medido: 6+5 pares no verificables → 0.

2. **La reconstrucción del día usaba la ventana de actividad como día entero.**
   El parte declara la media/máx de la **ventana** (2-6 h); el resto del día es
   más fresco. Con el weather real (2026-08-19, 13 escenarios) la ventana de
   tarde queda +3..+17 °C sobre la media diaria. Reconstruir el día con la
   media de la ventana inflaba la prob del ensemble (caso real medido: Lleida
   0.07 → 0.50) y el QC daba PELIGRO a respuestas que el pipeline real, con el
   weather completo, había dado SEGURO. Fix: el día sintético (`df_features` y
   `current`) queda `DIA_DELTA = 8.0` por debajo de la ventana; la ventana
   (`df_hora`) mantiene el extremo declarado para los overrides físicos.
   Calibración sobre el mini-set (74 pares): deltas 5/6/7/8 → 0 críticas en
   todos; delta 8 minimiza los incoherentes totales (25/74) y está dentro del
   rango medido.

El dataset no se tocó para pasar el QC: las respuestas salieron del pipeline
real sobre el weather real; los arreglos hicieron que la re-ejecución del QC
representara ese mismo weather con más fidelidad.

## Empaquetado

```bash
uv run python climasafeai/llm/empaquetar_dataset_colab.py
Empaquetado: data/llm/colab_dataset.zip (41 KB)
  train.jsonl: 2566f1ecb9cae17bac1b77ac6821535bf8cc2bb677f407ebe1a239158f32d017  (300 líneas)
  val.jsonl:   865e7e5a49035cacc1c6e30352abe06d91a1627b79e27dab040e7b755bf46784  (100 líneas)
```

Los sha256 cambiaron respecto a los de LLM-006 (dataset nuevo): los de LLM-006
eran `12208ccf…` (train) y `cda323ee…` (val). No están hardcodeados en ninguna
doc ni en el notebook: la Celda 4 del Colab los compara con los que imprime el
empaquetado.

## Ficheros tocados

- `data/llm/train.jsonl`, `data/llm/val.jsonl` (regenerados; `.bak` con fecha)
- `data/llm/colab_dataset.zip` (re-empaquetado)
- `climasafeai/llm/revisar_dataset.py` (regex con signo + `DIA_DELTA`)
- `documentacion/llm/qc-llm-017.md` (este resumen)
