# Agentes de IA — ClimaSafe

Se han añadido un conjunto de agentes para facilitar tareas de ingeniería
del repositorio (análisis, gestión de Git, documentación, revisión y
soporte para datos). Los agentes implementados en `agents/agents/` son:

- `data_agent` — herramientas para análisis y manipulación de datasets.
- `git_agent` — operaciones automatizadas de git (generar changelog, comprobar diffs).
- `documentation_agent` — sincroniza README/Makefile, genera `CHANGELOG.md` y construye docs.
- `docker_agent` — helpers para construir/ejecutar imágenes y contenedores.
- `graph_agent` — generación y análisis de grafos (depende del submódulo de gráficas).
- `ml_agent` — tareas relacionadas con entrenamiento, evaluación y tuning.
- `review_agent` — asistente para revisar código y generar sugerencias.

Estas incorporaciones permiten automatizar tareas repetitivas y mantener
la documentación y notebooks coherentes con el código. Revisa la carpeta
`agents/` para ver las acciones disponibles y su documentación.

---

## Paper Scout — Búsqueda y clasificación de literatura

`agents/paper_scout.py` busca papers en OpenAlex/arXiv, los clasifica con Gemini
y extrae factores de riesgo estructurados. Ver `documentacion/proximos_pasos.md`
para el roadmap completo.

### Aprendizaje activo (ActiveLearner)

Para evitar llamar a Gemini en cada paper, el scout incorpora un clasificador
ligero que filta irrelevantes antes del LLM.

**Arquitectura:**

```
Paper nuevo (título + abstract)
         │
         ▼
    all-MiniLM-L6-v2 ──→ embedding [384d]
         │
         ▼
    LogisticRegression (entrenada con scikit-learn)
         │
         ├── confianza > 0.75 y "irrelevante" → descartado (sin LLM)
         ├── confianza > 0.75 y "aceptable"   → pasa a Gemini para extraer coef/DOI
         └── confianza < 0.75                 → pasa a Gemini para clasificar
                                                   │
                                                   ▼
                                           Veredicto del LLM
                                                   │
                                                   ▼
                                      Almacenado en scout_entrenamiento (SQLite)
                                                   │
                                                   ▼
                                      retrain() → mejora LogisticRegression
```

**Componentes:**

| Archivo | Rol |
|---------|-----|
| `climasafeai/ml/active_learner.py` | Clase `ActiveLearner`: embed, predict, store, retrain, stats |
| `agents/paper_scout.py` | Integración: filtro previo a `_classify_batch()`, almacena ejemplos post-LLM |
| `scripts/entrenar_active_learner.py` | Entrenamiento inicial desde factores_riesgo + negativos sintéticos |
| Tabla `scout_entrenamiento` (SQLite) | Almacena título, abstract, embedding (blob f32[384]), veredicto, fuente |

**Comportamiento progresivo:**

| Ejecuciones scout | Labels | Comportamiento |
|---|---|---|
| 0 (inicial) | 42 | Filtra ~50% de irrelevantes obvios |
| 1 | ~60-70 | Mejora con los primeros veredictos reales |
| 5+ | ~150+ | Filtra ~70% sin llamar a Gemini |

**Re-entrenar:**
```bash
.venv/bin/python3 scripts/entrenar_active_learner.py
```

**Doc completa:** `documentacion/ml/active_learner.md`

**Ver estado:**
```python
from climasafeai.ml.active_learner import ActiveLearner
print(ActiveLearner().stats())
```

