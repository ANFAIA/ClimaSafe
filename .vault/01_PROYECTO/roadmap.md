---
type: proyecto
created: 2026-07-09
tags:
  - proyecto
  - roadmap
status: draft
---

# Roadmap

## Completado ✅
- Core ML funcional con XGBoost y RandomForest
- Pipeline completo: descarga → preprocesado → entrenamiento (3 modelos) → evaluación → tabla comparativa
- LSTM province_hybrid integrada en `main.py` con peso_riesgo_extra=8.0
- Rec_riesgo: LSTM Calor **0.7367**, Frío **0.7082** (supera a tabulares)
- Thresholds separados por tipo de modelo (LSTM vs tabulares)
- 15 agentes auto-registrados con CLI y orquestador
- Vault Obsidian actualizado con documentación de todos los modelos
- Graphify actualizado con última versión del código (2462 nodos, 4392 aristas)

## Pendiente 🧱
- [ ] FastAPI para servir predicciones
- [ ] Docker + docker-compose
- [ ] Monitoring avanzado en producción