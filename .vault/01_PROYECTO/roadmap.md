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
- LSTM province_hybrid integrada con peso_riesgo_extra=8.0
- Rec_riesgo: LSTM Calor **0.7367**, Frío **0.7082** (supera a tabulares)
- Thresholds separados por tipo de modelo (LSTM vs tabulares)
- 15 agentes auto-registrados con CLI y orquestador
- Vault Obsidian actualizado con documentación de todos los modelos
- Graphify actualizado con última versión del código
- **Ensemble 4 modelos** (XGBoost + RF + LSTM + fórmula) con guardarraíl físico y override
- **Explicabilidad**: SHAP + FEATURE_NAME_MAP (31 entradas) + HI ventana de actividad
- **Recomendaciones**: filtradas por riesgo dominante (calor/frío/ambos)
- **Fusión LSTM**: `lstm_hybrid.py` + `lstm_province.py` → `lstm_province_hybrid.py`
- **Open-Meteo inlined** en `weather_fetcher.py` con perfil_horario
- **Personalización**: `producto_bruto` y `capado` visibles en output
- **Documentación**: sección 7 en `diseño_modelo.md` con modelos avanzados evaluados y descartados (WeatherNext, TimesFM, TTM-R3, Prithvi-EO) — solo open-source gratuito

## Pendiente 🧱
- [ ] FastAPI para servir predicciones
- [ ] Docker + docker-compose
- [ ] Monitoring avanzado en producción