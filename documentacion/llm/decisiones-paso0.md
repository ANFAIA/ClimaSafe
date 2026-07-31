# Paso 0 — Test de Qwen 2.5 raw + Ollama

**Fecha:** 2026-07-30 · **Feature:** LLM-001

## Objetivo

Probar si Qwen 2.5 responde adecuadamente **sin fine-tuning** (solo con prompting)
para el caso de uso de ClimaSafe: convertir datos estructurados de predicción de riesgo
en texto legible en español.

## Instalación

Ollama se instaló desde los binarios precompilados de GitHub:

```bash
# Descargar Ollama v0.32.5
curl -L "https://github.com/ollama/ollama/releases/download/v0.32.5/ollama-linux-amd64.tar.zst" \
  -o ollama.tar.zst
mkdir -p ~/.local/bin && tar --zstd -xf ollama.tar.zst -C ~/.local/bin/
# Iniciar servidor
ollama serve &

# Descargar modelos
ollama pull qwen2.5:7b     # 4.7 GB — para GPU
ollama pull qwen2.5:1.5b   # ~1 GB — para CPU
```

## Resultados (CPU, sin GPU)

| Modelo | Tiempo | Velocidad | Calidad |
|--------|--------|-----------|---------|
| Qwen 2.5 1.5B | 7.8 s | 15.9 tok/s | Buena: clase, factores, contrafactual, español correcto |
| Qwen 2.5 7B | 43.2 s | 3.3 tok/s | Ligeramente mejor pero 5.5× más lento |

Ambos modelos responden en español, identifican correctamente la clase de riesgo
(PELIGRO/PRECAUCIÓN), mencionan los factores que contribuyen y el contrafactual.
Ninguno alucina valores ni inventa fuentes.

## Decisiones de diseño

### 1. Qwen 2.5 1.5B como modelo para CPU, Qwen 2.5 7B para GPU

La diferencia de calidad entre 1.5B y 7B es marginal para esta tarea acotada.
El 7B es ligeramente más preciso pero 5.5× más lento en CPU. Para un bot de
Telegram, 43 segundos por respuesta es inasumible.

**Decisión:** En CPU → Qwen 2.5 1.5B. En GPU → Qwen 2.5 7B.

### 2. Fine-tuning es mejora, no necesidad

El modelo raw con prompting responde correctamente. El fine-tuning (LoRA con
Unsloth) lo haría más conciso y eliminaría pequeñas repeticiones, pero el raw
ya vale como capa intermedia (tier 2 del sistema de tres capas).

### 3. Tres capas de servicio

1. **Qwen fine-tuneado** (LoRA + Unsloth) — mejor experiencia, respuestas más
   naturales y adaptadas al tono del proyecto
2. **Qwen raw con prompting** — funciona sin GPU ni fine-tuning. Suficiente
   para un bot funcional
3. **Bot determinista** (actual) — sin LLM, responde con plantilla. Siempre
   disponible como fallback

El sistema elige automáticamente el mejor modo según disponibilidad del modelo.

## Próximos pasos

- Paso 1: Indexar documentacion/ en el RAG (sqlite-vec)
- Paso 2: Dataset sintético desde conversaciones reales del bot
- Paso 3: Fine-tuning LoRA con Unsloth
- Paso 4: Integración LLM + RAG + bot
- Paso 5: Publicar skill + deploy manifest
