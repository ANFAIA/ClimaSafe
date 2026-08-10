#! /usr/bin/env python
"""
ClimaSafeAI — Fine‑tuning LoRA de Qwen 2.5 7B con Unsloth.

Uso:
    # Entrenar
    python climasafeai/llm/fine_tune.py \
        --model qwen2.5-7b \
        --train-file data/llm/train.jsonl \
        --val-file data/llm/val.jsonl \
        --output-dir models/llm/qwen-climasafe-lora

    # Exportar LoRA → GGUF
    python climasafeai/llm/fine_tune.py \
        --export-gguf \
        --lora-path models/llm/qwen-climasafe-lora \
        --gguf-path models/llm/qwen-climasafe-q4_k_m.gguf

    # Solo evaluar
    python climasafeai/llm/fine_tune.py \
        --eval-only \
        --lora-path models/llm/qwen-climasafe-lora \
        --val-file data/llm/val.jsonl

Requisitos: Python 3.10–3.11, CUDA 12.1+, Unsloth instalado.
Correr en entorno aislado (conda create -n unsloth python=3.10 ...).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MODEL_NAMES = {
    "qwen2.5-7b": "unsloth/Qwen2.5-7B",
    "qwen2.5-1.5b": "unsloth/Qwen2.5-1.5B",
    "qwen2.5-7b-instruct": "unsloth/Qwen2.5-7B-Instruct",
    "qwen2.5-1.5b-instruct": "unsloth/Qwen2.5-1.5B-Instruct",
}

DEFAULT_LORA_RANK = 16
# Medido sobre data/llm/*.jsonl con el tokenizer de Qwen2.5: 150 ejemplos, mediana
# 412 tokens, máximo 562, p95 509. Ninguno pasa de 1024, así que 4096 solo servía
# para reservar memoria de más — y la VRAM es justo lo que falta aquí.
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRADIENT_ACCUM = 4
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine‑tuning LoRA de Qwen con Unsloth")

    # Modo
    mode = p.add_argument_group("Modo de ejecución")
    mode.add_argument("--export-gguf", action="store_true",
                      help="Exportar LoRA entrenado a GGUF y salir")
    mode.add_argument("--check", action="store_true",
                      help="Solo comprobar si el entorno permite entrenar, y salir")
    mode.add_argument("--eval-only", action="store_true",
                      help="Evaluar LoRA contra val-set y salir")

    # Datos
    data = p.add_argument_group("Datos")
    data.add_argument("--model", default="qwen2.5-7b",
                      choices=list(MODEL_NAMES.keys()),
                      help="Modelo base (default: qwen2.5-7b)")
    data.add_argument("--train-file", default="data/llm/train.jsonl",
                      help="JSONL de entrenamiento (instruction/input/output)")
    data.add_argument("--val-file", default="data/llm/val.jsonl",
                      help="JSONL de validación")

    # LoRA
    lora = p.add_argument_group("LoRA")
    lora.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK,
                      help=f"Rango LoRA (default: {DEFAULT_LORA_RANK})")
    lora.add_argument("--lora-path",
                      help="Ruta al checkpoint LoRA (para export/eval)")

    # Salida
    out = p.add_argument_group("Salida")
    out.add_argument("--output-dir", default="models/llm/qwen-climasafe-lora",
                     help="Directorio donde guardar el adaptador LoRA")
    out.add_argument("--gguf-path", default="models/llm/qwen-climasafe.gguf",
                     help="Ruta del GGUF de salida")

    # Hiperparámetros
    hp = p.add_argument_group("Hiperparámetros")
    hp.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help=f"Batch por dispositivo (default: {DEFAULT_BATCH_SIZE})")
    hp.add_argument("--gradient-accum", type=int, default=DEFAULT_GRADIENT_ACCUM,
                    help=f"Pasos de acumulación (default: {DEFAULT_GRADIENT_ACCUM})")
    hp.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                    help=f"Épocas (default: {DEFAULT_EPOCHS})")
    hp.add_argument("--lr", type=float, default=DEFAULT_LR,
                    help=f"Learning rate (default: {DEFAULT_LR})")
    hp.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN,
                    help=f"Contexto máximo en tokens (default: {DEFAULT_MAX_SEQ_LEN})")
    hp.add_argument("--use-wandb", action="store_true",
                    help="Activar logging con WandB")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _formatear_chat(example: dict) -> list[dict]:
    """Convierte un ejemplo Alpaca a mensajes chat."""
    messages = []
    if example.get("input"):
        messages.append({"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"})
    else:
        messages.append({"role": "user", "content": example["instruction"]})
    messages.append({"role": "assistant", "content": example["output"]})
    return messages


def cargar_dataset(path: str | Path) -> list[dict]:
    """Carga un JSONL de entrenamiento en formato Alpaca."""
    examples = []
    path = Path(path)
    if not path.exists():
        print(f"ERROR: No existe {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"  Dataset: {path} → {len(examples)} ejemplos")
    return examples


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------


def entrenar(args: argparse.Namespace) -> None:
    """Entrena LoRA con Unsloth."""
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from datasets import Dataset as HFDataset

    model_name = MODEL_NAMES[args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Modelo base: {model_name}")
    print(f"  LoRA rank:   {args.lora_rank}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Batch:       {args.batch_size} × {args.gradient_accum} accum")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Output:      {output_dir}")
    print(f"{'='*60}\n")

    # 1. Cargar modelo base + tokenizer
    print("[1/5] Cargando modelo base...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        dtype=None,  # auto-detecta fp16/bf16
        load_in_4bit=True,  # QLoRA: modelo en 4 bits
        device_map="auto",
    )
    # Unsloth no garantiza tokenizer.chat_template y transformers 5.5.0 lo
    # exige para apply_chat_template: aplicamos el de Qwen 2.5.
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # 2. Añadir LoRA
    print("[2/5] Añadiendo LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 3. Preparar dataset
    print("[3/5] Preparando dataset...")
    raw_examples = cargar_dataset(args.train_file)
    formatted = []
    for ex in raw_examples:
        messages = _formatear_chat(ex)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted.append({"text": text})
    dataset = HFDataset.from_list(formatted)

    val_dataset = None
    if args.val_file and Path(args.val_file).exists():
        val_raw = cargar_dataset(args.val_file)
        val_formatted = []
        for ex in val_raw:
            messages = _formatear_chat(ex)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            val_formatted.append({"text": text})
        val_dataset = HFDataset.from_list(val_formatted)
        print(f"  Validación: {len(val_dataset)} ejemplos")

    # 4. Configurar entrenamiento
    print("[4/5] Configurando trainer...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="wandb" if args.use_wandb else "none",
        # BUG-005: sin checkpoints intermedios. Unsloth compila el trainer y el
        # SFTConfig de trl duplicado no se puede picklear; save_strategy="epoch"
        # dispara _save_checkpoint → torch.save(self.args) y peta. El LoRA final
        # se guarda con model.save_pretrained tras entrenar (sin pickle).
        save_strategy="no",
        load_best_model_at_end=False,
        # Eval por epoch sigue activo: solo evalúa, no guarda nada.
        eval_strategy="epoch" if val_dataset else "no",
    )

    trainer = None
    try:
        from trl import SFTTrainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=val_dataset,
            args=training_args,
            max_seq_length=args.max_seq_len,
            dataset_text_field="text",
        )
    except ImportError:
        from transformers import Trainer
        data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_seq_len,
                padding=False,
            )
        tokenized_dataset = dataset.map(tokenize_fn, batched=True)
        if val_dataset:
            tokenized_val = val_dataset.map(tokenize_fn, batched=True)
        else:
            tokenized_val = None

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_val,
            args=training_args,
            data_collator=data_collator,
        )

    # 5. Entrenar
    print("[5/5] Entrenando...")
    trainer.train()

    # Guardar LoRA
    print(f"\nGuardando LoRA en {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("¡Fine‑tuning completado!")


# ---------------------------------------------------------------------------
# Exportar a GGUF
# ---------------------------------------------------------------------------


def exportar_gguf(args: argparse.Namespace) -> None:
    """Fusiona LoRA con el modelo base y exporta a GGUF."""
    lora_path = Path(args.lora_path)
    gguf_path = Path(args.gguf_path)

    if not lora_path.exists():
        print(f"ERROR: No existe el LoRA en {lora_path}", file=sys.stderr)
        sys.exit(1)

    gguf_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LoRA:  {lora_path}")
    print(f"  GGUF:  {gguf_path}")
    print(f"{'='*60}\n")

    # Intentar con Unsloth (carga modelo base + fusión automática)
    try:
        from unsloth import FastLanguageModel

        print("[1/3] Cargando modelo base con LoRA...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,  # Unsloth carga base + LoRA si el path tiene adapter_config
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=False,  # Necesitamos precisión completa para export
            device_map="auto",
        )

        print("[2/3] Fusionando LoRA con el modelo base...")
        model = model.merge_and_unload()

    except Exception as exc:
        print(f"  Error cargando con Unsloth: {exc}", file=sys.stderr)
        print("  Puedes exportar manualmente con llama.cpp/convert.py", file=sys.stderr)
        sys.exit(1)

    # Guardar como GGUF usando llama.cpp
    print("[3/3] Exportando a GGUF...")
    try:
        # Opción A: usar la utilidad de unsloth si existe
        model.save_pretrained_gguf(
            str(gguf_path),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"  ✓ GGUF guardado: {gguf_path}")
        file_size = gguf_path.stat().st_size / (1024**3)
        print(f"  Tamaño: {file_size:.1f} GB")

    except AttributeError:
        # Opción B: guardar en HF format y convertir con llama.cpp
        hf_path = gguf_path.with_suffix("")
        print(f"  Guardando en formato HF en {hf_path}...")
        model.save_pretrained(str(hf_path))
        tokenizer.save_pretrained(str(hf_path))
        print(f"\n  Para convertir a GGUF:")
        print(f"  pip install llama-cpp-python")
        print(f"  python -m llama_cpp.convert --outtype q4_k_m \\")
        print(f"    {hf_path} {gguf_path}")
        print(f"\n  O clona llama.cpp:")
        print(f"  git clone https://github.com/ggml-org/llama.cpp")
        print(f"  python llama.cpp/convert.py {hf_path} --outfile {gguf_path} --outtype q4_k_m")


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------


def evaluar(args: argparse.Namespace) -> None:
    """Evalúa un LoRA contra el conjunto de validación."""
    lora_path = Path(args.lora_path)
    val_path = Path(args.val_file)

    if not lora_path.exists():
        print(f"ERROR: No existe {lora_path}", file=sys.stderr)
        sys.exit(1)
    if not val_path.exists():
        print(f"ERROR: No existe {val_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        print("ERROR: Unsloth no está instalado. Ejecuta en el entorno unsloth.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Evaluando LoRA: {lora_path}")
    print(f"  Validación:     {val_path}")
    print(f"{'='*60}\n")

    # Cargar modelo + LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )
    # Mismo requisito que en entrenar: apply_chat_template necesita
    # chat_template configurado en el tokenizer (transformers 5.5.0).
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)

    # Cargar dataset de validación
    examples = cargar_dataset(val_path)

    # Evaluar pérdida
    import torch
    from datasets import Dataset as HFDataset

    formatted = []
    for ex in examples:
        messages = _formatear_chat(ex)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted.append({"text": text})
    hf_dataset = HFDataset.from_list(formatted)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=4096, padding=False)
    tokenized = hf_dataset.map(tokenize_fn, batched=True)

    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    dataloader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=False,
    )

    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items() if k != "text"}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"\n  Pérdida media:   {avg_loss:.4f}")
    print(f"  Perplejidad:     {perplexity:.2f}")
    print(f"  (en {n_batches} batches)\n")

    # Mostrar algunas generaciones de ejemplo
    print(f"  Ejemplos generados:\n")
    for i, ex in enumerate(examples[:3]):
        messages = _formatear_chat(ex)
        inputs = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

        print(f"  --- Ejemplo {i+1} ---")
        print(f"  INPUT:  {ex['instruction'][:100]}...")
        print(f"  ESPERADO: {ex['output'][:150]}...")
        print(f"  GENERADO: {response[:200]}...")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def comprobar_entorno(model: str = "qwen2.5-7b") -> list[str]:
    """Devuelve la lista de problemas que impiden entrenar. Vacía = se puede.

    Existe porque el fallo por defecto era un `ModuleNotFoundError: unsloth` a los
    tres segundos, que no dice si falta el paquete, el driver o la GPU entera. Aquí
    se comprueban las tres cosas de golpe y se dice cuál falla.
    """
    problemas: list[str] = []

    # 1. Paquetes
    import importlib.util
    faltan = [m for m in ("unsloth", "peft", "trl", "datasets", "bitsandbytes", "accelerate")
              if importlib.util.find_spec(m) is None]
    if faltan:
        problemas.append(
            f"paquetes sin instalar: {', '.join(faltan)}. "
            "Unsloth pide su propio entorno (Python 3.10-3.11 + CUDA), no el .venv del proyecto."
        )

    # 2. GPU utilizable
    try:
        import torch
    except ImportError:
        problemas.append("torch no está instalado")
        return problemas

    if not torch.cuda.is_available():
        detalle = "CUDA no disponible para torch"
        if not Path("/proc/driver/nvidia/version").exists():
            detalle += " (no hay driver NVIDIA cargado: falta /proc/driver/nvidia)"
        problemas.append(
            f"{detalle}. Unsloth y la cuantización en 4 bits NO funcionan en CPU: "
            "sin GPU no hay entrenamiento, ni lento ni rápido."
        )
        return problemas

    # 3. VRAM suficiente para el modelo pedido
    libre = torch.cuda.get_device_properties(0).total_memory / 1024**3
    necesaria = 8.0 if "7b" in model else 4.0
    if libre < necesaria:
        problemas.append(
            f"VRAM insuficiente: {libre:.1f} GB en {torch.cuda.get_device_name(0)}, "
            f"y {model} en QLoRA pide ~{necesaria:.0f} GB. Prueba con --model qwen2.5-1.5b."
        )
    return problemas


def main() -> None:
    args = parse_args()

    if getattr(args, "check", False):
        problemas = comprobar_entorno(args.model)
        if problemas:
            print("No se puede entrenar todavía:")
            for i, p_ in enumerate(problemas, 1):
                print(f"  {i}. {p_}")
            sys.exit(1)
        print("Entorno listo para entrenar.")
        return

    if not (args.export_gguf or args.eval_only):
        problemas = comprobar_entorno(args.model)
        if problemas:
            print("No se puede entrenar todavía:")
            for i, p_ in enumerate(problemas, 1):
                print(f"  {i}. {p_}")
            print("\nEl dataset y el resto del script sí están listos: "
                  "`--check` vuelve a comprobarlo cuando lo arregles.")
            sys.exit(1)

    if args.export_gguf:
        exportar_gguf(args)
    elif args.eval_only:
        evaluar(args)
    else:
        entrenar(args)


if __name__ == "__main__":
    main()
