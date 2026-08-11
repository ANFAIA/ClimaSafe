"""
test_fine_tune.py — Tests para climasafeai/llm/fine_tune.py

Regresión BUG-004: transformers 5.5.0 lanza
`ValueError: Cannot use chat template functions because tokenizer.chat_template
is not set` en apply_chat_template. La máquina local no tiene GPU ni unsloth,
así que se mockea todo el entorno y se verifica que tras cada from_pretrained
que alimenta apply_chat_template (entrenar y evaluar) se aplica
get_chat_template(tokenizer, chat_template="qwen-2.5").
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import climasafeai.llm.fine_tune as ft


@pytest.fixture
def env_llm():
    """Entorno mínimo mockeado (torch, unsloth, transformers, datasets, trl).

    El from_pretrained devuelve un tokenizer SIN chat_template (tok_sin_template)
    y get_chat_template lo sustituye por uno con template (tok_con_template).
    Así un apply_chat_template sobre el tokenizer equivocado se detecta.
    """
    model = MagicMock()
    param = MagicMock()
    param.numel.return_value = 10
    param.requires_grad = True
    model.parameters.return_value = [param]

    tok_sin_template = MagicMock()  # lo que devuelve from_pretrained (sin chat_template)
    tok_con_template = MagicMock()  # lo que devuelve get_chat_template

    unsloth = MagicMock()
    unsloth.FastLanguageModel.from_pretrained.return_value = (model, tok_sin_template)
    unsloth.FastLanguageModel.get_peft_model.return_value = model
    unsloth.FastLanguageModel.for_inference.return_value = None

    chat_templates = MagicMock()
    chat_templates.get_chat_template.return_value = tok_con_template

    torch_mock = MagicMock()
    torch_mock.utils.data.DataLoader.return_value.__iter__.return_value = iter(
        [{"input_ids": MagicMock(), "attention_mask": MagicMock()}]
    )
    model.return_value.loss.item.return_value = 0.5  # outputs de model(**batch)
    torch_mock.exp.return_value.item.return_value = 1.6  # perplexity del print

    mods = {
        "torch": torch_mock,
        "unsloth": unsloth,
        "unsloth.chat_templates": chat_templates,
        "transformers": MagicMock(),
        "datasets": MagicMock(),
        "trl": MagicMock(),
    }
    return {
        "model": model,
        "tok_sin_template": tok_sin_template,
        "tok_con_template": tok_con_template,
        "unsloth": unsloth,
        "chat_templates": chat_templates,
        "mods": mods,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BUG-004 criterio 3: sintaxis válida sin GPU
# ─────────────────────────────────────────────────────────────────────────────


def test_fine_tune_sintaxis_valida():
    """El módulo compila (y se puede importar) sin GPU ni unsloth."""
    src = Path(ft.__file__).read_text(encoding="utf-8")
    compile(src, str(ft.__file__), "exec")
    assert callable(ft.entrenar)
    assert callable(ft.evaluar)


def test_cada_ruta_con_apply_chat_template_aplica_get_chat_template():
    """Criterios 1-2 (red estática): las dos rutas que llaman a
    apply_chat_template (entrenar y evaluar) aplican get_chat_template.
    exportar_gguf también hace from_pretrained pero no usa apply_chat_template."""
    src = Path(ft.__file__).read_text(encoding="utf-8")
    assert src.count('get_chat_template(tokenizer, chat_template="qwen-2.5")') == 2
    assert src.count("tokenizer.apply_chat_template") == 4


# ─────────────────────────────────────────────────────────────────────────────
# entrenar
# ─────────────────────────────────────────────────────────────────────────────


class TestEntrenarChatTemplate:
    def _args(self, tmp_path, val_file=""):
        return argparse.Namespace(
            model="qwen2.5-1.5b",
            train_file=str(tmp_path / "train.jsonl"),
            val_file=val_file,
            output_dir=str(tmp_path / "out"),
            lora_rank=16,
            max_seq_len=1024,
            batch_size=2,
            gradient_accum=4,
            epochs=1,
            lr=2e-4,
            use_wandb=False,
        )

    def test_get_chat_template_justo_tras_from_pretrained(self, env_llm, tmp_path):
        """Criterio 1: get_chat_template se aplica justo tras el from_pretrained
        y antes del primer apply_chat_template."""
        orden = []
        tok_sin = env_llm["tok_sin_template"]
        tok_con = env_llm["tok_con_template"]
        tok_con.apply_chat_template.side_effect = lambda *a, **k: (
            orden.append("apply_chat_template") or "<text>"
        )

        def _from_pretrained(**kwargs):
            orden.append("from_pretrained")
            return env_llm["model"], tok_sin

        def _get_chat_template(tokenizer, **kwargs):
            orden.append("get_chat_template")
            assert kwargs["chat_template"] == "qwen-2.5"
            return tok_con

        env_llm["unsloth"].FastLanguageModel.from_pretrained.side_effect = _from_pretrained
        env_llm["chat_templates"].get_chat_template.side_effect = _get_chat_template

        with (
            patch.dict(sys.modules, env_llm["mods"]),
            patch.object(
                ft,
                "cargar_dataset",
                return_value=[{"instruction": "¿Qué hago?", "input": "", "output": "Hidrátate"}],
            ),
        ):
            ft.entrenar(self._args(tmp_path))

        assert orden[0] == "from_pretrained"
        assert orden[1] == "get_chat_template"
        assert orden[2] == "apply_chat_template"

    def test_apply_chat_template_solo_sobre_tokenizer_con_template(self, env_llm, tmp_path):
        """Criterio 2: ninguna llamada a apply_chat_template cae sobre el
        tokenizer sin template (ni en train ni en val)."""
        tok_sin = env_llm["tok_sin_template"]
        tok_con = env_llm["tok_con_template"]
        tok_con.apply_chat_template.return_value = "<text>"
        val_file = tmp_path / "val.jsonl"
        val_file.write_text('{"instruction": "i", "input": "", "output": "o"}\n')

        with (
            patch.dict(sys.modules, env_llm["mods"]),
            patch.object(
                ft,
                "cargar_dataset",
                return_value=[{"instruction": "i", "input": "", "output": "o"}],
            ),
        ):
            ft.entrenar(self._args(tmp_path, val_file=str(val_file)))

        tok_sin.apply_chat_template.assert_not_called()
        assert tok_con.apply_chat_template.call_count == 2  # train + val


# ─────────────────────────────────────────────────────────────────────────────
# BUG-005 criterios 2-4: sin checkpoints intermedios
# ─────────────────────────────────────────────────────────────────────────────


class TestEntrenarTrainingArguments:
    def _args(self, tmp_path):
        val_file = tmp_path / "val.jsonl"
        val_file.write_text('{"instruction": "i", "input": "", "output": "o"}\n')
        return argparse.Namespace(
            model="qwen2.5-1.5b",
            train_file=str(tmp_path / "train.jsonl"),
            val_file=str(val_file),
            output_dir=str(tmp_path / "out"),
            lora_rank=16,
            max_seq_len=1024,
            batch_size=2,
            gradient_accum=4,
            epochs=1,
            lr=2e-4,
            use_wandb=False,
        )

    def test_sin_checkpoints_intermedios(self, env_llm, tmp_path):
        """Criterios 2-4 (BUG-005): TrainingArguments no dispara el guardado de
        checkpoints intermedios — save_strategy="no", load_best_model_at_end=False
        y sin save_total_limit/metric_for_best_model — pero eval_strategy sigue en
        "epoch" con val_dataset. El LoRA final se guarda con model.save_pretrained."""
        args = self._args(tmp_path)
        with (
            patch.dict(sys.modules, env_llm["mods"]),
            patch.object(
                ft,
                "cargar_dataset",
                return_value=[{"instruction": "i", "input": "", "output": "o"}],
            ),
        ):
            ft.entrenar(args)

        kwargs = env_llm["mods"]["transformers"].TrainingArguments.call_args.kwargs
        assert kwargs["save_strategy"] == "no"
        assert kwargs["load_best_model_at_end"] is False
        assert kwargs["eval_strategy"] == "epoch"  # sigue evaluando por epoch
        assert "save_total_limit" not in kwargs
        assert "metric_for_best_model" not in kwargs
        env_llm["model"].save_pretrained.assert_called_once_with(str(Path(args.output_dir)))


# ─────────────────────────────────────────────────────────────────────────────
# evaluar
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluarChatTemplate:
    def _args(self, tmp_path):
        lora = tmp_path / "lora"
        lora.mkdir()
        val = tmp_path / "val.jsonl"
        val.write_text('{"instruction": "i", "input": "", "output": "o"}\n')
        return argparse.Namespace(lora_path=str(lora), val_file=str(val))

    def test_get_chat_template_justo_tras_from_pretrained(self, env_llm, tmp_path):
        """Criterio 1: igual que en entrenar, antes de cualquier apply_chat_template."""
        orden = []
        tok_sin = env_llm["tok_sin_template"]
        tok_con = env_llm["tok_con_template"]
        tok_con.apply_chat_template.side_effect = lambda *a, **k: (
            orden.append("apply_chat_template") or MagicMock()
        )

        def _from_pretrained(**kwargs):
            orden.append("from_pretrained")
            return env_llm["model"], tok_sin

        def _get_chat_template(tokenizer, **kwargs):
            orden.append("get_chat_template")
            assert kwargs["chat_template"] == "qwen-2.5"
            return tok_con

        env_llm["unsloth"].FastLanguageModel.from_pretrained.side_effect = _from_pretrained
        env_llm["chat_templates"].get_chat_template.side_effect = _get_chat_template

        with (
            patch.dict(sys.modules, env_llm["mods"]),
            patch.object(
                ft,
                "cargar_dataset",
                return_value=[{"instruction": "i", "input": "", "output": "o"}],
            ),
        ):
            ft.evaluar(self._args(tmp_path))

        assert orden[0] == "from_pretrained"
        assert orden[1] == "get_chat_template"
        assert orden[2] == "apply_chat_template"
        assert len(orden) == 4  # dos apply_chat_template (pérdida + generación)

    def test_apply_chat_template_solo_sobre_tokenizer_con_template(self, env_llm, tmp_path):
        """Criterio 2: tampoco en evaluar cae ninguna llamada sobre el tokenizer sin template."""
        tok_sin = env_llm["tok_sin_template"]
        tok_con = env_llm["tok_con_template"]
        tok_con.apply_chat_template.return_value = MagicMock()

        with (
            patch.dict(sys.modules, env_llm["mods"]),
            patch.object(
                ft,
                "cargar_dataset",
                return_value=[{"instruction": "i", "input": "", "output": "o"}],
            ),
        ):
            ft.evaluar(self._args(tmp_path))

        tok_sin.apply_chat_template.assert_not_called()
        assert tok_con.apply_chat_template.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# BUG-006: model_name str (no Path) en exportar_gguf y evaluar
# ─────────────────────────────────────────────────────────────────────────────


class TestBug006ModelNameStr:
    def test_exportar_gguf_pasa_model_name_str(self, env_llm, tmp_path):
        """Criterio 1 (BUG-006): exportar_gguf pasa model_name como str a
        FastLanguageModel.from_pretrained, no como Path — Unsloth hace
        internamente model_name.lower() y PosixPath revienta."""
        lora = tmp_path / "lora"
        lora.mkdir()
        gguf = tmp_path / "out" / "model.gguf"
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"\x00" * 1024)  # simula el fichero que deja el mock de save_pretrained_gguf
        args = argparse.Namespace(lora_path=str(lora), gguf_path=str(gguf))
        capturado = {}

        def _from_pretrained(**kwargs):
            capturado["model_name"] = kwargs["model_name"]
            return env_llm["model"], env_llm["tok_sin_template"]

        env_llm["unsloth"].FastLanguageModel.from_pretrained.side_effect = _from_pretrained
        env_llm["model"].merge_and_unload.return_value = env_llm["model"]

        with patch.dict(sys.modules, env_llm["mods"]):
            ft.exportar_gguf(args)

        assert isinstance(capturado["model_name"], str)
        assert not isinstance(capturado["model_name"], Path)

    def test_evaluar_pasa_model_name_str(self, env_llm, tmp_path):
        """Criterio 2 (BUG-006): evaluar pasa model_name como str (mismo bug)."""
        lora = tmp_path / "lora"
        lora.mkdir()
        val = tmp_path / "val.jsonl"
        val.write_text('{"instruction": "i", "input": "", "output": "o"}\n')
        args = argparse.Namespace(lora_path=str(lora), val_file=str(val))
        capturado = {}

        def _from_pretrained(**kwargs):
            capturado["model_name"] = kwargs["model_name"]
            return env_llm["model"], env_llm["tok_sin_template"]

        env_llm["unsloth"].FastLanguageModel.from_pretrained.side_effect = _from_pretrained

        with (
            patch.dict(sys.modules, env_llm["mods"]),
            patch.object(
                ft,
                "cargar_dataset",
                return_value=[{"instruction": "i", "input": "", "output": "o"}],
            ),
        ):
            ft.evaluar(args)

        assert isinstance(capturado["model_name"], str)
        assert not isinstance(capturado["model_name"], Path)
