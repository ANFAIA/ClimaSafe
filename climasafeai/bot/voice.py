"""
climasafeai.bot.voice — Reconocimiento de voz (STT) y síntesis de voz (TTS).

BOT-018: permite al bot de Telegram recibir notas de audio (transcripción con
faster-whisper) y devolver el parte como audio (TTS con gTTS). Ambos motores
son opcionales: si no están instalados o fallan, el bot sigue funcionando en
texto exactamente como antes.

Motor de STT: faster-whisper (CTranslate2 + Whisper). Modelo base (~140 MB),
buen rendimiento en español con CPU. Latencia típica: 2-5s en un audio de 30s
en un portátil medio.

Motor de TTS: gTTS (Google Text-to-Speech). HTTP a la API pública de Google,
sin API key. Calidad buena para español. Latencia típica: 0.5-2s para un
texto de 200 caracteres.

Ambos motores se cargan bajo demanda (lazy) para no penalizar el arranque
del bot cuando no se usan.

Uso:
    from climasafeai.bot.voice import transcribe_voice, text_to_speech
    texto = transcribe_voice("/tmp/audio.ogg")
    audio_path = text_to_speech("Tu parte del día...")
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── STT: faster-whisper ────────────────────────────────────────────────────

_whisper_model: Any = None
_whisper_available: bool | None = None  # None = no comprobado aún


def _load_whisper() -> Any:
    """Carga el modelo de Whisper bajo demanda. Devuelve None si no está disponible.

    Usa faster-whisper (CTranslate2) por defecto: es más rápido que openai-whisper
    en CPU y consume menos RAM (~500 MB con modelo 'base' en float32). Si
    faster-whisper no está instalado, intenta openai-whisper como fallback.
    """
    global _whisper_model, _whisper_available

    if _whisper_available is not None:
        return _whisper_model

    # Intentar faster-whisper primero (preferido)
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        logger.info("Cargando modelo faster-whisper 'base'…")
        t0 = time.monotonic()
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        elapsed = time.monotonic() - t0
        logger.info(
            "Modelo faster-whisper 'base' cargado en %.1fs "
            "(~140 MB, CPU int8)", elapsed,
        )
        _whisper_available = True
        return _whisper_model
    except ImportError:
        logger.debug("faster-whisper no instalado, intentando openai-whisper")
    except Exception:
        logger.warning("faster-whisper no pudo cargarse", exc_info=True)

    # Fallback: openai-whisper
    try:
        import whisper  # type: ignore[import-untyped]

        logger.info("Cargando modelo openai-whisper 'base'…")
        t0 = time.monotonic()
        _whisper_model = whisper.load_model("base")
        elapsed = time.monotonic() - t0
        logger.info(
            "Modelo openai-whisper 'base' cargado en %.1fs "
            "(~140 MB, CPU)", elapsed,
        )
        _whisper_available = True
        return _whisper_model
    except ImportError:
        logger.debug("openai-whisper no instalado")
    except Exception:
        logger.warning("openai-whisper no pudo cargarse", exc_info=True)

    _whisper_available = False
    return None


def transcribe_voice(audio_path: str | Path, language: str = "es") -> str | None:
    """Transcribe un fichero de audio a texto.

    Acepta cualquier formato que soporte ffmpeg (ogg, mp3, wav, m4a, opus...).
    Telegram envía notas de voz en formato OGG/Opus.

    Args:
        audio_path: Ruta al fichero de audio.
        language: Código de idioma ISO 639-1 (default: 'es' para español).

    Returns:
        Texto transcrito, o None si el motor no está disponible o falla.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.warning("Fichero de audio no encontrado: %s", audio_path)
        return None

    model = _load_whisper()
    if model is None:
        logger.debug("Sin motor STT: audio %s no transcrito", audio_path.name)
        return None

    try:
        t0 = time.monotonic()
        # faster-whisper API
        if hasattr(model, "transcribe"):
            # Detectar si es faster-whisper o openai-whisper
            try:
                # faster-whisper: devuelve (segments, info)
                segments, info = model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=5,
                )
                texto = " ".join(seg.text.strip() for seg in segments)
            except (TypeError, ValueError):
                # openai-whisper: devuelve dict
                result = model.transcribe(str(audio_path), language=language)
                texto = result.get("text", "").strip()
        else:
            logger.error("Modelo de whisper inesperado: %s", type(model))
            return None

        elapsed = time.monotonic() - t0
        logger.info(
            "Transcripción completada en %.1fs (%d caracteres): %s…",
            elapsed, len(texto), texto[:80] if texto else "(vacío)",
        )
        return texto if texto else None

    except Exception:
        logger.exception("Error en transcripción de %s", audio_path.name)
        return None


# ── TTS: gTTS ───────────────────────────────────────────────────────────────

_gtts_available: bool | None = None
_gTTS_cls: Any = None  # clase gTTS cacheada para tests


def _check_gtts() -> bool:
    """Comprueba si gTTS está disponible (lazy, una vez)."""
    global _gtts_available, _gTTS_cls
    if _gtts_available is not None:
        return _gtts_available
    try:
        from gtts import gTTS  # type: ignore[import-untyped]
        _gTTS_cls = gTTS
        _gtts_available = True
    except ImportError:
        _gtts_available = False
        _gTTS_cls = None
        logger.debug("gTTS no instalado: TTS no disponible")
    return _gtts_available


def text_to_speech(
    text: str,
    language: str = "es",
    tld: str = "es",
) -> Path | None:
    """Convierte texto a audio con gTTS y devuelve la ruta al fichero OGG/MP3.

    El fichero se crea en un directorio temporal y debe ser borrado por el
    llamador después de usarlo (ver `cleanup_audio`).

    Args:
        text: Texto a sintetizar. Si es muy largo, se trunca a 5000 caracteres
              (límite práctico de gTTS).
        language: Código de idioma ISO 639-1.
        tld: Dominio de nivel superior para gTTS (es → español).

    Returns:
        Path al fichero de audio generado, o None si gTTS no está disponible o falla.
    """
    if not _check_gtts():
        return None

    if not text or not text.strip():
        return None

    # gTTS tiene problemas con textos muy largos
    texto_corto = text[:5000]

    try:
        t0 = time.monotonic()
        tts = _gTTS_cls(text=texto_corto, lang=language, tld=tld)

        # Crear fichero temporal
        tmp = tempfile.NamedTemporaryFile(
            suffix=".ogg", prefix="tts_", delete=False,
        )
        tmp_path = Path(tmp.name)
        tmp.close()

        tts.save(str(tmp_path))

        elapsed = time.monotonic() - t0
        size_kb = tmp_path.stat().st_size / 1024
        logger.info(
            "TTS generado en %.1fs (%.0f KB): %s",
            elapsed, size_kb, tmp_path.name,
        )
        return tmp_path

    except Exception:
        logger.exception("Error generando TTS")
        return None


def cleanup_audio(*paths: Path | None) -> None:
    """Borra ficheros de audio temporales de forma segura.

    Se ignora None y ficheros que no existen. Se loguea si hay error
    al borrar pero no se propaga.
    """
    for p in paths:
        if p is None:
            continue
        try:
            p = Path(p)
            if p.exists():
                p.unlink()
                logger.debug("Audio temporal borrado: %s", p.name)
        except Exception:
            logger.warning("No se pudo borrar audio temporal: %s", p, exc_info=True)
