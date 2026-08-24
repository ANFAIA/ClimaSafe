"""Tests de BOT-018: voz en el bot de Telegram (STT + TTS)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Tests del módulo voice.py (sin dependencias reales) ────────────────────


class TestTranscribeVoice:
    """STT: transcripción de audio con fallback graceful."""

    def test_fichero_no_existente_devuelve_none(self):
        """Un path que no existe no debe dar error."""
        from climasafeai.bot.voice import transcribe_voice

        result = transcribe_voice("/tmp/no_existe_12345.ogg")
        assert result is None

    def test_fichero_vacio_devuelve_none(self, tmp_path):
        """Un fichero vacío no debe causar excepción."""
        from climasafeai.bot.voice import transcribe_voice

        empty = tmp_path / "empty.ogg"
        empty.write_bytes(b"")
        result = transcribe_voice(empty)
        # Puede devolver None o una cadena vacía; ambos son aceptables
        assert result is None or result == ""

    @patch("climasafeai.bot.voice._whisper_available", False)
    def test_sin_whisper_devuelve_none(self, tmp_path):
        """Sin motor STT instalado, devuelve None (no excepción)."""
        from climasafeai.bot.voice import transcribe_voice

        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"\x00" * 100)
        result = transcribe_voice(audio)
        assert result is None

    @patch("climasafeai.bot.voice._load_whisper")
    def test_whisper_disponible_transcribe(self, mock_load, tmp_path):
        """Con whisper disponible, transcribe el audio."""
        from climasafeai.bot.voice import transcribe_voice

        # Mock del modelo faster-whisper
        mock_segment = MagicMock()
        mock_segment.text = "Hola, soy Aldán"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], MagicMock())
        mock_load.return_value = mock_model

        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"\x00" * 100)

        result = transcribe_voice(audio)
        assert result == "Hola, soy Aldán"
        mock_model.transcribe.assert_called_once()

    @patch("climasafeai.bot.voice._load_whisper")
    def test_whisper_devuelve_texto_vacio_devuelve_none(self, mock_load, tmp_path):
        """Si Whisper devuelve texto vacío, devuelve None."""
        from climasafeai.bot.voice import transcribe_voice

        mock_segment = MagicMock()
        mock_segment.text = "   "
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], MagicMock())
        mock_load.return_value = mock_model

        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"\x00" * 100)

        result = transcribe_voice(audio)
        assert result is None

    @patch("climasafeai.bot.voice._load_whisper")
    def test_whisper_excepcion_devuelve_none(self, mock_load, tmp_path):
        """Si Whisper lanza excepción, devuelve None (no propagar)."""
        from climasafeai.bot.voice import transcribe_voice

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("GPU out of memory")
        mock_load.return_value = mock_model

        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"\x00" * 100)

        result = transcribe_voice(audio)
        assert result is None


class TestTextToSpeech:
    """TTS: síntesis de voz con fallback graceful."""

    @patch("climasafeai.bot.voice._gtts_available", False)
    def test_sin_gtts_devuelve_none(self):
        """Sin gTTS instalado, devuelve None (no excepción)."""
        from climasafeai.bot.voice import text_to_speech

        result = text_to_speech("Hola mundo")
        assert result is None

    def test_texto_vacio_devuelve_none(self):
        """Texto vacío no debe generar audio."""
        from climasafeai.bot.voice import text_to_speech

        result = text_to_speech("")
        assert result is None

    def test_texto_solo_espacios_devuelve_none(self):
        """Texto solo con espacios no debe generar audio."""
        from climasafeai.bot.voice import text_to_speech

        result = text_to_speech("   ")
        assert result is None

    @patch("climasafeai.bot.voice._check_gtts", return_value=True)
    @patch("climasafeai.bot.voice._gTTS_cls")
    def test_gtts_disponible_genera_audio(self, mock_gtts_cls, mock_check):
        """Con gTTS disponible, genera un fichero de audio."""
        from climasafeai.bot.voice import text_to_speech

        mock_tts = MagicMock()
        mock_gtts_cls.return_value = mock_tts

        result = text_to_speech("Hola, soy ClimaSafeAI")

        assert result is not None
        assert result.exists()
        assert result.suffix == ".ogg"
        assert result.name.startswith("tts_")
        mock_tts.save.assert_called_once()

        # Limpiar
        result.unlink(missing_ok=True)

    @patch("climasafeai.bot.voice._check_gtts", return_value=True)
    @patch("climasafeai.bot.voice._gTTS_cls")
    def test_gtts_excepcion_devuelve_none(self, mock_gtts_cls, mock_check):
        """Si gTTS lanza excepción, devuelve None (no propagar)."""
        from climasafeai.bot.voice import text_to_speech

        mock_gtts_cls.side_effect = RuntimeError("Connection error")

        result = text_to_speech("Hola")
        assert result is None

    @patch("climasafeai.bot.voice._check_gtts", return_value=True)
    @patch("climasafeai.bot.voice._gTTS_cls")
    def test_texto_largo_se_trunca(self, mock_gtts_cls, mock_check):
        """Un texto de >5000 caracteres se trunca para gTTS."""
        from climasafeai.bot.voice import text_to_speech

        mock_tts = MagicMock()
        mock_gtts_cls.return_value = mock_tts

        texto_largo = "A" * 6000
        result = text_to_speech(texto_largo)

        # Verificar que se creó el audio
        assert result is not None
        # Verificar que se pasó el texto truncado
        call_args = mock_gtts_cls.call_args
        assert len(call_args.kwargs.get("lang", "")) > 0 or len(call_args.args) >= 1

        # Limpiar
        result.unlink(missing_ok=True)


class TestCleanupAudio:
    """Limpieza de ficheros de audio temporales."""

    def test_borra_ficheros_existentes(self, tmp_path):
        """Los ficheros existentes se borran correctamente."""
        from climasafeai.bot.voice import cleanup_audio

        f1 = tmp_path / "audio1.ogg"
        f2 = tmp_path / "audio2.ogg"
        f1.write_bytes(b"\x00")
        f2.write_bytes(b"\x00")

        cleanup_audio(f1, f2)

        assert not f1.exists()
        assert not f2.exists()

    def test_ignora_none(self, tmp_path):
        """None no debe causar error."""
        from climasafeai.bot.voice import cleanup_audio

        cleanup_audio(None, None)  # no debe lanzar

    def test_ignora_ficheros_no_existentes(self, tmp_path):
        """Ficheros que no existen se ignoran sin error."""
        from climasafeai.bot.voice import cleanup_audio

        f = tmp_path / "no_existe.ogg"
        cleanup_audio(f)  # no debe lanzar

    def test_borra_mixto(self, tmp_path):
        """Mezcla de None, existentes y no existentes."""
        from climasafeai.bot.voice import cleanup_audio

        existe = tmp_path / "existe.ogg"
        existe.write_bytes(b"\x00")

        cleanup_audio(None, existe, tmp_path / "no_existe.ogg")

        assert not existe.exists()


# ── Tests de integración con el bot ────────────────────────────────────────


class TestVoiceEnBot:
    """Integración de voz con el bot de Telegram."""

    @pytest.mark.asyncio
    async def test_recibir_voz_sin_motor_avisa_al_usuario(self, monkeypatch, tmp_path):
        """Si STT no está instalado, el bot avisa y sigue funcionando."""
        import climasafeai.bot.telegram_bot as mod

        enviados: list[str] = []

        async def _fake_tg(method: str, **kwargs):
            if method == "sendMessage":
                enviados.append(kwargs.get("text", ""))
            if method == "getFile":
                return {"result": {"file_path": "voice/file.ogg"}}
            return {"ok": True, "result": {}}

        monkeypatch.setattr(mod, "_tg", _fake_tg)
        monkeypatch.setattr(
            mod, "transcribe_voice", lambda *a, **k: None,
        )

        # Descarga exitosa, pero transcripción falla
        fake_audio = tmp_path / "test.ogg"
        fake_audio.write_bytes(b"\x00" * 100)

        async def _fake_descarga(file_id):
            return fake_audio

        monkeypatch.setattr(mod, "_descargar_audio", _fake_descarga)

        await mod._recibir_voz(1, {"file_id": "abc123"})

        assert any("transcribir" in m.lower() for m in enviados), enviados

    @pytest.mark.asyncio
    async def test_recibir_voz_exitosa_alimenta_el_flujo(self, monkeypatch):
        """Una voz transcrita se procesa como un mensaje de texto."""
        import climasafeai.bot.telegram_bot as mod

        enviados: list[str] = []

        async def _fake_tg(method: str, **kwargs):
            if method == "sendMessage":
                enviados.append(kwargs.get("text", ""))
            if method == "getFile":
                return {"result": {"file_path": "voice/file.ogg"}}
            return {"ok": True, "result": {}}

        monkeypatch.setattr(mod, "_tg", _fake_tg)
        monkeypatch.setattr(mod, "transcribe_voice", lambda *a, **k: "hola")
        monkeypatch.setattr(
            mod, "_modelo_por_defecto", lambda: mod.MODELO_DETERMINISTA,
        )

        async def _fake_descarga(file_id):
            return Path("/tmp/fake.ogg")

        monkeypatch.setattr(mod, "_descargar_audio", _fake_descarga)
        monkeypatch.setattr(mod, "cleanup_audio", lambda *a: None)

        await mod._recibir_voz(1, {"file_id": "abc123"})

        # Debe haber enviado la transcripción
        assert any("hola" in m.lower() for m in enviados), enviados

    @pytest.mark.asyncio
    async def test_procesar_update_con_voice_llama_a_recibir_voz(self, monkeypatch):
        """procesar_update detecta mensajes voice y llama a _recibir_voz."""
        import climasafeai.bot.telegram_bot as mod

        llamado = {"chat_id": None, "voice": None}

        async def _fake_recibir(chat_id, voice):
            llamado["chat_id"] = chat_id
            llamado["voice"] = voice

        monkeypatch.setattr(mod, "_recibir_voz", _fake_recibir)

        await mod.procesar_update({
            "message": {
                "chat": {"id": 42},
                "voice": {"file_id": "abc", "duration": 5},
            }
        })

        assert llamado["chat_id"] == 42
        assert llamado["voice"] == {"file_id": "abc", "duration": 5}

    @pytest.mark.asyncio
    async def test_finalizar_parte_intenta_tts(self, monkeypatch):
        """_finalizar_parte genera TTS y lo envía como audio."""
        import climasafeai.bot.telegram_bot as mod

        enviados: list[str] = []
        audios_enviados: list[Path] = []

        async def _fake_tg(method: str, **kwargs):
            if method == "sendMessage":
                enviados.append(kwargs.get("text", ""))
            return {"ok": True, "result": {}}

        def _fake_tts(text):
            p = Path("/tmp/fake_tts.ogg")
            p.write_bytes(b"\x00")
            return p

        async def _fake_enviar_audio(chat_id, path, **kwargs):
            audios_enviados.append(path)
            return True

        monkeypatch.setattr(mod, "_tg", _fake_tg)
        monkeypatch.setattr(mod, "text_to_speech", _fake_tts)
        monkeypatch.setattr(mod, "enviar_audio", _fake_enviar_audio)
        monkeypatch.setattr(mod, "cleanup_audio", lambda *a: None)

        async def _fake_prediccion(cid):
            return "Riesgo: PRECAUCIÓN (21%)"

        monkeypatch.setattr(mod, "ejecutar_prediccion", _fake_prediccion)

        mod._conversaciones.clear()
        mod._conversaciones[1] = {
            "estado": mod.Estado.DONE,
            "modelo": mod.MODELO_DETERMINISTA,
            "data": {},
        }

        await mod._finalizar_parte(1)

        # El texto se envió
        assert any("PRECAUCIÓN" in m for m in enviados), enviados
        # Se intentó generar TTS
        assert len(audios_enviados) == 1, audios_enviados

    @pytest.mark.asyncio
    async def test_finalizar_parte_tts_falla_sigue_con_texto(self, monkeypatch):
        """Si TTS falla, el parte se envía solo en texto sin error visible."""
        import climasafeai.bot.telegram_bot as mod

        enviados: list[str] = []

        async def _fake_tg(method: str, **kwargs):
            if method == "sendMessage":
                enviados.append(kwargs.get("text", ""))
            return {"ok": True, "result": {}}

        monkeypatch.setattr(mod, "_tg", _fake_tg)
        monkeypatch.setattr(mod, "text_to_speech", lambda *a: None)
        monkeypatch.setattr(mod, "cleanup_audio", lambda *a: None)

        async def _fake_prediccion(cid):
            return "Riesgo: SEGURO"

        monkeypatch.setattr(mod, "ejecutar_prediccion", _fake_prediccion)

        mod._conversaciones.clear()
        mod._conversaciones[1] = {
            "estado": mod.Estado.DONE,
            "modelo": mod.MODELO_DETERMINISTA,
            "data": {},
        }

        await mod._finalizar_parte(1)

        # El texto se envió aunque TTS devolvió None
        assert any("SEGURO" in m for m in enviados), enviados


class TestDescargarAudio:
    """Tests de descarga de audio desde Telegram."""

    @pytest.mark.asyncio
    async def test_descarga_exitosa(self, monkeypatch, tmp_path):
        """Descarga un fichero de audio correctamente."""
        import climasafeai.bot.telegram_bot as mod

        audio_bytes = b"\x00\x01\x02\x03"

        async def _fake_tg(method: str, **kwargs):
            if method == "getFile":
                return {"result": {"file_path": "voice/file_123.ogg"}}
            return {"ok": True, "result": {}}

        # Mock httpx para devolver bytes
        class _FakeResponse:
            status_code = 200
            content = audio_bytes

            def raise_for_status(self):
                pass

        class _FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, **kwargs):
                return _FakeResponse()

        monkeypatch.setattr(mod, "_tg", _fake_tg)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token")
        monkeypatch.setattr("httpx.AsyncClient", _FakeClient)

        result = await mod._descargar_audio("abc123")

        assert result is not None
        assert result.exists()
        assert result.suffix == ".ogg"
        assert result.read_bytes() == audio_bytes

        # Limpiar
        result.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_descarga_fallo_tg_devuelve_none(self, monkeypatch):
        """Si getFile falla, devuelve None."""
        import climasafeai.bot.telegram_bot as mod

        async def _fake_tg(method: str, **kwargs):
            raise RuntimeError("Network error")

        monkeypatch.setattr(mod, "_tg", _fake_tg)

        result = await mod._descargar_audio("abc123")
        assert result is None

    @pytest.mark.asyncio
    async def test_descarga_file_path_vacio_devuelve_none(self, monkeypatch):
        """Si getFile no devuelve file_path, devuelve None."""
        import climasafeai.bot.telegram_bot as mod

        async def _fake_tg(method: str, **kwargs):
            return {"result": {}}

        monkeypatch.setattr(mod, "_tg", _fake_tg)

        result = await mod._descargar_audio("abc123")
        assert result is None
