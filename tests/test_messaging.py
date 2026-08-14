"""Tests de la abstracción de mensajería (MSG-001).

Cada adaptador se prueba con transporte mockeado: Telegram delegando en
`enviar_mensaje` (parcheada), Hermes/Webhook con `httpx.AsyncClient.post`
parcheado — nunca se toca la red real.
"""

from __future__ import annotations

import httpx
import pytest

import climasafeai.bot.messaging as messaging
from climasafeai.bot.messaging import (
    HermesAdapter,
    MessageAdapter,
    TelegramAdapter,
    WebhookAdapter,
)


class TestMessageAdapter:
    def test_es_abstracta(self):
        """No se puede instanciar la interfaz: solo se usan adaptadores."""
        with pytest.raises(TypeError):
            MessageAdapter()


# ── TelegramAdapter ──────────────────────────────────────────────────────────


class TestTelegramAdapter:
    async def test_send_delega_en_enviar_mensaje(self, monkeypatch):
        """`send` llama a enviar_mensaje con el destino normalizado a int."""
        llamadas: list[tuple] = []

        async def _fake_enviar(chat_id, texto, **kwargs):
            llamadas.append((chat_id, texto, kwargs))

        monkeypatch.setattr(messaging, "enviar_mensaje", _fake_enviar)
        adapter = TelegramAdapter()

        await adapter.send("123", "hola", kb=None)

        assert llamadas == [(123, "hola", {"kb": None})]

    async def test_send_batch_envia_todos_en_secuencia(self, monkeypatch):
        llamadas: list[tuple] = []

        async def _fake_enviar(chat_id, texto, **kwargs):
            llamadas.append((chat_id, texto))

        monkeypatch.setattr(messaging, "enviar_mensaje", _fake_enviar)
        adapter = TelegramAdapter()

        await adapter.send_batch([("1", "a"), ("2", "b")])

        assert llamadas == [(1, "a"), (2, "b")]


# ── HermesAdapter ────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=None,
                response=self,  # type: ignore[arg-type]
            )


class TestHermesAdapter:
    async def test_send_hace_post_a_base_send(self, monkeypatch):
        llamadas: list[tuple] = []

        async def _fake_post(self, url, **kwargs):
            llamadas.append((url, kwargs.get("json")))
            return _FakeResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
        adapter = HermesAdapter("https://hermes.test")

        await adapter.send("u1", "texto", prioridad="alta")
        await adapter.close()

        assert llamadas == [
            ("https://hermes.test/send", {"chat_id": "u1", "text": "texto", "prioridad": "alta"})
        ]

    async def test_send_batch_hace_un_post_por_mensaje(self, monkeypatch):
        urls: list[str] = []

        async def _fake_post(self, url, **kwargs):
            urls.append(url)
            return _FakeResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
        adapter = HermesAdapter("https://hermes.test")

        await adapter.send_batch([("u1", "a"), ("u2", "b")])
        await adapter.close()

        assert urls == ["https://hermes.test/send", "https://hermes.test/send"]

    def test_base_url_desde_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_BASE_URL", "https://env.test/")
        adapter = HermesAdapter()
        assert adapter._base_url == "https://env.test"


# ── WebhookAdapter ───────────────────────────────────────────────────────────


class TestWebhookAdapter:
    async def test_send_hace_post_a_la_url(self, monkeypatch):
        llamadas: list[tuple] = []

        async def _fake_post(self, url, **kwargs):
            llamadas.append((url, kwargs.get("json")))
            return _FakeResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
        adapter = WebhookAdapter("https://hooks.test/alert")

        await adapter.send("c1", "texto")
        await adapter.close()

        assert llamadas == [("https://hooks.test/alert", {"to": "c1", "text": "texto"})]

    async def test_send_batch_hace_un_post_por_mensaje(self, monkeypatch):
        urls: list[str] = []

        async def _fake_post(self, url, **kwargs):
            urls.append(url)
            return _FakeResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
        adapter = WebhookAdapter("https://hooks.test/alert")

        await adapter.send_batch([("c1", "a"), ("c2", "b")])
        await adapter.close()

        assert urls == ["https://hooks.test/alert", "https://hooks.test/alert"]

    def test_url_desde_env(self, monkeypatch):
        monkeypatch.setenv("WEBHOOK_URL", "https://env.test/hook")
        adapter = WebhookAdapter()
        assert adapter._url == "https://env.test/hook"
