"""
climasafeai.bot.messaging — Abstracción común de mensajería (MSG-001).

Interfaz única para enviar mensajes por cualquier canal: Telegram (el bot
actual), Hermes o un webhook genérico. Los consumidores programan contra
`MessageAdapter` y no saben qué canal hay detrás.

Cada adaptador abre su propio `httpx.AsyncClient` (mismo patrón de keep-alive
que `_HTTP_CLIENT` en `telegram_bot.py`); `close()` cierra ese cliente.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

from climasafeai.bot.telegram_bot import enviar_mensaje


class MessageAdapter(ABC):
    """Interfaz común de envío: un canal, un contrato.

    `send` envía un mensaje de texto a un destino (chat_id, user id...);
    `send_batch` envía varios en secuencia con el mismo canal.
    """

    @abstractmethod
    async def send(self, destino: str, texto: str, **kwargs: Any) -> None:
        """Envía `texto` a `destino`. Los kwargs son específicos del canal."""

    @abstractmethod
    async def send_batch(self, mensajes: list[tuple[str, str]], **kwargs: Any) -> None:
        """Envía cada par (destino, texto) de `mensajes` en secuencia."""


class TelegramAdapter(MessageAdapter):
    """Canal Telegram: delega en `enviar_mensaje` del bot actual.

    El bot ya tiene toda la robustez del envío (fallback de formato ante 400,
    partición por 4096, recorte con aviso); el adapter la reutiliza en vez de
    duplicarla. `destino` llega como str (así se guarda en la DB) y se
    normaliza a int, que es lo que espera `enviar_mensaje`.
    """

    async def send(self, destino: str, texto: str, **kwargs: Any) -> None:
        await enviar_mensaje(int(destino), texto, **kwargs)

    async def send_batch(self, mensajes: list[tuple[str, str]], **kwargs: Any) -> None:
        for destino, texto in mensajes:
            await self.send(destino, texto, **kwargs)


class HermesAdapter(MessageAdapter):
    """Canal Hermes: POST JSON a {base_url}/send.

    La base sale de la variable de entorno HERMES_BASE_URL; se puede inyectar
    en el constructor (los tests no tocan la red).
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or os.getenv("HERMES_BASE_URL", "")).rstrip("/")
        self._client = httpx.AsyncClient(timeout=20)

    async def send(self, destino: str, texto: str, **kwargs: Any) -> None:
        payload = {"chat_id": destino, "text": texto, **kwargs}
        r = await self._client.post(f"{self._base_url}/send", json=payload)
        r.raise_for_status()

    async def send_batch(self, mensajes: list[tuple[str, str]], **kwargs: Any) -> None:
        for destino, texto in mensajes:
            await self.send(destino, texto, **kwargs)

    async def close(self) -> None:
        await self._client.aclose()


class WebhookAdapter(MessageAdapter):
    """Canal webhook: POST JSON a WEBHOOK_URL.

    Contrato mínimo del receptor: `{"to": destino, "text": texto, ...}`.
    La URL sale de la variable de entorno WEBHOOK_URL; se puede inyectar en el
    constructor (los tests no tocan la red).
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url or os.getenv("WEBHOOK_URL", "")
        self._client = httpx.AsyncClient(timeout=20)

    async def send(self, destino: str, texto: str, **kwargs: Any) -> None:
        payload = {"to": destino, "text": texto, **kwargs}
        r = await self._client.post(self._url, json=payload)
        r.raise_for_status()

    async def send_batch(self, mensajes: list[tuple[str, str]], **kwargs: Any) -> None:
        for destino, texto in mensajes:
            await self.send(destino, texto, **kwargs)

    async def close(self) -> None:
        await self._client.aclose()
