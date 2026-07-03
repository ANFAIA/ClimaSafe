"""
agents.tools.rest_tool — Cliente HTTP mínimo con `urllib.request` (librería
estándar). `httpx` solo se instala si el proyecto activó `use_api=true`, así
que esta herramienta no depende de él — cubre el caso general con lo que ya
viene con Python.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from agents.exceptions import ToolExecutionError
from agents.tools.registry import register_tool

DEFAULT_TIMEOUT_SECONDS = 15


@dataclass
class RestResponse:
    status: int
    headers: dict[str, str]
    text: str

    def json(self) -> Any:
        return json.loads(self.text)


@register_tool("rest")
class RestTool:
    @staticmethod
    def request(
        url: str,
        *,
        method: str = "GET",
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> RestResponse:
        data = None
        req_headers = dict(headers or {})
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")

        request = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return RestResponse(
                    status=response.status,
                    headers=dict(response.headers.items()),
                    text=response.read().decode("utf-8"),
                )
        except urllib.error.HTTPError as exc:
            return RestResponse(status=exc.code, headers=dict(exc.headers.items()), text=exc.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ToolExecutionError(f"No se pudo conectar a '{url}': {exc.reason}") from exc

    @staticmethod
    def get(url: str, **kwargs: Any) -> RestResponse:
        return RestTool.request(url, method="GET", **kwargs)

    @staticmethod
    def post(url: str, json_body: Any = None, **kwargs: Any) -> RestResponse:
        return RestTool.request(url, method="POST", json_body=json_body, **kwargs)
