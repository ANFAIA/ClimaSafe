"""
agents.tools.parallel_tool — Ejecución paralela con barras de progreso.

joblib y tqdm están en dependencias base del template, se importan directo.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import joblib
from tqdm import tqdm

from agents.tools.registry import register_tool


@register_tool("parallel")
class ParallelTool:
    @staticmethod
    def parallel_map(func: Callable, iterable: Iterable, n_jobs: int = -1,
                     desc: str | None = None, backend: str = "loky") -> list[Any]:
        """
        Map paralelo con barra de progreso.

        func     : función a aplicar a cada elemento
        iterable : elementos a procesar
        n_jobs   : -1 = todos los cores, 1 = secuencial
        desc     : texto descriptivo para la barra (None = sin barra)
        backend  : 'loky' | 'threading' | 'multiprocessing'
        """
        items = list(iterable)
        if desc:
            return list(tqdm(
                joblib.Parallel(n_jobs=n_jobs, backend=backend)(
                    joblib.delayed(func)(item) for item in items
                ),
                total=len(items), desc=desc,
            ))
        return joblib.Parallel(n_jobs=n_jobs, backend=backend)(
            joblib.delayed(func)(item) for item in items
        )

    @staticmethod
    def parallel_starmap(func: Callable, iterable: Iterable, n_jobs: int = -1,
                         desc: str | None = None, backend: str = "loky") -> list[Any]:
        """
        Starmap paralelo para funciones que reciben tuplas.

        func(*args) se aplica a cada tupla del iterable.
        """
        items = list(iterable)
        if desc:
            return list(tqdm(
                joblib.Parallel(n_jobs=n_jobs, backend=backend)(
                    joblib.delayed(func)(*item) for item in items
                ),
                total=len(items), desc=desc,
            ))
        return joblib.Parallel(n_jobs=n_jobs, backend=backend)(
            joblib.delayed(func)(*item) for item in items
        )

    @staticmethod
    def chunked(iterable: Iterable, n_chunks: int) -> list[list]:
        """Divide un iterable en n_chunks aproximadamente iguales."""
        items = list(iterable)
        n = len(items)
        chunk_size = (n + n_chunks - 1) // n_chunks
        return [items[i:i + chunk_size] for i in range(0, n, chunk_size)]
