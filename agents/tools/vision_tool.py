"""
agents.tools.vision_tool — Inspección estructural de gráficos en reports/figures/.

Límite honesto de esta herramienta: NO hace comprensión visual real (eso
requeriría un modelo multimodal, que no forma parte de este template). Lo
que sí hace, con matplotlib (dependencia base del template, sin añadir PIL
ni opencv), es leer los píxeles de la imagen y calcular métricas objetivas
que correlacionan con problemas típicos:
  - gráfico casi vacío / mal renderizado -> varianza de píxeles muy baja
  - aspect ratio raro -> asimetría fuera de rango típico (0.5–3.0)
  - archivo corrupto o vacío -> falla la lectura

`GraphAgent` usa estas métricas como señales, no como veredictos. Si
necesitas interpretación semántica real ("¿qué tendencia muestra este
gráfico?"), esa parte queda fuera de esta herramienta a propósito — la
opción honesta es conectar un modelo multimodal por fuera del sistema de
agentes, no fingir que un heurístico de píxeles hace ese trabajo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agents.tools.registry import register_tool

SUPPORTED_EXTENSIONS = (".png",)


@dataclass
class FigureMetrics:
    path: Path
    width: int
    height: int
    aspect_ratio: float
    pixel_std: float
    mostly_blank: bool
    warnings: list[str]


@register_tool("vision")
class VisionTool:
    @staticmethod
    def list_figures(figures_dir: Path) -> list[Path]:
        if not figures_dir.exists():
            return []
        return sorted(
            p for p in figures_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    @staticmethod
    def inspect(image_path: Path) -> FigureMetrics:
        """
        Lee un PNG con `matplotlib.image.imread` (no requiere PIL) y calcula
        métricas estructurales básicas.

        Nota de precisión: `matplotlib.image.imread` es una función estable y
        documentada de matplotlib, pero si tu versión instalada difiere de la
        que se usó al escribir este código, vale la pena confirmar la firma
        exacta en la documentación de matplotlib antes de asumir el
        comportamiento en casos borde (p. ej. PNGs en escala de grises).
        """
        import matplotlib.image as mpimg

        warnings: list[str] = []
        try:
            array = mpimg.imread(image_path)
        except Exception as exc:  # noqa: BLE001 — cualquier fallo de lectura es "no se pudo inspeccionar"
            return FigureMetrics(
                path=image_path, width=0, height=0, aspect_ratio=0.0,
                pixel_std=0.0, mostly_blank=False,
                warnings=[f"No se pudo leer la imagen: {exc}"],
            )

        height, width = array.shape[0], array.shape[1]
        aspect_ratio = round(width / height, 3) if height else 0.0
        pixel_std = float(np.std(array))

        # Umbral empírico: un PNG de matplotlib con contenido real (líneas, barras,
        # texto) casi siempre tiene std > 0.02 en escala [0, 1]. Por debajo de eso
        # suele ser una figura vacía o un fondo liso sin datos dibujados.
        mostly_blank = pixel_std < 0.02
        if mostly_blank:
            warnings.append(
                f"Varianza de píxeles muy baja ({pixel_std:.4f}) — la figura podría "
                f"estar vacía o no haberse renderizado correctamente."
            )

        if not (0.5 <= aspect_ratio <= 3.0):
            warnings.append(
                f"Aspect ratio inusual ({aspect_ratio}) — revisa figsize en la "
                f"función que generó '{image_path.name}'."
            )

        return FigureMetrics(
            path=image_path, width=width, height=height, aspect_ratio=aspect_ratio,
            pixel_std=round(pixel_std, 5), mostly_blank=mostly_blank, warnings=warnings,
        )
