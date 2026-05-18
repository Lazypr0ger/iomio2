from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import PLOTS_DIR
from src.functions.base import ObjectiveFunction
from src.models import OptimizationResult


def _get_track(result: OptimizationResult) -> np.ndarray:
    """Массив точек траектории метода."""

    return np.array([record.x for record in result.history], dtype=float)


def _get_plot_bounds(
    function: ObjectiveFunction,
    results: list[OptimizationResult],
    margin: float = 1.5,
) -> tuple[float, float, float, float]:
    """Границы области графика по траекториям и известному минимуму."""

    points = [function.default_x0]

    for result in results:
        track = _get_track(result)
        points.extend(track)

    for known_minimum in function.known_minima:
        points.append(known_minimum)

    points_array = np.array(points, dtype=float)

    x_min = float(np.min(points_array[:, 0]) - margin)
    x_max = float(np.max(points_array[:, 0]) + margin)
    y_min = float(np.min(points_array[:, 1]) - margin)
    y_max = float(np.max(points_array[:, 1]) + margin)

    if abs(x_max - x_min) < 1e-6:
        x_min -= 1.0
        x_max += 1.0

    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0

    return x_min, x_max, y_min, y_max


def plot_contour_with_tracks(
    function: ObjectiveFunction,
    results: list[OptimizationResult],
    output_path: Path | None = None,
) -> Path:
    """
    Контурный график функции с траекториями трёх методов.

    Требования:
    - линии уровня;
    - треки всех методов;
    - начальная точка;
    - найденные минимумы;
    - точный минимум, если известен;
    - легенда;
    - подписи осей x1, x2.
    """

    if function.dimension != 2:
        raise ValueError("Контурный график строится только для 2D-функций.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = PLOTS_DIR / f"contour_tracks_{function.name}.png"

    x_min, x_max, y_min, y_max = _get_plot_bounds(function, results)

    grid_x = np.linspace(x_min, x_max, 400)
    grid_y = np.linspace(y_min, y_max, 400)
    xx, yy = np.meshgrid(grid_x, grid_y)

    zz = np.zeros_like(xx, dtype=float)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j]], dtype=float)
            zz[i, j] = function._value(point)

    finite_values = zz[np.isfinite(zz)]

    if finite_values.size == 0:
        raise ValueError(f"Не удалось построить значения функции {function.name}.")

    min_level = float(np.nanmin(finite_values))
    max_level = float(np.nanpercentile(finite_values, 95))

    if max_level <= min_level:
        max_level = float(np.nanmax(finite_values))

    levels = np.linspace(min_level, max_level, 35)

    plt.figure(figsize=(10, 8))

    contour = plt.contour(xx, yy, zz, levels=levels, linewidths=0.8)
    plt.clabel(contour, inline=True, fontsize=7, fmt="%.1f")

    for result in results:
        track = _get_track(result)

        plt.plot(
            track[:, 0],
            track[:, 1],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{result.method_name}, k={result.iterations}",
        )

        plt.scatter(
            result.x_star[0],
            result.x_star[1],
            marker="x",
            s=80,
        )

        plt.text(
            result.x_star[0],
            result.x_star[1],
            f" {result.method_name}",
            fontsize=8,
        )

    plt.scatter(
        function.default_x0[0],
        function.default_x0[1],
        marker="s",
        s=90,
        label="Начальная точка",
    )

    for index, known_minimum in enumerate(function.known_minima, start=1):
        label = "Точный минимум" if index == 1 else None

        plt.scatter(
            known_minimum[0],
            known_minimum[1],
            marker="*",
            s=150,
            label=label,
        )

        if len(function.known_minima) > 1:
            plt.text(
                known_minimum[0],
                known_minimum[1],
                f" m{index}",
                fontsize=8,
            )

    plt.title(f"{function.name}: линии уровня и траектории методов")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def plot_all_2d_contours(
    functions: list[ObjectiveFunction],
    results: list[OptimizationResult],
) -> list[Path]:
    """Построение контурных графиков для всех 2D-функций."""

    paths: list[Path] = []

    for function in functions:
        if function.dimension != 2:
            continue

        function_results = [
            result
            for result in results
            if result.function_name == function.name
        ]

        if not function_results:
            continue

        path = plot_contour_with_tracks(function, function_results)
        paths.append(path)

    return paths