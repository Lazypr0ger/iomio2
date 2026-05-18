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


def plot_polynomial_2d_section(
    function: ObjectiveFunction,
    results: list[OptimizationResult],
    variable_i: int = 0,
    variable_j: int = 1,
    output_path: Path | None = None,
) -> Path:
    """
    2D-сечение 10D-полинома.

    Остальные 8 переменных фиксируются в лучшем найденном x*.
    На график также наносятся проекции траекторий методов.
    """

    if function.dimension != 10:
        raise ValueError("2D-сечение предназначено для функции размерности 10.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = (
            PLOTS_DIR
            / f"polynomial_section_{function.name}_x{variable_i + 1}_x{variable_j + 1}.png"
        )

    best_result = min(results, key=lambda result: result.f_star)
    fixed_point = best_result.x_star.copy()

    projected_points = []

    for result in results:
        track = _get_track(result)
        projected_points.extend(track[:, [variable_i, variable_j]])

    projected_points.append(function.default_x0[[variable_i, variable_j]])
    projected_points.append(fixed_point[[variable_i, variable_j]])

    projected_points_array = np.array(projected_points, dtype=float)

    x_min = float(np.min(projected_points_array[:, 0]) - 1.0)
    x_max = float(np.max(projected_points_array[:, 0]) + 1.0)
    y_min = float(np.min(projected_points_array[:, 1]) - 1.0)
    y_max = float(np.max(projected_points_array[:, 1]) + 1.0)

    grid_x = np.linspace(x_min, x_max, 300)
    grid_y = np.linspace(y_min, y_max, 300)
    xx, yy = np.meshgrid(grid_x, grid_y)

    zz = np.zeros_like(xx, dtype=float)

    for row in range(xx.shape[0]):
        for col in range(xx.shape[1]):
            point = fixed_point.copy()
            point[variable_i] = xx[row, col]
            point[variable_j] = yy[row, col]
            zz[row, col] = function._value(point)

    finite_values = zz[np.isfinite(zz)]
    min_level = float(np.nanmin(finite_values))
    max_level = float(np.nanpercentile(finite_values, 95))

    if max_level <= min_level:
        max_level = float(np.nanmax(finite_values))

    levels = np.linspace(min_level, max_level, 35)

    plt.figure(figsize=(10, 8))

    contour = plt.contour(xx, yy, zz, levels=levels, linewidths=0.8)
    plt.clabel(contour, inline=True, fontsize=7, fmt="%.2f")

    for result in results:
        track = _get_track(result)
        projected_track = track[:, [variable_i, variable_j]]

        plt.plot(
            projected_track[:, 0],
            projected_track[:, 1],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{result.method_name}, k={result.iterations}",
        )

        plt.scatter(
            result.x_star[variable_i],
            result.x_star[variable_j],
            marker="x",
            s=80,
        )

    plt.scatter(
        function.default_x0[variable_i],
        function.default_x0[variable_j],
        marker="s",
        s=90,
        label="Начальная точка",
    )

    plt.scatter(
        fixed_point[variable_i],
        fixed_point[variable_j],
        marker="*",
        s=150,
        label="Лучшая найденная точка",
    )

    plt.title(
        f"{function.name}: 2D-сечение по x{variable_i + 1}, x{variable_j + 1}"
    )
    plt.xlabel(f"x{variable_i + 1}")
    plt.ylabel(f"x{variable_j + 1}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def plot_polynomial_sections(
    functions: list[ObjectiveFunction],
    results: list[OptimizationResult],
) -> list[Path]:
    """Построение 2D-сечений для всех 10D-функций."""

    paths: list[Path] = []

    for function in functions:
        if function.dimension != 10:
            continue

        function_results = [
            result
            for result in results
            if result.function_name == function.name
        ]

        if not function_results:
            continue

        paths.append(
            plot_polynomial_2d_section(
                function=function,
                results=function_results,
                variable_i=0,
                variable_j=1,
            )
        )

    return paths