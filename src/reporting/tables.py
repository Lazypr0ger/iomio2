from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import OptimizationResult


def result_to_summary_row(result: OptimizationResult) -> dict[str, object]:
    """Строка сводной таблицы для одного результата."""

    x_star_text = np.array2string(
        result.x_star,
        precision=6,
        separator=", ",
        suppress_small=True,
    )

    return {
        "function": result.function_name,
        "method": result.method_name,
        "converged": result.converged,
        "iterations": result.iterations,
        "function_evaluations": result.function_evaluations,
        "gradient_evaluations": result.gradient_evaluations,
        "hessian_evaluations": result.hessian_evaluations,
        "x_star": x_star_text,
        "f_star": result.f_star,
        "grad_norm": result.grad_norm,
        "message": result.message,
    }


def results_to_summary_dataframe(
    results: list[OptimizationResult],
) -> pd.DataFrame:
    """Сводная таблица по всем функциям и методам."""

    rows = [result_to_summary_row(result) for result in results]
    return pd.DataFrame(rows)


def result_history_to_dataframe(result: OptimizationResult) -> pd.DataFrame:
    """Таблица итераций одного метода."""

    rows: list[dict[str, object]] = []

    for record in result.history:
        row: dict[str, object] = {
            "k": record.k,
            "f_value": record.f_value,
            "grad_norm": record.grad_norm,
            "step_size": record.step_size,
        }

        if result.x_star.size == 2:
            row["x1"] = record.x[0]
            row["x2"] = record.x[1]
        else:
            row["x"] = np.array2string(
                record.x,
                precision=6,
                separator=", ",
                suppress_small=True,
            )
            if record.k > 0:
                previous_x = result.history[record.k - 1].x
                row["step_norm"] = float(np.linalg.norm(record.x - previous_x))
            else:
                row["step_norm"] = None

        for key, value in record.extra.items():
            row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)