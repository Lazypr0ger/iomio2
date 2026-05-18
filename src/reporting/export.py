from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import TABLES_DIR
from src.models import OptimizationResult
from src.reporting.tables import (
    result_history_to_dataframe,
    results_to_summary_dataframe,
)


def safe_filename(text: str) -> str:
    """Безопасное имя файла."""

    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def export_summary_csv(results: list[OptimizationResult]) -> Path:
    """Экспорт сводной таблицы в CSV."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = results_to_summary_dataframe(results)
    path = TABLES_DIR / "summary_variant_3.csv"

    dataframe.to_csv(path, index=False, encoding="utf-8-sig")

    return path


def export_iteration_tables_csv(results: list[OptimizationResult]) -> list[Path]:
    """Экспорт таблиц итераций в CSV."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    for result in results:
        dataframe = result_history_to_dataframe(result)

        file_name = (
            f"iterations_{safe_filename(result.function_name)}_"
            f"{safe_filename(result.method_name)}.csv"
        )

        path = TABLES_DIR / file_name
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)

    return paths


def export_summary_excel(results: list[OptimizationResult]) -> Path:
    """Экспорт всех таблиц в один Excel-файл."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    path = TABLES_DIR / "results_variant_3.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_dataframe = results_to_summary_dataframe(results)
        summary_dataframe.to_excel(writer, sheet_name="summary", index=False)

        for result in results:
            dataframe = result_history_to_dataframe(result)

            sheet_name = safe_filename(
                f"{result.function_name}_{result.method_name}"
            )[:31]

            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

    return path


def export_all_tables(results: list[OptimizationResult]) -> None:
    """Экспорт всех табличных результатов."""

    summary_csv = export_summary_csv(results)
    excel_path = export_summary_excel(results)
    iteration_paths = export_iteration_tables_csv(results)

    print("=" * 80)
    print("Экспорт таблиц завершён")
    print(f"Сводная CSV-таблица: {summary_csv}")
    print(f"Excel-файл: {excel_path}")
    print(f"CSV-таблицы итераций: {len(iteration_paths)} шт.")