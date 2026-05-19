from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from src.config import REPORTS_DIR
from src.functions.base import ObjectiveFunction
from src.models import OptimizationResult
from src.reporting.tables import (
    result_history_to_dataframe,
    results_to_summary_dataframe,
)
from src.visualization.contours import plot_contour_with_tracks
from src.visualization.convergence import (
    plot_function_decrease,
    plot_gradient_norm_convergence,
)
from src.visualization.polynomial_sections import plot_polynomial_2d_section


FUNCTION_RU_NAMES = {
    "Himmelblau": "Химмельблау",
    "Rosenbrock": "Розенброк",
    "Booth": "Бут",
    "Beale": "Бил",
    "Matyas": "Матьяс",
    "ThreeHumpCamel": "Трёхгорбый верблюд",
    "Sphere": "Сфера",
    "Rastrigin": "Растригин",
    "Polynomial10DVariant3": "Полином 10D, вариант 3",
    "CustomPolynomial10D": "Произвольный полином 10D",
}


METHOD_RU_NAMES = {
    "CG Fletcher-Reeves": "CG (Фл.-Ривз)",
    "Modified Newton": "Ньютон (мод.)",
    "BFGS": "BFGS",
}


METHOD_FULL_RU_NAMES = {
    "CG Fletcher-Reeves": "Сопряжённые градиенты Флетчера–Ривза",
    "Modified Newton": "Модифицированный метод Ньютона",
    "BFGS": "BFGS",
}


SUMMARY_COLUMNS_RU = {
    "function": "Функция",
    "method": "Метод",
    "converged": "Сошёлся",
    "iterations": "Итер.",
    "function_evaluations": "Выч. f",
    "gradient_evaluations": "Выч. ∇f",
    "hessian_evaluations": "Выч. H",
    "x_star": "x*",
    "f_star": "f(x*)",
    "grad_norm": "||∇f||",
    "distance_to_known_minimum": "||x* − x*_т||",
}


def get_function_ru_name(name: str) -> str:
    """Русское название функции."""

    return FUNCTION_RU_NAMES.get(name, name)


def get_method_ru_name(name: str) -> str:
    """Краткое русское название метода для таблиц по шаблону."""

    return METHOD_RU_NAMES.get(name, name)


def get_method_full_ru_name(name: str) -> str:
    """Полное русское название метода."""

    return METHOD_FULL_RU_NAMES.get(name, name)


def format_array(array: np.ndarray, precision: int = 6) -> str:
    """Форматирование вектора."""

    return np.array2string(
        array,
        precision=precision,
        separator=", ",
        suppress_small=True,
    )


def format_value(value: object) -> str:
    """Форматирование значения для ячейки таблицы."""

    if value is None:
        return "—"

    if isinstance(value, bool):
        return "Да" if value else "Нет"

    if isinstance(value, float):
        if abs(value) < 1e-4 and value != 0:
            return f"{value:.3e}"
        return f"{value:.6f}".rstrip("0").rstrip(".")

    if isinstance(value, np.floating):
        float_value = float(value)

        if abs(float_value) < 1e-4 and float_value != 0:
            return f"{float_value:.3e}"

        return f"{float_value:.6f}".rstrip("0").rstrip(".")

    if isinstance(value, np.integer):
        return str(int(value))

    return str(value)


def format_extra_info(extra: dict[str, object]) -> str:
    """Форматирование дополнительной информации итерации."""

    if not extra:
        return "—"

    if "beta" in extra and extra["beta"] is not None:
        return f"β={format_value(extra['beta'])}"

    if "regularization" in extra and extra["regularization"] is not None:
        return f"reg={format_value(extra['regularization'])}"

    if "rho" in extra and extra["rho"] is not None:
        return f"ρ={format_value(extra['rho'])}"

    if "update" in extra and extra["update"] is not None:
        return str(extra["update"])

    return "—"


def get_distance_to_known_minimum(
    function: ObjectiveFunction,
    result: OptimizationResult,
) -> float | None:
    """
    Расстояние до ближайшего точного минимума.

    Для полинома точный минимум неизвестен, поэтому возвращается None.
    """

    if not function.known_minima:
        return None

    distances = [
        float(np.linalg.norm(result.x_star - known_minimum))
        for known_minimum in function.known_minima
    ]

    return min(distances)


def add_heading(document: Document, text: str, level: int = 1) -> None:
    """Добавление заголовка."""

    document.add_heading(text, level=level)


def add_paragraph(document: Document, text: str = "") -> None:
    """Добавление обычного абзаца."""

    paragraph = document.add_paragraph(text)
    paragraph_format = paragraph.paragraph_format
    paragraph_format.space_after = Pt(6)


def add_centered_title(document: Document, text: str) -> None:
    """Добавление центрированного заголовка."""

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(16)


def set_table_font_size(table, font_size: int = 9) -> None:
    """Установка размера шрифта в таблице."""

    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)


def add_dataframe_table(
    document: Document,
    dataframe: pd.DataFrame,
    max_rows: int | None = None,
    font_size: int = 9,
) -> None:
    """Добавление DataFrame как таблицы Word."""

    if dataframe.empty:
        add_paragraph(document, "Нет данных для таблицы.")
        return

    table_dataframe = dataframe.copy()

    if max_rows is not None and len(table_dataframe) > max_rows:
        head = table_dataframe.head(max_rows // 2)
        tail = table_dataframe.tail(max_rows - len(head))
        separator = pd.DataFrame(
            [
                {
                    column: "..."
                    for column in table_dataframe.columns
                }
            ]
        )
        table_dataframe = pd.concat([head, separator, tail], ignore_index=True)

    table = document.add_table(
        rows=1,
        cols=len(table_dataframe.columns),
    )
    table.style = "Table Grid"

    header_cells = table.rows[0].cells
    for index, column_name in enumerate(table_dataframe.columns):
        header_cells[index].text = str(column_name)

    for _, row in table_dataframe.iterrows():
        cells = table.add_row().cells

        for index, value in enumerate(row):
            cells[index].text = format_value(value)

    set_table_font_size(table, font_size=font_size)


def add_image_if_exists(
    document: Document,
    image_path: Path,
    width_inches: float = 6.5,
) -> None:
    """Добавление изображения в отчёт."""

    if not image_path.exists():
        add_paragraph(document, f"График не найден: {image_path}")
        return

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))


def prepare_2d_result_template_dataframe(
    results: list[OptimizationResult],
) -> pd.DataFrame:
    """
    Таблица результата для 2D-функции в формате шаблона презентации.

    Шаблон:
        Метод | k | x₁* | x₂* | f(x*) | ||∇f|| | Итер. | Выч. f
    """

    rows: list[dict[str, object]] = []

    for result in results:
        rows.append(
            {
                "Метод": get_method_ru_name(result.method_name),
                "k": result.iterations,
                "x₁*": result.x_star[0],
                "x₂*": result.x_star[1],
                "f(x*)": result.f_star,
                "||∇f||": result.grad_norm,
                "Итер.": result.iterations,
                "Выч. f": result.function_evaluations,
            }
        )

    return pd.DataFrame(rows)


def prepare_polynomial_result_template_dataframe(
    results: list[OptimizationResult],
) -> pd.DataFrame:
    """
    Таблица результата для полинома 10D в формате шаблона презентации.

    Шаблон:
        Метод | k | f(x*) | ||∇f(x*)|| | Итерации | Выч. f | Выч. ∇f
    """

    rows: list[dict[str, object]] = []

    for result in results:
        rows.append(
            {
                "Метод": get_method_ru_name(result.method_name),
                "k": result.iterations,
                "f(x*)": result.f_star,
                "||∇f(x*)||": result.grad_norm,
                "Итерации": result.iterations,
                "Выч. f": result.function_evaluations,
                "Выч. ∇f": result.gradient_evaluations,
            }
        )

    return pd.DataFrame(rows)


def prepare_history_dataframe(
    result: OptimizationResult,
    max_rows: int = 12,
) -> pd.DataFrame:
    """
    Таблица итераций в формате шаблонов презентации.

    Для 2D:
        k | x₁ᵏ | x₂ᵏ | f(xᵏ) | ||∇f(xᵏ)|| | λᵏ | βᵏ / доп.

    Для 10D:
        k | f(xᵏ) | ||∇f(xᵏ)|| | λᵏ | ||xᵏ − xᵏ⁻¹||
    """

    rows: list[dict[str, object]] = []

    for index, record in enumerate(result.history):
        if result.x_star.size == 2:
            row: dict[str, object] = {
                "k": record.k,
                "x₁ᵏ": record.x[0],
                "x₂ᵏ": record.x[1],
                "f(xᵏ)": record.f_value,
                "||∇f(xᵏ)||": record.grad_norm,
                "λᵏ": record.step_size,
                "βᵏ / доп.": format_extra_info(record.extra),
            }
        else:
            if index == 0:
                step_norm = None
            else:
                previous_x = result.history[index - 1].x
                step_norm = float(np.linalg.norm(record.x - previous_x))

            row = {
                "k": record.k,
                "f(xᵏ)": record.f_value,
                "||∇f(xᵏ)||": record.grad_norm,
                "λᵏ": record.step_size,
                "||xᵏ - xᵏ⁻¹||": step_norm,
            }

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    if len(dataframe) > max_rows:
        head = dataframe.head(max_rows // 2)
        tail = dataframe.tail(max_rows - len(head))
        separator = pd.DataFrame(
            [
                {
                    column: "..."
                    for column in dataframe.columns
                }
            ]
        )
        dataframe = pd.concat([head, separator, tail], ignore_index=True)

    return dataframe


def prepare_global_summary_dataframe(
    functions: list[ObjectiveFunction],
    results: list[OptimizationResult],
) -> pd.DataFrame:
    """
    Общая сводная таблица в формате, близком к шаблону.

    Колонки:
        Функция | Метод | Итер. | Выч. f | Выч. ∇f | x* | f(x*) | ||x* − x*_т||
    """

    rows: list[dict[str, object]] = []

    function_by_name = {
        function.name: function
        for function in functions
    }

    for result in results:
        function = function_by_name.get(result.function_name)
        distance = (
            get_distance_to_known_minimum(function, result)
            if function is not None
            else None
        )

        rows.append(
            {
                "Функция": get_function_ru_name(result.function_name),
                "Метод": get_method_ru_name(result.method_name),
                "Итер.": result.iterations,
                "Выч. f": result.function_evaluations,
                "Выч. ∇f": result.gradient_evaluations,
                "x*": format_array(result.x_star),
                "f(x*)": result.f_star,
                "||x*−x*_т||": distance,
            }
        )

    return pd.DataFrame(rows)


def get_results_for_function(
    function: ObjectiveFunction,
    results: list[OptimizationResult],
) -> list[OptimizationResult]:
    """Получение результатов для одной функции."""

    return [
        result
        for result in results
        if result.function_name == function.name
    ]


def add_title_page(document: Document) -> None:
    """Титульная часть."""

    add_centered_title(document, "Лабораторная работа №2")
    add_centered_title(document, "Многомерная нелинейная оптимизация")

    add_paragraph(document)
    add_paragraph(document, "Вариант: 3")
    add_paragraph(
        document,
        "Методы: сопряжённые градиенты Флетчера–Ривза, "
        "модифицированный метод Ньютона, BFGS.",
    )
    add_paragraph(
        document,
        "Цель работы: реализовать и сравнить методы безусловной "
        "минимизации на тестовых функциях и полиноме 10D.",
    )

    document.add_page_break()


def add_problem_statement(document: Document, epsilon: float) -> None:
    """Постановка задачи."""

    add_heading(document, "Постановка задачи", level=1)

    add_paragraph(
        document,
        "Необходимо решить задачу безусловной минимизации функции f(x) "
        "тремя методами: методом сопряжённых градиентов, модифицированным "
        "методом Ньютона и методом BFGS.",
    )

    add_paragraph(
        document,
        f"Критерий остановки: ||∇f(xᵏ)|| ≤ ε, где ε = {epsilon:g}. "
        "Длина шага выбирается с помощью бэктрекинга Армихо.",
    )

    add_paragraph(
        document,
        "Для двумерных тестовых функций используются аналитические "
        "градиенты и Гессианы. Для полинома 10D в программе также "
        "реализованы аналитический градиент и аналитический Гессиан.",
    )


def add_variant_description(document: Document) -> None:
    """Описание варианта 3."""

    add_heading(document, "Функции варианта 3", level=1)

    dataframe = pd.DataFrame(
        [
            {
                "№": 1,
                "Функция": "Бут",
                "Начальная точка": "(-5, -5)",
            },
            {
                "№": 2,
                "Функция": "Сфера",
                "Начальная точка": "(-3, -3)",
            },
            {
                "№": 3,
                "Функция": "Розенброк",
                "Начальная точка": "(-2, 2)",
            },
            {
                "№": 4,
                "Функция": "Химмельблау",
                "Начальная точка": "(-1, 0)",
            },
            {
                "№": 5,
                "Функция": "Полином 10D",
                "Начальная точка": "(-1.5, -2.4, 2.5, 0, ..., 0)",
            },
        ]
    )

    add_dataframe_table(document, dataframe, font_size=10)


def add_method_summary(document: Document) -> None:
    """Описание методов."""

    add_heading(document, "Краткое описание методов", level=1)

    add_heading(document, "Метод сопряжённых градиентов Флетчера–Ривза", level=2)
    add_paragraph(
        document,
        "В методе используется направление p₀ = -g₀, а последующие "
        "направления строятся по формуле pₖ = -gₖ + βₖpₖ₋₁. "
        "Для варианта Флетчера–Ривза βₖ = ||gₖ||² / ||gₖ₋₁||².",
    )

    add_heading(document, "Модифицированный метод Ньютона", level=2)
    add_paragraph(
        document,
        "На каждой итерации решается система Hₖpₖ = -gₖ. "
        "Если Гессиан не является положительно определённым, "
        "используется регуляризация Hₖ + μI.",
    )

    add_heading(document, "Метод BFGS", level=2)
    add_paragraph(
        document,
        "BFGS является квазиньютоновским методом. В реализации хранится "
        "приближение обратного Гессиана, которое обновляется по формуле BFGS "
        "при выполнении условия кривизны yₖᵀsₖ > 0.",
    )


def add_2d_function_section(
    document: Document,
    function: ObjectiveFunction,
    function_results: list[OptimizationResult],
    epsilon: float,
) -> None:
    """Раздел результата для 2D-функции по шаблону презентации."""

    function_name = get_function_ru_name(function.name)

    add_heading(document, f"Результат для 2D-функции: {function_name}", level=1)

    add_paragraph(
        document,
        f"Функция: {function_name}     x⁰ = {format_array(function.default_x0)}",
    )

    if function.known_minima:
        minima_text = "; ".join(
            format_array(point)
            for point in function.known_minima
        )
        add_paragraph(document, f"Точные минимумы: {minima_text}.")

    function_summary = prepare_2d_result_template_dataframe(function_results)
    add_dataframe_table(document, function_summary, font_size=9)

    add_heading(document, "График: линии уровня + треки трёх методов", level=2)
    contour_path = plot_contour_with_tracks(function, function_results)
    add_image_if_exists(document, contour_path)

    add_heading(document, "График сходимости", level=2)
    convergence_path = plot_gradient_norm_convergence(
        function=function,
        results=function_results,
        epsilon=epsilon,
    )
    add_image_if_exists(document, convergence_path)

    add_heading(document, "График убывания значения функции", level=2)
    decrease_path = plot_function_decrease(function, function_results)
    add_image_if_exists(document, decrease_path)

    add_heading(document, "Таблицы итераций методов", level=2)
    add_paragraph(
        document,
        "В таблицах показаны первые и последние итерации. "
        "Полные таблицы сохраняются в outputs/tables.",
    )

    for result in function_results:
        add_heading(document, get_method_full_ru_name(result.method_name), level=3)
        add_paragraph(
            document,
            f"Функция: {function_name}     Метод: "
            f"{get_method_full_ru_name(result.method_name)}     "
            f"x⁰ = {format_array(function.default_x0)}",
        )
        history_dataframe = prepare_history_dataframe(result)
        add_dataframe_table(document, history_dataframe, font_size=8)

        add_paragraph(
            document,
            f"Результат: x* ≈ {format_array(result.x_star)}, "
            f"f(x*) ≈ {format_value(result.f_star)}, "
            f"||∇f(x*)|| = {format_value(result.grad_norm)}. "
            f"Число итераций: {result.iterations}.",
        )

    best_result = min(function_results, key=lambda item: item.f_star)

    add_paragraph(
        document,
        f"Вывод: наименьшее значение функции получено методом "
        f"{get_method_full_ru_name(best_result.method_name)}: "
        f"f(x*) ≈ {format_value(best_result.f_star)}, "
        f"||∇f(x*)|| ≈ {format_value(best_result.grad_norm)}.",
    )

    document.add_page_break()


def add_polynomial_section(
    document: Document,
    function: ObjectiveFunction,
    function_results: list[OptimizationResult],
    epsilon: float,
) -> None:
    """Раздел результата для полинома 10D по шаблону презентации."""

    function_name = get_function_ru_name(function.name)

    add_heading(document, f"Результат для полинома 10D: {function_name}", level=1)

    add_paragraph(
        document,
        f"Полином: вариант 3     x⁰ = {format_array(function.default_x0)}",
    )

    function_summary = prepare_polynomial_result_template_dataframe(function_results)
    add_dataframe_table(document, function_summary, font_size=9)

    add_heading(document, "Полные найденные векторы x*", level=2)

    rows = []
    for result in function_results:
        rows.append(
            {
                "Метод": get_method_ru_name(result.method_name),
                "x*": format_array(result.x_star),
            }
        )

    add_dataframe_table(document, pd.DataFrame(rows), font_size=8)

    add_heading(document, "График сходимости", level=2)
    convergence_path = plot_gradient_norm_convergence(
        function=function,
        results=function_results,
        epsilon=epsilon,
    )
    add_image_if_exists(document, convergence_path)

    add_heading(document, "График убывания значения функции", level=2)
    decrease_path = plot_function_decrease(function, function_results)
    add_image_if_exists(document, decrease_path)

    add_heading(document, "2D-сечение полинома", level=2)
    section_path = plot_polynomial_2d_section(
        function=function,
        results=function_results,
        variable_i=0,
        variable_j=1,
    )
    add_image_if_exists(document, section_path)

    add_heading(document, "Таблицы итераций методов", level=2)
    add_paragraph(
        document,
        "В таблицах показаны первые и последние итерации. "
        "Полные таблицы сохраняются в outputs/tables.",
    )

    for result in function_results:
        add_heading(document, get_method_full_ru_name(result.method_name), level=3)
        history_dataframe = prepare_history_dataframe(result)
        add_dataframe_table(document, history_dataframe, font_size=8)

        add_paragraph(
            document,
            f"Результат: x* = {format_array(result.x_star)}, "
            f"f(x*) ≈ {format_value(result.f_star)}, "
            f"||∇f(x*)|| = {format_value(result.grad_norm)}. "
            f"Число итераций: {result.iterations}.",
        )

    best_result = min(function_results, key=lambda item: item.f_star)

    add_paragraph(
        document,
        f"Вывод: лучшее значение для полинома получено методом "
        f"{get_method_full_ru_name(best_result.method_name)}: "
        f"f(x*) ≈ {format_value(best_result.f_star)}, "
        f"||∇f(x*)|| ≈ {format_value(best_result.grad_norm)}.",
    )

    document.add_page_break()


def add_global_summary(
    document: Document,
    functions: list[ObjectiveFunction],
    results: list[OptimizationResult],
) -> None:
    """Общая сводная таблица сравнения."""

    add_heading(document, "Сводная таблица сравнения", level=1)

    summary_dataframe = prepare_global_summary_dataframe(functions, results)
    add_dataframe_table(document, summary_dataframe, font_size=8)


def add_final_conclusion(
    document: Document,
    results: list[OptimizationResult],
) -> None:
    """Общий вывод."""

    add_heading(document, "Общий вывод", level=1)

    add_paragraph(
        document,
        "Все три обязательных метода были реализованы и применены к функциям "
        "варианта 3. Для всех двумерных функций построены линии уровня с "
        "траекториями методов, а также графики сходимости и убывания значения "
        "целевой функции.",
    )

    add_paragraph(
        document,
        "Модифицированный метод Ньютона обычно требует меньше итераций, "
        "поскольку использует информацию второго порядка. Однако одна итерация "
        "метода Ньютона дороже из-за вычисления и обработки Гессиана.",
    )

    add_paragraph(
        document,
        "Метод BFGS показывает устойчивую сходимость и часто требует число "
        "итераций, близкое к методу Ньютона, но не требует явного вычисления "
        "Гессиана.",
    )

    add_paragraph(
        document,
        "Метод сопряжённых градиентов имеет простую структуру и низкую стоимость "
        "итерации, но на овражных функциях, например на функции Розенброка, "
        "может требовать существенно больше итераций.",
    )

    converged_count = sum(result.converged for result in results)
    add_paragraph(
        document,
        f"Количество успешных запусков: {converged_count} из {len(results)}.",
    )


def configure_document_styles(document: Document) -> None:
    """Базовые стили документа."""

    styles = document.styles

    normal_style = styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)

    for style_name in ["Heading 1", "Heading 2", "Heading 3"]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style.font.bold = True


def generate_docx_report(
    functions: list[ObjectiveFunction],
    results: list[OptimizationResult],
    epsilon: float,
    output_path: Path | None = None,
) -> Path:
    """
    Создание DOCX-отчёта по результатам лабораторной работы.

    Отчёт строится по структуре шаблонов презентации:
    - результат для каждой 2D-функции;
    - таблица итераций методов;
    - результат для полинома 10D;
    - таблица итераций 10D;
    - сводная таблица сравнения;
    - графики сходимости и убывания.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = REPORTS_DIR / "report_variant_3.docx"

    document = Document()
    configure_document_styles(document)

    add_title_page(document)
    add_problem_statement(document, epsilon)
    add_variant_description(document)
    add_method_summary(document)

    for function in functions:
        function_results = get_results_for_function(function, results)

        if not function_results:
            continue

        if function.dimension == 2:
            add_2d_function_section(
                document=document,
                function=function,
                function_results=function_results,
                epsilon=epsilon,
            )
        elif function.dimension == 10:
            add_polynomial_section(
                document=document,
                function=function,
                function_results=function_results,
                epsilon=epsilon,
            )

    add_global_summary(document, functions, results)
    add_final_conclusion(document, results)

    document.save(output_path)

    return output_path