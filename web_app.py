from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import streamlit as st

from src.config import ensure_output_dirs
from src.functions.base import ObjectiveFunction
from src.functions.builtin_2d import get_all_builtin_functions, get_variant_3_functions
from src.functions.polynomial_10d import (
    CrossTerm,
    HighPowerTerm,
    Polynomial10DFunction,
    get_variant_3_polynomial,
)
from src.methods.bfgs import bfgs
from src.methods.conjugate_gradient import conjugate_gradient_fletcher_reeves
from src.methods.newton import modified_newton
from src.models import OptimizationParams, OptimizationResult
from src.reporting.export import export_summary_csv, export_summary_excel
from src.reporting.tables import result_history_to_dataframe, results_to_summary_dataframe
from src.visualization.contours import plot_contour_with_tracks
from src.visualization.convergence import (
    plot_function_decrease,
    plot_gradient_norm_convergence,
)
from src.visualization.polynomial_sections import plot_polynomial_2d_section


def get_methods():
    """Три обязательных метода лабораторной работы."""

    return [
        conjugate_gradient_fletcher_reeves,
        modified_newton,
        bfgs,
    ]


def get_default_params() -> OptimizationParams:
    """Параметры методов по умолчанию."""

    return OptimizationParams(
        epsilon=1e-3,
        max_iterations=1000,
        initial_step=1.0,
        armijo_c1=1e-4,
        backtracking_factor=0.5,
        min_step=1e-12,
        cg_restart_frequency=None,
        newton_regularization=1e-6,
    )


def run_methods(
    function: ObjectiveFunction,
    params: OptimizationParams,
) -> list[OptimizationResult]:
    """Запуск всех обязательных методов для одной функции."""

    results: list[OptimizationResult] = []

    for method in get_methods():
        result = method(function, function.default_x0, params)
        results.append(result)

    return results


def get_function_catalog() -> dict[str, ObjectiveFunction]:
    """Каталог встроенных функций и полинома варианта 3."""

    functions = get_all_builtin_functions()
    functions["Polynomial10DVariant3"] = get_variant_3_polynomial()
    return functions


def parse_vector(raw_value: str, expected_size: int) -> np.ndarray:
    """Ввод вектора через пробел или запятую."""

    cleaned = raw_value.replace(",", " ")
    values = [float(item) for item in cleaned.split()]
    vector = np.array(values, dtype=float)

    if vector.shape != (expected_size,):
        raise ValueError(
            f"Ожидался вектор размера {expected_size}, получен размер {vector.size}."
        )

    return vector


def parse_float_vector(raw_value: str, expected_size: int) -> np.ndarray:
    """Ввод числового вектора через пробел, запятую или точку с запятой."""

    cleaned = raw_value.replace(",", " ").replace(";", " ")
    values = [float(item) for item in cleaned.split()]
    vector = np.array(values, dtype=float)

    if vector.shape != (expected_size,):
        raise ValueError(
            f"Ожидался вектор размера {expected_size}, получен размер {vector.size}."
        )

    return vector


def parse_cross_terms(raw_value: str) -> list[CrossTerm]:
    """
    Парсинг перекрёстных членов.

    Формат:
        i j coefficient; i j coefficient

    Индексы вводятся с 1, как в математической записи.

    Пример:
        1 2 -1.3; 2 3 0.8; 4 5 -1.1
    """

    if not raw_value.strip():
        return []

    terms: list[CrossTerm] = []

    for raw_term in raw_value.split(";"):
        raw_term = raw_term.strip()

        if not raw_term:
            continue

        parts = raw_term.replace(",", " ").split()

        if len(parts) != 3:
            raise ValueError(
                "Каждый перекрёстный член должен иметь формат: i j coefficient."
            )

        i = int(parts[0]) - 1
        j = int(parts[1]) - 1
        coefficient = float(parts[2])

        if not 0 <= i < 10 or not 0 <= j < 10:
            raise ValueError("Индексы перекрёстных членов должны быть от 1 до 10.")

        if i == j:
            raise ValueError("Для перекрёстного члена индексы i и j должны отличаться.")

        terms.append(
            CrossTerm(
                i=i,
                j=j,
                coefficient=coefficient,
            )
        )

    return terms


def build_custom_polynomial_from_ui() -> ObjectiveFunction | None:
    """Форма ввода произвольного полинома 10D."""

    st.subheader("Произвольный полином 10D")

    st.markdown(
        """
        Формат полинома:

        `Σ dᵢxᵢ² + Σ cᵢⱼxᵢxⱼ + Σ lᵢxᵢ + αxₖᵖ`

        Индексы переменных вводятся с 1: `x1, x2, ..., x10`.
        """
    )

    diagonal_text = st.text_input(
        "Диагональные коэффициенты dᵢ, 10 чисел",
        value="9.8 5.1 7.5 4.2 6.4 3.8 8.1 2.9 5.7 4.6",
    )

    cross_terms_text = st.text_area(
        "Перекрёстные члены: i j coefficient; ...",
        value="1 2 -1.3; 2 3 0.8; 4 5 -1.1; 6 8 1.4; 9 10 -0.9",
        help="Пример: 1 2 -1.3; 2 3 0.8",
    )

    linear_text = st.text_input(
        "Линейные коэффициенты lᵢ, 10 чисел",
        value="1.2 -2.0 0.7 0.0 -1.5 2.1 0.0 -0.8 1.4 0.0",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        high_power_index = st.number_input(
            "Индекс xₖ",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

    with col2:
        high_power_coefficient = st.number_input(
            "Коэффициент α",
            value=0.043,
            format="%.6f",
        )

    with col3:
        high_power_power = st.number_input(
            "Степень p",
            min_value=2,
            max_value=10,
            value=6,
            step=1,
        )

    x0_text = st.text_input(
        "Начальная точка x0, 10 чисел",
        value="-1.5 -2.4 2.5 0 0 0 0 0 0 0",
    )

    try:
        diagonal = parse_float_vector(diagonal_text, 10)
        cross_terms = parse_cross_terms(cross_terms_text)
        linear = parse_float_vector(linear_text, 10)
        x0 = parse_float_vector(x0_text, 10)

        return Polynomial10DFunction(
            name="CustomPolynomial10D",
            diagonal=diagonal,
            cross_terms=cross_terms,
            linear=linear,
            high_power_term=HighPowerTerm(
                index=int(high_power_index) - 1,
                coefficient=float(high_power_coefficient),
                power=int(high_power_power),
            ),
            default_x0=x0,
        )

    except ValueError as error:
        st.error(str(error))
        return None


def make_download_zip(file_paths: list[Path], zip_name: str) -> Path:
    """Создание ZIP-архива с графиками."""

    output_path = Path("outputs") / zip_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, "w") as zip_file:
        for file_path in file_paths:
            if file_path.exists():
                zip_file.write(file_path, arcname=file_path.name)

    return output_path


def download_file_button(path: Path, label: str, mime: str) -> None:
    """Кнопка скачивания одного файла."""

    if not path.exists():
        st.warning(f"Файл не найден: {path}")
        return

    with path.open("rb") as file:
        st.download_button(
            label=label,
            data=file,
            file_name=path.name,
            mime=mime,
        )


def display_downloads(
    results: list[OptimizationResult],
    graph_paths: list[Path],
    prefix: str,
) -> None:
    """Кнопки скачивания результатов."""

    st.header("Скачивание результатов")

    summary_csv = export_summary_csv(results)
    summary_excel = export_summary_excel(results)

    col1, col2, col3 = st.columns(3)

    with col1:
        download_file_button(
            path=summary_csv,
            label="Скачать сводную таблицу CSV",
            mime="text/csv",
        )

    with col2:
        download_file_button(
            path=summary_excel,
            label="Скачать таблицы Excel",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col3:
        zip_path = make_download_zip(
            file_paths=graph_paths,
            zip_name=f"{prefix}_plots.zip",
        )

        download_file_button(
            path=zip_path,
            label="Скачать графики ZIP",
            mime="application/zip",
        )


def display_result_cards(results: list[OptimizationResult]) -> None:
    """Карточки результатов по методам."""

    columns = st.columns(3)

    for column, result in zip(columns, results):
        with column:
            st.subheader(result.method_name)
            st.metric("Итерации", result.iterations)
            st.metric("f(x*)", f"{result.f_star:.6g}")
            st.metric("||∇f(x*)||", f"{result.grad_norm:.6g}")

            if result.converged:
                st.success("Сошёлся")
            else:
                st.warning("Не сошёлся")

            st.caption(result.message)


def display_iteration_tables(results: list[OptimizationResult]) -> None:
    """Таблицы итераций."""

    st.header("Таблицы итераций")

    tabs = st.tabs([result.method_name for result in results])

    for tab, result in zip(tabs, results):
        with tab:
            dataframe = result_history_to_dataframe(result)
            st.dataframe(dataframe, use_container_width=True)


def display_graphs(
    function: ObjectiveFunction,
    results: list[OptimizationResult],
    params: OptimizationParams,
) -> list[Path]:
    """Построение и отображение графиков."""

    st.header("Графики")

    graph_paths: list[Path] = []

    if function.dimension == 2:
        contour_path = plot_contour_with_tracks(function, results)
        graph_paths.append(contour_path)

        st.subheader("Линии уровня и траектории методов")
        st.image(str(contour_path), use_container_width=True)

    gradient_path = plot_gradient_norm_convergence(
        function=function,
        results=results,
        epsilon=params.epsilon,
    )
    graph_paths.append(gradient_path)

    st.subheader("Сходимость по норме градиента")
    st.image(str(gradient_path), use_container_width=True)

    decrease_path = plot_function_decrease(
        function=function,
        results=results,
    )
    graph_paths.append(decrease_path)

    st.subheader("Убывание значения функции")
    st.image(str(decrease_path), use_container_width=True)

    if function.dimension == 10:
        section_path = plot_polynomial_2d_section(
            function=function,
            results=results,
            variable_i=0,
            variable_j=1,
        )
        graph_paths.append(section_path)

        st.subheader("2D-сечение полинома по x₁ и x₂")
        st.image(str(section_path), use_container_width=True)

    return graph_paths


def sidebar_params(default_params: OptimizationParams) -> OptimizationParams:
    """Параметры методов в боковой панели."""

    st.sidebar.header("Гиперпараметры")

    epsilon = st.sidebar.number_input(
        "epsilon",
        min_value=1e-12,
        value=default_params.epsilon,
        format="%.12f",
    )

    max_iterations = st.sidebar.number_input(
        "max_iterations",
        min_value=1,
        max_value=10000,
        value=default_params.max_iterations,
        step=100,
    )

    initial_step = st.sidebar.number_input(
        "initial_step",
        min_value=1e-12,
        value=default_params.initial_step,
        format="%.6f",
    )

    armijo_c1 = st.sidebar.number_input(
        "armijo_c1",
        min_value=1e-12,
        max_value=0.5,
        value=default_params.armijo_c1,
        format="%.8f",
    )

    backtracking_factor = st.sidebar.number_input(
        "backtracking_factor",
        min_value=0.01,
        max_value=0.99,
        value=default_params.backtracking_factor,
        format="%.4f",
    )

    min_step = st.sidebar.number_input(
        "min_step",
        min_value=1e-16,
        value=default_params.min_step,
        format="%.16f",
    )

    use_restart = st.sidebar.checkbox("Задать частоту рестарта CG вручную", value=False)

    if use_restart:
        cg_restart_frequency_value = st.sidebar.number_input(
            "cg_restart_frequency",
            min_value=1,
            max_value=1000,
            value=2,
            step=1,
        )
        cg_restart_frequency = int(cg_restart_frequency_value)
    else:
        cg_restart_frequency = None

    newton_regularization = st.sidebar.number_input(
        "newton_regularization",
        min_value=0.0,
        value=default_params.newton_regularization,
        format="%.12f",
    )

    return OptimizationParams(
        epsilon=float(epsilon),
        max_iterations=int(max_iterations),
        initial_step=float(initial_step),
        armijo_c1=float(armijo_c1),
        backtracking_factor=float(backtracking_factor),
        min_step=float(min_step),
        cg_restart_frequency=cg_restart_frequency,
        newton_regularization=float(newton_regularization),
    )


def render_single_function_mode(params: OptimizationParams) -> None:
    """Режим запуска одной функции."""

    function_catalog = get_function_catalog()
    function_names = [*list(function_catalog.keys()), "CustomPolynomial10D"]

    selected_name = st.selectbox(
        "Функция",
        function_names,
        index=function_names.index("Booth"),
    )

    if selected_name == "CustomPolynomial10D":
        function = build_custom_polynomial_from_ui()

        if function is None:
            return
    else:
        function = function_catalog[selected_name]

        default_x0_text = " ".join(str(value) for value in function.default_x0)

        x0_text = st.text_input(
            "Начальная точка x0",
            value=default_x0_text,
            help="Введите координаты через пробел или запятую.",
        )

        try:
            function.default_x0 = parse_vector(x0_text, function.dimension)
        except ValueError as error:
            st.error(str(error))
            return

    st.info(f"Выбрана функция: {function.name}. x0 = {function.default_x0}")

    if st.button("Запустить оптимизацию", type="primary"):
        with st.spinner("Выполняется оптимизация..."):
            results = run_methods(function, params)

        st.header("Результаты")
        display_result_cards(results)

        st.subheader("Сводная таблица")
        st.dataframe(results_to_summary_dataframe(results), use_container_width=True)

        graph_paths = display_graphs(function, results, params)
        display_iteration_tables(results)
        display_downloads(results, graph_paths, prefix=function.name)


def render_variant_3_mode(params: OptimizationParams) -> None:
    """Режим полного запуска варианта 3."""

    variant_functions = [
        *get_variant_3_functions(),
        get_variant_3_polynomial(),
    ]

    st.markdown(
        """
        Вариант 3 включает:

        - Booth, x0 = (-5, -5)
        - Sphere, x0 = (-3, -3)
        - Rosenbrock, x0 = (-2, 2)
        - Himmelblau, x0 = (-1, 0)
        - Polynomial10DVariant3
        """
    )

    if st.button("Запустить вариант 3", type="primary"):
        all_results: list[OptimizationResult] = []
        all_graph_paths: list[Path] = []

        with st.spinner("Выполняются расчёты для варианта 3..."):
            for function in variant_functions:
                all_results.extend(run_methods(function, params))

        st.header("Сводная таблица по варианту 3")
        st.dataframe(
            results_to_summary_dataframe(all_results),
            use_container_width=True,
        )

        for function in variant_functions:
            function_results = [
                result
                for result in all_results
                if result.function_name == function.name
            ]

            st.divider()
            st.header(function.name)
            st.write(f"Начальная точка: `{function.default_x0}`")

            display_result_cards(function_results)

            graph_paths = display_graphs(function, function_results, params)
            all_graph_paths.extend(graph_paths)

            with st.expander(f"Таблицы итераций: {function.name}"):
                display_iteration_tables(function_results)

        display_downloads(
            results=all_results,
            graph_paths=all_graph_paths,
            prefix="variant_3",
        )


def main() -> None:
    """Главная функция Streamlit-приложения."""

    ensure_output_dirs()

    st.set_page_config(
        page_title="ЛР2: Нелинейная оптимизация",
        layout="wide",
    )

    st.title("Лабораторная работа №2")
    st.subheader("Многомерная нелинейная оптимизация. Вариант 3")

    st.markdown(
        """
        Реализованы три метода безусловной минимизации:

        - сопряжённые градиенты Флетчера–Ривза;
        - модифицированный метод Ньютона;
        - BFGS.

        Для 2D-функций используются аналитические градиенты и Гессианы.
        Для полинома 10D используются аналитические градиент и Гессиан.
        """
    )

    params = sidebar_params(get_default_params())

    mode = st.radio(
        "Режим запуска",
        [
            "Одна функция",
            "Вариант 3 полностью",
        ],
        horizontal=True,
    )

    if mode == "Одна функция":
        render_single_function_mode(params)
    else:
        render_variant_3_mode(params)


if __name__ == "__main__":
    main()