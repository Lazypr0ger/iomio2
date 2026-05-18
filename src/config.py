from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"

DEFAULT_EPSILON = 1e-3
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_INITIAL_STEP = 1.0
DEFAULT_ARMIJO_C1 = 1e-4
DEFAULT_BACKTRACKING_FACTOR = 0.5
DEFAULT_MIN_STEP = 1e-12

VARIANT_NUMBER = 3


def ensure_output_dirs() -> None:
    """Создание папок для результатов, если они отсутствуют."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)