"""Public API of the report module (workflow step 6): metrics, benchmarks, HTML report."""

from trading.report.benchmarks import buy_and_hold  # noqa: F401
from trading.report.performance import daily_equity, metrics  # noqa: F401
from trading.report.report import build_report  # noqa: F401
