"""Shared display formatters, used everywhere a percentage or duration is
rendered so numbers look consistent across KPI cards, tables, charts and
the AI panel."""


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def format_duration(seconds: float) -> str:
    return f"{seconds:.1f}s"
