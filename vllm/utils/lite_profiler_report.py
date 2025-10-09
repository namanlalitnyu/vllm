# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a single vLLM lite-profiler log in tabular form.

The script consumes the pipe-separated records emitted by `vllm.lite_profiler`
It expects log lines in the format: "<scope_name>|<elapsed_microseconds>"
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from vllm.logger import init_logger

logger = init_logger(__name__)


def _extract_event_us(filename: str) -> dict[str, list[int]]:
    """Collect the microsecond timings for every scope in ``filenames``."""

    all_event_us: dict[str, list[int]] = defaultdict(list)
    try:
        with open(filename, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                # Parse the format: "scope_name|elapsed_microseconds"
                if "|" in line:
                    try:
                        scope_name, elapsed_us_str = line.split("|", 1)
                        elapsed_us = int(elapsed_us_str)
                        all_event_us[scope_name].append(elapsed_us)
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Lite-profiler log not found: {filename}") from None
    return all_event_us


def _sum_events(event_us: dict[str, list[int]]) -> dict[str, int]:
    return {event: sum(values) for event, values in event_us.items()}


def _format_duration_us(value_us: int, total_us: int) -> str:
    # Convert microseconds to seconds
    seconds = value_us / 1e6 if value_us else 0.0
    percent = (value_us * 100.0 / total_us) if total_us else 0.0
    return f"{seconds:.2f}s ({percent:.2f}%)"


def _render_table(title: str, headers: Sequence[str],
                  rows: Iterable[Sequence[str]]) -> None:
    table = [list(headers)] + [list(row) for row in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    logger.info("")
    logger.info(title)
    separator = "-" * sum(widths) + "-" * (len(widths) - 1)
    logger.info(separator)

    def _fmt(row: Sequence[str]) -> str:
        return " ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    logger.info(_fmt(table[0]))
    logger.info(" ".join("-" * w for w in widths))
    for row in table[1:]:
        logger.info(_fmt(row))


TOP_EVENTS = [
    "Input:Process",
    "Step:Schedule",
    "Step:Model",
    "Step:Output",
]

MODEL_EVENTS = [
    "Model:UpdateState",
    "Model:PrepareInput",
    "Model:Forward",
    "Model:Postprocess",
    "Model:Sample",
    "Model:Bookkeep",
    "Model:EPLB",
    "Model:Draft",
]


def _compute_table_rows(
    event_us_sum: dict[str, int],
    events: Sequence[str],
) -> list[str]:
    total_us = sum(event_us_sum.get(event, 0) for event in events)
    cells = []
    for event in events:
        cells.append(_format_duration_us(event_us_sum.get(event, 0), total_us))
    # Convert microseconds to seconds
    total_seconds = total_us / 1_000_000 if total_us else 0.0
    cells.append(f"{total_seconds:.2f}s")
    return cells


def _print_breakdown_tables(event_us_sum: dict[str, int]) -> None:
    for title, events in (
        ("Top-level pipeline events", TOP_EVENTS),
        ("Model events breakdown (only includes the main key events)",
         MODEL_EVENTS),
    ):
        headers = [*events, "TOTAL"]
        rows = [_compute_table_rows(event_us_sum, events)]
        _render_table(title, headers, rows)


def summarize_log(log_path: str) -> None:
    event_us = _extract_event_us(log_path)
    event_us_sum = _sum_events(event_us)
    _print_breakdown_tables(event_us_sum)
