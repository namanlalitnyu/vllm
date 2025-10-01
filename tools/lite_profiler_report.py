"""Summarize a single vLLM lite-profiler log in tabular form.

The script consumes the JSONL records emitted by :mod:`vllm.lite_profiler`
It expects log lines prefixed with ``===LITE`` where the payload contains a
``metrics`` dictionary whose values are ``{"ns": int, "count": int}``.

Usage examples::

    # Use the log file pointed to by VLLM_LITE_PROFILER_LOG_PATH
    python -m tools.lite_profiler_report

    # Provide an explicit path
    python -m tools.lite_profiler_report /tmp/vllm-lite-profiler.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple, TextIO


def _extract_event_ns(filenames: Sequence[str]) -> Dict[str, List[int]]:
    """Collect the nanosecond timings for every scope contained in ``filenames``."""

    all_event_ns: Dict[str, List[int]] = defaultdict(list)
    for filename in filenames:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.lstrip()
                    if not line.startswith("===LITE "):
                        continue
                    try:
                        payload = json.loads(line.split("===LITE ", 1)[1].strip())
                    except json.JSONDecodeError:
                        continue
                    metrics = payload.get("metrics")
                    if not isinstance(metrics, dict):
                        continue
                    for event, meta in metrics.items():
                        if isinstance(meta, dict) and "ns" in meta:
                            all_event_ns[event].append(int(meta["ns"]))
        except FileNotFoundError:
            raise FileNotFoundError(f"Lite-profiler log not found: {filename}")
    return all_event_ns


def _sum_events(event_ns: Dict[str, List[int]]) -> Dict[str, int]:
    return {event: sum(values) for event, values in event_ns.items()}


def _format_duration_ns(value_ns: int, total_ns: int) -> str:
    seconds = value_ns / 1_000_000_000 if value_ns else 0.0
    percent = (value_ns * 100.0 / total_ns) if total_ns else 0.0
    return f"{seconds:.2f}s ({percent:.2f}%)"


def _render_table(title: str,
                  headers: Sequence[str],
                  rows: Iterable[Sequence[str]],
                  *,
                  stream: TextIO) -> None:
    table = [list(headers)] + [list(row) for row in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    print(f"\n{title}", file=stream)
    print("-" * sum(widths) + "-" * (len(widths) - 1), file=stream)

    def _fmt(row: Sequence[str]) -> str:
        return " ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(_fmt(table[0]), file=stream)
    print(" ".join("-" * w for w in widths), file=stream)
    for row in table[1:]:
        print(_fmt(row), file=stream)


TOP_EVENTS = [
    "Input:Process",
    "Step:Schedule",
    "Step:Model",
    "Step:Output",
    "Model:Preprocess",
    "Model:Sample",
    "Model:Bookkeep",
]

SCHEDULE_EVENTS = [
    "Scheduler:AllocateSave",
    "Scheduler:AllocateNew",
    "Scheduler:AllocateCache",
]

MODEL_EVENTS = [
    "Model:PrepareInput",
    "Model:Preprocess",
    "Model:Forward",
    "Model:Postprocess",
    "Model:Sample",
    "Model:Bookkeep",
]

NON_FORWARD_EVENTS = [
    "Input:Process",
    "Step:Schedule",
    "Model:PrepareInput",
    "Model:UpdateState",
    "Model:Preprocess",
    "Model:Forward",
    "Model:Postprocess",
    "Model:Sample",
    "Model:Bookkeep",
    "Step:Model",
    "Step:Output",
]


def _compute_table_rows(
    name: str,
    event_ns_sum: Dict[str, int],
    events: Sequence[str],
) -> List[str]:
    total_ns = sum(event_ns_sum.get(event, 0) for event in events)
    cells = [name]
    for event in events:
        cells.append(_format_duration_ns(event_ns_sum.get(event, 0), total_ns))
    total_seconds = total_ns / 1_000_000_000 if total_ns else 0.0
    cells.append(f"{total_seconds:.2f}s")
    return cells


def _print_breakdown_tables(name: str,
                            event_ns_sum: Dict[str, int],
                            *,
                            stream: TextIO) -> None:
    for title, events in (
        ("Breakdown (non-forward events)", NON_FORWARD_EVENTS),
        ("Schedule breakdown", SCHEDULE_EVENTS),
        ("Model events breakdown", MODEL_EVENTS),
        ("Topline events", TOP_EVENTS),
    ):
        headers = ["Log", *events, "TOTAL"]
        rows = [_compute_table_rows(name, event_ns_sum, events)]
        _render_table(title, headers, rows, stream=stream)


def summarize_log(log_path: str, *, stream: TextIO) -> None:
    event_ns = _extract_event_ns([log_path])
    event_ns_sum = _sum_events(event_ns)

    available_events = sorted(event_ns_sum)
    if available_events:
        print("Available events:", file=stream)
        print(", ".join(available_events), file=stream)

    _print_breakdown_tables(os.path.basename(log_path),
                            event_ns_sum,
                            stream=stream)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_path",
        nargs="?",
        help=(
            "Lite-profiler log to summarise. Defaults to VLLM_LITE_PROFILER_LOG_PATH"
            " when omitted."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    log_path = args.log_path or os.getenv("VLLM_LITE_PROFILER_LOG_PATH")
    if not log_path:
        raise SystemExit(
            "No log file specified. Provide LOG_PATH or set VLLM_LITE_PROFILER_LOG_PATH"
        )

    summarize_log(log_path, stream=sys.stdout)


if __name__ == "__main__":
    main()

