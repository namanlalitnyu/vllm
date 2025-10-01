"""Summarize vLLM lite-profiler logs in tabular form.

This script converts the JSONL records emitted by :mod:`vllm.lite_profiler`
into human-readable tables that match the notebook workflow used during
development.  It expects log lines prefixed with ``===LITE`` where the payload
contains a ``metrics`` dictionary whose values are ``{"ns": int, "count": int}``.

Usage examples::

    # Use the log file pointed to by VLLM_LITE_PROFILER_LOG_PATH
    python -m tools.lite_profiler_report

    # Compare two runs with custom names
    python -m tools.lite_profiler_report \
        --group opt125m=/tmp/opt.log --group llama3=/tmp/llama3.log

    # Aggregate multiple shards under one name
    python -m tools.lite_profiler_report \
        --group blended=/tmp/run_*.log

When no groups are supplied the script falls back to the current value of
``VLLM_LITE_PROFILER_LOG_PATH``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Log ingestion


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


# ---------------------------------------------------------------------------
# Tabular formatting helpers


def _format_duration_ns(value_ns: int, total_ns: int) -> str:
    seconds = value_ns / 1_000_000_000 if value_ns else 0.0
    percent = (value_ns * 100.0 / total_ns) if total_ns else 0.0
    return f"{seconds:.2f}s ({percent:.2f}%)"


def _render_table(title: str, headers: Sequence[str], rows: Iterable[Sequence[str]]
                  ) -> None:
    table = [list(headers)] + [list(row) for row in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    print(f"\n{title}")
    print("-" * sum(widths) + "-" * (len(widths) - 1))

    def _fmt(row: Sequence[str]) -> str:
        return " ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(_fmt(table[0]))
    print(" ".join("-" * w for w in widths))
    for row in table[1:]:
        print(_fmt(row))


# ---------------------------------------------------------------------------
# Event groups mirrored from the notebook prototype


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
    group_name: str,
    event_ns_sum: Dict[str, int],
    events: Sequence[str],
) -> List[str]:
    total_ns = sum(event_ns_sum.get(event, 0) for event in events)
    cells = [group_name]
    for event in events:
        cells.append(_format_duration_ns(event_ns_sum.get(event, 0), total_ns))
    total_seconds = total_ns / 1_000_000_000 if total_ns else 0.0
    cells.append(f"{total_seconds:.2f}s")
    return cells


def _print_breakdown_tables(
    groups: Sequence[Tuple[str, Dict[str, int]]],
) -> None:
    for title, events in (
        ("Breakdown (non-forward events)", NON_FORWARD_EVENTS),
        ("Schedule breakdown", SCHEDULE_EVENTS),
        ("Model events breakdown", MODEL_EVENTS),
        ("Topline events", TOP_EVENTS),
    ):
        headers = ["Group", *events, "TOTAL"]
        rows = [
            _compute_table_rows(name, event_ns_sum, events)
            for name, event_ns_sum in groups
        ]
        _render_table(title, headers, rows)


# ---------------------------------------------------------------------------
# CLI


def _expand_group_arg(arg: str) -> Tuple[str, List[str]]:
    if "=" not in arg:
        raise ValueError("Group arguments must use the NAME=PATH[,PATH...] syntax")
    name, raw_paths = arg.split("=", 1)
    if not name:
        raise ValueError("Group name cannot be empty")
    paths: List[str] = []
    for pattern in raw_paths.split(","):
        pattern = pattern.strip()
        if not pattern:
            continue
        expanded = sorted(glob.glob(pattern)) or [pattern]
        paths.extend(expanded)
    if not paths:
        raise ValueError(f"No log files matched for group '{name}'")
    return name, paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=PATH[,PATH...]",
        help="Named group of lite-profiler logs to aggregate",
    )
    parser.add_argument(
        "--log",
        default=None,
        help=(
            "Single lite-profiler log to summarise. Defaults to "
            "VLLM_LITE_PROFILER_LOG_PATH when omitted and no --group is provided."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    groups: List[Tuple[str, Dict[str, int]]] = []

    if args.group:
        for raw_group in args.group:
            name, paths = _expand_group_arg(raw_group)
            event_ns = _extract_event_ns(paths)
            groups.append((name, _sum_events(event_ns)))
    else:
        log_path = args.log or os.getenv("VLLM_LITE_PROFILER_LOG_PATH")
        if not log_path:
            raise SystemExit(
                "No log file specified. Use --log or set VLLM_LITE_PROFILER_LOG_PATH"
            )
        event_ns = _extract_event_ns([log_path])
        groups.append((os.path.basename(log_path), _sum_events(event_ns)))

    available_events = sorted({event for _, summary in groups for event in summary})
    if available_events:
        print("Available events:")
        print(", ".join(available_events))

    _print_breakdown_tables(groups)


if __name__ == "__main__":
    main()

