"""Minimal helpers for opt-in lightweight timing collection."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Dict, Iterable, Optional

import vllm.envs as envs
from vllm.logger import init_logger

_PREFIX = "===LITE"


class _LiteScope:
    def __init__(self, transaction: "_LiteTransaction", name: str) -> None:
        self._transaction = transaction
        self._name = name
        self._start_ns: Optional[int] = None

    def __enter__(self) -> None:
        self._start_ns = time.monotonic_ns()
        return None

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        if self._start_ns is not None and exc_type is None:
            elapsed = time.monotonic_ns() - self._start_ns
            self._transaction.record(self._name, elapsed)
        return False


class _LiteTransaction:
    def __init__(self, profiler: "LiteProfiler", tag: str) -> None:
        self._profiler = profiler
        self.tag = tag
        self._metrics: Dict[str, list[int]] = {}

    def __enter__(self) -> "_LiteTransaction":
        self._profiler._push(self)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self._profiler._pop(self)
        if exc_type is None:
            self._profiler._emit(self)
        return False

    # Public helpers -----------------------------------------------------
    def scope(self, name: str):
        return _LiteScope(self, name)

    def record(self, name: str, elapsed_ns: int, *, count: int = 1) -> None:
        bucket = self._metrics.get(name)
        if bucket is None:
            bucket = [0, 0]
            self._metrics[name] = bucket
        bucket[0] += int(elapsed_ns)
        bucket[1] += int(count)

    # Internal accessors -------------------------------------------------
    @property
    def metrics(self) -> Dict[str, list[int]]:
        return self._metrics


class _NullTransaction:
    def __enter__(self) -> "_NullTransaction":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        return False

    def scope(self, _name: str):
        return nullcontext()

    def record(self, _name: str, _elapsed_ns: int, *, count: int = 1) -> None:
        return None


class LiteProfiler:
    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._logger = init_logger("vllm.lite_profiler")
        self._log_path: Optional[str] = None

    # ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        try:
            return bool(envs.VLLM_LITE_PROFILER)
        except AttributeError:  # pragma: no cover - env not wired in tests
            return False

    # Transaction handling ---------------------------------------------
    def transaction(self, tag: str):
        if not self.is_enabled():
            return _NullTransaction()
        return _LiteTransaction(self, tag)

    def _push(self, transaction: _LiteTransaction) -> None:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        stack.append(transaction)

    def _pop(self, transaction: _LiteTransaction) -> None:
        stack = getattr(self._local, "stack", None)
        if stack and stack[-1] is transaction:
            stack.pop()

    def _current(self) -> Optional[_LiteTransaction]:
        stack = getattr(self._local, "stack", None)
        if stack:
            return stack[-1]
        return None

    # Scope helpers -----------------------------------------------------
    def scope(self, name: str):
        if not self.is_enabled():
            return None
        transaction = self._current()
        if transaction is None:
            return None
        return transaction.scope(name)

    # Emission ----------------------------------------------------------
    def _ensure_log_handler(self) -> None:
        log_path = getattr(envs, "VLLM_LITE_PROFILER_LOG_PATH", None)
        if not log_path:
            return
        log_path = os.path.abspath(log_path)
        if log_path == self._log_path:
            return
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._logger.propagate = False
        self._log_path = log_path

    def _emit(self, transaction: _LiteTransaction) -> None:
        metrics = {
            name: {
                "ns": values[0],
                "count": values[1],
            }
            for name, values in transaction.metrics.items()
        }
        payload = {
            "tag": transaction.tag,
            "ts": time.time(),
            "metrics": metrics,
        }

        self._ensure_log_handler()
        message = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            self._logger.info("%s %s", _PREFIX, message)


lite_profiler = LiteProfiler()


def context_logger(tag: str):
    return lite_profiler.transaction(tag)


@contextmanager
def combine_contexts(contexts: Iterable):
    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield
