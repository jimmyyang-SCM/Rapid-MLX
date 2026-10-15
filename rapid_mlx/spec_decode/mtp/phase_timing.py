"""Opt-in synchronized phase timing for the continuous self-MTP cohort (#4589).

Disabled unless ``RAPID_MLX_MTP_PHASE_TIMING=1``.  When enabled, every phase
boundary evaluates the arrays that phase produced and calls ``mx.synchronize()``
before reading the clock, so each phase's wall time includes its own GPU work
and nothing that was queued earlier.  That removes the overlap the uninstrumented
path relies on, so phase sums are an *attribution*, not a throughput number --
throughput must come from a separate run with the variable unset.

Output: a cumulative JSON snapshot written atomically (at most once a second)
to ``RAPID_MLX_MTP_PHASE_TIMING_OUT`` (default
``/tmp/rapid_mlx_mtp_phase_timing.json``).  Creating ``<OUT>.reset`` clears the
accumulated stats at the next cycle boundary (used by benchmark drivers to
isolate one concurrency point).

Records are keyed ``<mode>|B=<lanes>|K=<draft depth>`` where mode is
``cohort`` (one ``ContinuousMTPGenerationBatch.next_burst`` proposal/commit
cycle) or ``plain`` (one mlx-lm ``GenerationBatch._step``).  The phase
``scheduler_between`` is the wall time from the end of the previous record of
the same mode to the start of this one: everything the scheduler does between
two model steps (draining queued tokens one per ``next()`` call, detokenizing,
response delivery, stop checks).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import Any, Iterable

ENABLED = os.environ.get("RAPID_MLX_MTP_PHASE_TIMING", "") not in ("", "0")
OUT = os.environ.get(
    "RAPID_MLX_MTP_PHASE_TIMING_OUT", "/tmp/rapid_mlx_mtp_phase_timing.json"
)
_RESERVOIR = 4096

_lock = threading.Lock()
_stats: dict[str, dict[str, Any]] = {}
_cur: list[Any] | None = None  # [mode, lanes, depth, t_last, {phase: seconds}, t_begin]
_last_end: dict[str, float] = {}
_driver_calls = 0


def _flatten(values: Iterable[Any]) -> list[Any]:
    import mlx.core as mx

    out: list[Any] = []
    stack = list(values)
    while stack:
        value = stack.pop()
        if value is None:
            continue
        if isinstance(value, mx.array):
            out.append(value)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, dict):
            stack.extend(value.values())
    return out


def cache_arrays(caches: Iterable[Any]) -> list[Any]:
    """Best-effort list of the arrays a cache group currently holds."""
    arrays: list[Any] = []
    for cache in caches or ():
        try:
            state = cache.state
        except Exception:  # noqa: BLE001 - timing must never break serving
            continue
        arrays.extend(_flatten([state]))
    return arrays


def _sync(arrays: Iterable[Any]) -> None:
    import mlx.core as mx

    flat = _flatten(arrays)
    if flat:
        mx.eval(*flat)
    mx.synchronize()


def begin(mode: str, lanes: int, depth: int = -1) -> None:
    global _cur, _driver_calls
    if not ENABLED:
        return
    t_enter = time.perf_counter()
    _sync(())
    now = time.perf_counter()
    phases: dict[str, float] = {}
    prev = _last_end.get(mode)
    if prev is not None:
        # Gap measured to the moment we entered, before our own sync.
        phases["scheduler_between"] = t_enter - prev
        phases["driver_next_calls"] = float(_driver_calls)
    phases["entry_sync"] = now - t_enter
    _driver_calls = 0
    _cur = [mode, lanes, depth, now, phases, now]


def set_depth(depth: int) -> None:
    if ENABLED and _cur is not None:
        _cur[2] = int(depth)


def mark(name: str, *arrays: Any) -> None:
    if not ENABLED or _cur is None:
        return
    _sync(arrays)
    now = time.perf_counter()
    phases = _cur[4]
    phases[name] = phases.get(name, 0.0) + (now - _cur[3])
    _cur[3] = now


def note_driver_call() -> None:
    global _driver_calls
    if ENABLED:
        _driver_calls += 1


def end(tokens: int = 0) -> None:
    global _cur
    if not ENABLED or _cur is None:
        return
    mode, lanes, depth, _t_last, phases, t_begin = _cur
    now = time.perf_counter()
    _cur = None
    _last_end[mode] = now
    key = f"{mode}|B={lanes}|K={depth}"
    with _lock:
        if os.path.exists(OUT + ".reset"):
            _stats.clear()
            try:
                os.unlink(OUT + ".reset")
            except OSError:
                pass
        rec = _stats.setdefault(key, {"cycles": 0, "tokens": 0, "phases": {}})
        rec["cycles"] += 1
        rec["tokens"] += int(tokens)
        phases = dict(phases)
        phases["cycle_measured"] = now - t_begin
        for name, value in phases.items():
            ph = rec["phases"].setdefault(
                name, {"n": 0, "sum": 0.0, "sumsq": 0.0, "max": 0.0, "samples": []}
            )
            ph["n"] += 1
            ph["sum"] += value
            ph["sumsq"] += value * value
            ph["max"] = max(ph["max"], value)
            samples = ph["samples"]
            if len(samples) < _RESERVOIR:
                samples.append(value)
            else:
                slot = random.randrange(ph["n"])
                if slot < _RESERVOIR:
                    samples[slot] = value
        global _dirty
        _dirty = True
    _ensure_dumper()


_dirty = False
_dumper: threading.Thread | None = None


def _dump_locked() -> None:
    tmp = OUT + ".tmp"
    with open(tmp, "w") as handle:
        json.dump({"written_at": time.time(), "stats": _stats}, handle)
    os.replace(tmp, OUT)


def _dump_loop() -> None:
    global _dirty
    while True:
        time.sleep(1.0)
        with _lock:
            if _dirty:
                _dirty = False
                try:
                    _dump_locked()
                except OSError:
                    pass


def _ensure_dumper() -> None:
    global _dumper
    if _dumper is None:
        _dumper = threading.Thread(target=_dump_loop, name="mtp-phase-timing-dump", daemon=True)
        _dumper.start()


def install_plain_step_hook() -> bool:
    """Wrap mlx-lm ``GenerationBatch._step`` so plain decode steps are timed."""
    if not ENABLED:
        return False
    import importlib

    gen = importlib.import_module("mlx_lm.generate")
    cls = getattr(gen, "GenerationBatch", None)
    if cls is None or getattr(cls._step, "_anp_phase_timed", False):
        return False
    original = cls._step

    def _timed_step(self, *args, **kwargs):
        if _cur is not None:  # nested inside a cohort record: do not split it
            return original(self, *args, **kwargs)
        lanes = len(getattr(self, "uids", ()) or ())
        begin("plain", lanes, 0)
        result = original(self, *args, **kwargs)
        mark("plain_step", result)
        end(tokens=lanes)
        return result

    _timed_step._anp_phase_timed = True  # type: ignore[attr-defined]
    cls._step = _timed_step
    return True
