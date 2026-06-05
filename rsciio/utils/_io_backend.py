# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# RosettaSciIO is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RosettaSciIO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

"""
Per-machine I/O backend selection for read_binary_distributed.

Two backends are available:

``"memmap"``
    Classic numpy.memmap path — maps the full file into the process address
    space on each chunk call.  Reliable everywhere; throughput limited by
    VMA setup overhead on large files (~450-700 MB/s on a RAID array).

``"sequential"``
    open() + seek + readinto() directly into a pre-shaped numpy buffer.
    Avoids full-file VMA; lets the OS issue sequential read-ahead across
    all RAID stripes (~1000-1700 MB/s on the same array).  Works on both
    Windows and Linux.

The system default is determined once by a short benchmark and stored in
``platformdirs.user_config_dir("rosettasciio") / io_backend.json``.
Subsequent imports load the cached result in microseconds.

Public API
----------
get_default_backend() -> str
    Return the cached system default, running the benchmark if needed.

set_default_backend(backend: str)
    Persist a specific backend as the system default.

benchmark_backends(path=None, n_mb=64) -> dict
    Run a timed comparison and return {backend: MB/s}.
    If ``path`` is None a temporary file is used.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time

import numpy as np

_log = logging.getLogger(__name__)

# Valid backend names
BACKENDS = ("memmap", "sequential")

# Config file location: platform-appropriate user config directory
_CONFIG_FILENAME = "io_backend.json"
_BENCHMARK_N_BYTES = 64 * 1024 * 1024  # 64 MB -- fast but representative


def _config_path() -> str:
    try:
        import platformdirs
        config_dir = platformdirs.user_config_dir("rosettasciio")
    except ImportError:
        config_dir = os.path.join(os.path.expanduser("~"), ".config", "rosettasciio")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, _CONFIG_FILENAME)


def _load_config() -> dict:
    path = _config_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_config(data: dict) -> None:
    path = _config_path()
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        _log.warning(f"Could not save io_backend config to {path}: {e}")


# Module-level cache: None means not yet determined this session
_cached_backend: str | None = None


# ---------------------------------------------------------------------------
# Backend implementations used during benchmarking
# ---------------------------------------------------------------------------

def _read_memmap(path: str, offset: int, dtype: np.dtype, shape: tuple) -> None:
    """Read via np.memmap — one full-file map per call (existing behaviour)."""
    mm = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
    out = np.empty(shape, dtype=dtype)
    out[:] = mm
    return out


def _read_sequential(path: str, offset: int, dtype: np.dtype, shape: tuple) -> None:
    """Read via open+readinto — no full-file map, OS read-ahead hint."""
    out = np.empty(shape, dtype=dtype)
    with open(path, "rb") as fh:
        fh.seek(offset)
        fh.readinto(out)
    return out


_BACKEND_READERS = {
    "memmap": _read_memmap,
    "sequential": _read_sequential,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def benchmark_backends(
    path: str | None = None,
    n_mb: int = 64,
) -> dict[str, float]:
    """
    Benchmark each backend on sequential reads and return MB/s per backend.

    Parameters
    ----------
    path : str or None
        File to read.  If None, a temporary file of ``n_mb`` MB is created,
        used, and deleted.  Pass a path on the target storage device (e.g.
        your data RAID) for a realistic measurement.
    n_mb : int
        Approximate size of the test read in megabytes.

    Returns
    -------
    dict mapping backend name -> MB/s (float)
    """
    n_bytes = n_mb * 1024 * 1024
    # Round to whole float32 elements and a square-ish shape
    n_elements = n_bytes // 4
    side = max(1, int(n_elements ** 0.5))
    n_rows = max(1, n_elements // side)
    shape = (n_rows, side)
    actual_bytes = n_rows * side * 4
    dtype = np.dtype("f4")

    cleanup = False
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        cleanup = True
        data = np.random.rand(*shape).astype(dtype)
        data.tofile(path)
    else:
        # Use the first n_bytes of the existing file after a 1024-byte header
        actual_bytes = min(actual_bytes, os.path.getsize(path) - 1024)
        n_rows = max(1, actual_bytes // (side * 4))
        shape = (n_rows, side)
        actual_bytes = n_rows * side * 4

    offset = 0 if cleanup else 1024

    results: dict[str, float] = {}
    try:
        for name in BACKENDS:
            reader = _BACKEND_READERS[name]
            # Warm-up pass (fills OS cache for memmap, measures cold for sequential)
            try:
                reader(path, offset, dtype, shape)
            except Exception:
                continue

            n_reps = 3
            t0 = time.perf_counter()
            for _ in range(n_reps):
                try:
                    reader(path, offset, dtype, shape)
                except Exception:
                    break
            elapsed = time.perf_counter() - t0
            mb_s = actual_bytes / 1e6 * n_reps / elapsed
            results[name] = mb_s
            _log.debug(f"  {name}: {mb_s:.0f} MB/s")
    finally:
        if cleanup:
            try:
                os.unlink(path)
            except OSError:
                pass

    return results


def _detect_best_backend(path: str | None = None) -> str:
    """Run benchmark and return the faster backend name."""
    _log.info("rsciio: probing I/O backends (one-time, ~1 s)...")
    results = benchmark_backends(path=path)
    if not results:
        return "memmap"
    best = max(results, key=results.get)
    _log.info(
        f"rsciio: selected backend '{best}'  "
        + "  ".join(f"{k}={v:.0f} MB/s" for k, v in results.items())
    )
    return best


def get_default_backend() -> str:
    """
    Return the system default I/O backend, detecting it if necessary.

    Detection runs once per machine: result is stored in
    ``platformdirs.user_config_dir("rosettasciio")/io_backend.json`` and
    loaded on future calls in microseconds.

    Returns
    -------
    str
        One of ``"memmap"`` or ``"sequential"``.
    """
    global _cached_backend

    # 1. In-process cache (fastest -- subsequent calls in the same session)
    if _cached_backend is not None:
        return _cached_backend

    # 2. On-disk cache (subsequent sessions on this machine)
    cfg = _load_config()
    stored = cfg.get("default_backend")
    if stored in BACKENDS:
        _cached_backend = stored
        return _cached_backend

    # 3. First time: benchmark and persist
    best = _detect_best_backend()
    set_default_backend(best)
    return best


def set_default_backend(backend: str) -> None:
    """
    Set and persist the system default I/O backend.

    Parameters
    ----------
    backend : str
        One of ``"memmap"`` or ``"sequential"``.

    Raises
    ------
    ValueError
        If ``backend`` is not a recognised backend name.
    """
    global _cached_backend

    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Must be one of {BACKENDS}."
        )
    _cached_backend = backend
    cfg = _load_config()
    cfg["default_backend"] = backend
    _save_config(cfg)
    _log.info(f"rsciio: default I/O backend set to '{backend}'")


def reset_default_backend() -> None:
    """
    Clear the cached backend so it will be re-detected on next use.

    Removes the persisted config entry.  Useful when moving data to a
    different storage device.
    """
    global _cached_backend
    _cached_backend = None
    cfg = _load_config()
    cfg.pop("default_backend", None)
    _save_config(cfg)
