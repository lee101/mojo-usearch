"""Build and load the Mojo dense-search ABI."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_USEARCH_LIB") or os.path.join(ROOT, "dist", "libmojo-usearch.so")
I = ctypes.c_int64
F32 = ctypes.c_float


class BuildError(RuntimeError):
    pass


def _mojo() -> list[str]:
    override = os.environ.get("MOJO_USEARCH_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_USEARCH_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if os.environ.get("MOJO_USEARCH_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo() + ["build", "--emit", "shared-lib", source, "-o", LIB]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        fn = _loaded.mus_search_f32
        fn.argtypes = [I, I, I, I, I, I, I, I, I, F32, I, I, I, I, I]
        fn.restype = I
    return _loaded


def f32_matrix(values, dimensions: int | None = None) -> np.ndarray:
    original = np.asarray(values)
    if isinstance(values, np.ndarray) and original.dtype != np.float32:
        raise TypeError("NumPy vector arrays must have dtype float32; convert explicitly to avoid narrowing")
    array = np.ascontiguousarray(original, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or (dimensions is not None and array.shape[1] != dimensions):
        raise ValueError("vectors must be a 2-dimensional array with the index dimensionality")
    if array.shape[1] == 0:
        raise ValueError("vectors must have at least one dimension")
    return array


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)
