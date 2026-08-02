"""Exact-search benchmark against the upstream native usearch package."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

from usearch.index import Index as MojoIndex


def upstream_index():
    local_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
    saved = {name: module for name, module in list(sys.modules.items()) if name == "usearch" or name.startswith("usearch.")}
    for name in saved:
        del sys.modules[name]
    sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry) != os.path.abspath(local_python)]
    from usearch.index import Index
    for name in [name for name in list(sys.modules) if name == "usearch" or name.startswith("usearch.")]:
        del sys.modules[name]
    sys.modules.update(saved)
    return Index


UpstreamIndex = upstream_index()


def timeit(call, repeat: int = 5) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    rng = np.random.default_rng(7)
    vectors = np.ascontiguousarray(rng.normal(size=(12_000, 128)).astype(np.float32))
    queries = np.ascontiguousarray(rng.normal(size=(96, 128)).astype(np.float32))
    keys = np.arange(vectors.shape[0], dtype=np.uint64)
    print(f"Machine: {cpu_name()}")
    print()
    print("| case | mojo-usearch | upstream usearch | ratio | result |")
    print("| --- | ---: | ---: | ---: | --- |")
    for metric in ("l2sq", "cos", "ip"):
        ours = MojoIndex(ndim=128, metric=metric, dtype="f32")
        theirs = UpstreamIndex(ndim=128, metric=metric, dtype="f32")
        ours.add(keys, vectors)
        theirs.add(keys, vectors)
        ours.search(queries, count=10, exact=True)
        theirs.search(queries, count=10, exact=True)
        ours_seconds = timeit(lambda: ours.search(queries, count=10, exact=True))
        upstream_seconds = timeit(lambda: theirs.search(queries, count=10, exact=True))
        ratio = upstream_seconds / ours_seconds
        result = "faster" if ratio > 1 else "slower"
        print(f"| exact {metric}, 12k x 128, 96 queries | {ours_seconds * 1e3:.1f} ms | "
              f"{upstream_seconds * 1e3:.1f} ms | {ratio:.2f}x | {result} |")


if __name__ == "__main__":
    main()
