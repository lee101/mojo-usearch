# mojo-usearch

`mojo-usearch` is a Mojo implementation of exact SIMD dense-vector search with
the common Python `usearch.index` surface. It is useful when reproducible,
exact nearest neighbours matter more than the recall/latency trade-off of an
approximate graph. The Python package is named `usearch`, so code using the
covered subset can keep its import path.

## Covered subset

`Index` stores float32 dense vectors and supports `add`, `search`, `get`,
`contains`, `count`, `remove`, `rename`, `reset`/`clear`, and `save`/`load`/
`restore`. Searches accept one query or a batch, `count`, `radius`, `exact`,
and the usual `threads`, `log`, `progress`, and `dtype` keyword parameters.
NumPy vector arrays must already use `float32`; the port rejects implicit dtype
narrowing. Python sequences are converted to contiguous `float32` buffers before
the native call.
`Matches`, `BatchMatches`, `MetricKind`, `ScalarKind`, and top-level
`usearch.index.search` are included. Covered metrics are `l2sq`, `cos`, and
`ip`; their returned distances match upstream's definitions.

This is intentionally not an HNSW implementation. It does not yet cover
graph persistence interoperable with native usearch, quantized/binary scalar
types, custom metrics, clustering/join APIs, graph statistics, or parallel
index construction. `connectivity` and expansion constructor arguments are retained as
metadata for call-site compatibility, but do not alter an exact scan.

## Install

```bash
pixi install
pixi run build
pixi run test
```

The first Python call also builds `dist/libmojo-usearch.so` when it is absent
or older than `src/capi.mojo`.

## Usage

```python
import numpy as np
from usearch.index import Index

index = Index(ndim=3, metric="cos", dtype="f32")
index.add([101, 202, 303], np.array([
    [1, 0, 0], [0, 1, 0], [1, 1, 0],
], dtype=np.float32))

matches = index.search(np.array([1, 0, 0], dtype=np.float32), count=2, exact=True)
print(matches.keys)       # [101 303]
print(matches.distances)  # [0.        0.29289323]
```

## How it works

The index ownership, keys, and persistence live in Python as contiguous NumPy
arrays. A batch search passes their raw addresses through `ctypes` to one Mojo
shared library. `src/capi.mojo` rebuilds those addresses as mutable
`UnsafePointer[Float32, AnyOrigin[mut=True]]` values, walks each row with
native-width SIMD accumulators, and retains the best `k` positions in a sorted
fixed-size output buffer. Large batched searches run independent queries in
parallel; small searches stay serial to avoid task-launch overhead. Python maps
positions back to user keys. There are no allocations in the Mojo kernel, and
the buffer lifetime remains Python's responsibility.

## Correctness

The test suite installs the real PyPI `usearch` package and compares exact
nearest-neighbor keys and distances for all three metrics, scalar and batched
queries, on the same float32 inputs. It also exercises radius filtering,
mutation, multi-key mode, persistence, and the top-level helper.

```bash
pixi run test
```

## Benchmark

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 @ 2.30GHz. Times
are best of five complete exact batch searches; upstream's native package uses
its mature vector kernels, so it remains faster here.

| case | mojo-usearch | upstream usearch | ratio | result |
| --- | ---: | ---: | ---: | --- |
| exact l2sq, 12k x 128, 96 queries | 8.9 ms | 6.4 ms | 0.72x | slower |
| exact cos, 12k x 128, 96 queries | 9.5 ms | 7.3 ms | 0.77x | slower |
| exact ip, 12k x 128, 96 queries | 10.0 ms | 6.4 ms | 0.64x | slower |

Run it under Pixi so the task's machine-wide benchmark lock is held:

```bash
pixi run bench
```
