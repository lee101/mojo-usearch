"""A compact, exact dense-vector subset of the upstream ``usearch.index`` API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Iterator, Sequence

import numpy as np

from ._lib import addr, f32_matrix, lib


def _f32_dtype(dtype) -> None:
    if dtype is None:
        return
    name = str(dtype.value if isinstance(dtype, Enum) else dtype).lower()
    if name not in {"f32", "float32"}:
        raise ValueError("this port currently supports only f32 vectors")


class MetricKind(str, Enum):
    Unknown = "unknown"
    L2sq = "l2sq"
    Cos = "cos"
    IP = "ip"
    Cosine = "cos"
    InnerProduct = "ip"


class ScalarKind(str, Enum):
    Unknown = "unknown"
    F32 = "f32"


_METRICS = {
    "l2sq": 0, "l2": 0, "euclidean": 0,
    "cos": 1, "cosine": 1,
    "ip": 2, "inner_product": 2,
}


@dataclass(frozen=True)
class Matches:
    keys: np.ndarray
    distances: np.ndarray

    @property
    def count(self) -> int:
        return int(self.keys.size)

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[tuple[int, float]]:
        return iter(zip(self.keys.tolist(), self.distances.tolist()))


@dataclass(frozen=True)
class BatchMatches:
    keys: np.ndarray
    distances: np.ndarray
    counts: np.ndarray

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def __getitem__(self, item: int) -> Matches:
        count = int(self.counts[item])
        return Matches(self.keys[item, :count], self.distances[item, :count])


class Index:
    """Exact float32 dense index with upstream-compatible common methods.

    ``connectivity`` and expansion values are retained as metadata for source
    compatibility. Searches are always exact scans, not an HNSW traversal.
    """

    def __init__(
        self, *, ndim: int = 0, metric: str | MetricKind = MetricKind.Cos,
        dtype: str | ScalarKind | None = None, connectivity: int | None = None,
        expansion_add: int | None = None, expansion_search: int | None = None,
        multi: bool = False, path: str | None = None, view: bool = False,
        enable_key_lookups: bool = True,
    ) -> None:
        metric_name = str(metric.value if isinstance(metric, Enum) else metric).lower()
        if metric_name not in _METRICS:
            raise ValueError("covered metrics are: l2sq, cos, ip")
        dtype_name = "f32" if dtype is None else str(dtype.value if isinstance(dtype, Enum) else dtype).lower()
        if dtype_name not in {"f32", "float32"}:
            raise ValueError("this port currently stores only f32 vectors")
        if ndim < 0:
            raise ValueError("ndim must be non-negative")
        self.ndim = int(ndim)
        self.metric = MetricKind(("l2sq" if _METRICS[metric_name] == 0 else
                                  "cos" if _METRICS[metric_name] == 1 else "ip"))
        self.dtype = ScalarKind.F32
        self.connectivity = 16 if connectivity is None else int(connectivity)
        self.expansion_add = 128 if expansion_add is None else int(expansion_add)
        self.expansion_search = 64 if expansion_search is None else int(expansion_search)
        self.multi = bool(multi)
        self.enable_key_lookups = bool(enable_key_lookups)
        self._vectors = np.empty((0, self.ndim), dtype=np.float32)
        self._keys = np.empty(0, dtype=np.uint64)
        if path is not None:
            self.load(path)

    @property
    def size(self) -> int:
        return int(self._keys.size)

    @property
    def capacity(self) -> int:
        return self.size

    @property
    def dimensions(self) -> int:
        return self.ndim

    @property
    def metric_kind(self) -> MetricKind:
        return self.metric

    @property
    def keys(self) -> np.ndarray:
        return self._keys.copy()

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors.copy()

    @property
    def memory_usage(self) -> int:
        return int(self._keys.nbytes + self._vectors.nbytes)

    @property
    def serialized_length(self) -> int:
        return len(self.save())

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: int) -> bool:
        return self.contains(key)

    def contains(self, key) -> bool | np.ndarray:
        values = np.asarray(key, dtype=np.uint64)
        result = np.isin(values, self._keys)
        return bool(result) if values.ndim == 0 else result

    def count(self, keys) -> int | np.ndarray:
        values = np.asarray(keys, dtype=np.uint64)
        result = np.array([np.count_nonzero(self._keys == key) for key in values.reshape(-1)], dtype=np.int64)
        return int(result[0]) if values.ndim == 0 else result.reshape(values.shape)

    def add(self, keys, vectors, *, copy: bool = True, threads: int = 0,
            log: str | bool = False, progress=None, dtype=None) -> int | np.ndarray:
        _f32_dtype(dtype)
        vectors = f32_matrix(vectors, self.ndim or None)
        keys = np.asarray(keys, dtype=np.uint64)
        scalar = keys.ndim == 0
        keys = np.full(vectors.shape[0], keys.item(), dtype=np.uint64) if scalar else keys.reshape(-1)
        if keys.size != vectors.shape[0]:
            raise ValueError("one key is required for each vector")
        if self.ndim == 0:
            self.ndim = int(vectors.shape[1])
            self._vectors = np.empty((0, self.ndim), dtype=np.float32)
        if not self.multi and self.size:
            duplicate = np.isin(self._keys, keys)
            if np.any(duplicate):
                self._vectors = self._vectors[~duplicate]
                self._keys = self._keys[~duplicate]
        self._vectors = np.concatenate((self._vectors, vectors.copy() if copy else vectors))
        self._keys = np.concatenate((self._keys, keys))
        return int(keys[0]) if scalar else keys.copy()

    def search(self, vectors, count: int = 10, radius: float = np.inf, *, threads: int = 0,
               exact: bool = False, log: str | bool = False, progress=None, dtype=None):
        _f32_dtype(dtype)
        input_ndim = np.asarray(vectors).ndim
        queries = f32_matrix(vectors, self.ndim)
        requested = int(count)
        if requested < 1:
            raise ValueError("count must be positive")
        radius_value = np.float32(radius)
        slots = min(requested, self.size)
        if slots == 0:
            empty = Matches(np.empty(0, np.uint64), np.empty(0, np.float32))
            return empty if input_ndim == 1 else BatchMatches(
                np.empty((queries.shape[0], 0), np.uint64), np.empty((queries.shape[0], 0), np.float32),
                np.zeros(queries.shape[0], np.int64))
        positions = np.empty((queries.shape[0], slots), dtype=np.int64)
        distances = np.empty((queries.shape[0], slots), dtype=np.float32)
        status = lib().mus_search_f32(
            addr(self._vectors), addr(queries), addr(positions), addr(distances),
            self.size, self.ndim, queries.shape[0], slots, _METRICS[self.metric.value], radius_value,
            int(threads), self._vectors.size, queries.size, positions.size, distances.size,
        )
        if status:
            raise RuntimeError("Mojo search ABI rejected its validated buffer contract")
        valid = positions >= 0
        keys = np.zeros(positions.shape, dtype=np.uint64)
        keys[valid] = self._keys[positions[valid]]
        counts = valid.sum(axis=1, dtype=np.int64)
        if input_ndim == 1:
            n = int(counts[0])
            return Matches(keys[0, :n], distances[0, :n])
        return BatchMatches(keys, distances, counts)

    def get(self, keys, dtype=None):
        wanted = np.asarray(keys, dtype=np.uint64)
        scalar = wanted.ndim == 0
        wanted = wanted.reshape(-1)
        result = []
        for key in wanted:
            found = np.flatnonzero(self._keys == key)
            result.append(None if not found.size else self._vectors[found[0]].copy())
        return result[0] if scalar else tuple(result)

    def remove(self, keys, *, compact: bool = False, threads: int = 0) -> int | np.ndarray:
        wanted = np.asarray(keys, dtype=np.uint64)
        scalar = wanted.ndim == 0
        wanted = wanted.reshape(-1)
        removed = np.array([np.count_nonzero(self._keys == key) for key in wanted], dtype=np.int64)
        keep = ~np.isin(self._keys, wanted)
        self._keys, self._vectors = self._keys[keep], self._vectors[keep]
        return int(removed[0]) if scalar else removed

    def rename(self, from_, to) -> int | np.ndarray:
        old = np.asarray(from_, dtype=np.uint64)
        new = np.asarray(to, dtype=np.uint64)
        old, new = np.broadcast_arrays(old, new)
        scalar = old.ndim == 0
        changed = np.empty(old.size, dtype=np.int64)
        for i, (source, target) in enumerate(zip(old.reshape(-1), new.reshape(-1))):
            mask = self._keys == source
            changed[i] = np.count_nonzero(mask)
            if changed[i]:
                if not self.multi and source != target:
                    self.remove(target)
                self._keys[mask] = target
        return int(changed[0]) if scalar else changed.reshape(old.shape)

    def reset(self) -> None:
        self._keys = np.empty(0, dtype=np.uint64)
        self._vectors = np.empty((0, self.ndim), dtype=np.float32)

    clear = reset

    def save(self, path_or_buffer=None, progress=None) -> bytes | None:
        target = BytesIO() if path_or_buffer is None else path_or_buffer
        close = not hasattr(target, "write")
        file = open(target, "wb") if close else target
        try:
            np.savez(file, keys=self._keys, vectors=self._vectors, ndim=self.ndim,
                     metric=self.metric.value, multi=self.multi)
        finally:
            if close:
                file.close()
        return target.getvalue() if path_or_buffer is None else None

    def load(self, path_or_buffer=None, progress=None) -> "Index":
        if path_or_buffer is None:
            raise ValueError("a path or serialized index buffer is required")
        source = BytesIO(path_or_buffer) if isinstance(path_or_buffer, bytes) else path_or_buffer
        with np.load(source, allow_pickle=False) as data:
            self._keys = np.ascontiguousarray(data["keys"], dtype=np.uint64)
            self._vectors = np.ascontiguousarray(data["vectors"], dtype=np.float32)
            self.ndim = int(data["ndim"])
            self.metric = MetricKind(str(data["metric"]))
            self.multi = bool(data["multi"])
        return self

    def view(self, path_or_buffer=None, progress=None) -> "Index":
        return self.load(path_or_buffer, progress=progress)

    def copy(self) -> "Index":
        clone = Index(ndim=self.ndim, metric=self.metric, dtype=self.dtype,
                      connectivity=self.connectivity, expansion_add=self.expansion_add,
                      expansion_search=self.expansion_search, multi=self.multi,
                      enable_key_lookups=self.enable_key_lookups)
        clone._keys, clone._vectors = self._keys.copy(), self._vectors.copy()
        return clone

    @classmethod
    def restore(cls, path_or_buffer, view: bool = False, **kwargs) -> "Index":
        index = cls(**kwargs)
        return index.view(path_or_buffer) if view else index.load(path_or_buffer)

    @staticmethod
    def metadata(path: str) -> dict:
        with np.load(path, allow_pickle=False) as data:
            return {"dimensions": int(data["ndim"]), "metric": str(data["metric"]),
                    "size": int(data["keys"].size)}


def search(dataset, query, count: int = 10, metric: str | MetricKind = MetricKind.Cos,
           *, exact: bool = False, threads: int = 0, log: str | bool = False, progress=None, dtype=None):
    vectors = f32_matrix(dataset)
    index = Index(ndim=vectors.shape[1], metric=metric, dtype=dtype)
    index.add(np.arange(vectors.shape[0], dtype=np.uint64), vectors)
    return index.search(query, count=count, exact=exact, threads=threads, log=log, progress=progress, dtype=dtype)
