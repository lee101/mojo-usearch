"""Parity checks against the PyPI usearch package on the same float32 data."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from usearch.index import Index as MojoIndex
from usearch.index import MetricKind, ScalarKind
from usearch.index import search as mojo_search


def _upstream_index():
    """Import the installed distribution after retaining direct references to ours."""
    local_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
    saved = {name: module for name, module in list(sys.modules.items()) if name == "usearch" or name.startswith("usearch.")}
    for name in saved:
        del sys.modules[name]
    old_path = sys.path[:]
    sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry) != os.path.abspath(local_python)]
    try:
        from usearch.index import Index
        return Index
    finally:
        for name in [name for name in list(sys.modules) if name == "usearch" or name.startswith("usearch.")]:
            del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = old_path


UpstreamIndex = _upstream_index()


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(42)
    return np.ascontiguousarray(rng.normal(size=(257, 31)).astype(np.float32))


@pytest.mark.parametrize("metric", ["l2sq", "cos", "ip"])
def test_exact_search_matches_upstream(vectors, metric):
    keys = np.arange(10_000, 10_000 + len(vectors), dtype=np.uint64)
    query = vectors[17] + np.float32(0.03125)
    ours = MojoIndex(ndim=vectors.shape[1], metric=metric, dtype="f32")
    theirs = UpstreamIndex(ndim=vectors.shape[1], metric=metric, dtype="f32")
    ours.add(keys, vectors)
    theirs.add(keys, vectors)
    got, expected = ours.search(query, count=20, exact=True), theirs.search(query, count=20, exact=True)
    assert np.array_equal(got.keys, expected.keys)
    assert np.allclose(got.distances, expected.distances, rtol=2e-5, atol=2e-5)
    assert got.count == len(expected.keys)


@pytest.mark.parametrize("metric", ["l2sq", "cos", "ip"])
def test_batched_search_matches_upstream(vectors, metric):
    keys = np.arange(len(vectors), dtype=np.uint64)
    queries = np.ascontiguousarray(vectors[[0, 91, 192]] + np.float32(0.007))
    ours = MojoIndex(ndim=vectors.shape[1], metric=metric, dtype="f32")
    theirs = UpstreamIndex(ndim=vectors.shape[1], metric=metric, dtype="f32")
    ours.add(keys, vectors)
    theirs.add(keys, vectors)
    got, expected = ours.search(queries, count=11, exact=True), theirs.search(queries, count=11, exact=True)
    assert np.array_equal(got.keys, expected.keys)
    assert np.allclose(got.distances, expected.distances, rtol=2e-5, atol=2e-5)
    assert np.array_equal(got.counts, np.full(queries.shape[0], 11))


def test_simd_tail_and_parallel_threshold_are_deterministic():
    rng = np.random.default_rng(9)
    vectors = np.ascontiguousarray(rng.normal(size=(4097, 31)).astype(np.float32))
    queries = np.ascontiguousarray(rng.normal(size=(9, 31)).astype(np.float32))
    index = MojoIndex(ndim=31, metric="cos", dtype="f32")
    index.add(np.arange(len(vectors), dtype=np.uint64), vectors)

    serial = index.search(queries, count=7, exact=True, threads=1)
    automatic = index.search(queries, count=7, exact=True)
    parallel = index.search(queries, count=7, exact=True, threads=4)

    for result in (automatic, parallel):
        assert np.array_equal(result.keys, serial.keys)
        assert np.allclose(result.distances, serial.distances, rtol=2e-5, atol=2e-5)

    small_vectors = vectors[:64]
    small = MojoIndex(ndim=31, metric="cos", dtype="f32")
    small.add(np.arange(len(small_vectors), dtype=np.uint64), small_vectors)
    small_serial = small.search(queries[:2], count=7, exact=True, threads=1)
    small_automatic = small.search(queries[:2], count=7, exact=True)
    assert np.array_equal(small_automatic.keys, small_serial.keys)
    assert np.allclose(
        small_automatic.distances, small_serial.distances,
        rtol=2e-5, atol=2e-5,
    )


def test_radius_and_empty_index_behaviour(vectors):
    index = MojoIndex(ndim=vectors.shape[1], metric="l2sq")
    assert index.search(vectors[0], count=4).count == 0
    index.add(np.arange(len(vectors)), vectors)
    matches = index.search(vectors[0], count=20, radius=0.001, exact=True)
    assert matches.keys.tolist() == [0]
    assert matches.distances.tolist() == pytest.approx([0.0])


def test_mutation_get_and_key_lookup_contract(vectors):
    index = MojoIndex(ndim=vectors.shape[1], metric="cos")
    index.add([7, 8], vectors[:2])
    assert index.contains(7) and index.contains([7, 9]).tolist() == [True, False]
    assert index.count(7) == 1
    assert np.array_equal(index.get(8), vectors[1])
    assert index.get([8, 99])[1] is None
    assert index.rename(8, 12) == 1
    assert index.contains(12) and not index.contains(8)
    assert index.remove([7, 99]).tolist() == [1, 0]
    assert index.size == 1
    index.reset()
    assert index.size == 0 and index.capacity == 0
    index.add(3, vectors[0])
    index.clear()
    assert index.size == 0


def test_multi_key_mode_preserves_multiple_vectors(vectors):
    index = MojoIndex(ndim=vectors.shape[1], metric="l2sq", multi=True)
    index.add([5, 5], vectors[:2])
    assert index.count(5) == 2
    assert index.remove(5) == 2


def test_top_level_search_matches_upstream(vectors):
    query = vectors[10]
    got = mojo_search(vectors, query, count=8, metric="cos", exact=True)
    expected = UpstreamIndex(ndim=vectors.shape[1], metric="cos", dtype="f32")
    expected.add(np.arange(len(vectors)), vectors)
    reference = expected.search(query, count=8, exact=True)
    assert np.array_equal(got.keys, reference.keys)
    assert np.allclose(got.distances, reference.distances, atol=2e-5)


def test_save_load_round_trip(vectors, tmp_path):
    path = tmp_path / "dense.usearch"
    index = MojoIndex(ndim=vectors.shape[1], metric="ip")
    index.add(np.arange(100, 100 + len(vectors)), vectors)
    index.save(path)
    restored = MojoIndex.restore(path)
    assert restored.metadata(path) == {"dimensions": 31, "metric": "ip", "size": 257}
    assert np.array_equal(restored.search(vectors[2], count=5).keys, index.search(vectors[2], count=5).keys)
    assert np.array_equal(MojoIndex(ndim=31).view(path).keys, index.keys)


def test_float32_contract_and_documented_example():
    index = MojoIndex(ndim=3, metric=MetricKind.Cos, dtype=ScalarKind.F32)
    index.add([101, 202, 303], np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32))
    matches = index.search(np.array([1, 0, 0], dtype=np.float32), count=2, exact=True)
    assert matches.keys.tolist() == [101, 303]
    assert matches.distances.tolist() == pytest.approx([0.0, 0.29289323])
    with pytest.raises(TypeError, match="float32"):
        index.search(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    with pytest.raises(ValueError, match="only f32"):
        index.search([1, 0, 0], dtype="f64")


def test_ffi_rejects_inconsistent_lengths(vectors):
    from usearch._lib import addr, lib

    query = vectors[:1]
    positions = np.empty((1, 1), dtype=np.int64)
    distances = np.empty((1, 1), dtype=np.float32)
    status = lib().mus_search_f32(
        addr(vectors), addr(query), addr(positions), addr(distances),
        len(vectors), vectors.shape[1], 1, 1, 0, np.inf, 1,
        vectors.size - 1, query.size, positions.size, distances.size,
    )
    assert status != 0
