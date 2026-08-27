"""SIMD exact-search kernels exposed through a stable C ABI."""

from max.algorithm import parallelize
from std.math import sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime W = simdwidthof[DType.float32]()
comptime PARALLEL_WORK_THRESHOLD = 262144
comptime AUTO_WORKERS = 24
comptime Ptr = Pointer[Float32, AnyOrigin[mut=True]]
comptime IndexPtr = Pointer[Int64, AnyOrigin[mut=True]]


def product_matches(left: Int, right: Int, actual: Int) -> Bool:
    """Check `actual == left * right` without overflowing the multiplication."""
    if left < 0 or right < 0 or actual < 0:
        return False
    if left == 0 or right == 0:
        return actual == 0
    if right > actual // left:
        return False
    return left * right == actual


def l2sq(a: Ptr, b: Ptr, n: Int) -> Float32:
    var acc0 = SIMD[DType.float32, W](0.0)
    var acc1 = SIMD[DType.float32, W](0.0)
    var acc2 = SIMD[DType.float32, W](0.0)
    var acc3 = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + 4 * W <= n:
        var delta0 = a.unsafe_load[width=W](i) - b.unsafe_load[width=W](i)
        var delta1 = a.unsafe_load[width=W](i + W) - b.unsafe_load[width=W](i + W)
        var delta2 = a.unsafe_load[width=W](i + 2 * W) - b.unsafe_load[width=W](i + 2 * W)
        var delta3 = a.unsafe_load[width=W](i + 3 * W) - b.unsafe_load[width=W](i + 3 * W)
        acc0 += delta0 * delta0
        acc1 += delta1 * delta1
        acc2 += delta2 * delta2
        acc3 += delta3 * delta3
        i += 4 * W
    var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
    while i + W <= n:
        var delta = a.unsafe_load[width=W](i) - b.unsafe_load[width=W](i)
        total += (delta * delta).reduce_add()
        i += W
    while i < n:
        var delta = a.unsafe_load(i) - b.unsafe_load(i)
        total += delta * delta
        i += 1
    return total


def squared_norm(a: Ptr, n: Int) -> Float32:
    var acc0 = SIMD[DType.float32, W](0.0)
    var acc1 = SIMD[DType.float32, W](0.0)
    var acc2 = SIMD[DType.float32, W](0.0)
    var acc3 = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + 4 * W <= n:
        var av0 = a.unsafe_load[width=W](i)
        var av1 = a.unsafe_load[width=W](i + W)
        var av2 = a.unsafe_load[width=W](i + 2 * W)
        var av3 = a.unsafe_load[width=W](i + 3 * W)
        acc0 += av0 * av0
        acc1 += av1 * av1
        acc2 += av2 * av2
        acc3 += av3 * av3
        i += 4 * W
    var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
    while i + W <= n:
        var av = a.unsafe_load[width=W](i)
        total += (av * av).reduce_add()
        i += W
    while i < n:
        total += a.unsafe_load(i) * a.unsafe_load(i)
        i += 1
    return total


def cosine_with_query_norm(a: Ptr, b: Ptr, n: Int, an: Float32) -> Float32:
    var dots0 = SIMD[DType.float32, W](0.0)
    var dots1 = SIMD[DType.float32, W](0.0)
    var dots2 = SIMD[DType.float32, W](0.0)
    var dots3 = SIMD[DType.float32, W](0.0)
    var bb0 = SIMD[DType.float32, W](0.0)
    var bb1 = SIMD[DType.float32, W](0.0)
    var bb2 = SIMD[DType.float32, W](0.0)
    var bb3 = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + 4 * W <= n:
        var av0 = a.unsafe_load[width=W](i)
        var bv0 = b.unsafe_load[width=W](i)
        var av1 = a.unsafe_load[width=W](i + W)
        var bv1 = b.unsafe_load[width=W](i + W)
        var av2 = a.unsafe_load[width=W](i + 2 * W)
        var bv2 = b.unsafe_load[width=W](i + 2 * W)
        var av3 = a.unsafe_load[width=W](i + 3 * W)
        var bv3 = b.unsafe_load[width=W](i + 3 * W)
        dots0 += av0 * bv0
        dots1 += av1 * bv1
        dots2 += av2 * bv2
        dots3 += av3 * bv3
        bb0 += bv0 * bv0
        bb1 += bv1 * bv1
        bb2 += bv2 * bv2
        bb3 += bv3 * bv3
        i += 4 * W
    var dot = (dots0 + dots1 + dots2 + dots3).reduce_add()
    var bn = (bb0 + bb1 + bb2 + bb3).reduce_add()
    while i + W <= n:
        var av = a.unsafe_load[width=W](i)
        var bv = b.unsafe_load[width=W](i)
        dot += (av * bv).reduce_add()
        bn += (bv * bv).reduce_add()
        i += W
    while i < n:
        dot += a.unsafe_load(i) * b.unsafe_load(i)
        bn += b.unsafe_load(i) * b.unsafe_load(i)
        i += 1
    if an == 0.0 or bn == 0.0:
        return 1.0
    return 1.0 - dot / sqrt(an * bn)


def inner_product(a: Ptr, b: Ptr, n: Int) -> Float32:
    var acc0 = SIMD[DType.float32, W](0.0)
    var acc1 = SIMD[DType.float32, W](0.0)
    var acc2 = SIMD[DType.float32, W](0.0)
    var acc3 = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + 4 * W <= n:
        acc0 += a.unsafe_load[width=W](i) * b.unsafe_load[width=W](i)
        acc1 += a.unsafe_load[width=W](i + W) * b.unsafe_load[width=W](i + W)
        acc2 += a.unsafe_load[width=W](i + 2 * W) * b.unsafe_load[width=W](i + 2 * W)
        acc3 += a.unsafe_load[width=W](i + 3 * W) * b.unsafe_load[width=W](i + 3 * W)
        i += 4 * W
    var dot = (acc0 + acc1 + acc2 + acc3).reduce_add()
    while i + W <= n:
        dot += (a.unsafe_load[width=W](i) * b.unsafe_load[width=W](i)).reduce_add()
        i += W
    while i < n:
        dot += a.unsafe_load(i) * b.unsafe_load(i)
        i += 1
    return 1.0 - dot


def search_one(
    vectors: Ptr,
    query: Ptr,
    positions: IndexPtr,
    distances: Ptr,
    rows: Int,
    dimensions: Int,
    count: Int,
    metric: Int,
    radius: Float32,
    q: Int,
):
    var base = q * count
    for slot in range(count):
        positions.unsafe_store(base + slot, -1)
        distances.unsafe_store(base + slot, 3.402823466e38)
    var query_row = query.unsafe_offset(q * dimensions)
    var query_norm = Float32(0.0)
    if metric == 1:
        query_norm = squared_norm(query_row, dimensions)
    for row in range(rows):
        var vector = vectors.unsafe_offset(row * dimensions)
        var score = Float32(0.0)
        if metric == 0:
            score = l2sq(query_row, vector, dimensions)
        elif metric == 1:
            score = cosine_with_query_norm(query_row, vector, dimensions, query_norm)
        else:
            score = inner_product(query_row, vector, dimensions)
        if score > radius or score >= distances.unsafe_load(base + count - 1):
            continue
        var slot = count - 1
        while slot > 0 and distances.unsafe_load(base + slot - 1) > score:
            distances.unsafe_store(base + slot, distances.unsafe_load(base + slot - 1))
            positions.unsafe_store(base + slot, positions.unsafe_load(base + slot - 1))
            slot -= 1
        distances.unsafe_store(base + slot, score)
        positions.unsafe_store(base + slot, Int64(row))


@export("mus_search_f32")
def mus_search_f32(
    vectors_addr: Int,
    queries_addr: Int,
    positions_addr: Int,
    distances_addr: Int,
    rows: Int,
    dimensions: Int,
    queries: Int,
    count: Int,
    metric: Int,
    radius: Float32,
    workers: Int,
    vectors_length: Int,
    queries_length: Int,
    positions_length: Int,
    distances_length: Int,
) abi("C") -> Int:
    if dimensions <= 0 or count <= 0 or metric < 0 or metric > 2:
        return 1
    if not product_matches(rows, dimensions, vectors_length):
        return 1
    if not product_matches(queries, dimensions, queries_length):
        return 1
    if not product_matches(queries, count, positions_length):
        return 1
    if not product_matches(queries, count, distances_length):
        return 1
    if queries == 0 or rows == 0:
        return 0
    if vectors_addr == 0 or queries_addr == 0 or positions_addr == 0 or distances_addr == 0:
        return 1
    var vectors = Ptr(unsafe_from_address=vectors_addr)
    var query = Ptr(unsafe_from_address=queries_addr)
    var positions = IndexPtr(unsafe_from_address=positions_addr)
    var distances = Ptr(unsafe_from_address=distances_addr)
    @__parameter
    def search_query(q: Int):
        search_one(
            vectors, query, positions, distances, rows, dimensions,
            count, metric, radius, q,
        )

    var worker_count = workers
    if worker_count <= 0:
        worker_count = AUTO_WORKERS
    worker_count = min(worker_count, queries)
    if (
        worker_count > 1
        and rows * dimensions >= PARALLEL_WORK_THRESHOLD // queries
    ):
        parallelize[search_query](queries, worker_count)
    else:
        for q in range(queries):
            search_query(q)
    return 0
