"""
Basic tests for zero-copy vector search engine.
"""

import numpy as np
import pytest
import os
import tempfile
from zero_copy_search import VectorStore, HNSWIndex, brute_force_search, calculate_distance


@pytest.fixture
def temp_vector_file():
    """Create a temporary file path for testing."""
    fd, path = tempfile.mkstemp(suffix='.vec')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    np.random.seed(42)
    return np.random.randn(100, 32).astype(np.float32).tolist()


def test_vector_store_create_and_open(temp_vector_file, sample_vectors):
    """Test creating and opening a vector store."""
    # Create store
    VectorStore.create(temp_vector_file, sample_vectors)

    # Open store
    store = VectorStore.open(temp_vector_file)

    # Verify
    assert len(store) == len(sample_vectors)
    assert store.dimension() == len(sample_vectors[0])

    # Check individual vectors
    for i in range(min(10, len(sample_vectors))):
        retrieved = store.get(i)
        assert len(retrieved) == len(sample_vectors[i])
        assert all(abs(a - b) < 1e-5 for a, b in zip(retrieved, sample_vectors[i]))


def test_vector_store_repr(temp_vector_file, sample_vectors):
    """Test vector store string representation."""
    VectorStore.create(temp_vector_file, sample_vectors)
    store = VectorStore.open(temp_vector_file)
    repr_str = repr(store)
    assert "VectorStore" in repr_str
    assert str(len(sample_vectors)) in repr_str
    assert str(len(sample_vectors[0])) in repr_str


def test_hnsw_build_and_search(temp_vector_file, sample_vectors):
    """Test HNSW index building and search."""
    # Create and open store
    VectorStore.create(temp_vector_file, sample_vectors)
    store = VectorStore.open(temp_vector_file)

    # Build index
    index = HNSWIndex(store, m=8, ef_construction=100, metric="l2")
    index.build()

    # Search
    query = sample_vectors[0]
    results = index.search(query, k=5, ef=50)

    # Verify
    assert len(results) == 5
    assert results[0][0] == 0  # First result should be the query itself
    assert results[0][1] < 1e-5  # Distance should be ~0


def test_hnsw_different_metrics(temp_vector_file, sample_vectors):
    """Test HNSW with different distance metrics."""
    VectorStore.create(temp_vector_file, sample_vectors)
    store = VectorStore.open(temp_vector_file)

    metrics = ["l2", "cosine", "dot"]

    for metric in metrics:
        index = HNSWIndex(store, m=8, ef_construction=50, metric=metric)
        index.build()

        query = sample_vectors[0]
        results = index.search(query, k=3)

        assert len(results) <= 3
        assert all(isinstance(idx, int) for idx, _ in results)
        assert all(isinstance(dist, float) for _, dist in results)


def test_brute_force_search(temp_vector_file, sample_vectors):
    """Test brute-force search."""
    VectorStore.create(temp_vector_file, sample_vectors)
    store = VectorStore.open(temp_vector_file)

    query = sample_vectors[0]
    results = brute_force_search(store, query, k=5, metric="l2")

    assert len(results) == 5
    assert results[0][0] == 0  # First result should be the query itself
    assert results[0][1] < 1e-5

    # Results should be sorted by distance
    distances = [dist for _, dist in results]
    assert distances == sorted(distances)


def test_calculate_distance():
    """Test distance calculation."""
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]

    # L2 distance
    l2_dist = calculate_distance(a, b, "l2")
    expected_l2 = np.sqrt(2)
    assert abs(l2_dist - expected_l2) < 1e-5

    # Cosine distance
    cos_dist = calculate_distance(a, b, "cosine")
    assert abs(cos_dist - 1.0) < 1e-5  # Orthogonal vectors


def test_hnsw_stats(temp_vector_file, sample_vectors):
    """Test HNSW index statistics."""
    VectorStore.create(temp_vector_file, sample_vectors)
    store = VectorStore.open(temp_vector_file)

    index = HNSWIndex(store, m=8, ef_construction=50)
    index.build()

    stats = index.stats()

    assert stats["num_vectors"] == len(sample_vectors)
    assert stats["dimension"] == len(sample_vectors[0])
    assert stats["num_layers"] > 0
    assert stats["total_connections"] > 0
    assert stats["avg_connections"] > 0


def test_empty_vectors():
    """Test handling of empty vectors list."""
    with tempfile.NamedTemporaryFile(suffix='.vec', delete=False) as f:
        temp_file = f.name

    try:
        with pytest.raises(Exception):
            VectorStore.create(temp_file, [])
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_mismatched_dimensions():
    """Test handling of vectors with mismatched dimensions."""
    vectors = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0],  # Wrong dimension
    ]

    with tempfile.NamedTemporaryFile(suffix='.vec', delete=False) as f:
        temp_file = f.name

    try:
        with pytest.raises(Exception):
            VectorStore.create(temp_file, vectors)
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_index_out_of_bounds(temp_vector_file, sample_vectors):
    """Test accessing vector with out-of-bounds index."""
    VectorStore.create(temp_vector_file, sample_vectors)
    store = VectorStore.open(temp_vector_file)

    with pytest.raises(Exception):
        store.get(len(sample_vectors) + 10)


@pytest.mark.benchmark
def test_search_performance(temp_vector_file, benchmark):
    """Benchmark search performance."""
    # Create larger dataset
    np.random.seed(42)
    vectors = np.random.randn(1000, 128).astype(np.float32).tolist()

    VectorStore.create(temp_vector_file, vectors)
    store = VectorStore.open(temp_vector_file)

    index = HNSWIndex(store, m=16, ef_construction=100)
    index.build()

    query = vectors[0]

    # Benchmark
    result = benchmark(lambda: index.search(query, k=10, ef=50))
    assert len(result) == 10
