"""
Zero-Copy Vector Search Engine

A high-performance vector search engine using Rust and memory-mapped files.
Supports approximate nearest neighbor search with HNSW index and SIMD-accelerated
distance calculations.
"""

from ._zero_copy import (
    PyVectorStore as VectorStore,
    PyHNSWIndex as HNSWIndex,
    brute_force_search,
    calculate_distance,
)

__version__ = "0.1.0"
__all__ = [
    "VectorStore",
    "HNSWIndex",
    "brute_force_search",
    "calculate_distance",
]
