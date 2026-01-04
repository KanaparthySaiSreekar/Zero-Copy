"""
Basic usage example of the zero-copy vector search engine.

This example demonstrates:
1. Creating a vector store from random data
2. Building an HNSW index
3. Performing approximate nearest neighbor search
4. Comparing with brute-force search
"""

import numpy as np
import time
from zero_copy_search import VectorStore, HNSWIndex, brute_force_search

def main():
    # Configuration
    num_vectors = 10000
    dimension = 128
    k = 10  # Number of nearest neighbors to find

    print("=" * 60)
    print("Zero-Copy Vector Search - Basic Usage Example")
    print("=" * 60)

    # Generate random vectors
    print(f"\n1. Generating {num_vectors} random vectors of dimension {dimension}...")
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    vectors_list = [vec.tolist() for vec in vectors]

    # Create vector store
    store_path = "/tmp/example_vectors.vec"
    print(f"2. Creating vector store at {store_path}...")
    VectorStore.create(store_path, vectors_list)

    # Open vector store
    print("3. Opening vector store (using memory-mapped file)...")
    store = VectorStore.open(store_path)
    print(f"   Loaded: {store}")

    # Build HNSW index
    print("\n4. Building HNSW index...")
    start_time = time.time()
    index = HNSWIndex(store, m=16, ef_construction=200, metric="l2")
    index.build()
    build_time = time.time() - start_time
    print(f"   Build time: {build_time:.2f} seconds")
    print(f"   Index stats: {index.stats()}")

    # Generate query vector
    print("\n5. Performing search...")
    query = np.random.randn(dimension).astype(np.float32).tolist()

    # HNSW search
    print(f"   a) HNSW search (approximate, k={k})...")
    start_time = time.time()
    hnsw_results = index.search(query, k=k, ef=50)
    hnsw_time = time.time() - start_time
    print(f"      Search time: {hnsw_time * 1000:.2f} ms")
    print(f"      Top 5 results: {hnsw_results[:5]}")

    # Brute-force search for comparison
    print(f"\n   b) Brute-force search (exact, k={k})...")
    start_time = time.time()
    exact_results = brute_force_search(store, query, k=k, metric="l2")
    exact_time = time.time() - start_time
    print(f"      Search time: {exact_time * 1000:.2f} ms")
    print(f"      Top 5 results: {exact_results[:5]}")

    # Calculate recall
    hnsw_ids = set(idx for idx, _ in hnsw_results)
    exact_ids = set(idx for idx, _ in exact_results)
    recall = len(hnsw_ids & exact_ids) / k

    print(f"\n6. Results:")
    print(f"   Speedup: {exact_time / hnsw_time:.2f}x")
    print(f"   Recall@{k}: {recall:.2%}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
