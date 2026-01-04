"""
Semantic search example using sentence-transformers with zero-copy vector search.

This example demonstrates:
1. Using sentence-transformers to embed text documents
2. Storing embeddings in a zero-copy vector store
3. Building an HNSW index for fast semantic search
4. Performing semantic queries
"""

import time
from sentence_transformers import SentenceTransformer
from zero_copy_search import VectorStore, HNSWIndex

def main():
    print("=" * 60)
    print("Zero-Copy Vector Search - Semantic Search Example")
    print("=" * 60)

    # Sample documents (you can replace with your own corpus)
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be, that is the question",
        "Python is a high-level programming language",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing enables computers to understand human language",
        "Vector databases are optimized for similarity search",
        "Embeddings represent text as dense numerical vectors",
        "Transformers revolutionized natural language understanding",
        "BERT is a pre-trained transformer model",
        "GPT models are autoregressive language models",
        "Semantic search finds results based on meaning, not just keywords",
        "The cat sat on the mat",
        "Dogs are loyal companions",
        "Artificial neural networks are inspired by biological neurons",
        "Gradient descent is an optimization algorithm",
        "Backpropagation is used to train neural networks",
        "Attention mechanisms allow models to focus on relevant information",
        "Transfer learning leverages pre-trained models for new tasks",
    ]

    print(f"\n1. Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"   Model loaded: {model}")

    print(f"\n2. Encoding {len(documents)} documents...")
    start_time = time.time()
    embeddings = model.encode(documents, show_progress_bar=True)
    encode_time = time.time() - start_time
    print(f"   Encoding time: {encode_time:.2f} seconds")
    print(f"   Embedding dimension: {embeddings.shape[1]}")

    # Convert to list of lists for vector store
    embeddings_list = [emb.tolist() for emb in embeddings]

    # Create vector store
    store_path = "/tmp/semantic_vectors.vec"
    print(f"\n3. Creating vector store at {store_path}...")
    VectorStore.create(store_path, embeddings_list)
    store = VectorStore.open(store_path)
    print(f"   Vector store: {store}")

    # Build HNSW index
    print("\n4. Building HNSW index...")
    start_time = time.time()
    index = HNSWIndex(store, m=16, ef_construction=100, metric="cosine")
    index.build()
    build_time = time.time() - start_time
    print(f"   Build time: {build_time:.2f} seconds")

    # Perform semantic queries
    print("\n5. Performing semantic searches:")
    print("-" * 60)

    queries = [
        "programming languages",
        "artificial intelligence and neural networks",
        "famous quotes and sayings",
        "animals and pets",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")

        # Encode query
        query_embedding = model.encode([query])[0].tolist()

        # Search
        start_time = time.time()
        results = index.search(query_embedding, k=3, ef=50)
        search_time = time.time() - start_time

        print(f"Search time: {search_time * 1000:.2f} ms")
        print("Top 3 results:")
        for rank, (idx, distance) in enumerate(results, 1):
            similarity = 1 - distance  # Convert cosine distance to similarity
            print(f"  {rank}. [{similarity:.3f}] {documents[idx]}")

    print("\n" + "=" * 60)
    print("Semantic search example completed!")
    print("=" * 60)

    # Performance summary
    print("\n📊 Performance Summary:")
    print(f"  • Documents: {len(documents)}")
    print(f"  • Embedding dimension: {embeddings.shape[1]}")
    print(f"  • Encoding time: {encode_time:.2f}s")
    print(f"  • Index build time: {build_time:.2f}s")
    print(f"  • Average search time: {search_time * 1000:.2f}ms")

    # Calculate storage efficiency
    import os
    file_size_mb = os.path.getsize(store_path) / (1024 * 1024)
    print(f"  • Vector store size: {file_size_mb:.2f} MB")
    print(f"  • Storage per vector: {file_size_mb * 1024 / len(documents):.2f} KB")

if __name__ == "__main__":
    main()
