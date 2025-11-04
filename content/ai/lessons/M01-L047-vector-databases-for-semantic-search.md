# Vector Databases for Semantic Search

## Core Concepts

Vector databases store and retrieve data based on semantic similarity rather than exact matches. Unlike traditional databases that compare strings or indexed keys, vector databases compare high-dimensional numerical representations (embeddings) that encode meaning.

### Traditional vs. Vector-Based Search

```python
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

# Traditional keyword search
class TraditionalSearch:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.inverted_index = self._build_index()
    
    def _build_index(self) -> dict:
        index = {}
        for doc_id, doc in enumerate(self.documents):
            for word in doc.lower().split():
                if word not in index:
                    index[word] = set()
                index[word].add(doc_id)
        return index
    
    def search(self, query: str) -> List[int]:
        words = query.lower().split()
        if not words:
            return []
        result = self.inverted_index.get(words[0], set())
        for word in words[1:]:
            result &= self.inverted_index.get(word, set())
        return list(result)

# Vector-based semantic search
@dataclass
class Document:
    id: int
    text: str
    embedding: np.ndarray

class VectorSearch:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.embeddings = np.vstack([doc.embedding for doc in documents])
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        # Cosine similarity: dot product of normalized vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]

# Demonstration
docs = [
    "Python programming language",
    "Machine learning algorithms",
    "Neural network architectures"
]

# Traditional search fails on semantic queries
trad = TraditionalSearch(docs)
print(trad.search("ML models"))  # Returns [] - no exact match

# Vector search succeeds (assuming embeddings exist)
# Note: In practice, you'd use a real embedding model
mock_embeddings = np.random.randn(3, 384)  # 384-dim embeddings
vec_docs = [Document(i, text, emb) for i, (text, emb) in enumerate(zip(docs, mock_embeddings))]
vec_search = VectorSearch(vec_docs)
query_emb = np.random.randn(384)
results = vec_search.search(query_emb, top_k=2)
print(f"Top results: {results}")  # Returns semantically similar docs
```

### Key Engineering Insights

**1. Dimensionality Drives Everything**: Vector dimensionality (typically 384-1536) fundamentally impacts memory usage, query latency, and index structure. A million 1536-dimensional float32 vectors require 6GB of raw storage before indexing overhead.

**2. Approximate is Good Enough**: Production vector databases trade perfect recall for speed using approximate nearest neighbor (ANN) algorithms. You'll typically achieve 95-99% recall at 10-100x faster query times compared to exact search.

**3. Embedding Quality Trumps Infrastructure**: A better embedding model (higher semantic quality) will outperform a worse model in a more sophisticated vector database. Focus on embedding selection first, then optimize infrastructure.

### Why This Matters Now

Vector databases have shifted from research curiosities to production requirements because:
- LLM context windows remain limited (4K-128K tokens), requiring efficient retrieval of relevant information
- Traditional search fails at semantic understanding: "car accident" vs "vehicle collision" are different strings but identical meanings
- Real-time requirements: Users expect <100ms query response times even across millions of documents
- Cost efficiency: Retrieving 5 relevant documents is cheaper than passing 10,000 documents to an LLM

## Technical Components

### 1. Vector Embeddings and Dimensionality

Embeddings are dense numerical representations where semantic similarity correlates with geometric proximity. Each dimension captures learned features from training data.

```python
import numpy as np
from typing import List, Dict
import hashlib

class EmbeddingAnalysis:
    """Analyze embedding characteristics for production planning"""
    
    @staticmethod
    def memory_footprint(num_vectors: int, dimensions: int, dtype: str = 'float32') -> Dict[str, float]:
        """Calculate storage requirements"""
        bytes_per_element = {'float32': 4, 'float16': 2, 'int8': 1}[dtype]
        raw_bytes = num_vectors * dimensions * bytes_per_element
        
        # Approximate index overhead (varies by algorithm)
        hnsw_overhead = raw_bytes * 0.5  # HNSW typically adds 50% overhead
        ivf_overhead = raw_bytes * 0.2   # IVF typically adds 20% overhead
        
        return {
            'raw_gb': raw_bytes / (1024**3),
            'hnsw_total_gb': (raw_bytes + hnsw_overhead) / (1024**3),
            'ivf_total_gb': (raw_bytes + ivf_overhead) / (1024**3)
        }
    
    @staticmethod
    def similarity_metrics(vec1: np.ndarray, vec2: np.ndarray) -> Dict[str, float]:
        """Compare different distance metrics"""
        # Cosine similarity (most common for semantic search)
        cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Euclidean distance (L2)
        euclidean = np.linalg.norm(vec1 - vec2)
        
        # Dot product (assumes normalized vectors)
        dot_product = np.dot(vec1, vec2)
        
        return {
            'cosine_similarity': float(cosine),
            'euclidean_distance': float(euclidean),
            'dot_product': float(dot_product)
        }

# Example: Planning for production scale
analysis = EmbeddingAnalysis()

# Scenario: 1M documents with different embedding dimensions
for dims in [384, 768, 1536]:
    memory = analysis.memory_footprint(1_000_000, dims)
    print(f"\n{dims}D embeddings for 1M vectors:")
    print(f"  Raw storage: {memory['raw_gb']:.2f} GB")
    print(f"  With HNSW index: {memory['hnsw_total_gb']:.2f} GB")
    print(f"  With IVF index: {memory['ivf_total_gb']:.2f} GB")

# Understanding similarity metrics
vec_a = np.random.randn(384)
vec_b = vec_a + np.random.randn(384) * 0.1  # Similar vector
vec_c = np.random.randn(384)  # Random vector

print("\nSimilarity between related vectors:")
print(analysis.similarity_metrics(vec_a, vec_b))
print("\nSimilarity between unrelated vectors:")
print(analysis.similarity_metrics(vec_a, vec_c))
```

**Practical Implications**:
- **384 dimensions** (e.g., MiniLM models): 1.5GB per million vectors, faster queries, sufficient for most use cases
- **768 dimensions** (e.g., BERT-base): 3GB per million vectors, better semantic understanding
- **1536 dimensions** (e.g., OpenAI ada-002): 6GB per million vectors, highest quality but expensive

**Trade-offs**: Higher dimensions provide richer semantic representations but require more memory and slower query times. In practice, 384-768 dimensions offer the best performance/quality balance.

### 2. Indexing Algorithms (ANN)

Approximate Nearest Neighbor algorithms make vector search tractable at scale by organizing vectors for efficient retrieval.

```python
import numpy as np
from typing import List, Tuple, Set
from collections import defaultdict
import time

class HNSWIndex:
    """Simplified HNSW (Hierarchical Navigable Small World) implementation"""
    
    def __init__(self, dimensions: int, max_connections: int = 16, ef_construction: int = 200):
        self.dimensions = dimensions
        self.max_connections = max_connections
        self.ef_construction = ef_construction
        self.vectors = []
        self.graph = defaultdict(set)  # Adjacency list
        
    def add(self, vector: np.ndarray, vector_id: int):
        """Add vector to index"""
        self.vectors.append((vector_id, vector))
        
        if len(self.vectors) == 1:
            return
        
        # Find nearest neighbors during construction
        candidates = self._search_layer(vector, self.ef_construction)
        
        # Connect to M nearest neighbors
        neighbors = sorted(candidates, key=lambda x: x[1])[:self.max_connections]
        for neighbor_id, _ in neighbors:
            self.graph[vector_id].add(neighbor_id)
            self.graph[neighbor_id].add(vector_id)
            
            # Prune connections if needed
            if len(self.graph[neighbor_id]) > self.max_connections:
                self._prune_connections(neighbor_id)
    
    def _search_layer(self, query: np.ndarray, ef: int) -> List[Tuple[int, float]]:
        """Search within a layer"""
        visited = set()
        candidates = []
        
        # Start from random entry point
        if self.vectors:
            entry_id, entry_vec = self.vectors[0]
            dist = self._distance(query, entry_vec)
            candidates.append((entry_id, dist))
            visited.add(entry_id)
        
        # Greedy search
        while candidates:
            current_id, current_dist = min(candidates, key=lambda x: x[1])
            candidates.remove((current_id, current_dist))
            
            # Explore neighbors
            for neighbor_id in self.graph[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_vec = next(v for vid, v in self.vectors if vid == neighbor_id)
                    dist = self._distance(query, neighbor_vec)
                    candidates.append((neighbor_id, dist))
            
            if len(visited) >= ef:
                break
        
        return [(vid, d) for vid, d in candidates][:ef]
    
    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine distance (1 - cosine similarity)"""
        return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _prune_connections(self, node_id: int):
        """Keep only the best connections"""
        node_vec = next(v for vid, v in self.vectors if vid == node_id)
        neighbors = list(self.graph[node_id])
        
        # Sort by distance and keep closest
        neighbor_distances = [
            (nid, self._distance(node_vec, next(v for vid, v in self.vectors if vid == nid)))
            for nid in neighbors
        ]
        neighbor_distances.sort(key=lambda x: x[1])
        
        self.graph[node_id] = set(nid for nid, _ in neighbor_distances[:self.max_connections])
    
    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for nearest neighbors"""
        candidates = self._search_layer(query, max(self.ef_construction, top_k))
        return sorted(candidates, key=lambda x: x[1])[:top_k]


class IVFIndex:
    """Inverted File Index - partitions space into clusters"""
    
    def __init__(self, dimensions: int, num_clusters: int = 100):
        self.dimensions = dimensions
        self.num_clusters = num_clusters
        self.centroids = None
        self.clusters = defaultdict(list)  # cluster_id -> [(vector_id, vector)]
        
    def build(self, vectors: List[Tuple[int, np.ndarray]]):
        """Build index using k-means clustering"""
        if len(vectors) < self.num_clusters:
            self.num_clusters = max(1, len(vectors) // 10)
        
        # Simple k-means for centroid initialization
        all_vectors = np.vstack([v for _, v in vectors])
        indices = np.random.choice(len(all_vectors), self.num_clusters, replace=False)
        self.centroids = all_vectors[indices]
        
        # Assign vectors to clusters
        for vector_id, vector in vectors:
            cluster_id = self._nearest_centroid(vector)
            self.clusters[cluster_id].append((vector_id, vector))
    
    def _nearest_centroid(self, vector: np.ndarray) -> int:
        """Find closest centroid"""
        distances = np.linalg.norm(self.centroids - vector, axis=1)
        return int(np.argmin(distances))
    
    def search(self, query: np.ndarray, top_k: int = 5, num_probes: int = 3) -> List[Tuple[int, float]]:
        """Search nearest neighbors in closest clusters"""
        # Find nearest centroids
        centroid_distances = np.linalg.norm(self.centroids - query, axis=1)
        probe_clusters = np.argsort(centroid_distances)[:num_probes]
        
        # Search within selected clusters
        candidates = []
        for cluster_id in probe_clusters:
            for vector_id, vector in self.clusters[cluster_id]:
                dist = np.linalg.norm(query - vector)
                candidates.append((vector_id, dist))
        
        # Return top-k
        candidates.sort(key=lambda x: x[1])
        return candidates[:top_k]


# Performance comparison
def benchmark_indexes(num_vectors: int = 10000, dimensions: int = 384):
    """Compare index performance"""
    print(f"Benchmarking with {num_vectors} vectors of {dimensions} dimensions\n")
    
    # Generate test data
    vectors = [(i, np.random.randn(dimensions)) for i in range(num_vectors)]
    query = np.random.randn(dimensions)
    
    # Brute force (baseline)
    start = time.time()
    distances = [(i, np.linalg.norm(query - v)) for i, v in vectors]
    distances.sort(key=lambda x: x[1])
    exact_results = distances[:5]
    brute_time = time.time() - start
    
    # HNSW Index
    start = time.time()
    hnsw = HNSWIndex(dimensions)
    for vid, vec in vectors:
        hnsw.add(vec, vid)
    build_time = time.time() - start
    
    start = time.time()
    hnsw_results = hnsw.search(query, top_k=5)
    query_time = time.time() - start
    
    print(f"Brute Force:")
    print(f"  Query time: {brute_time*1000:.2f}ms")
    print(f"\nHNSW Index:")
    print(f"  Build time: {build_time:.2f}s")
    print(f"  Query time: {query_time*1000:.2f}ms")
    print(f"  Speedup: {brute_time/query_time:.1f}x")
    
    # IVF Index
    start = time.time()
    ivf = IVFIndex(