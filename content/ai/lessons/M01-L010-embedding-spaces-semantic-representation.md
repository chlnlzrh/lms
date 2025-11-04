# Embedding Spaces & Semantic Representation

## Core Concepts

Embedding spaces transform discrete data—words, sentences, images, code—into continuous vector representations where semantic similarity corresponds to geometric proximity. Unlike symbolic representations that treat "dog" and "puppy" as completely different tokens, embeddings place semantically related concepts near each other in high-dimensional space.

### Traditional vs. Modern Approaches

**Traditional approach:** Exact string matching with manual rules

```python
# Traditional keyword matching
def find_similar_queries_old(query: str, database: list[str]) -> list[str]:
    """Find similar queries using exact string matching."""
    results = []
    query_lower = query.lower()
    keywords = set(query_lower.split())
    
    for item in database:
        item_lower = item.lower()
        # Only matches if exact words appear
        if any(keyword in item_lower for keyword in keywords):
            results.append(item)
    
    return results

# Fails on semantic similarity
queries = [
    "How do I fix a leaky faucet?",
    "Repairing dripping tap",
    "Stop water coming from sink",
]

print(find_similar_queries_old("faucet leak repair", queries))
# Output: ['How do I fix a leaky faucet?']
# Misses semantically identical queries with different words
```

**Modern approach:** Semantic similarity in embedding space

```python
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

def cosine_similarity(a: NDArray, b: NDArray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_similar_queries_embeddings(
    query: str,
    database: List[str],
    embed_fn,
    threshold: float = 0.7
) -> List[Tuple[str, float]]:
    """Find similar queries using semantic embeddings."""
    query_embedding = embed_fn(query)
    results = []
    
    for item in database:
        item_embedding = embed_fn(item)
        similarity = cosine_similarity(query_embedding, item_embedding)
        
        if similarity >= threshold:
            results.append((item, similarity))
    
    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# With embeddings (conceptual - we'll implement real ones below)
# All three queries now match because they're semantically similar
# Output would include all queries with high similarity scores
```

### Key Engineering Insights

**1. Geometry encodes meaning:** The spatial relationship between vectors captures semantic relationships. "King" - "Man" + "Woman" ≈ "Queen" isn't magic—it's learned patterns in high-dimensional space.

**2. Dimensionality is a design choice:** More dimensions capture more nuance but increase compute and storage. 384 dimensions might suffice for basic semantic search; 1536+ dimensions enable finer distinctions.

**3. Distance metrics matter:** Cosine similarity measures angular distance (direction), Euclidean distance measures magnitude. For normalized embeddings, both work; for unnormalized, cosine is typically more robust.

### Why This Matters Now

Embeddings power every modern semantic system: search engines retrieving conceptually relevant documents, recommendation engines finding similar items, RAG systems matching queries to knowledge bases, and clustering algorithms grouping related data. Without understanding embedding spaces, you're building on abstractions you can't debug or optimize.

The cost implications are immediate: a 1536-dimensional float32 embedding costs 6KB per item. For 10M documents, that's 60GB just for vectors—before indexing. Understanding the embedding-quality-cost tradeoff directly impacts your infrastructure budget.

## Technical Components

### 1. Vector Dimensions and Capacity

Embedding dimensions represent independent features the model can encode. More dimensions = more representational capacity, but with diminishing returns.

**Technical explanation:** Each dimension is a learned feature that might capture semantic properties like "formality," "technical depth," "temporal reference," etc. The model learns these features during training, not through explicit programming.

**Practical implications:**

```python
import numpy as np

# Different dimensional embeddings
embedding_small = np.random.randn(128)    # 128 dimensions
embedding_medium = np.random.randn(384)   # 384 dimensions  
embedding_large = np.random.randn(1536)   # 1536 dimensions

# Storage cost comparison (float32)
print(f"128d: {embedding_small.nbytes} bytes")      # 512 bytes
print(f"384d: {embedding_medium.nbytes} bytes")     # 1,536 bytes
print(f"1536d: {embedding_large.nbytes} bytes")     # 6,144 bytes

# For 1 million documents
docs = 1_000_000
print(f"\n1M documents:")
print(f"128d: {(embedding_small.nbytes * docs) / (1024**3):.2f} GB")
print(f"384d: {(embedding_medium.nbytes * docs) / (1024**3):.2f} GB")
print(f"1536d: {(embedding_large.nbytes * docs) / (1024**3):.2f} GB")
```

**Real constraints:** Higher dimensions provide better semantic granularity but increase:
- Storage costs linearly
- Similarity computation costs (O(d) for d dimensions)
- Index build time for vector databases
- Memory bandwidth requirements

**Decision framework:** Use 128-384 dimensions for basic semantic similarity, 768-1024 for nuanced domain-specific tasks, 1536+ when you need maximum fidelity and have the resources.

### 2. Distance Metrics and Similarity

Different metrics measure "closeness" differently, affecting retrieval quality and computational cost.

```python
import numpy as np
from typing import Callable

def cosine_similarity(a: NDArray, b: NDArray) -> float:
    """Measures angle between vectors (0-1 for normalized vectors)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a: NDArray, b: NDArray) -> float:
    """Measures straight-line distance."""
    return np.linalg.norm(a - b)

def dot_product_similarity(a: NDArray, b: NDArray) -> float:
    """Fast similarity for normalized vectors."""
    return np.dot(a, b)

# Example vectors
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([2.0, 4.0, 6.0])  # Same direction, different magnitude
v3 = np.array([1.0, -2.0, 3.0])  # Different direction

print("v1 vs v2 (same direction, different magnitude):")
print(f"  Cosine: {cosine_similarity(v1, v2):.3f}")  # 1.0 (identical)
print(f"  Euclidean: {euclidean_distance(v1, v2):.3f}")  # ~3.74 (different)

print("\nv1 vs v3 (different direction):")
print(f"  Cosine: {cosine_similarity(v1, v3):.3f}")  # ~0.29 (different)
print(f"  Euclidean: {euclidean_distance(v1, v3):.3f}")  # ~4.0 (different)

# Normalized vectors: cosine = dot product
v1_norm = v1 / np.linalg.norm(v1)
v2_norm = v2 / np.linalg.norm(v2)

print("\nNormalized vectors:")
print(f"  Cosine: {cosine_similarity(v1_norm, v2_norm):.6f}")
print(f"  Dot product: {dot_product_similarity(v1_norm, v2_norm):.6f}")
# Same result, but dot product is faster (no division)
```

**Trade-offs:**
- **Cosine similarity:** Direction matters more than magnitude. Ideal for text where "very technical document" shouldn't be penalized vs. "technical document."
- **Euclidean distance:** Magnitude matters. Better when scale is meaningful (e.g., embeddings of measurements).
- **Dot product:** Fastest for normalized vectors, equivalent to cosine. Use this in production with normalized embeddings.

### 3. Embedding Model Selection

Different models optimize for different objectives: speed, quality, domain-specificity, multilingual support.

```python
from typing import Protocol, List
import numpy as np

class EmbeddingModel(Protocol):
    """Interface for embedding models."""
    
    def embed(self, texts: List[str]) -> NDArray:
        """Generate embeddings for input texts."""
        ...
    
    @property
    def dimensions(self) -> int:
        """Number of dimensions in output embeddings."""
        ...

class FastLocalModel:
    """Simulates a fast, small local model (e.g., all-MiniLM-L6-v2)."""
    
    def __init__(self):
        self.dimensions = 384
        self._model_size_mb = 80  # Approximate
    
    def embed(self, texts: List[str]) -> NDArray:
        # Placeholder: real implementation would use sentence-transformers
        return np.random.randn(len(texts), self.dimensions)
    
    @property
    def speed_tokens_per_sec(self) -> int:
        return 10000  # Fast local inference

class LargeQualityModel:
    """Simulates a large, high-quality model (e.g., text-embedding-3-large)."""
    
    def __init__(self):
        self.dimensions = 1536
        self._latency_ms = 200  # API call latency
    
    def embed(self, texts: List[str]) -> NDArray:
        # Placeholder: real implementation would call API
        return np.random.randn(len(texts), self.dimensions)
    
    @property
    def speed_tokens_per_sec(self) -> int:
        return 2000  # API rate limits

# Model selection based on requirements
def choose_model(
    volume_docs: int,
    latency_requirement_ms: int,
    quality_priority: str  # "speed", "balanced", "quality"
) -> EmbeddingModel:
    """Select appropriate model based on requirements."""
    
    if quality_priority == "speed" or volume_docs > 10_000_000:
        return FastLocalModel()
    elif quality_priority == "quality" and latency_requirement_ms > 500:
        return LargeQualityModel()
    else:
        return FastLocalModel()  # Balanced choice

# Example decision
model = choose_model(
    volume_docs=1_000_000,
    latency_requirement_ms=100,
    quality_priority="balanced"
)
print(f"Selected model with {model.dimensions} dimensions")
```

**Real-world pattern:**
- **Local models (384-768d):** 100-1000x cheaper at scale, <10ms latency, sufficient for 80% of use cases
- **API models (1024-3072d):** Better quality, maintained/updated, but $0.0001-0.0004 per 1K tokens adds up
- **Domain-specific models:** Fine-tuned for legal, medical, code—5-15% quality improvement in-domain

### 4. Normalization and Preprocessing

Raw embeddings need normalization for consistent similarity calculations and optimal storage.

```python
import numpy as np
from typing import Optional

def normalize_embedding(embedding: NDArray) -> NDArray:
    """Normalize embedding to unit length (L2 norm = 1)."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def quantize_embedding(
    embedding: NDArray,
    bits: int = 8
) -> NDArray:
    """Quantize embedding to reduce storage (8-bit common)."""
    # Scale to [0, 2^bits - 1]
    min_val, max_val = embedding.min(), embedding.max()
    scaled = (embedding - min_val) / (max_val - min_val)
    quantized = np.round(scaled * (2**bits - 1)).astype(np.uint8)
    return quantized

def dequantize_embedding(
    quantized: NDArray,
    original_min: float,
    original_max: float,
    bits: int = 8
) -> NDArray:
    """Restore quantized embedding (approximate)."""
    scaled = quantized.astype(np.float32) / (2**bits - 1)
    return scaled * (original_max - original_min) + original_min

# Example workflow
original = np.random.randn(384)

# Normalize for consistent similarity
normalized = normalize_embedding(original)
print(f"Original norm: {np.linalg.norm(original):.3f}")
print(f"Normalized norm: {np.linalg.norm(normalized):.3f}")  # Should be 1.0

# Quantize for storage
quantized = quantize_embedding(normalized, bits=8)
print(f"\nStorage reduction:")
print(f"  Float32: {original.nbytes} bytes")
print(f"  Uint8: {quantized.nbytes} bytes")
print(f"  Savings: {(1 - quantized.nbytes/original.nbytes)*100:.1f}%")

# Quality impact (approximate)
dequantized = dequantize_embedding(
    quantized,
    normalized.min(),
    normalized.max()
)
similarity_loss = 1 - cosine_similarity(normalized, dequantized)
print(f"  Similarity loss: {similarity_loss:.4f} (typically <0.01)")
```

**Trade-offs:**
- **Normalization:** Essential for dot-product similarity, enables faster computation, minimal quality impact
- **Quantization:** 75% storage reduction (float32→uint8), <1% quality loss for most tasks, required for some vector databases
- **Dimension reduction:** PCA or matryoshka embeddings can reduce dimensions post-hoc, 20-40% reduction possible with ~2-5% quality loss

### 5. Batch Processing and Caching

Efficient embedding generation requires batching and strategic caching.

```python
import numpy as np
from typing import List, Dict, Tuple
import hashlib

class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""
    
    def __init__(self, model: EmbeddingModel):
        self.model = model
        self._cache: Dict[str, NDArray] = {}
        self._hits = 0
        self._misses = 0
    
    def _hash(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> NDArray:
        """Embed texts with caching and batching."""
        results = []
        to_compute = []
        to_compute_indices = []
        
        # Check cache
        for i, text in enumerate(texts):
            key = self._hash(text)
            if key in self._cache:
                results.append((i, self._cache[key]))
                self._hits += 1
            else:
                to_compute.append(text)
                to_compute_indices.append(i)
                self._misses += 1
        
        # Compute missing embeddings in batches
        if to_compute:
            for i in range(0, len(to_compute), batch_size):
                batch = to_compute[i:i + batch_size]
                embeddings = self.model.embed(batch)
                
                # Cache results
                for text, embedding in zip(batch, embeddings):
                    key = self._hash(text)
                    self._cache[key] = embedding
                    
                # Store with original indices
                for idx, embedding in zip(
                    to_compute_indices[i:i + batch_size],
                    embeddings
                ):
                    results.append((idx, embedding))
        
        # Sort by original order and return
        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results