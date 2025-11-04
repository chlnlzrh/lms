# RAG Pattern Architecture: Building Context-Aware LLM Systems

## Core Concepts

Retrieval-Augmented Generation (RAG) is an architectural pattern that combines information retrieval with language model generation to overcome the fundamental constraint of LLMs: their training data cutoff and limited context window. Instead of relying solely on the model's parametric knowledge (what it learned during training), RAG dynamically retrieves relevant information from external sources and injects it into the generation context.

### Engineering Analogy: Database-Backed Applications

Consider how we evolved from hardcoded application logic to database-backed systems:

**Traditional LLM Approach (Hardcoded Knowledge):**
```python
def answer_question(question: str) -> str:
    """LLM with only parametric knowledge"""
    # Model can only use what it learned during training
    # No access to recent data, private documents, or specialized knowledge
    response = llm.generate(question)
    return response

# Limitations:
# - Knowledge frozen at training time
# - No company-specific information
# - Hallucinates when uncertain
# - Can't cite sources
```

**RAG Approach (Database-Backed):**
```python
from typing import List, Dict
import numpy as np

def answer_question_with_rag(
    question: str,
    knowledge_base: VectorStore,
    llm: LanguageModel
) -> Dict[str, any]:
    """RAG: Retrieve relevant context, then generate"""
    
    # 1. Retrieve relevant documents (like a database query)
    relevant_docs = knowledge_base.similarity_search(
        query=question,
        top_k=5
    )
    
    # 2. Construct context-enriched prompt
    context = "\n\n".join([doc.content for doc in relevant_docs])
    prompt = f"""Based on the following information:

{context}

Question: {question}

Provide a detailed answer citing the specific information above."""
    
    # 3. Generate with grounded context
    response = llm.generate(prompt)
    
    return {
        "answer": response,
        "sources": [doc.metadata for doc in relevant_docs],
        "confidence": calculate_relevance_score(relevant_docs, question)
    }

# Advantages:
# - Fresh, updateable knowledge
# - Private/specialized data
# - Citable sources
# - Reduced hallucination
```

### Key Insights That Change Engineering Thinking

1. **Separation of Knowledge and Reasoning**: RAG decouples the "database" (vector store) from the "application logic" (LLM). You can update knowledge without retraining the model, just as you update databases without redeploying applications.

2. **Retrieval Quality Dominates**: In production RAG systems, 70% of answer quality comes from retrieval precision. The best LLM with poor retrieval performs worse than a mediocre LLM with excellent retrieval.

3. **Context Window as Working Memory**: Think of the LLM's context window as RAM—limited and expensive. RAG is your paging mechanism, fetching only what's needed from "disk" (vector store).

4. **Latency Trade-offs**: RAG adds 100-500ms of retrieval latency. For interactive systems, this matters. Caching and pre-retrieval strategies become critical.

### Why This Matters NOW

LLMs are commoditizing rapidly, but your proprietary data and domain expertise aren't. RAG is the primary pattern for creating defensible AI applications that leverage your unique knowledge. As context windows grow to millions of tokens, RAG evolves from necessity to optimization—you still retrieve to reduce costs and improve precision rather than dumping entire databases into context.

## Technical Components

### 1. Document Processing Pipeline

Raw documents must be transformed into retrievable chunks with semantic meaning preserved.

**Technical Explanation:**

Document chunking is the process of splitting large documents into smaller segments that fit within embedding model limits (typically 512-8192 tokens) while maintaining semantic coherence. Poor chunking destroys retrieval quality.

```python
from typing import List, Iterator
import re

class DocumentChunker:
    """Semantic-aware document chunking"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Hierarchical separators: paragraphs > sentences > words
        self.separators = separators or ["\n\n", "\n", ". ", " "]
    
    def chunk_document(self, text: str, metadata: Dict) -> List[Dict]:
        """Split document into semantic chunks with preserved context"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split by hierarchical separators
        segments = self._split_text(text, self.separators)
        
        for segment in segments:
            segment_length = len(segment)
            
            if current_length + segment_length > self.chunk_size:
                if current_chunk:
                    # Create chunk with metadata
                    chunk_text = "".join(current_chunk)
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_id": len(chunks),
                            "char_count": len(chunk_text)
                        }
                    })
                    
                    # Implement overlap for context continuity
                    overlap_text = chunk_text[-self.chunk_overlap:]
                    current_chunk = [overlap_text, segment]
                    current_length = len(overlap_text) + segment_length
                else:
                    # Segment too large, force split
                    current_chunk = [segment[:self.chunk_size]]
                    current_length = self.chunk_size
            else:
                current_chunk.append(segment)
                current_length += segment_length
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "content": "".join(current_chunk),
                "metadata": {**metadata, "chunk_id": len(chunks)}
            })
        
        return chunks
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split by separator hierarchy"""
        if not separators:
            return [text]
        
        separator = separators[0]
        segments = text.split(separator)
        
        if len(segments) == 1:
            # Try next separator in hierarchy
            return self._split_text(text, separators[1:])
        
        # Re-add separator to maintain original formatting
        return [seg + separator for seg in segments[:-1]] + [segments[-1]]


# Example usage
chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)

document = """
Machine learning models require careful hyperparameter tuning.

The learning rate is the most critical hyperparameter. A learning rate 
that's too high causes divergence, while too low leads to slow convergence.

Common values range from 0.001 to 0.1 depending on the optimizer used.
Adam optimizer typically works well with higher learning rates than SGD.
"""

chunks = chunker.chunk_document(
    document,
    metadata={"source": "ml_guide.txt", "section": "hyperparameters"}
)

for chunk in chunks:
    print(f"Chunk {chunk['metadata']['chunk_id']}: {len(chunk['content'])} chars")
    print(chunk['content'][:100] + "...\n")
```

**Practical Implications:**
- Chunk size affects retrieval precision (smaller) vs. context completeness (larger)
- Overlap prevents information loss at chunk boundaries
- Metadata inheritance enables filtering and source tracking

**Real Constraints:**
- Embedding models have token limits (512-8192)
- Smaller chunks = more embeddings = higher storage costs
- Too small chunks lose context; too large chunks reduce retrieval precision

### 2. Vector Embeddings & Similarity Search

Embeddings transform text into high-dimensional vectors where semantic similarity correlates with geometric proximity.

**Technical Explanation:**

Vector embeddings are dense numerical representations (typically 384-1536 dimensions) where semantically similar text has high cosine similarity. This enables "fuzzy" semantic search vs. exact keyword matching.

```python
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Document:
    content: str
    embedding: np.ndarray
    metadata: Dict

class VectorStore:
    """In-memory vector store with similarity search"""
    
    def __init__(self, embedding_dimension: int = 384):
        self.embedding_dim = embedding_dimension
        self.documents: List[Document] = []
        self.embeddings_matrix: np.ndarray = None
    
    def add_documents(
        self,
        documents: List[Dict],
        embedding_function
    ) -> None:
        """Embed and store documents"""
        for doc in documents:
            # Generate embedding (in production, batch this)
            embedding = embedding_function(doc["content"])
            
            self.documents.append(Document(
                content=doc["content"],
                embedding=embedding,
                metadata=doc.get("metadata", {})
            ))
        
        # Build matrix for efficient similarity computation
        self.embeddings_matrix = np.vstack([
            doc.embedding for doc in self.documents
        ])
    
    def similarity_search(
        self,
        query: str,
        embedding_function,
        top_k: int = 5,
        filter_metadata: Dict = None
    ) -> List[Tuple[Document, float]]:
        """Find most similar documents using cosine similarity"""
        
        # Embed query
        query_embedding = embedding_function(query)
        
        # Apply metadata filters
        candidate_indices = self._apply_filters(filter_metadata)
        candidate_embeddings = self.embeddings_matrix[candidate_indices]
        
        # Compute cosine similarity: (A·B) / (||A|| ||B||)
        similarities = self._cosine_similarity(
            query_embedding,
            candidate_embeddings
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Map back to original document indices
        original_indices = [candidate_indices[i] for i in top_indices]
        
        results = [
            (self.documents[idx], float(similarities[top_indices[i]]))
            for i, idx in enumerate(original_indices)
        ]
        
        return results
    
    def _cosine_similarity(
        self,
        query_vec: np.ndarray,
        doc_vecs: np.ndarray
    ) -> np.ndarray:
        """Vectorized cosine similarity computation"""
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        
        # Dot product of normalized vectors
        return np.dot(doc_norms, query_norm)
    
    def _apply_filters(self, filter_metadata: Dict) -> np.ndarray:
        """Filter documents by metadata"""
        if not filter_metadata:
            return np.arange(len(self.documents))
        
        indices = []
        for i, doc in enumerate(self.documents):
            if all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                indices.append(i)
        
        return np.array(indices)


# Mock embedding function (in production, use sentence-transformers or OpenAI)
def mock_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Deterministic mock for demonstration"""
    np.random.seed(hash(text) % (2**32))
    return np.random.randn(dim)

# Example usage
store = VectorStore(embedding_dimension=384)

docs = [
    {"content": "Python is a programming language", "metadata": {"type": "definition"}},
    {"content": "Python uses dynamic typing", "metadata": {"type": "feature"}},
    {"content": "JavaScript is a scripting language", "metadata": {"type": "definition"}},
]

store.add_documents(docs, mock_embedding)

results = store.similarity_search(
    "What is Python?",
    mock_embedding,
    top_k=2
)

for doc, score in results:
    print(f"Score: {score:.3f} | {doc.content}")
```

**Practical Implications:**
- Cosine similarity range: [-1, 1], typically threshold at 0.7+ for relevance
- Embedding models encode semantic meaning, enabling cross-lingual and paraphrase matching
- Metadata filtering narrows search space before similarity computation

**Real Constraints:**
- Embedding dimension trades quality (higher) vs. storage/compute (lower)
- Similarity search is O(n) for brute force; use approximate nearest neighbors (ANN) for scale
- Embedding models have domain biases; general models underperform on specialized domains

### 3. Prompt Construction & Context Injection

Retrieved documents must be formatted into prompts that maximize LLM comprehension and attribution.

**Technical Explanation:**

Context injection is the process of structuring retrieved information in the prompt to guide the LLM's attention and enable source attribution without overwhelming the context window.

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    content: str
    score: float
    metadata: Dict

class RAGPromptBuilder:
    """Construct optimized prompts for RAG"""
    
    def __init__(
        self,
        max_context_tokens: int = 2000,
        tokens_per_char: float = 0.25  # Rough estimate
    ):
        self.max_context_tokens = max_context_tokens
        self.max_context_chars = int(max_context_tokens / tokens_per_char)
    
    def build_prompt(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        system_instructions: str = None
    ) -> Dict[str, str]:
        """Build context-aware prompt with token budget"""
        
        # Filter and rank by relevance
        relevant_docs = [
            doc for doc in retrieved_docs
            if doc.score >= 0.7  # Relevance threshold
        ]
        
        if not relevant_docs:
            return self._build_fallback_prompt(query)
        
        # Allocate context budget
        context_parts = []
        remaining_chars = self.max_context_chars
        
        for i, doc in enumerate(relevant_docs):
            source_id = f"[{i+1}]"
            
            # Add citation marker
            doc_text = f"{source_id} {doc.content}"
            
            if len(doc_text) <= remaining_chars:
                context_parts.append(doc_text)
                remaining_chars -= len(doc_text)
            else:
                # Truncate last document to fit budget
                truncated = doc_text[:remaining_chars-3] + "..."
                context_parts.append(truncated)
                break
        
        # Construct structured prompt
        context_block = "\n\n".join(context_parts)
        
        system_prompt = system_instructions or """You are a helpful assistant that answers questions based solely on the provided context. 
Always cite sources using the [N] format. If the context doesn't contain enough information, say so."""
        
        user_prompt = f"""Context:
{context_block}

Question: {query}

Provide a detailed answer based on the context above, citing sources with [N] notation."""
        
        return {
            "system": system_prompt,
            "user": user_prompt,
            "metadata": {
                "sources": [doc.metadata for doc in relevant_docs[:len(context_parts)]],
                "context_chars": self.max_context_chars - remaining_chars
            }
        }
    
    def _build_fallback_prompt(self, query: str) -> Dict[str, str]:
        """Fallback when no relevant docs found"""
        return {
            "system": "You are a helpful assistant. Be honest when you don't have information.",
            "user": query,
            "metadata": {"sources": [], "context_chars": 0}
        }


# Example usage
builder = RAGPromptBuilder(max_context_tokens=1000)

retrieved = [
    RetrievalResult(