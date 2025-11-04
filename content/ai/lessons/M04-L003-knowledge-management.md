# Knowledge Management for LLM Applications

## Core Concepts

Knowledge management in LLM systems is the engineering discipline of providing external information to language models at inference time. Unlike training data, which is baked into model weights, managed knowledge is dynamic, verifiable, and updateable without model retraining.

### Traditional vs. Modern Approaches

```python
# Traditional approach: Hardcoded domain knowledge
class CustomerSupportBot:
    def __init__(self):
        # Knowledge embedded in code logic
        self.policies = {
            "return_window": 30,
            "shipping_cost": 5.99,
            "premium_threshold": 100
        }
    
    def answer_question(self, question: str) -> str:
        # Rule-based matching
        if "return" in question.lower():
            return f"Return window is {self.policies['return_window']} days"
        elif "shipping" in question.lower():
            return f"Shipping costs ${self.policies['shipping_cost']}"
        return "I don't understand your question"

# Modern LLM approach: Dynamic knowledge retrieval
from typing import List, Dict
import json

class LLMKnowledgeBot:
    def __init__(self, llm_client, knowledge_base: List[Dict[str, str]]):
        self.llm = llm_client
        self.knowledge = knowledge_base
    
    def retrieve_relevant_context(self, question: str, top_k: int = 3) -> str:
        """Retrieve relevant knowledge chunks"""
        # Simplified retrieval - real implementation uses embeddings
        relevant = [doc for doc in self.knowledge 
                   if any(term in doc['content'].lower() 
                   for term in question.lower().split())][:top_k]
        return "\n\n".join([doc['content'] for doc in relevant])
    
    def answer_question(self, question: str) -> str:
        context = self.retrieve_relevant_context(question)
        prompt = f"""Use the following information to answer the question.

Context:
{context}

Question: {question}

Answer based only on the provided context:"""
        
        return self.llm.complete(prompt)

# Knowledge stored separately from code
knowledge_base = [
    {"id": "policy_001", "content": "Return policy: Customers can return items within 30 days of purchase with original receipt. Refund processed within 5-7 business days."},
    {"id": "policy_002", "content": "Shipping: Standard shipping is $5.99. Free shipping on orders over $100. Express shipping available for $15.99."},
    {"id": "policy_003", "content": "Premium membership: $49/year. Benefits include free shipping, 60-day returns, and 24/7 priority support."}
]
```

### Key Engineering Insights

**1. Separation of Knowledge and Intelligence**: LLMs provide reasoning capability; knowledge management provides facts. This separation enables updating information without model retraining, costs measured in seconds rather than GPU-days.

**2. Context Window as Working Memory**: The model can only reason over information in its context window (typically 4K-128K tokens). Knowledge management is fundamentally about selecting which information deserves this limited space.

**3. Retrieval Quality Dominates Response Quality**: An LLM given wrong information will produce confident, wrong answers. The retrieval system's precision directly determines output reliability—garbage in, garbage out applies with unprecedented confidence.

### Why This Matters Now

Modern LLMs have sufficient reasoning capability that the bottleneck has shifted from "can it understand?" to "does it have the right information?". Organizations with years of documentation, customer data, and domain expertise can leverage these assets only through effective knowledge management. Without it, you're building on quicksand—capable models producing unreliable outputs.

## Technical Components

### 1. Knowledge Representation and Chunking

Knowledge exists as unstructured text (documents, wikis, tickets) but must be transformed into structured, retrievable units.

**Technical Explanation**: Chunking divides documents into semantic units small enough for embedding and retrieval but large enough to maintain context. Each chunk becomes an atomic unit of retrieval.

```python
from typing import List, Dict
import re

class DocumentChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Args:
            chunk_size: Target characters per chunk
            overlap: Characters to overlap between chunks for context continuity
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str, metadata: Dict) -> List[Dict]:
        """Split on sentence boundaries to preserve semantic units"""
        # Split into sentences while preserving the delimiter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'metadata': metadata,
                    'length': len(chunk_text)
                })
                
                # Keep last sentences for overlap
                overlap_text = ' '.join(current_chunk)
                if len(overlap_text) > self.overlap:
                    # Start new chunk with overlap
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'content': ' '.join(current_chunk),
                'metadata': metadata,
                'length': current_length
            })
        
        return chunks

# Example usage
chunker = DocumentChunker(chunk_size=200, overlap=50)
document = """
Machine learning models require careful monitoring in production. 
Key metrics include latency, throughput, and accuracy drift. 
Latency should be measured at p50, p95, and p99 percentiles.
Throughput indicates system capacity under load.
Accuracy drift occurs when model performance degrades over time due to data distribution shifts.
"""

chunks = chunker.chunk_by_sentences(
    document, 
    metadata={'source': 'ml_ops_guide.md', 'section': 'monitoring'}
)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({chunk['length']} chars):\n{chunk['content']}\n")
```

**Practical Implications**: 
- Too small: Fragments lose context, retrieval becomes noisy
- Too large: Waste context window space, reduce retrieval precision
- Optimal size depends on your domain: legal contracts need larger chunks (1000+ chars), FAQs work with smaller (200-400 chars)

**Real Constraints**: Overlapping chunks consume more storage (1.2-1.5x) and increase retrieval complexity but significantly improve context continuity for queries spanning chunk boundaries.

### 2. Semantic Search via Embeddings

Traditional keyword search fails for semantic queries. Embeddings convert text to vectors where semantic similarity equals geometric proximity.

**Technical Explanation**: Embedding models encode text into high-dimensional vectors (768-1536 dimensions). Cosine similarity between query and document vectors measures semantic relevance.

```python
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    content: str
    embedding: np.ndarray
    metadata: dict

class SemanticSearchEngine:
    def __init__(self, embedding_function):
        """
        Args:
            embedding_function: Function that converts text to embedding vector
        """
        self.embed = embedding_function
        self.documents: List[Document] = []
    
    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """Add document to searchable index"""
        embedding = self.embed(content)
        doc = Document(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        self.documents.append(doc)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product > 0 else 0.0
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for most relevant documents"""
        query_embedding = self.embed(query)
        
        # Calculate similarity scores
        results = []
        for doc in self.documents:
            score = self.cosine_similarity(query_embedding, doc.embedding)
            results.append((doc, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# Mock embedding function for demonstration
# In production, use actual embedding models (OpenAI, Sentence-Transformers, etc.)
def mock_embed(text: str) -> np.ndarray:
    """Simplified embedding - real implementations use neural networks"""
    # Create deterministic vector based on text characteristics
    words = text.lower().split()
    vector = np.zeros(384)
    for i, word in enumerate(words[:50]):
        vector[i % 384] += hash(word) % 100
    return vector / np.linalg.norm(vector)

# Example usage
search_engine = SemanticSearchEngine(embedding_function=mock_embed)

# Add documents
docs = [
    ("doc1", "Python is a high-level programming language with dynamic typing", 
     {"category": "programming"}),
    ("doc2", "Machine learning models learn patterns from training data",
     {"category": "ml"}),
    ("doc3", "APIs provide interfaces for software components to communicate",
     {"category": "architecture"}),
    ("doc4", "Neural networks consist of layers of interconnected nodes",
     {"category": "ml"})
]

for doc_id, content, metadata in docs:
    search_engine.add_document(doc_id, content, metadata)

# Search with semantic query
results = search_engine.search("How do AI models learn?", top_k=2)
print("Search results:")
for doc, score in results:
    print(f"Score: {score:.3f} | {doc.content}")
```

**Practical Implications**:
- Embedding models cost ~$0.0001-0.0004 per 1K tokens
- Embeddings are cached; compute once per document chunk
- Vector similarity search scales to millions of documents with vector databases

**Real Constraints**: Different embedding models produce incompatible vectors. Changing models requires re-embedding entire corpus. Choose carefully: general-purpose models (e.g., text-embedding-3-small) work well for most domains, but domain-specific models (legal, medical) can improve precision by 15-30%.

### 3. Retrieval Strategies and Ranking

Not all retrieved documents are equally useful. Multi-stage retrieval and reranking improve precision.

**Technical Explanation**: Initial retrieval casts a wide net (top-20 to top-50), then reranking models score context relevance more accurately, selecting the best 3-5 for the context window.

```python
from typing import List, Tuple, Dict
import numpy as np

class HybridRetriever:
    def __init__(self, semantic_engine, keyword_index, reranker=None):
        self.semantic = semantic_engine
        self.keyword = keyword_index
        self.reranker = reranker
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Simple keyword matching with TF-IDF scoring"""
        query_terms = set(query.lower().split())
        scores = []
        
        for doc in self.semantic.documents:
            doc_terms = set(doc.content.lower().split())
            # Jaccard similarity
            intersection = len(query_terms & doc_terms)
            union = len(query_terms | doc_terms)
            score = intersection / union if union > 0 else 0
            scores.append((doc.id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 5, 
                     semantic_weight: float = 0.7) -> List[Dict]:
        """Combine semantic and keyword search with weighted scoring"""
        # Get results from both methods
        semantic_results = self.semantic.search(query, top_k=20)
        keyword_results = self.keyword_search(query, top_k=20)
        
        # Normalize scores to 0-1 range
        def normalize_scores(results):
            scores = [score for _, score in results]
            max_score = max(scores) if scores else 1
            return [(doc_id, score/max_score) for doc_id, score in results]
        
        # Create score dictionary
        combined_scores = {}
        
        # Add semantic scores
        for doc, score in semantic_results:
            combined_scores[doc.id] = semantic_weight * score
        
        # Add keyword scores
        keyword_norm = dict(normalize_scores(keyword_results))
        for doc_id, score in keyword_norm.items():
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - semantic_weight) * score
            else:
                combined_scores[doc_id] += (1 - semantic_weight) * score
        
        # Sort by combined score
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return full documents
        doc_map = {doc.id: doc for doc in self.semantic.documents}
        results = []
        for doc_id, score in ranked[:top_k]:
            if doc_id in doc_map:
                results.append({
                    'document': doc_map[doc_id],
                    'score': score,
                    'content': doc_map[doc_id].content
                })
        
        return results

# Example usage
retriever = HybridRetriever(search_engine, keyword_index=None)

query = "What programming concepts should beginners learn?"
results = retriever.hybrid_search(query, top_k=3, semantic_weight=0.8)

print(f"Query: {query}\n")
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.3f}")
    print(f"   {result['content']}\n")
```

**Practical Implications**:
- Semantic-only retrieval: Best for conceptual queries ("how does authentication work?")
- Keyword-heavy: Better for exact matches (product codes, error messages)
- Hybrid: Balances both, typically 70-80% semantic weight

**Real Constraints**: Reranking adds 50-200ms latency per query but improves relevance significantly. Use for customer-facing applications where quality matters more than speed.

### 4. Metadata Filtering and Access Control

Knowledge isn't one-size-fits-all. Metadata enables filtering by freshness, authority, user permissions.

```python
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum

class AccessLevel(Enum):
    PUBLIC = 1
    INTERNAL = 2
    CONFIDENTIAL = 3

class MetadataFilter:
    def __init__(self):
        self.filters: List[Callable] = []
    
    def add_date_filter(self, max_age_days: Optional[int] = None):
        """Filter documents by recency"""
        if max_age_days:
            cutoff = datetime.now() - timedelta(days=max_age_days)
            self.filters.append(
                lambda doc: doc.metadata.get('date', datetime.min) >= cutoff
            )
        return self
    
    def add_access_filter(self, user_level: AccessLevel):
        """Filter by access permissions"""
        self.