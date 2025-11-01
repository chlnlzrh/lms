# Retrieval Evaluation & Optimization

## Core Concepts

Retrieval evaluation is the systematic measurement and optimization of how well your system finds relevant information from a document corpus. Unlike traditional search systems that optimize for keyword matching and PageRank-style signals, modern retrieval for LLM applications must optimize for semantic relevance, factual grounding, and downstream generation quality.

### Traditional vs. Modern Retrieval Comparison

```python
# Traditional: Keyword-based search with simple ranking
class TraditionalRetrieval:
    def __init__(self, documents: list[str]):
        self.inverted_index = self._build_index(documents)
        self.documents = documents
    
    def _build_index(self, documents: list[str]) -> dict[str, set[int]]:
        index = {}
        for doc_id, doc in enumerate(documents):
            for word in doc.lower().split():
                index.setdefault(word, set()).add(doc_id)
        return index
    
    def search(self, query: str, top_k: int = 3) -> list[str]:
        query_words = set(query.lower().split())
        scores = {}
        for word in query_words:
            for doc_id in self.inverted_index.get(word, set()):
                scores[doc_id] = scores.get(doc_id, 0) + 1
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.documents[doc_id] for doc_id, _ in ranked[:top_k]]


# Modern: Semantic embedding-based retrieval with evaluation metrics
from typing import Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    document: str
    score: float
    doc_id: str
    metadata: Optional[dict] = None

class SemanticRetrieval:
    def __init__(self, documents: list[str], embeddings: np.ndarray):
        self.documents = documents
        self.embeddings = embeddings  # Shape: (num_docs, embedding_dim)
        self.doc_ids = [f"doc_{i}" for i in range(len(documents))]
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 3,
        score_threshold: float = 0.0
    ) -> list[RetrievalResult]:
        # Cosine similarity
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Filter by threshold and get top-k
        valid_indices = np.where(scores >= score_threshold)[0]
        top_indices = valid_indices[np.argsort(scores[valid_indices])[-top_k:][::-1]]
        
        return [
            RetrievalResult(
                document=self.documents[idx],
                score=float(scores[idx]),
                doc_id=self.doc_ids[idx],
                metadata={"rank": rank}
            )
            for rank, idx in enumerate(top_indices)
        ]
```

The fundamental shift: traditional retrieval optimizes for term matching, modern retrieval optimizes for semantic similarity and downstream task performance. This means evaluation must measure not just "did we find similar text" but "did we find information that helps the LLM answer correctly."

### Key Insights

1. **Retrieval quality compounds**: A 10% improvement in retrieval precision typically yields 20-30% better generation accuracy because the LLM has better context to work with.

2. **Offline metrics rarely predict online performance**: High cosine similarity scores don't guarantee the retrieved chunks help the LLM. You must measure end-to-end task success.

3. **The retrieval-generation boundary is porous**: Sometimes poor generation indicates retrieval failure; sometimes good retrieval masks generation issues. Evaluation must isolate failure modes.

4. **Chunking strategy dominates embedding choice**: A 30% improvement from better chunking outperforms a 5% gain from swapping embedding models. Test chunking first.

### Why This Matters Now

Production RAG systems fail silently. Your LLM will confidently hallucinate when retrieval misses relevant context, and without systematic evaluation, you won't know whether to fix chunking, embeddings, ranking, or prompting. Advanced retrieval evaluation gives you the instrumentation to diagnose and optimize each component independently.

## Technical Components

### 1. Evaluation Metrics Hierarchy

Retrieval metrics form a hierarchy from basic similarity to end-to-end task performance.

```python
from typing import Any
import numpy as np
from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    """Comprehensive retrieval evaluation metrics"""
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    mean_average_precision: float
    
    def f1_score(self) -> float:
        if self.precision_at_k + self.recall_at_k == 0:
            return 0.0
        return 2 * (self.precision_at_k * self.recall_at_k) / (
            self.precision_at_k + self.recall_at_k
        )

class RetrievalEvaluator:
    """Production-grade retrieval evaluation"""
    
    def __init__(self, ground_truth: dict[str, set[str]]):
        """
        Args:
            ground_truth: Map of query_id -> set of relevant doc_ids
        """
        self.ground_truth = ground_truth
    
    def precision_at_k(
        self, 
        retrieved: list[str], 
        relevant: set[str], 
        k: int
    ) -> float:
        """Fraction of top-k results that are relevant"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return relevant_retrieved / k if k > 0 else 0.0
    
    def recall_at_k(
        self, 
        retrieved: list[str], 
        relevant: set[str], 
        k: int
    ) -> float:
        """Fraction of relevant docs found in top-k results"""
        if not relevant:
            return 0.0
        retrieved_k = set(retrieved[:k])
        relevant_retrieved = len(retrieved_k & relevant)
        return relevant_retrieved / len(relevant)
    
    def mean_reciprocal_rank(
        self, 
        retrieved: list[str], 
        relevant: set[str]
    ) -> float:
        """Inverse rank of first relevant document"""
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0
    
    def ndcg_at_k(
        self, 
        retrieved: list[str], 
        relevant: set[str], 
        k: int,
        relevance_scores: Optional[dict[str, float]] = None
    ) -> float:
        """Normalized Discounted Cumulative Gain"""
        if relevance_scores is None:
            # Binary relevance: relevant=1, not relevant=0
            relevance_scores = {doc_id: 1.0 for doc_id in relevant}
        
        def dcg(doc_ids: list[str]) -> float:
            return sum(
                relevance_scores.get(doc_id, 0.0) / np.log2(rank + 1)
                for rank, doc_id in enumerate(doc_ids[:k], 1)
            )
        
        # DCG of retrieved documents
        dcg_score = dcg(retrieved)
        
        # Ideal DCG (best possible ordering)
        ideal_ordering = sorted(
            relevant, 
            key=lambda x: relevance_scores.get(x, 0.0), 
            reverse=True
        )
        idcg_score = dcg(ideal_ordering)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0
    
    def mean_average_precision(
        self, 
        retrieved: list[str], 
        relevant: set[str]
    ) -> float:
        """Average of precision values at each relevant document position"""
        if not relevant:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                relevant_count += 1
                precision = relevant_count / rank
                precisions.append(precision)
        
        return sum(precisions) / len(relevant) if precisions else 0.0
    
    def evaluate(
        self, 
        query_id: str, 
        retrieved: list[str], 
        k: int = 10
    ) -> EvaluationMetrics:
        """Compute all metrics for a single query"""
        relevant = self.ground_truth.get(query_id, set())
        
        return EvaluationMetrics(
            precision_at_k=self.precision_at_k(retrieved, relevant, k),
            recall_at_k=self.recall_at_k(retrieved, relevant, k),
            mean_reciprocal_rank=self.mean_reciprocal_rank(retrieved, relevant),
            ndcg_at_k=self.ndcg_at_k(retrieved, relevant, k),
            mean_average_precision=self.mean_average_precision(retrieved, relevant)
        )
    
    def batch_evaluate(
        self, 
        results: dict[str, list[str]], 
        k: int = 10
    ) -> dict[str, float]:
        """Evaluate across multiple queries and aggregate"""
        all_metrics = [
            self.evaluate(query_id, retrieved, k)
            for query_id, retrieved in results.items()
            if query_id in self.ground_truth
        ]
        
        if not all_metrics:
            return {}
        
        return {
            "precision@k": np.mean([m.precision_at_k for m in all_metrics]),
            "recall@k": np.mean([m.recall_at_k for m in all_metrics]),
            "mrr": np.mean([m.mean_reciprocal_rank for m in all_metrics]),
            "ndcg@k": np.mean([m.ndcg_at_k for m in all_metrics]),
            "map": np.mean([m.mean_average_precision for m in all_metrics]),
            "f1@k": np.mean([m.f1_score() for m in all_metrics])
        }
```

**Practical Implications**: Start with precision@k and recall@k for basic sanity checks. Use MRR when you care about finding any relevant document quickly (FAQ systems). Use NDCG when you have graded relevance (some docs more relevant than others). MAP provides a single metric balancing precision across all recall levels.

**Trade-offs**: Computing metrics requires ground truth labels, which are expensive to create. Budget 30-50 high-quality query-document pairs for initial evaluation, expanding to 200+ for production confidence.

### 2. Ground Truth Generation Strategies

Ground truth is your labeled dataset of which documents are relevant for which queries. Quality matters more than quantity.

```python
from enum import Enum
from typing import Callable
import json

class LabelingStrategy(Enum):
    EXPLICIT = "explicit"  # Manual human labeling
    SYNTHETIC = "synthetic"  # LLM-generated queries
    IMPLICIT = "implicit"  # Usage logs
    HYBRID = "hybrid"  # Combination

class GroundTruthBuilder:
    """Build evaluation datasets using different strategies"""
    
    def __init__(self, documents: list[dict[str, str]]):
        """
        Args:
            documents: List of dicts with 'id', 'content', optional 'metadata'
        """
        self.documents = documents
    
    def explicit_labeling(
        self, 
        queries: list[str],
        labeling_function: Callable[[str, dict], bool]
    ) -> dict[str, set[str]]:
        """
        Manual labeling with custom function
        
        Example labeling_function:
            def is_relevant(query: str, doc: dict) -> bool:
                # Custom logic, can use LLM or human review
                return query.lower() in doc['content'].lower()
        """
        ground_truth = {}
        for query in queries:
            relevant_docs = {
                doc['id'] 
                for doc in self.documents 
                if labeling_function(query, doc)
            }
            ground_truth[query] = relevant_docs
        return ground_truth
    
    def synthetic_queries(
        self,
        generate_query_fn: Callable[[dict], list[str]],
        queries_per_doc: int = 3
    ) -> dict[str, set[str]]:
        """
        Generate synthetic queries from documents
        
        Example generate_query_fn using an LLM:
            def generate(doc: dict) -> list[str]:
                prompt = f"Generate {queries_per_doc} questions answered by:\\n{doc['content']}"
                return llm.generate(prompt).split('\\n')
        """
        ground_truth = {}
        for doc in self.documents:
            queries = generate_query_fn(doc)
            for query in queries:
                ground_truth.setdefault(query, set()).add(doc['id'])
        return ground_truth
    
    def implicit_from_logs(
        self,
        query_log: list[dict[str, Any]],
        relevance_threshold: float = 0.5
    ) -> dict[str, set[str]]:
        """
        Extract ground truth from usage logs
        
        Args:
            query_log: List of {query, doc_id, clicked, dwell_time_sec, ...}
            relevance_threshold: Minimum score to consider relevant
        """
        ground_truth = {}
        
        for entry in query_log:
            query = entry['query']
            doc_id = entry['doc_id']
            
            # Compute implicit relevance signal
            relevance_score = 0.0
            if entry.get('clicked', False):
                relevance_score += 0.3
            if entry.get('dwell_time_sec', 0) > 30:
                relevance_score += 0.5
            if entry.get('copied_text', False):
                relevance_score += 0.2
            
            if relevance_score >= relevance_threshold:
                ground_truth.setdefault(query, set()).add(doc_id)
        
        return ground_truth
    
    def hybrid_approach(
        self,
        seed_queries: list[tuple[str, set[str]]],  # (query, relevant_doc_ids)
        augment_fn: Callable[[str], list[str]],  # Generate similar queries
        confidence_filter_fn: Callable[[str, str], float]  # Score relevance
    ) -> dict[str, set[str]]:
        """
        Start with seed labels, augment with synthetic data, filter by confidence
        """
        ground_truth = {query: docs for query, docs in seed_queries}
        
        # Augment each seed query
        for seed_query, seed_docs in seed_queries:
            augmented_queries = augment_fn(seed_query)
            
            for aug_query in augmented_queries:
                # Filter docs by confidence
                confident_docs = {
                    doc_id for doc_id in seed_docs
                    if confidence_filter_fn(aug_query, doc_id) > 0.7
                }
                if confident_docs:
                    ground_truth[aug_query] = confident_docs
        
        return ground_truth
    
    def save(self, ground_truth: dict[str, set[str]], filepath: str):
        """Save ground truth to JSON"""
        serializable = {
            query: list(docs) for query, docs in ground_truth.items()
        }
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def load(self, filepath: str) -> dict[str, set[str]]:
        """Load ground truth from JSON"""
        with open(filepath