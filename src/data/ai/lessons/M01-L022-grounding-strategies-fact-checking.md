# Grounding Strategies & Fact-Checking

## Core Concepts

Language models are sophisticated pattern matchers trained on massive text corpora. They excel at generating fluent, contextually appropriate text but have a critical flaw: they don't inherently distinguish between factual information and plausible-sounding fabrications. They generate tokens based on statistical patterns, not truth.

**Grounding** is the practice of anchoring model outputs to verifiable, authoritative sources. Instead of allowing the model to rely solely on its training data (which is static, potentially outdated, and sometimes incorrect), you provide it with specific, current information that constrains its responses.

### Engineering Analogy: Cache vs. Database

Traditional approach (ungrounded generation):

```python
def answer_question_ungrounded(question: str, model) -> str:
    """Model relies entirely on training data (like a cache)"""
    prompt = f"Answer this question: {question}"
    return model.generate(prompt)

# Problem: "cache" is stale, incomplete, and sometimes wrong
result = answer_question_ungrounded(
    "What is the current interest rate?",
    model
)
# May return outdated or hallucinated information
```

Modern approach (grounded generation):

```python
from typing import List, Dict
import datetime

def answer_question_grounded(
    question: str,
    model,
    knowledge_retriever
) -> Dict[str, str]:
    """Model uses retrieved facts (like querying a database)"""
    # Retrieve current, verified information
    relevant_docs = knowledge_retriever.search(question, top_k=3)
    
    # Build context from verified sources
    context = "\n\n".join([
        f"Source {i+1}: {doc['content']}"
        for i, doc in enumerate(relevant_docs)
    ])
    
    prompt = f"""Using ONLY the information provided below, answer the question.
If the answer is not in the provided information, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    answer = model.generate(prompt)
    
    return {
        "answer": answer,
        "sources": [doc['url'] for doc in relevant_docs],
        "retrieved_at": datetime.datetime.utcnow().isoformat()
    }
```

The grounded approach treats the model as a reasoning engine over verified data rather than as the source of truth itself.

### Key Insights

**1. Models don't "know" facts—they predict tokens.** When a model outputs "Paris is the capital of France," it's not retrieving a stored fact; it's predicting that this token sequence has high probability given the input. This distinction is critical when facts matter.

**2. Grounding transforms generation from creative writing to constrained synthesis.** You shift the model's role from "answer from memory" to "reason over provided evidence."

**3. The retrieval quality bottleneck is real.** A perfectly prompted model with wrong source documents produces confidently wrong answers. Grounding is only as good as your retrieval pipeline.

**4. Verification must be architectural, not instructional.** You cannot reliably prompt a model to "be accurate." Accuracy comes from system design: retrieval quality, source authority, output validation, and explicit citation requirements.

### Why This Matters Now

Production AI applications increasingly handle high-stakes queries where hallucinations are unacceptable: customer support, medical information, legal research, financial advice. The gap between "sounds right" and "is right" is a liability risk. Grounding strategies provide:

- **Auditability**: Citations enable verification
- **Freshness**: Retrieval provides current data beyond training cutoff
- **Controllability**: You define what sources are authoritative
- **Risk mitigation**: Verifiable claims reduce hallucination impact

## Technical Components

### 1. Retrieval-Augmented Generation (RAG)

RAG separates knowledge storage from reasoning. Instead of encoding all facts in model weights, you maintain an external knowledge base and retrieve relevant passages at inference time.

**Technical Implementation:**

```python
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    content: str
    metadata: dict
    embedding: np.ndarray

class RAGPipeline:
    def __init__(self, embedding_model, generator_model):
        self.embedding_model = embedding_model
        self.generator_model = generator_model
        self.knowledge_base: List[Document] = []
    
    def index_documents(self, documents: List[Dict]) -> None:
        """Convert documents to searchable embeddings"""
        for doc in documents:
            embedding = self.embedding_model.encode(doc['content'])
            self.knowledge_base.append(Document(
                id=doc['id'],
                content=doc['content'],
                metadata=doc.get('metadata', {}),
                embedding=embedding
            ))
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """Semantic search over knowledge base"""
        query_embedding = self.embedding_model.encode(query)
        
        # Compute similarity scores
        scores = []
        for doc in self.knowledge_base:
            similarity = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
            )
            scores.append((similarity, doc))
        
        # Return top-k most relevant
        scores.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scores[:top_k]]
    
    def generate(self, query: str, top_k: int = 3) -> Dict:
        """RAG: retrieve then generate"""
        # Retrieval phase
        relevant_docs = self.retrieve(query, top_k)
        
        # Generation phase with grounding
        context = "\n\n".join([
            f"[{i+1}] {doc.content}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        prompt = f"""Answer the question using the provided context.
Cite sources using [number] notation.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.generator_model.generate(prompt)
        
        return {
            "answer": answer,
            "sources": [
                {
                    "id": doc.id,
                    "excerpt": doc.content[:200],
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ]
        }
```

**Practical Implications:**

- Retrieval latency adds 50-200ms to response time
- Embedding model quality directly impacts relevance
- Top-k parameter trades coverage for noise (typical: 3-5)
- Context window limits how many documents you can include

**Constraints & Trade-offs:**

- **Chunk size dilemma**: Small chunks (100-300 tokens) provide precise retrieval but fragment context. Large chunks (500-1000 tokens) preserve context but reduce precision.
- **Embedding model selection**: Faster models (e.g., 384-dimensional) retrieve quickly but miss semantic nuances. Larger models (768+ dimensional) capture meaning better but increase latency and storage.

### 2. Citation and Attribution

Citations enable verification and build trust. Effective attribution goes beyond appending URLs—it requires structured linking between claims and sources.

**Technical Implementation:**

```python
from typing import List, Dict, Optional
import re

class CitationTracker:
    def __init__(self):
        self.sources: Dict[str, Dict] = {}
        self.citation_pattern = re.compile(r'\[(\d+)\]')
    
    def add_source(self, source_id: str, content: str, url: str) -> str:
        """Add source and return citation marker"""
        citation_key = str(len(self.sources) + 1)
        self.sources[citation_key] = {
            "content": content,
            "url": url,
            "source_id": source_id
        }
        return f"[{citation_key}]"
    
    def build_grounded_prompt(
        self,
        query: str,
        documents: List[Dict]
    ) -> str:
        """Create prompt with numbered sources"""
        context_parts = []
        for i, doc in enumerate(documents):
            citation_key = self.add_source(
                source_id=doc['id'],
                content=doc['content'],
                url=doc.get('url', 'N/A')
            )
            context_parts.append(f"{citation_key} {doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        return f"""Answer using ONLY the provided sources.
Cite sources inline using [number] format.
If information isn't in the sources, state this explicitly.

Sources:
{context}

Question: {query}

Answer (with inline citations):"""
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citation markers from generated text"""
        return self.citation_pattern.findall(text)
    
    def validate_citations(self, text: str) -> Dict:
        """Check if all citations are valid"""
        cited = set(self.extract_citations(text))
        available = set(self.sources.keys())
        
        return {
            "valid_citations": list(cited & available),
            "invalid_citations": list(cited - available),
            "unused_sources": list(available - cited),
            "citation_count": len(cited)
        }
    
    def format_bibliography(self) -> str:
        """Generate formatted source list"""
        return "\n".join([
            f"[{key}] {source['url']}"
            for key, source in sorted(self.sources.items())
        ])

# Usage example
tracker = CitationTracker()

documents = [
    {"id": "1", "content": "Earth's atmosphere is 78% nitrogen.", "url": "nasa.gov/atm"},
    {"id": "2", "content": "Oxygen comprises 21% of the atmosphere.", "url": "noaa.gov/air"},
]

prompt = tracker.build_grounded_prompt(
    "What is the composition of Earth's atmosphere?",
    documents
)

# Simulated model output
response = "Earth's atmosphere consists primarily of nitrogen at 78% [1] and oxygen at 21% [2]."

validation = tracker.validate_citations(response)
print(validation)
# {"valid_citations": ["1", "2"], "invalid_citations": [], ...}

bibliography = tracker.format_bibliography()
```

**Practical Implications:**

- Explicit citation requirements reduce hallucination by 40-60%
- Structured citation enables automated fact verification
- Users can validate claims without re-querying

### 3. Temporal Awareness and Freshness

Language models have a training data cutoff. Questions about current events, pricing, or status require real-time retrieval.

**Technical Implementation:**

```python
from datetime import datetime, timedelta
from typing import Optional
import hashlib

class FreshnessEnforcer:
    def __init__(self, max_age_hours: int = 24):
        self.max_age = timedelta(hours=max_age_hours)
        self.cache: Dict[str, Dict] = {}
    
    def cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def is_fresh(self, timestamp: datetime) -> bool:
        """Check if data is within freshness window"""
        age = datetime.utcnow() - timestamp
        return age < self.max_age
    
    def retrieve_with_freshness(
        self,
        query: str,
        retriever,
        force_refresh: bool = False
    ) -> Dict:
        """Retrieve data with freshness guarantees"""
        key = self.cache_key(query)
        
        # Check cache
        if not force_refresh and key in self.cache:
            cached = self.cache[key]
            if self.is_fresh(cached['timestamp']):
                cached['cache_hit'] = True
                return cached
        
        # Fetch fresh data
        documents = retriever.search(query)
        
        result = {
            "documents": documents,
            "timestamp": datetime.utcnow(),
            "cache_hit": False
        }
        
        self.cache[key] = result
        return result
    
    def build_temporal_context(
        self,
        query: str,
        retrieval_result: Dict
    ) -> str:
        """Add temporal metadata to context"""
        timestamp = retrieval_result['timestamp']
        age_minutes = (datetime.utcnow() - timestamp).seconds // 60
        
        temporal_note = f"""Data retrieved at: {timestamp.isoformat()}
Data age: {age_minutes} minutes
Training data cutoff: [model's cutoff date]

Note: For time-sensitive information, this answer is current as of {timestamp.strftime('%Y-%m-%d %H:%M UTC')}.
"""
        
        return temporal_note

# Example: Time-sensitive query handling
enforcer = FreshnessEnforcer(max_age_hours=1)

def answer_with_freshness(query: str, model, retriever) -> Dict:
    """Answer with explicit freshness guarantees"""
    # Ensure fresh data
    retrieval = enforcer.retrieve_with_freshness(query, retriever)
    
    # Build temporally-aware context
    temporal_context = enforcer.build_temporal_context(query, retrieval)
    
    context = "\n\n".join([
        doc['content'] for doc in retrieval['documents']
    ])
    
    prompt = f"""{temporal_context}

Context:
{context}

Question: {query}

Answer (note if information may be time-sensitive):"""
    
    answer = model.generate(prompt)
    
    return {
        "answer": answer,
        "data_timestamp": retrieval['timestamp'].isoformat(),
        "cache_hit": retrieval['cache_hit'],
        "freshness_guaranteed": True
    }
```

**Practical Implications:**

- Freshness requirements increase retrieval costs
- Cache invalidation strategies balance cost vs. accuracy
- Temporal metadata helps users assess answer reliability

### 4. Confidence Scoring and Uncertainty

Models should communicate when they're uncertain. Confidence scores help systems decide when to escalate to human review.

**Technical Implementation:**

```python
from typing import List, Tuple
import numpy as np

class ConfidenceScorer:
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
    
    def score_retrieval_confidence(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: List[np.ndarray],
        top_k_scores: List[float]
    ) -> Dict:
        """Score confidence based on retrieval quality"""
        # Metric 1: Top result similarity
        top_score = top_k_scores[0] if top_k_scores else 0.0
        
        # Metric 2: Score distribution (is there a clear best match?)
        score_std = np.std(top_k_scores) if len(top_k_scores) > 1 else 0.0
        
        # Metric 3: Gap between top and second result
        score_gap = (
            top_k_scores[0] - top_k_scores[1]
            if len(top_k_scores) > 1
            else 0.0
        )
        
        confidence = {
            "retrieval_quality": top_score,
            "result_clarity": score_gap,
            "score_variance": score_std,
            "overall_confidence": (
                top_score * 0.6 +
                score_gap * 0.3 +
                (1 - score_std) * 0.1
            )
        }
        
        return confidence
    
    def detect_uncertainty_markers(self, text: str) -> Dict:
        """Detect linguistic uncertainty in generated text"""
        uncertainty_phrases = [
            "may", "might", "possibly", "perhaps", "unclear",
            "not certain", "not sure", "appears to", "seems to",
            "could be", "it is possible"
        ]
        
        text_lower = text.lower()
        found_markers = [
            phrase for phrase in uncertainty_phrases
            if phrase in text_lower
        ]
        
        return {
            "uncertainty_markers": found_markers,
            "marker_count": len(found_markers),
            "contains_uncertainty": len(found_markers) >