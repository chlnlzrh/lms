# Context Engineering: Architecting Information for Language Models

## Core Concepts

Context engineering is the practice of structuring, selecting, and delivering information to language models to maximize output quality, reliability, and efficiency. Unlike prompt engineering—which focuses on instruction design—context engineering treats the entire input as an information architecture problem.

### The Engineering Shift

Traditional software engineering relies on explicit state management:

```python
# Traditional API: Explicit state and schema
class PaymentProcessor:
    def __init__(self, config: PaymentConfig):
        self.config = config
        self.db = Database(config.db_url)
    
    def process_refund(self, transaction_id: str, amount: Decimal) -> RefundResult:
        # Strongly typed, explicit parameters
        transaction = self.db.get_transaction(transaction_id)
        if transaction.amount < amount:
            raise ValueError("Refund exceeds transaction amount")
        return self._execute_refund(transaction, amount)
```

With language models, state becomes implicit through context:

```python
# LLM-based approach: Context as implicit state
def process_refund_request(user_request: str, context: Dict) -> str:
    """
    Context must contain: transaction history, business rules,
    user permissions, refund policies, previous conversation
    """
    
    # All "state" lives in the context string
    context_prompt = f"""
You are a payment processing system with access to:

TRANSACTION DATABASE:
{json.dumps(context['recent_transactions'], indent=2)}

REFUND POLICIES:
- Maximum refund: 100% of transaction
- Refund window: 90 days
- Requires manager approval: >$500

USER PERMISSIONS:
Role: {context['user_role']}
Approval limit: ${context['approval_limit']}

PREVIOUS ACTIONS THIS SESSION:
{context['conversation_history']}

USER REQUEST: {user_request}

Process this refund request according to policies. Return structured JSON.
"""
    
    return llm.complete(context_prompt)
```

### Critical Insights

**1. Context is your runtime memory.** Unlike traditional functions where you pass discrete parameters, LLMs reconstruct their understanding from unstructured text. Poor context architecture causes the model to "forget," hallucinate, or misinterpret requirements.

**2. Context windows are expensive and finite.** At $0.01-0.10 per 1K tokens for input (varying by model), a 100K context costs $1-10 per request. Context engineering directly impacts operational costs.

**3. Position matters as much as content.** Models demonstrate recency bias (better recall of recent context) and primacy bias (higher weight to initial instructions). Strategic positioning dramatically affects output quality.

**4. Context is adversarial to latency.** Processing time scales roughly linearly with context size. A 50K token context may add 2-5 seconds of latency compared to 1K tokens.

### Why This Matters Now

Modern models (GPT-4, Claude, Gemini) support 100K-1M+ token contexts, making context engineering the primary bottleneck for:
- **RAG systems** where retrieval quality determines output accuracy
- **Agentic workflows** that accumulate context across multi-step operations
- **Long-document analysis** where entire codebases or reports are processed
- **Stateful conversations** that maintain coherence across hundreds of turns

Poor context engineering manifests as high costs, slow responses, degraded quality, and unpredictable failures—problems that don't appear until production scale.

## Technical Components

### 1. Context Budget Management

Context budgets define hard limits on how much information fits in model input. Exceeding the limit causes truncation or errors; underutilizing wastes opportunity for quality improvement.

**Technical mechanism:** Models use tokenizers that split text into subword units. "Context window" refers to the maximum number of tokens (input + output) the model processes. Common limits:
- GPT-4 Turbo: 128K tokens (~300 pages)
- Claude 3: 200K tokens (~500 pages)
- Gemini 1.5 Pro: 1M tokens (~2,500 pages)

**Practical implications:**

```python
from typing import List, Dict
import tiktoken

class ContextBudgetManager:
    def __init__(self, model: str = "gpt-4", max_tokens: int = 8000):
        self.encoder = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
        self.reserved_output = 1000  # Reserve for model response
        self.reserved_system = 500   # Reserve for system prompt
        self.available = max_tokens - self.reserved_output - self.reserved_system
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def fits_budget(self, contexts: List[str]) -> bool:
        total = sum(self.count_tokens(c) for c in contexts)
        return total <= self.available
    
    def truncate_to_fit(self, contexts: List[Dict[str, str]], 
                        priorities: List[int]) -> List[Dict[str, str]]:
        """
        Truncate contexts by priority until budget is met.
        Priority: higher number = more important (keep longer)
        """
        sorted_contexts = sorted(
            zip(contexts, priorities), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        result = []
        used_tokens = 0
        
        for ctx, priority in sorted_contexts:
            ctx_tokens = self.count_tokens(ctx['content'])
            if used_tokens + ctx_tokens <= self.available:
                result.append(ctx)
                used_tokens += ctx_tokens
            else:
                # Partial include for highest priority item
                if priority >= 9 and not result:
                    remaining = self.available - used_tokens
                    truncated = self._truncate_text(ctx['content'], remaining)
                    result.append({**ctx, 'content': truncated})
                break
        
        return result
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        # Keep beginning and end, remove middle
        keep_tokens = max_tokens - 50  # Buffer for truncation marker
        half = keep_tokens // 2
        truncated_tokens = tokens[:half] + tokens[-half:]
        return self.encoder.decode(truncated_tokens[:half]) + \
               "\n\n[... content truncated ...]\n\n" + \
               self.encoder.decode(truncated_tokens[half:])

# Usage
budget = ContextBudgetManager(model="gpt-4", max_tokens=8000)

contexts = [
    {"role": "system", "content": "You are a code reviewer."},
    {"role": "user", "content": large_codebase},  # 10K tokens
    {"role": "user", "content": recent_changes},   # 2K tokens
    {"role": "user", "content": style_guide},      # 3K tokens
]

priorities = [10, 9, 8, 5]  # System prompt most important

if not budget.fits_budget([c['content'] for c in contexts]):
    contexts = budget.truncate_to_fit(contexts, priorities)
```

**Real constraints:**
- Token counting is model-specific; use correct tokenizer
- Output tokens count against budget; reserve appropriately
- Truncation mid-sentence breaks model understanding
- Cost scales linearly: 10K tokens = 10× the cost of 1K

### 2. Context Retrieval and Ranking

When relevant information exceeds context budget, retrieval systems select the most valuable content. Poor retrieval is the #1 cause of hallucination in production RAG systems.

**Technical mechanism:** Semantic search using embeddings (vector representations of text meaning) to find documents similar to the query. Ranking determines which retrieved documents actually enter context.

```python
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict
    embedding: np.ndarray = None

class SemanticRetriever:
    def __init__(self, embedding_function):
        self.embed = embedding_function
        self.documents: List[Document] = []
    
    def add_documents(self, docs: List[Document]):
        for doc in docs:
            doc.embedding = self.embed(doc.content)
        self.documents.extend(docs)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_embedding = self.embed(query)
        
        # Cosine similarity
        scores = [
            (doc, np.dot(query_embedding, doc.embedding) / 
             (np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)))
            for doc in self.documents
        ]
        
        # Sort by relevance
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:top_k]]

class HybridRanker:
    """Combines multiple ranking signals for better retrieval."""
    
    def __init__(self, semantic_weight: float = 0.6):
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
    
    def rank(self, query: str, documents: List[Document], 
             semantic_scores: List[float]) -> List[Tuple[Document, float]]:
        
        keyword_scores = [
            self._bm25_score(query, doc.content) 
            for doc in documents
        ]
        
        # Normalize scores to [0, 1]
        semantic_norm = self._normalize(semantic_scores)
        keyword_norm = self._normalize(keyword_scores)
        
        # Combine scores
        combined = [
            (doc, 
             self.semantic_weight * sem + self.keyword_weight * kw)
            for doc, sem, kw in zip(documents, semantic_norm, keyword_norm)
        ]
        
        # Re-rank by recency for tied scores
        combined = self._boost_recent(combined)
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined
    
    def _bm25_score(self, query: str, document: str, 
                    k1: float = 1.5, b: float = 0.75) -> float:
        """Simplified BM25 for keyword matching."""
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        doc_length = len(doc_terms)
        avg_length = 100  # Simplified
        
        score = 0.0
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf > 0:
                idf = 1.0  # Simplified (would need corpus stats)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / avg_length)
                score += idf * numerator / denominator
        
        return score
    
    def _normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
    
    def _boost_recent(self, scored_docs: List[Tuple[Document, float]], 
                      boost: float = 0.1) -> List[Tuple[Document, float]]:
        """Boost recent documents within tied score ranges."""
        result = []
        for doc, score in scored_docs:
            if 'timestamp' in doc.metadata:
                age_days = (datetime.now() - doc.metadata['timestamp']).days
                recency_boost = boost * max(0, 1 - age_days / 365)
                score += recency_boost
            result.append((doc, score))
        return result

# Usage in production context builder
class ContextBuilder:
    def __init__(self, retriever: SemanticRetriever, 
                 ranker: HybridRanker,
                 budget_manager: ContextBudgetManager):
        self.retriever = retriever
        self.ranker = ranker
        self.budget = budget_manager
    
    def build_context(self, query: str, max_docs: int = 10) -> str:
        # Retrieve candidates
        candidates = self.retriever.retrieve(query, top_k=max_docs * 2)
        
        # Get semantic scores
        query_emb = self.retriever.embed(query)
        semantic_scores = [
            np.dot(query_emb, doc.embedding) / 
            (np.linalg.norm(query_emb) * np.linalg.norm(doc.embedding))
            for doc in candidates
        ]
        
        # Re-rank with hybrid approach
        ranked = self.ranker.rank(query, candidates, semantic_scores)
        
        # Fill budget
        context_parts = []
        for doc, score in ranked[:max_docs]:
            doc_text = f"[Source: {doc.id}]\n{doc.content}\n"
            if self.budget.fits_budget(context_parts + [doc_text]):
                context_parts.append(doc_text)
            else:
                break
        
        return "\n---\n".join(context_parts)
```

**Practical implications:**
- Embedding models have their own biases; test on your domain
- Hybrid ranking (semantic + keyword + metadata) outperforms pure semantic search by 15-30% in most domains
- Retrieval latency compounds with model latency; cache embeddings aggressively
- Over-retrieval (more results than needed) enables better ranking

### 3. Context Structure and Ordering

The physical arrangement of information within context dramatically affects model interpretation. Models are not equally attentive to all context positions.

**Technical mechanism:** Transformer attention mechanisms create implicit weights across context. Research shows:
- **Primacy effect**: First ~10% of context receives higher attention
- **Recency effect**: Last ~20% of context strongly influences output
- **Middle dilution**: Center portions may receive 40-60% less attention

```python
from enum import Enum
from typing import List, Optional

class ContextSection(Enum):
    SYSTEM_PROMPT = 1    # Position: Start (highest priority)
    TASK_DEFINITION = 2  # Position: Start
    CRITICAL_RULES = 3   # Position: Start + End (repetition)
    REFERENCE_DATA = 4   # Position: Middle (least critical)
    EXAMPLES = 5         # Position: Middle-End
    RECENT_HISTORY = 6   # Position: End
    CURRENT_QUERY = 7    # Position: End (highest recency)

class StructuredContextBuilder:
    """
    Build context with attention-aware positioning.
    """
    
    def __init__(self):
        self.sections = {section: [] for section in ContextSection}
    
    def add_content(self, section: ContextSection, content: str, 
                    priority: int = 5):
        self.sections[section].append({
            'content': content,
            'priority': priority,
            'tokens': len(content.split())  # Simplified
        })
    
    def build(self, max_tokens: int = 8000) -> str:
        """
        Assemble context with strategic positioning.
        
        Structure:
        1. System prompt (fixed position: start)
        2. Task definition (fixed position: start)
        3. Critical rules (start, with reminder at end)
        4. Examples (middle-end for pattern learning)
        5. Reference data (middle, truncate first if needed)
        6. Recent history (end for recency)
        7. Current query (fixed position: end)
        """
        
        output = []
        used_tokens = 0
        
        # Section 1-2: System and task (never truncate)
        for section in [ContextSection.SYSTEM_PROMPT, 
                       ContextSection.TASK_DEFINITION]:
            content = self._format_section(section)
            output.append(content)
            used_tokens += self._count_tokens(