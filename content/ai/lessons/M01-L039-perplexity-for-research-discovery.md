# Perplexity for Research & Discovery: Technical Foundations for Engineers

## Core Concepts

### What Is Research-Augmented Generation?

Research-augmented generation combines large language models with real-time web search to produce answers grounded in current information. Unlike pure generative models that rely solely on training data, research-augmented systems query external sources, retrieve relevant documents, and synthesize responses with citations.

Think of it as the difference between these two approaches:

```python
# Traditional LLM approach - static knowledge cutoff
def answer_question_static(question: str, model_knowledge: dict) -> str:
    """
    Answer based only on training data (frozen at training time)
    """
    # Model has no knowledge beyond training cutoff date
    context = model_knowledge.get(question, "training_data_circa_2023")
    response = generate_from_memory(context)
    return response  # May be outdated, no citations

# Research-augmented approach - dynamic knowledge retrieval
def answer_question_augmented(question: str) -> dict:
    """
    Answer by searching current information and synthesizing with citations
    """
    # 1. Generate search queries from the question
    search_queries = expand_question_to_queries(question)
    
    # 2. Retrieve current documents from web
    documents = []
    for query in search_queries:
        results = web_search(query, recency_hours=168)  # Last week
        documents.extend(results)
    
    # 3. Rank and filter relevant content
    relevant_docs = rerank_by_relevance(documents, question)[:10]
    
    # 4. Generate answer grounded in retrieved content
    response = synthesize_with_citations(
        question=question,
        sources=relevant_docs,
        model=llm
    )
    
    return {
        'answer': response,
        'sources': [doc.url for doc in relevant_docs],
        'retrieved_at': datetime.now()
    }
```

### Engineering Mental Model

Traditional search engines return a list of links—you must click, read, evaluate, and synthesize manually. Pure LLMs generate fluent text but can hallucinate facts and lack current information. Research-augmented generation bridges this gap by:

1. **Query expansion**: Converting natural language questions into multiple targeted search queries
2. **Source retrieval**: Fetching and parsing relevant documents in real-time
3. **Contextual synthesis**: Generating coherent answers grounded in retrieved sources
4. **Citation tracking**: Maintaining provenance links between claims and sources

### Why This Matters Now

**Training data staleness** is a fundamental constraint of LLMs. A model trained in early 2024 knows nothing about events from mid-2024 onward. For technical research—checking API deprecations, finding recent vulnerabilities, understanding new frameworks—this lag is unacceptable.

**Hallucination reduction** through grounding. When LLMs generate from memory alone, they confidently produce plausible-sounding falsehoods. Tying generation to retrieved documents dramatically reduces fabrication.

**Verification speed** improves by orders of magnitude. Instead of spending 30 minutes reading five articles to answer a technical question, you get a synthesized answer with citations in 10 seconds, then selectively deep-dive into sources as needed.

The key insight: **Research-augmented generation transforms LLMs from creative text generators into reasoning engines over current information**. This fundamentally changes their utility for technical work.

## Technical Components

### 1. Query Decomposition and Expansion

**Technical Explanation:**  
A natural language question rarely maps to optimal search queries. "How do I implement rate limiting in a distributed system?" is too broad and vague for effective search. Query decomposition breaks complex questions into focused sub-queries that retrieve complementary information.

**Practical Implications:**  
The quality of retrieved documents depends entirely on query quality. Poor queries → irrelevant documents → hallucinated answers citing bad sources. Advanced systems generate 3-7 queries per question, each targeting different aspects.

```python
from typing import List, Dict
import json

def decompose_technical_question(question: str, llm_client) -> List[str]:
    """
    Break complex question into focused search queries
    """
    prompt = f"""Given this technical question, generate 5 specific search queries 
    that would retrieve complementary information to answer it comprehensively.
    
    Question: {question}
    
    Output as JSON array of strings. Make queries specific and searchable.
    Each query should target a different aspect: implementation, best practices, 
    edge cases, performance, security."""
    
    response = llm_client.generate(
        prompt=prompt,
        temperature=0.3,  # Lower temp for focused queries
        max_tokens=300
    )
    
    queries = json.loads(response)
    return queries

# Example usage
question = "How do I implement rate limiting in a distributed system?"
queries = decompose_technical_question(question, llm)

# Output might be:
# [
#   "distributed rate limiting algorithms token bucket",
#   "Redis rate limiting implementation patterns",
#   "rate limiting edge cases distributed systems",
#   "rate limiting performance benchmarks",
#   "distributed rate limiting race conditions"
# ]
```

**Real Constraints:**  
Too many queries increase latency and cost. Too few miss critical information. The trade-off: 3-5 queries balance coverage and speed for most technical questions.

### 2. Source Retrieval and Relevance Ranking

**Technical Explanation:**  
After generating queries, the system retrieves candidate documents from search APIs, then reranks them by relevance to the original question. Initial search ranking optimizes for general popularity and SEO; reranking optimizes for semantic relevance to the specific question.

**Practical Implications:**  
The top 3 documents disproportionately influence the final answer. If irrelevant sources rank highly, the answer quality degrades regardless of LLM quality.

```python
from dataclasses import dataclass
from typing import List
import requests

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    content: str
    published_date: str
    
def retrieve_and_rerank(
    queries: List[str],
    original_question: str,
    top_k: int = 10
) -> List[SearchResult]:
    """
    Retrieve documents and rerank by relevance to original question
    """
    # Step 1: Retrieve candidates from all queries
    candidates = []
    for query in queries:
        results = search_api(query, num_results=20)
        candidates.extend(results)
    
    # Step 2: Deduplicate by URL
    unique_docs = {doc.url: doc for doc in candidates}.values()
    
    # Step 3: Rerank by semantic similarity to original question
    ranked_docs = []
    question_embedding = embed_text(original_question)
    
    for doc in unique_docs:
        # Combine title and snippet for relevance scoring
        doc_text = f"{doc.title} {doc.snippet}"
        doc_embedding = embed_text(doc_text)
        
        # Cosine similarity
        similarity = cosine_similarity(question_embedding, doc_embedding)
        
        # Recency boost (prefer recent content)
        days_old = (datetime.now() - parse_date(doc.published_date)).days
        recency_factor = max(0.5, 1.0 - (days_old / 365))
        
        score = similarity * recency_factor
        ranked_docs.append((score, doc))
    
    # Return top K most relevant
    ranked_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in ranked_docs[:top_k]]
```

**Real Constraints:**  
- **Recency vs. authority trade-off**: Recent content may lack depth; authoritative sources may be outdated
- **Snippet vs. full content**: Ranking on snippets is fast but may miss key content; fetching full content for all candidates is slow
- **Computational cost**: Embedding hundreds of documents for reranking adds significant latency

### 3. Context Assembly and Truncation

**Technical Explanation:**  
LLMs have finite context windows (typically 4K-128K tokens). Retrieved documents often exceed this limit. The system must intelligently select and truncate content to fit the most relevant information within the context window.

**Practical Implications:**  
Naive truncation (first N characters) often cuts critical information. Strategic extraction (pull key sections based on question keywords) preserves answer quality while fitting constraints.

```python
from typing import List, Tuple
import tiktoken

def assemble_context(
    question: str,
    documents: List[SearchResult],
    max_tokens: int = 8000
) -> Tuple[str, List[str]]:
    """
    Intelligently fit documents into context window
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Reserve tokens for question and response
    question_tokens = len(encoding.encode(question))
    reserved_for_response = 1500
    available_tokens = max_tokens - question_tokens - reserved_for_response
    
    context_parts = []
    citations = []
    used_tokens = 0
    
    for i, doc in enumerate(documents):
        # Extract most relevant passages
        relevant_passages = extract_relevant_passages(
            doc.content,
            question,
            max_passages=3
        )
        
        # Build citation entry
        citation_id = i + 1
        doc_text = f"\n[Source {citation_id}: {doc.title}]\n"
        doc_text += "\n".join(relevant_passages)
        
        doc_tokens = len(encoding.encode(doc_text))
        
        # Check if we have room
        if used_tokens + doc_tokens > available_tokens:
            # Try with fewer passages
            if len(relevant_passages) > 1:
                doc_text = f"\n[Source {citation_id}: {doc.title}]\n"
                doc_text += relevant_passages[0]
                doc_tokens = len(encoding.encode(doc_text))
                
                if used_tokens + doc_tokens > available_tokens:
                    break  # Can't fit more
        
        context_parts.append(doc_text)
        citations.append(f"[{citation_id}] {doc.url}")
        used_tokens += doc_tokens
    
    context = "\n\n".join(context_parts)
    return context, citations

def extract_relevant_passages(
    content: str,
    question: str,
    max_passages: int = 3
) -> List[str]:
    """
    Extract passages most relevant to the question
    """
    # Split into paragraphs
    paragraphs = content.split('\n\n')
    
    # Score each paragraph by keyword overlap
    question_keywords = set(question.lower().split())
    scored_paragraphs = []
    
    for para in paragraphs:
        if len(para.split()) < 20:  # Skip very short paragraphs
            continue
            
        para_keywords = set(para.lower().split())
        overlap = len(question_keywords & para_keywords)
        scored_paragraphs.append((overlap, para))
    
    # Return top N paragraphs
    scored_paragraphs.sort(reverse=True, key=lambda x: x[0])
    return [para for score, para in scored_paragraphs[:max_passages]]
```

**Real Constraints:**  
- **Context stuffing**: More content ≠ better answers. Irrelevant content dilutes signal and increases "lost in the middle" effects
- **Citation granularity**: Track which sentences came from which sources for accurate attribution
- **Token counting accuracy**: Different tokenizers → different counts; always leave safety margin

### 4. Synthesis with Attribution

**Technical Explanation:**  
The final step generates an answer by reasoning over the assembled context while maintaining explicit links between claims and sources. This requires prompting the LLM to cite sources for factual claims and synthesize information rather than copy-paste.

**Practical Implications:**  
Without explicit citation instructions, LLMs either fail to cite sources or hallucinate source references. Clear structural prompts enforce attribution.

```python
def generate_answer_with_citations(
    question: str,
    context: str,
    citations: List[str],
    llm_client
) -> Dict[str, any]:
    """
    Generate answer grounded in sources with explicit citations
    """
    prompt = f"""You are a technical research assistant. Answer the question using ONLY 
    information from the provided sources. Cite sources using [1], [2], etc.
    
    IMPORTANT:
    - Every factual claim must cite a source number
    - If sources don't contain information to answer, say so explicitly
    - Synthesize information across sources; don't just quote
    - For code examples, cite where they came from
    
    Question: {question}
    
    Sources:
    {context}
    
    Answer the question comprehensively with citations:"""
    
    response = llm_client.generate(
        prompt=prompt,
        temperature=0.2,  # Lower temp for factual accuracy
        max_tokens=1500
    )
    
    return {
        'answer': response,
        'citations': citations,
        'source_count': len(citations)
    }
```

**Real Constraints:**  
- **Citation hallucination**: LLMs sometimes invent source numbers not in the context
- **Over-citation**: Citing every trivial claim reduces readability
- **Synthesis quality**: Poorly instructed models copy-paste quotes instead of synthesizing

### 5. Freshness and Caching Strategy

**Technical Explanation:**  
Not all queries require real-time search. "What is a binary tree?" has stable answers; caching results for hours or days is acceptable. "What's the latest Kubernetes version?" requires fresh data. Smart systems cache based on query type.

**Practical Implications:**  
Real-time search is expensive (latency, API costs). Unnecessary fresh searches waste resources and slow responses.

```python
from datetime import datetime, timedelta
from hashlib import sha256
import json

class ResearchCache:
    def __init__(self, redis_client):
        self.cache = redis_client
        
    def get_cached_response(
        self,
        question: str,
        max_age_hours: int = 24
    ) -> Dict | None:
        """
        Retrieve cached response if fresh enough
        """
        cache_key = self._cache_key(question)
        cached = self.cache.get(cache_key)
        
        if not cached:
            return None
            
        data = json.loads(cached)
        cached_at = datetime.fromisoformat(data['cached_at'])
        age_hours = (datetime.now() - cached_at).total_seconds() / 3600
        
        if age_hours <= max_age_hours:
            return data['response']
        
        return None
    
    def cache_response(
        self,
        question: str,
        response: Dict,
        ttl_hours: int = 24
    ):
        """
        Cache response with TTL
        """
        cache_key = self._cache_key(question)
        data = {
            'response': response,
            'cached_at': datetime.now().isoformat()
        }
        
        self.cache.setex(
            cache_key,
            timedelta(hours=ttl_hours),
            json.dumps(data)
        )
    
    def _cache_key(self, question: str) -> str:
        return f"research:{sha256(question.encode()).hexdigest()}"

def determine_freshness_requirement(question: str) -> int:
    """
    Determine how fresh data needs to be (in hours)
    """
    time_sensitive_keywords = [
        'latest', 'current', 'recent', 'today', 'now',
        'breaking', 'update', 'new', 'just', 'yesterday'
    ]
    
    question_lower = question.lower()
    
    # Time-sensitive query - require fresh data
    if any(keyword in question_lower for keyword in time_sensitive_keywords):
        return 1  # Max 1 hour old
    
    # Technical version queries - moderately fresh
    if any(word in question_lower for word in ['version', 'release', 'deprecated']):
        return 24  # Max 24 hours old
    
    # Conceptual queries - can be older
    if any(word in question_lower for word in ['what is', 'how does', 'explain', 'concept']):
        return 168  # Max 1 week old
    
    # Default: moderately fresh
    return 48  # Max 2 days old
```

## Hands-On Exercises

###