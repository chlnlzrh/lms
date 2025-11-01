# Performance Profiling & Optimization for LLM Applications

## Core Concepts

Performance profiling for LLM applications differs fundamentally from traditional software profiling. While conventional applications bottleneck on CPU cycles, memory allocation, or I/O operations, LLM applications introduce new bottlenecks: token processing latency, context management overhead, API rate limits, and non-deterministic execution paths that make traditional profiling tools inadequate.

### Traditional vs. Modern Profiling

```python
# Traditional API profiling
import time
import cProfile

def traditional_api_call(data: dict) -> dict:
    """Predictable, deterministic operation"""
    start = time.time()
    result = process_data(data)  # Consistent execution path
    elapsed = time.time() - start
    return result

# Profile with standard tools
cProfile.run('traditional_api_call(data)')
# Output: function calls, time per function, predictable hotspots

# LLM application profiling
from typing import Any
import asyncio
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LLMMetrics:
    prompt_tokens: int
    completion_tokens: int
    wall_time: float
    time_to_first_token: float
    tokens_per_second: float
    cache_hit: bool
    model_load_time: float

async def llm_api_call(prompt: str, context: list[str]) -> tuple[str, LLMMetrics]:
    """Non-deterministic with variable costs"""
    metrics = LLMMetrics(0, 0, 0.0, 0.0, 0.0, False, 0.0)
    start = time.time()
    
    # Token counting affects cost but not wall time
    metrics.prompt_tokens = count_tokens(prompt) + sum(count_tokens(c) for c in context)
    
    # First token latency matters for UX
    first_token_time = None
    tokens_generated = 0
    
    # Streaming response
    async for chunk in stream_llm_response(prompt, context):
        if first_token_time is None:
            first_token_time = time.time()
            metrics.time_to_first_token = first_token_time - start
        tokens_generated += 1
    
    metrics.wall_time = time.time() - start
    metrics.completion_tokens = tokens_generated
    metrics.tokens_per_second = tokens_generated / (metrics.wall_time - metrics.time_to_first_token)
    
    return chunk, metrics

# Standard profilers miss critical LLM-specific metrics:
# - Token usage (cost)
# - Time to first token (UX)
# - Cache effectiveness
# - Context window utilization
```

The fundamental insight: **LLM performance is multi-dimensional**. Optimizing wall-clock time alone may increase costs 10x. Optimizing tokens alone may degrade quality. You must profile and optimize across latency, throughput, cost, and quality simultaneously.

### Why This Matters Now

Production LLM applications face unique challenges:

1. **Non-linear cost scaling**: A 2x increase in context size doesn't double cost—it may 4x it due to quadratic attention complexity
2. **Latency unpredictability**: The same prompt may take 200ms or 2s depending on model load, cache state, and generation length
3. **Hidden bottlenecks**: 90% of "LLM latency" often occurs in preprocessing, embedding lookups, or result parsing—not the LLM call itself
4. **Rate limit economics**: Hitting rate limits wastes money on retries while increasing latency

Without specialized profiling, you'll optimize the wrong things.

## Technical Components

### 1. Token-Level Profiling

Token accounting is fundamental because tokens determine both cost and theoretical minimum latency.

```python
from typing import Protocol
import tiktoken
from collections import defaultdict

class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

class DetailedTokenProfiler:
    """Track token usage across application components"""
    
    def __init__(self, model: str = "gpt-4"):
        self.encoder = tiktoken.encoding_for_model(model)
        self.token_usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "total": 0}
        )
    
    def profile_component(self, component: str, prompt: str, completion: str) -> dict[str, int]:
        """Track tokens used by specific component"""
        prompt_tokens = len(self.encoder.encode(prompt))
        completion_tokens = len(self.encoder.encode(completion))
        total = prompt_tokens + completion_tokens
        
        self.token_usage[component]["prompt"] += prompt_tokens
        self.token_usage[component]["completion"] += completion_tokens
        self.token_usage[component]["total"] += total
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total
        }
    
    def get_cost_breakdown(self, prompt_cost_per_1k: float, completion_cost_per_1k: float) -> dict[str, float]:
        """Calculate cost per component"""
        costs = {}
        for component, usage in self.token_usage.items():
            prompt_cost = (usage["prompt"] / 1000) * prompt_cost_per_1k
            completion_cost = (usage["completion"] / 1000) * completion_cost_per_1k
            costs[component] = {
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "total_cost": prompt_cost + completion_cost
            }
        return costs
    
    def identify_wasteful_components(self, threshold: float = 0.3) -> list[str]:
        """Find components using disproportionate tokens"""
        total_tokens = sum(u["total"] for u in self.token_usage.values())
        wasteful = []
        
        for component, usage in self.token_usage.items():
            ratio = usage["total"] / total_tokens if total_tokens > 0 else 0
            if ratio > threshold:
                wasteful.append(f"{component}: {ratio:.1%} of total tokens")
        
        return wasteful

# Usage example
profiler = DetailedTokenProfiler()

# Profile different components
profiler.profile_component(
    "system_prompt",
    prompt="You are a helpful assistant with extensive knowledge...",
    completion=""
)

profiler.profile_component(
    "user_query",
    prompt="What is the capital of France?",
    completion="The capital of France is Paris."
)

profiler.profile_component(
    "few_shot_examples",
    prompt="Example 1: Q: ... A: ...\nExample 2: Q: ... A: ...",
    completion=""
)

# Analyze
costs = profiler.get_cost_breakdown(prompt_cost_per_1k=0.03, completion_cost_per_1k=0.06)
wasteful = profiler.identify_wasteful_components()

print(f"Cost breakdown: {costs}")
print(f"Wasteful components: {wasteful}")
```

**Practical implications**: Most applications discover that 60-80% of tokens come from few-shot examples or verbose system prompts. Profiling reveals that a 500-token system prompt costs $0.015 per request—$15 per 1,000 requests.

**Trade-offs**: Reducing token usage by simplifying prompts may reduce quality. The optimization target is cost-per-successful-outcome, not cost-per-request.

### 2. Latency Breakdown Profiling

LLM applications have multiple latency components that standard profilers don't separate.

```python
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator
import json

@dataclass
class LatencyProfile:
    """Detailed latency breakdown"""
    preprocessing: float = 0.0
    embedding_lookup: float = 0.0
    llm_queue_wait: float = 0.0
    time_to_first_token: float = 0.0
    token_generation: float = 0.0
    postprocessing: float = 0.0
    total: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}
    
    def bottleneck(self) -> tuple[str, float]:
        """Identify primary bottleneck"""
        times = {k: v for k, v in self.to_dict().items() if k != "total"}
        bottleneck_component = max(times, key=times.get)
        return bottleneck_component, times[bottleneck_component]

class LatencyProfiler:
    """Context manager for profiling latency components"""
    
    def __init__(self):
        self.profile = LatencyProfile()
        self._phase_start: float = 0.0
        self._total_start: float = 0.0
    
    @asynccontextmanager
    async def phase(self, name: str) -> AsyncIterator[None]:
        """Profile a specific phase"""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            setattr(self.profile, name, elapsed)
    
    async def profile_request(
        self,
        query: str,
        context_docs: list[str],
        llm_callable: callable
    ) -> tuple[str, LatencyProfile]:
        """Profile complete request"""
        self._total_start = time.time()
        
        # Preprocessing
        async with self.phase("preprocessing"):
            processed_query = query.lower().strip()
            await asyncio.sleep(0.01)  # Simulated processing
        
        # Embedding lookup (vector search)
        async with self.phase("embedding_lookup"):
            # Simulate vector DB lookup
            await asyncio.sleep(0.05)
            relevant_docs = context_docs[:3]
        
        # LLM call with internal timing
        async with self.phase("llm_queue_wait"):
            # Simulate queue wait
            await asyncio.sleep(0.02)
        
        # First token latency
        start_generation = time.time()
        response_chunks = []
        first_token = True
        
        async for chunk in llm_callable(processed_query, relevant_docs):
            if first_token:
                self.profile.time_to_first_token = time.time() - start_generation
                first_token = False
            response_chunks.append(chunk)
        
        self.profile.token_generation = time.time() - start_generation - self.profile.time_to_first_token
        
        response = "".join(response_chunks)
        
        # Postprocessing
        async with self.phase("postprocessing"):
            # Parse, validate, format
            parsed_response = json.loads(response) if response.startswith("{") else response
            await asyncio.sleep(0.01)
        
        self.profile.total = time.time() - self._total_start
        
        return response, self.profile

# Example usage
async def mock_llm_stream(query: str, docs: list[str]) -> AsyncIterator[str]:
    """Mock streaming LLM"""
    await asyncio.sleep(0.15)  # First token delay
    for i in range(20):  # Generate 20 tokens
        yield f"token_{i} "
        await asyncio.sleep(0.01)

async def profile_example():
    profiler = LatencyProfiler()
    
    response, profile = await profiler.profile_request(
        query="What is the capital?",
        context_docs=["doc1", "doc2", "doc3"],
        llm_callable=mock_llm_stream
    )
    
    print(f"Total latency: {profile.total:.3f}s")
    print(f"Breakdown: {json.dumps(profile.to_dict(), indent=2)}")
    bottleneck, time_spent = profile.bottleneck()
    print(f"Bottleneck: {bottleneck} ({time_spent:.3f}s, {time_spent/profile.total:.1%})")

# Run
# asyncio.run(profile_example())
```

**Practical implications**: In production systems, embedding lookups and postprocessing often contribute 40-60% of total latency. Developers optimize the LLM call while the real bottleneck is a slow vector database query.

**Constraints**: Profiling adds ~1-5ms overhead. For high-throughput systems, sample only a percentage of requests.

### 3. Cache Effectiveness Analysis

LLM applications use multiple cache layers: prompt caching, KV-cache, semantic cache. Measuring cache effectiveness is critical.

```python
from typing import Optional
import hashlib
from datetime import datetime, timedelta

@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    avg_hit_latency: float = 0.0
    avg_miss_latency: float = 0.0
    latency_improvement: float = 0.0
    cost_saved: float = 0.0

class SemanticCache:
    """Cache with similarity-based lookup"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache: dict[str, tuple[str, datetime, float]] = {}
        self.similarity_threshold = similarity_threshold
        self.metrics = CacheMetrics()
        self._hit_latencies: list[float] = []
        self._miss_latencies: list[float] = []
    
    def _hash_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def _compute_similarity(self, prompt1: str, prompt2: str) -> float:
        """Simplified similarity - use embeddings in production"""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        return len(words1 & words2) / len(words1 | words2) if words1 or words2 else 0.0
    
    async def get(self, prompt: str, cost_per_request: float) -> Optional[str]:
        """Retrieve from cache with similarity matching"""
        start = time.time()
        
        # Exact match
        key = self._hash_key(prompt)
        if key in self.cache:
            response, cached_at, _ = self.cache[key]
            latency = time.time() - start
            self._hit_latencies.append(latency)
            self.metrics.hits += 1
            self.metrics.cost_saved += cost_per_request
            return response
        
        # Similarity match
        for cached_key, (response, cached_at, original_latency) in self.cache.items():
            # Reconstruct original prompt (in production, store it)
            similarity = self._compute_similarity(prompt, response)
            if similarity >= self.similarity_threshold:
                latency = time.time() - start
                self._hit_latencies.append(latency)
                self.metrics.hits += 1
                self.metrics.cost_saved += cost_per_request * 0.8  # Partial cost saved
                return response
        
        # Miss
        latency = time.time() - start
        self._miss_latencies.append(latency)
        self.metrics.misses += 1
        return None
    
    async def set(self, prompt: str, response: str, generation_latency: float):
        """Store in cache"""
        key = self._hash_key(prompt)
        self.cache[key] = (response, datetime.now(), generation_latency)
    
    def get_metrics(self) -> CacheMetrics:
        """Calculate cache effectiveness metrics"""
        total = self.metrics.hits + self.metrics.misses
        if total > 0:
            self.metrics.hit_rate = self.metrics.hits / total
        
        if self._hit_latencies:
            self.metrics.avg_hit_latency = sum(self._hit_latencies) / len(self._hit_latencies)
        
        if self._miss_latencies:
            self.metrics.avg_miss_latency = sum(self._miss_latencies) / len(self._miss_latencies)
        
        if self.