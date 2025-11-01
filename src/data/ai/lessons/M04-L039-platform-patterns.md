# Platform Patterns: Architecting LLM-Powered Systems at Scale

## Core Concepts

Platform patterns for LLM systems represent the architectural blueprints that separate proof-of-concept demos from production-grade AI applications. Unlike traditional software platforms that primarily orchestrate deterministic services, LLM platforms must handle probabilistic outputs, manage expensive computational resources, and provide abstraction layers that accommodate rapid model evolution without breaking downstream consumers.

### Traditional vs. LLM Platform Architecture

**Traditional API Platform:**
```python
# Traditional deterministic platform
class PaymentPlatform:
    def __init__(self, config: dict):
        self.processor = PaymentProcessor(config)
        self.cache = RedisCache()
        
    def process_payment(self, amount: float, card: str) -> dict:
        # Deterministic: same input → same output
        cache_key = f"payment:{card}:{amount}"
        if cached := self.cache.get(cache_key):
            return cached
            
        result = self.processor.charge(amount, card)
        self.cache.set(cache_key, result, ttl=300)
        return result  # Predictable success/failure
```

**LLM Platform Pattern:**
```python
from typing import Protocol, Optional, List
from dataclasses import dataclass
from datetime import datetime
import hashlib

class LLMProvider(Protocol):
    def complete(self, prompt: str, **kwargs) -> str: ...

@dataclass
class LLMRequest:
    prompt: str
    model: str
    temperature: float
    max_tokens: int
    metadata: dict

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    cost_usd: float
    request_id: str
    cached: bool

class LLMPlatform:
    def __init__(self, providers: dict[str, LLMProvider]):
        self.providers = providers
        self.cache = SemanticCache()
        self.fallback_chain = ['primary', 'secondary', 'local']
        self.rate_limiter = AdaptiveRateLimiter()
        self.cost_tracker = CostTracker()
        
    async def complete(
        self, 
        request: LLMRequest,
        use_cache: bool = True,
        retry_strategy: str = 'fallback'
    ) -> LLMResponse:
        # Non-deterministic: same input can → different outputs
        # Must handle: costs, failures, rate limits, varying latencies
        
        request_id = self._generate_request_id(request)
        start_time = datetime.now()
        
        # Semantic caching (not exact match like traditional cache)
        if use_cache:
            if cached := await self.cache.get_similar(
                request.prompt, 
                threshold=0.95
            ):
                return self._build_response(
                    cached, request_id, start_time, cached=True
                )
        
        # Rate limiting with token bucket accounting
        await self.rate_limiter.acquire(
            estimated_tokens=request.max_tokens
        )
        
        # Fallback chain for reliability
        for provider_name in self.fallback_chain:
            try:
                provider = self.providers[provider_name]
                result = await provider.complete(
                    request.prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                response = self._build_response(
                    result, request_id, start_time, cached=False
                )
                
                # Track costs and cache result
                await self.cost_tracker.record(response)
                await self.cache.store(request.prompt, result)
                
                return response
                
            except (RateLimitError, TimeoutError) as e:
                if provider_name == self.fallback_chain[-1]:
                    raise PlatformError(f"All providers failed") from e
                continue
```

The key difference: LLM platforms must treat **every request as potentially expensive and unreliable**, requiring sophisticated cost management, semantic caching, fallback strategies, and observability that traditional platforms don't need.

### Why Platform Patterns Matter Now

Three converging factors make platform patterns critical:

1. **Model commoditization**: As models become interchangeable, platform capabilities (routing, caching, observability) become the differentiator
2. **Cost unpredictability**: Token-based pricing means unbounded costs without platform controls
3. **Multi-model reality**: Production systems increasingly use multiple models/providers, requiring abstraction layers

Without solid platform patterns, you'll spend more time firefighting API failures and cost overruns than building features.

## Technical Components

### 1. Provider Abstraction Layer

The provider abstraction layer normalizes interfaces across different LLM providers, enabling runtime switching without code changes throughout your application.

**Technical Implementation:**

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from enum import Enum
import asyncio

class ModelCapability(Enum):
    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDING = "embedding"
    FUNCTION_CALLING = "function_calling"

class ProviderInterface(ABC):
    """Base interface all providers must implement"""
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        pass
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        pass
    
    @abstractmethod
    def get_capabilities(self) -> set[ModelCapability]:
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pass

class OpenAIProvider(ProviderInterface):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.pricing = {
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-3.5-turbo": {"input": 0.001 / 1000, "output": 0.002 / 1000}
        }
    
    async def complete(
        self, prompt: str, temperature: float = 0.7, 
        max_tokens: int = 1000, **kwargs
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        async for chunk in stream:
            if content := chunk.choices[0].delta.content:
                yield content
    
    def get_capabilities(self) -> set[ModelCapability]:
        return {
            ModelCapability.COMPLETION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING
        }
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self.pricing[self.model]
        return (input_tokens * pricing["input"]) + \
               (output_tokens * pricing["output"])

class AnthropicProvider(ProviderInterface):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.pricing = {
            "claude-3-opus-20240229": {"input": 0.015 / 1000, "output": 0.075 / 1000},
            "claude-3-sonnet-20240229": {"input": 0.003 / 1000, "output": 0.015 / 1000}
        }
    
    async def complete(
        self, prompt: str, temperature: float = 0.7,
        max_tokens: int = 1000, **kwargs
    ) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get('max_tokens', 1000)
        ) as stream:
            async for text in stream.text_stream:
                yield text
    
    def get_capabilities(self) -> set[ModelCapability]:
        return {ModelCapability.COMPLETION, ModelCapability.CHAT}
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self.pricing[self.model]
        return (input_tokens * pricing["input"]) + \
               (output_tokens * pricing["output"])

# Usage: Switch providers without changing application code
async def generate_summary(text: str, provider: ProviderInterface) -> str:
    prompt = f"Summarize this text in 2 sentences:\n\n{text}"
    return await provider.complete(prompt, max_tokens=100)

# Runtime provider selection
provider = OpenAIProvider(api_key="...") if use_openai else AnthropicProvider(api_key="...")
summary = await generate_summary(long_text, provider)
```

**Practical Implications:**
- **A/B testing**: Route 10% of traffic to a new provider to compare quality/cost
- **Cost optimization**: Switch to cheaper models for non-critical requests
- **Reliability**: Fallback to secondary provider on primary failure
- **Vendor negotiation leverage**: Avoid lock-in with single provider

**Constraints:**
- Abstraction leaks for provider-specific features (e.g., Claude's XML tags)
- Performance overhead from interface layer (typically <5ms, negligible vs. LLM latency)
- Complexity in maintaining parity as providers add features

### 2. Semantic Caching Layer

Unlike traditional caching that requires exact key matches, semantic caching recognizes when prompts are "similar enough" to return cached results, dramatically reducing costs for high-traffic applications.

**Technical Implementation:**

```python
import numpy as np
from typing import Optional, Tuple
import hashlib
from datetime import datetime, timedelta

class SemanticCache:
    def __init__(
        self,
        embedding_provider: ProviderInterface,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600
    ):
        self.embedding_provider = embedding_provider
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        # In production: use vector DB (Pinecone, Weaviate, etc.)
        self.cache: dict[str, dict] = {}
        self.embeddings: dict[str, np.ndarray] = {}
    
    def _hash_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        # Cache embeddings to avoid recomputing
        key = self._hash_key(text)
        if key in self.embeddings:
            return self.embeddings[key]
        
        # Get embedding from provider
        embedding = await self.embedding_provider.embed(text)
        self.embeddings[key] = np.array(embedding)
        return self.embeddings[key]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    async def get_similar(
        self, 
        prompt: str,
        threshold: Optional[float] = None
    ) -> Optional[Tuple[str, float]]:
        threshold = threshold or self.threshold
        prompt_embedding = await self._get_embedding(prompt)
        
        best_match = None
        best_similarity = 0.0
        
        for cached_key, cached_data in self.cache.items():
            # Check TTL
            if datetime.now() > cached_data['expires_at']:
                continue
            
            cached_embedding = cached_data['embedding']
            similarity = self._cosine_similarity(
                prompt_embedding, 
                cached_embedding
            )
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = cached_data['response']
        
        if best_match:
            return best_match, best_similarity
        return None
    
    async def store(
        self, 
        prompt: str, 
        response: str,
        metadata: Optional[dict] = None
    ) -> None:
        key = self._hash_key(prompt)
        embedding = await self._get_embedding(prompt)
        
        self.cache[key] = {
            'prompt': prompt,
            'response': response,
            'embedding': embedding,
            'expires_at': datetime.now() + timedelta(seconds=self.ttl),
            'metadata': metadata or {},
            'hit_count': 0
        }
    
    def get_stats(self) -> dict:
        total = len(self.cache)
        expired = sum(
            1 for data in self.cache.values()
            if datetime.now() > data['expires_at']
        )
        return {
            'total_entries': total,
            'active_entries': total - expired,
            'total_hits': sum(
                data['hit_count'] for data in self.cache.values()
            )
        }

# Usage example with cost tracking
class CachedLLMPlatform:
    def __init__(self, provider: ProviderInterface):
        self.provider = provider
        self.cache = SemanticCache(
            embedding_provider=EmbeddingProvider(),
            similarity_threshold=0.92
        )
        self.stats = {'cache_hits': 0, 'cache_misses': 0, 'cost_saved': 0.0}
    
    async def complete(self, prompt: str, **kwargs) -> str:
        # Check cache first
        if cached := await self.cache.get_similar(prompt):
            result, similarity = cached
            self.stats['cache_hits'] += 1
            
            # Estimate cost savings
            estimated_tokens = len(prompt.split()) * 1.3  # rough estimate
            cost_saved = self.provider.estimate_cost(
                int(estimated_tokens), 
                len(result.split()) * 1.3
            )
            self.stats['cost_saved'] += cost_saved
            
            return result
        
        # Cache miss - call provider
        self.stats['cache_misses'] += 1
        result = await self.provider.complete(prompt, **kwargs)
        await self.cache.store(prompt, result)
        return result
    
    def get_cache_hit_rate(self) -> float:
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        return self.stats['cache_hits'] / total if total > 0 else 0.0
```

**Real-World Impact:**
- A chatbot answering FAQs with 60% cache hit rate saves $1,800/month at 100K requests/month
- Reduces p95 latency from 2000ms to 50ms for cached responses
- Lowers provider rate limit pressure by 60%

**Constraints:**
- Embedding costs (~$0.0001 per request) vs. savings must be favorable
- Cold start: cache needs population before providing value
- Semantic drift: prompts evolve over time, requiring cache invalidation strategies