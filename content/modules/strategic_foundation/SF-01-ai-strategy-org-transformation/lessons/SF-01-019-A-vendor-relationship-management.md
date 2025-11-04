# Vendor Relationship Management for AI/LLM Systems

## Core Concepts

In traditional software systems, vendor relationships are relatively static: you choose PostgreSQL or MySQL, AWS or GCP, and switching costs are high but predictable. In AI/LLM systems, vendor relationships are dynamic runtime dependencies where your application's core functionality—the intelligence itself—comes from external providers whose models, pricing, rate limits, and capabilities change monthly or even weekly.

### Technical Definition

Vendor Relationship Management (VRM) in AI systems is the architectural practice of abstracting, monitoring, and orchestrating dependencies on external AI service providers to maintain application reliability, cost predictability, and performance guarantees despite the volatile nature of the AI provider ecosystem.

### Engineering Analogy: Traditional vs. Modern Approach

**Traditional approach (tightly coupled):**

```python
import openai

def analyze_sentiment(text: str) -> dict:
    """Tightly coupled to a single provider"""
    openai.api_key = "sk-..."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Analyze sentiment: {text}"}]
    )
    
    return {"sentiment": response.choices[0].message.content}

# What happens when:
# - Provider raises prices 300%?
# - Rate limits hit during traffic spike?
# - Model gets deprecated?
# - Provider has an outage?
# Answer: Your application breaks, and fixing requires code changes across your codebase.
```

**Modern approach (abstracted and resilient):**

```python
from abc import ABC, abstractmethod
from typing import Protocol, Optional
from enum import Enum
import time
from dataclasses import dataclass

class ModelCapability(Enum):
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"

@dataclass
class ModelConfig:
    name: str
    cost_per_1k_tokens: float
    tokens_per_minute_limit: int
    latency_p95_ms: int
    capabilities: list[ModelCapability]

class LLMProvider(Protocol):
    """Provider abstraction allows swapping implementations"""
    
    def complete(self, prompt: str, max_tokens: int) -> str:
        ...
    
    def get_config(self) -> ModelConfig:
        ...

class ProviderRouter:
    """Intelligent routing based on requirements and constraints"""
    
    def __init__(self):
        self.providers: dict[str, LLMProvider] = {}
        self.fallback_chain: list[str] = []
        self.cost_tracker = CostTracker()
        
    def register_provider(self, name: str, provider: LLMProvider, 
                         is_fallback: bool = False):
        self.providers[name] = provider
        if is_fallback:
            self.fallback_chain.append(name)
    
    def route(self, prompt: str, capability: ModelCapability,
              max_cost_per_call: float = None,
              max_latency_ms: int = None) -> str:
        """Route to best available provider based on constraints"""
        
        eligible_providers = self._filter_providers(
            capability, max_cost_per_call, max_latency_ms
        )
        
        for provider_name in eligible_providers:
            try:
                provider = self.providers[provider_name]
                
                # Check rate limits before calling
                if self._check_rate_limit(provider_name):
                    result = provider.complete(prompt, max_tokens=500)
                    self.cost_tracker.record(provider_name, len(prompt))
                    return result
                    
            except Exception as e:
                # Try next provider in chain
                continue
        
        raise RuntimeError("All providers exhausted")
    
    def _filter_providers(self, capability: ModelCapability,
                         max_cost: Optional[float],
                         max_latency: Optional[int]) -> list[str]:
        eligible = []
        
        for name, provider in self.providers.items():
            config = provider.get_config()
            
            if capability not in config.capabilities:
                continue
            
            if max_cost and config.cost_per_1k_tokens > max_cost:
                continue
                
            if max_latency and config.latency_p95_ms > max_latency:
                continue
            
            eligible.append(name)
        
        # Sort by cost (cheapest first)
        eligible.sort(key=lambda x: self.providers[x].get_config().cost_per_1k_tokens)
        
        return eligible

class CostTracker:
    """Real-time cost monitoring and alerting"""
    
    def __init__(self):
        self.usage: dict[str, float] = {}
        self.budget_alerts: dict[str, float] = {}
    
    def record(self, provider: str, tokens: int):
        # Implementation tracks spending
        pass
    
    def set_alert_threshold(self, provider: str, daily_budget: float):
        self.budget_alerts[provider] = daily_budget
```

### Key Insights That Change Engineering Thinking

1. **AI providers are volatile infrastructure**: Unlike traditional databases that remain stable for years, AI models change capabilities, pricing, and availability constantly. Your architecture must assume change, not stability.

2. **Cost is a first-class runtime constraint**: In traditional systems, infrastructure cost is relatively fixed. In AI systems, a single model choice can make your cost 10x higher or lower. Cost must be part of your routing logic, not just a billing concern.

3. **Provider diversity is reliability**: Having fallback providers isn't just cost optimization—it's essential reliability engineering. No AI provider has the same uptime guarantees as traditional infrastructure providers.

4. **Rate limits are shared state**: Unlike API rate limits you control, AI provider rate limits affect all your customers simultaneously and can change without notice. You need sophisticated rate limit tracking and request queuing.

### Why This Matters NOW

The AI provider landscape is experiencing massive changes in 2024-2025:

- **Price wars**: Major providers have dropped prices by 90%+ in 18 months
- **Model proliferation**: New models release monthly with different strengths
- **Capability shifts**: Yesterday's best model for task X might be surpassed today
- **Regional availability**: Geopolitical factors affect model access
- **Rate limit volatility**: Providers adjust limits based on demand

Engineers building without VRM face:
- Unexpected 10x cost increases when usage scales
- Application downtime during provider outages
- Inability to leverage better/cheaper models as they emerge
- Manual firefighting instead of automated resilience

## Technical Components

### 1. Provider Abstraction Layer

**Technical Explanation:**

The abstraction layer creates a uniform interface across different AI providers, hiding implementation details while exposing necessary configuration for routing decisions.

**Practical Implementation:**

```python
from typing import Optional, AsyncIterator
import httpx
import asyncio
from datetime import datetime

class ProviderResponse:
    """Standardized response format"""
    def __init__(self, text: str, provider: str, 
                 tokens_used: int, latency_ms: float):
        self.text = text
        self.provider = provider
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.timestamp = datetime.utcnow()

class AnthropicProvider:
    """Concrete provider implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.client = httpx.AsyncClient()
        
    async def complete(self, prompt: str, max_tokens: int = 1000) -> ProviderResponse:
        start = time.time()
        
        response = await self.client.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-sonnet-20240229",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        latency_ms = (time.time() - start) * 1000
        data = response.json()
        
        return ProviderResponse(
            text=data["content"][0]["text"],
            provider="anthropic",
            tokens_used=data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
            latency_ms=latency_ms
        )
    
    def get_config(self) -> ModelConfig:
        return ModelConfig(
            name="claude-3-sonnet",
            cost_per_1k_tokens=0.015,
            tokens_per_minute_limit=40000,
            latency_p95_ms=2000,
            capabilities=[ModelCapability.SENTIMENT, 
                         ModelCapability.SUMMARIZATION]
        )

class OpenRouterProvider:
    """Alternative provider with multiple models"""
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-haiku"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient()
    
    async def complete(self, prompt: str, max_tokens: int = 1000) -> ProviderResponse:
        start = time.time()
        
        response = await self.client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://your-app.com"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
        )
        
        latency_ms = (time.time() - start) * 1000
        data = response.json()
        
        return ProviderResponse(
            text=data["choices"][0]["message"]["content"],
            provider=f"openrouter/{self.model}",
            tokens_used=data["usage"]["total_tokens"],
            latency_ms=latency_ms
        )
    
    def get_config(self) -> ModelConfig:
        # OpenRouter allows dynamic model selection
        return ModelConfig(
            name=self.model,
            cost_per_1k_tokens=0.004,  # Haiku pricing
            tokens_per_minute_limit=100000,
            latency_p95_ms=1500,
            capabilities=[ModelCapability.SENTIMENT, 
                         ModelCapability.CLASSIFICATION]
        )
```

**Real Constraints:**

- **Abstraction leakage**: Provider-specific features (function calling, vision) don't map cleanly
- **Error handling**: Each provider has different error formats and retry semantics
- **Authentication**: Various methods (API keys, OAuth, IP whitelisting) need normalization

**Trade-offs:**

- Adding abstraction increases code complexity by ~30% initially
- Saves 10x that effort when swapping providers or adding fallbacks
- Slight latency overhead (~10-50ms) from abstraction layer logic

### 2. Intelligent Request Routing

**Technical Explanation:**

Request routing makes real-time decisions about which provider to use based on current constraints: cost budgets, latency requirements, provider health, and rate limits.

**Practical Implementation:**

```python
from collections import deque
from threading import Lock
import time

class RateLimiter:
    """Token bucket rate limiter per provider"""
    
    def __init__(self, requests_per_minute: int):
        self.capacity = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = Lock()
    
    def acquire(self) -> bool:
        """Try to acquire permission for one request"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Refill tokens based on time passed
            self.tokens = min(
                self.capacity,
                self.tokens + (elapsed * self.capacity / 60.0)
            )
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            return False
    
    def wait_time(self) -> float:
        """Seconds until next token available"""
        with self.lock:
            if self.tokens >= 1:
                return 0.0
            tokens_needed = 1 - self.tokens
            return (tokens_needed * 60.0) / self.capacity

class ProviderHealthTracker:
    """Track provider health based on recent requests"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.recent_requests: dict[str, deque] = {}
        self.lock = Lock()
    
    def record_success(self, provider: str, latency_ms: float):
        self._record(provider, success=True, latency_ms=latency_ms)
    
    def record_failure(self, provider: str):
        self._record(provider, success=False, latency_ms=0)
    
    def _record(self, provider: str, success: bool, latency_ms: float):
        with self.lock:
            if provider not in self.recent_requests:
                self.recent_requests[provider] = deque(maxlen=self.window_size)
            
            self.recent_requests[provider].append({
                "success": success,
                "latency_ms": latency_ms,
                "timestamp": time.time()
            })
    
    def get_success_rate(self, provider: str) -> float:
        """Get success rate over recent window"""
        with self.lock:
            if provider not in self.recent_requests:
                return 1.0  # Assume healthy until proven otherwise
            
            requests = list(self.recent_requests[provider])
            if not requests:
                return 1.0
            
            successes = sum(1 for r in requests if r["success"])
            return successes / len(requests)
    
    def get_avg_latency(self, provider: str) -> float:
        """Get average latency over recent window"""
        with self.lock:
            if provider not in self.recent_requests:
                return 0.0
            
            requests = [r for r in self.recent_requests[provider] 
                       if r["success"]]
            if not requests:
                return 0.0
            
            return sum(r["latency_ms"] for r in requests) / len(requests)
    
    def is_healthy(self, provider: str, min_success_rate: float = 0.90) -> bool:
        """Check if provider meets health threshold"""
        return self.get_success_rate(provider) >= min_success_rate

class SmartRouter:
    """Production-ready routing with health and rate limit awareness"""
    
    def __init__(self):
        self.providers: dict[str, LLMProvider] = {}
        self.rate_limiters: dict[str, RateLimiter] = {}
        self.health_tracker = ProviderHealthTracker()
        self.cost_tracker = CostTracker()
    
    def register_provider(self, name: str, provider: LLMProvider):
        self.providers[name] = provider
        config = provider.get_config()
        self.rate_limiters[name] = RateLimiter(config.tokens_per_minute_limit)
    
    async def route(self, prompt: str, 
                    capability: ModelCapability,
                    max_cost_per_1k: Optional[float] = None,
                    max_latency_ms: Optional[int] = None) -> ProviderResponse:
        """
        Route request to optimal provider considering:
        - Capability match
        - Cost constraints
        - Latency requirements
        - Provider health
        - Rate limits
        """
        
        # Get eligible providers
        candidates = self._rank_providers(
            capability, max_cost_per_1k, max_latency_ms
        )
        
        if not candidates:
            raise RuntimeError(f"No providers available for {capability}")
        
        # Try providers in ranked order
        last_error = None
        
        for provider_name in candidates:
            # Skip unhealthy providers
            if not self.health_tracker.