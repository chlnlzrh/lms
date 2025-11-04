# Multi-Provider Failover Architecture

## Core Concepts

Multi-provider failover architecture is a system design pattern that distributes LLM inference requests across multiple AI service providers with automatic switching when primary providers fail, experience degraded performance, or exceed rate limits. Unlike traditional web service failover that primarily handles binary states (up/down), LLM provider failover must handle nuanced failure modes: rate limiting, quality degradation, context window mismatches, and cost optimization across heterogeneous APIs with different capabilities, pricing models, and performance characteristics.

### Engineering Analogy: Traditional vs. Modern Approach

**Traditional Database Failover:**
```python
import psycopg2
from typing import Optional

class DatabaseConnection:
    def __init__(self, primary_host: str, replica_host: str):
        self.primary = primary_host
        self.replica = replica_host
        self.conn: Optional[psycopg2.connection] = None
    
    def query(self, sql: str) -> list:
        try:
            if not self.conn:
                self.conn = psycopg2.connect(host=self.primary)
            return self.conn.execute(sql).fetchall()
        except psycopg2.OperationalError:
            # Simple binary failover: primary down, use replica
            self.conn = psycopg2.connect(host=self.replica)
            return self.conn.execute(sql).fetchall()
```

**LLM Multi-Provider Failover:**
```python
import httpx
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class FailureReason(Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONTEXT_LENGTH = "context_length"
    QUALITY_DEGRADATION = "quality_degradation"
    SERVICE_ERROR = "service_error"
    COST_THRESHOLD = "cost_threshold"

@dataclass
class ProviderConfig:
    name: str
    api_key: str
    endpoint: str
    model: str
    max_tokens: int
    cost_per_1k_tokens: float
    timeout_seconds: float
    quality_threshold: float

class IntelligentFailover:
    def __init__(self, providers: List[ProviderConfig]):
        self.providers = providers
        self.failure_counts: Dict[str, int] = {p.name: 0 for p in providers}
        self.last_failure_time: Dict[str, float] = {}
        self.cumulative_cost: float = 0.0
        
    def query(self, prompt: str, max_cost: float = 1.0) -> tuple[str, Dict[str, Any]]:
        """
        Attempts query with intelligent failover across multiple dimensions:
        - Rate limits (with exponential backoff)
        - Context length mismatches (route to appropriate model)
        - Cost constraints (skip expensive providers if budget exceeded)
        - Quality requirements (validate response meets threshold)
        """
        metadata = {"attempts": [], "total_cost": 0.0}
        
        for provider in self._prioritize_providers(prompt, max_cost):
            try:
                # Check if provider in cooldown from rate limit
                if self._in_cooldown(provider.name):
                    metadata["attempts"].append({
                        "provider": provider.name,
                        "skipped": "cooldown",
                        "wait_time": self._cooldown_remaining(provider.name)
                    })
                    continue
                
                # Check context window compatibility
                estimated_tokens = len(prompt) // 4  # rough estimate
                if estimated_tokens > provider.max_tokens:
                    metadata["attempts"].append({
                        "provider": provider.name,
                        "skipped": "context_length",
                        "required": estimated_tokens,
                        "available": provider.max_tokens
                    })
                    continue
                
                # Check cost constraints
                estimated_cost = (estimated_tokens / 1000) * provider.cost_per_1k_tokens
                if self.cumulative_cost + estimated_cost > max_cost:
                    metadata["attempts"].append({
                        "provider": provider.name,
                        "skipped": "cost_threshold",
                        "would_cost": estimated_cost,
                        "remaining_budget": max_cost - self.cumulative_cost
                    })
                    continue
                
                # Attempt the query
                start_time = time.time()
                response = self._execute_query(provider, prompt)
                latency = time.time() - start_time
                
                # Validate quality
                quality_score = self._assess_quality(response)
                if quality_score < provider.quality_threshold:
                    metadata["attempts"].append({
                        "provider": provider.name,
                        "failed": "quality_degradation",
                        "score": quality_score,
                        "threshold": provider.quality_threshold
                    })
                    continue
                
                # Success
                actual_cost = (len(response) / 1000) * provider.cost_per_1k_tokens
                self.cumulative_cost += actual_cost
                metadata["attempts"].append({
                    "provider": provider.name,
                    "success": True,
                    "latency": latency,
                    "cost": actual_cost,
                    "quality": quality_score
                })
                
                return response, metadata
                
            except httpx.TimeoutException:
                self._record_failure(provider.name, FailureReason.TIMEOUT)
                metadata["attempts"].append({
                    "provider": provider.name,
                    "failed": "timeout",
                    "timeout_threshold": provider.timeout_seconds
                })
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    self._record_failure(provider.name, FailureReason.RATE_LIMIT)
                    metadata["attempts"].append({
                        "provider": provider.name,
                        "failed": "rate_limit",
                        "retry_after": e.response.headers.get("Retry-After", "unknown")
                    })
                else:
                    self._record_failure(provider.name, FailureReason.SERVICE_ERROR)
                    metadata["attempts"].append({
                        "provider": provider.name,
                        "failed": "service_error",
                        "status_code": e.response.status_code
                    })
        
        raise Exception(f"All providers exhausted. Attempts: {metadata['attempts']}")
    
    def _prioritize_providers(self, prompt: str, max_cost: float) -> List[ProviderConfig]:
        """Sort providers by: success rate, cost efficiency, context window fit"""
        def score(p: ProviderConfig) -> float:
            failure_rate = self.failure_counts[p.name] / max(sum(self.failure_counts.values()), 1)
            cost_score = 1.0 / (p.cost_per_1k_tokens + 0.001)
            context_score = 1.0 if len(prompt) // 4 < p.max_tokens else 0.0
            return (1 - failure_rate) * cost_score * context_score
        
        return sorted(self.providers, key=score, reverse=True)
    
    def _in_cooldown(self, provider_name: str) -> bool:
        if provider_name not in self.last_failure_time:
            return False
        cooldown = min(300, 2 ** self.failure_counts[provider_name])  # exponential backoff, max 5min
        return time.time() - self.last_failure_time[provider_name] < cooldown
    
    def _cooldown_remaining(self, provider_name: str) -> float:
        cooldown = min(300, 2 ** self.failure_counts[provider_name])
        elapsed = time.time() - self.last_failure_time[provider_name]
        return max(0, cooldown - elapsed)
    
    def _record_failure(self, provider_name: str, reason: FailureReason):
        self.failure_counts[provider_name] += 1
        self.last_failure_time[provider_name] = time.time()
    
    def _execute_query(self, provider: ProviderConfig, prompt: str) -> str:
        # Implementation would call actual provider API
        raise NotImplementedError("Provider-specific implementation required")
    
    def _assess_quality(self, response: str) -> float:
        # Implementation would validate response quality
        # e.g., check for coherence, length, formatting
        raise NotImplementedError("Quality assessment implementation required")
```

### Key Insights That Change Engineering Thinking

1. **Failure modes are multidimensional**: Unlike traditional failover where services are binary (up/down), LLM providers exhibit partial failures—rate limits, quality degradation, cost spikes—requiring routing logic beyond simple health checks.

2. **Provider heterogeneity is architectural constraint**: Different providers support different context windows (4K to 1M+ tokens), model capabilities, and pricing models. Your failover logic must treat providers as non-fungible resources with capability matrices.

3. **Cost is a runtime constraint, not just a metric**: When a provider charges 100x more per token, cost becomes a failure condition. Budget-aware failover prevents cascading financial failures.

4. **Quality monitoring replaces simple health checks**: A 200 OK response doesn't guarantee useful output. Production systems need automated quality assessment to detect model degradation, refusals, or malicious responses.

### Why This Matters Now

As of 2024, LLM APIs have become critical infrastructure with three emerging challenges:

1. **Rate limit warfare**: Major providers aggressively rate limit during peak usage. Single-provider systems experience 10-40% request failures during US business hours.

2. **Model churn**: Providers update models monthly, causing quality regressions. Systems locked to single providers experience sudden accuracy drops without warning.

3. **Regional availability gaps**: Providers blacklist regions, implement GDPR restrictions, or experience localized outages. Multi-provider architecture provides geographic redundancy.

## Technical Components

### 1. Provider Abstraction Layer

The abstraction layer normalizes heterogeneous provider APIs into a unified interface while preserving provider-specific capabilities.

**Technical Explanation:**

Provider APIs differ in authentication (API keys, OAuth, custom headers), request formats (REST, gRPC, SSE), and response structures. The abstraction layer creates a common interface without losing provider-specific optimizations like streaming, function calling, or vision inputs.

**Practical Implementation:**

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict, Any
import httpx
import json

class ProviderAdapter(ABC):
    """Abstract base for all provider implementations"""
    
    def __init__(self, api_key: str, timeout: float = 30.0):
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    @abstractmethod
    async def complete(
        self, 
        prompt: str, 
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Synchronous completion"""
        pass
    
    @abstractmethod
    async def stream_complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Streaming completion"""
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD"""
        pass
    
    @abstractmethod
    def max_context_window(self) -> int:
        """Return max tokens for this provider/model"""
        pass

class OpenAIAdapter(ProviderAdapter):
    """OpenAI-compatible API adapter"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", base_url: str = "https://api.openai.com/v1"):
        super().__init__(api_key)
        self.model = model
        self.base_url = base_url
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
        }
    
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def stream_complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                **kwargs
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    if line.strip() == "data: [DONE]":
                        break
                    data = json.loads(line[6:])
                    if content := data["choices"][0]["delta"].get("content"):
                        yield content
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self.pricing.get(self.model, {"input": 0.03, "output": 0.06})
        return (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])
    
    def max_context_window(self) -> int:
        windows = {"gpt-4": 8192, "gpt-3.5-turbo": 16384, "gpt-4-turbo": 128000}
        return windows.get(self.model, 8192)

class AnthropicAdapter(ProviderAdapter):
    """Anthropic Claude API adapter"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        self.pricing = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015}
        }
    
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        response = await self.client.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]
    
    async def stream_complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float