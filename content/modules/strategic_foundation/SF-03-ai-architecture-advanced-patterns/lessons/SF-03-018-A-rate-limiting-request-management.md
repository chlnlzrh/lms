# Rate Limiting & Request Management for LLM Applications

## Core Concepts

Rate limiting for LLM APIs differs fundamentally from traditional REST API rate limiting. Instead of simply counting requests per time window, you're managing multiple constraint dimensions simultaneously: requests per minute (RPM), tokens per minute (TPM), concurrent requests, and often cost budgets. A single expensive request can consume your entire quota, while hundreds of small requests might sail through unaffected.

### Traditional vs. Modern Rate Limiting

**Traditional REST API approach:**

```python
import time
from collections import deque
from typing import Callable, Any

class SimpleRateLimiter:
    """Traditional request-count-based rate limiter"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
    
    def acquire(self) -> bool:
        now = time.time()
        # Remove requests outside the window
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        while not self.acquire():
            time.sleep(0.1)
        return func(*args, **kwargs)

# Usage: Simple but inadequate for LLMs
limiter = SimpleRateLimiter(max_requests=60, window_seconds=60)
# Problem: Doesn't account for token consumption or request cost
```

**LLM-aware rate limiting:**

```python
import time
from dataclasses import dataclass
from typing import Optional
import threading

@dataclass
class TokenBucket:
    """Multi-dimensional rate limiter for LLM APIs"""
    capacity: float
    refill_rate: float  # tokens per second
    current: float
    last_update: float
    lock: threading.Lock
    
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.current = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: float) -> bool:
        """Try to consume tokens, return True if successful"""
        with self.lock:
            self._refill()
            if self.current >= tokens:
                self.current -= tokens
                return True
            return False
    
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_update
        refill = elapsed * self.refill_rate
        self.current = min(self.capacity, self.current + refill)
        self.last_update = now
    
    def wait_time(self, tokens: float) -> float:
        """Calculate seconds to wait before tokens available"""
        with self.lock:
            self._refill()
            if self.current >= tokens:
                return 0.0
            needed = tokens - self.current
            return needed / self.refill_rate

class LLMRateLimiter:
    """Multi-constraint rate limiter for LLM APIs"""
    
    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        max_concurrent: int = 10
    ):
        # Convert to per-second rates for smoother limiting
        self.request_bucket = TokenBucket(
            capacity=requests_per_minute / 60.0 * 10,  # 10-second capacity
            refill_rate=requests_per_minute / 60.0
        )
        self.token_bucket = TokenBucket(
            capacity=tokens_per_minute / 60.0 * 10,
            refill_rate=tokens_per_minute / 60.0
        )
        self.concurrent_semaphore = threading.Semaphore(max_concurrent)
    
    def acquire(self, estimated_tokens: int) -> float:
        """
        Acquire permission to make request.
        Returns wait time in seconds (0 if immediate).
        """
        # Check both buckets, wait for the longer constraint
        request_wait = self.request_bucket.wait_time(1.0)
        token_wait = self.token_bucket.wait_time(float(estimated_tokens))
        max_wait = max(request_wait, token_wait)
        
        if max_wait > 0:
            time.sleep(max_wait)
        
        # Consume from both buckets
        self.request_bucket.consume(1.0)
        self.token_bucket.consume(float(estimated_tokens))
        
        return max_wait
```

### Key Engineering Insights

**1. Token consumption is bimodal and unpredictable:** Your prompt might be 100 tokens, but the response could be 10 or 4,000 tokens. You must estimate conservatively and handle overages gracefully.

**2. Rate limits compose multiplicatively, not additively:** If you have a 60 RPM limit and 90,000 TPM limit, you can't actually achieve 60 requests per minute if each request averages 2,000 tokens—you'll hit the token limit at 45 requests. Your effective limit is `min(rpm_limit, tpm_limit / avg_tokens)`.

**3. Burst capacity matters more than average throughput:** LLM rate limits typically use token buckets, not fixed windows. This means you can burst above your average rate if you've been idle, which is crucial for responsive user-facing applications.

### Why This Matters Now

LLM API costs scale with usage in ways that can surprise engineering teams. A misconfigured rate limiter can:

- **Waste 40-60% of your quota** on retries and exponential backoff
- **Create cascading failures** when one slow request blocks others
- **Cost 10-100x more than necessary** by failing to batch or optimize token usage
- **Degrade user experience** with unnecessary queueing or dropped requests

Production LLM applications need request management strategies, not just throttling. The difference between naïve and optimized approaches can mean 3-5x cost reduction and 2-3x latency improvement.

## Technical Components

### 1. Multi-Dimensional Token Buckets

Token bucket algorithms provide smooth rate limiting by accumulating "tokens" over time. For LLMs, you need separate buckets for different resource types.

**Technical Implementation:**

```python
from enum import Enum
from typing import Dict, NamedTuple
import asyncio

class ResourceType(Enum):
    REQUESTS = "requests"
    TOKENS = "tokens"
    COST = "cost"  # Optional: track dollar spend

class ResourceLimit(NamedTuple):
    capacity: float
    refill_rate: float  # per second

class MultiDimensionalLimiter:
    """Rate limiter tracking multiple resource constraints"""
    
    def __init__(self, limits: Dict[ResourceType, ResourceLimit]):
        self.buckets = {
            resource: TokenBucket(limit.capacity, limit.refill_rate)
            for resource, limit in limits.items()
        }
    
    async def acquire_async(
        self,
        costs: Dict[ResourceType, float]
    ) -> Dict[ResourceType, float]:
        """
        Async acquire that waits for all constraints.
        Returns actual wait times per resource.
        """
        # Calculate wait times for each resource
        wait_times = {
            resource: bucket.wait_time(costs.get(resource, 0))
            for resource, bucket in self.buckets.items()
        }
        
        # Wait for the longest constraint
        max_wait = max(wait_times.values())
        if max_wait > 0:
            await asyncio.sleep(max_wait)
        
        # Consume from all buckets
        for resource, cost in costs.items():
            if resource in self.buckets:
                self.buckets[resource].consume(cost)
        
        return wait_times

# Usage example
limiter = MultiDimensionalLimiter({
    ResourceType.REQUESTS: ResourceLimit(
        capacity=10.0,  # burst capacity
        refill_rate=1.0  # 60 RPM
    ),
    ResourceType.TOKENS: ResourceLimit(
        capacity=15000.0,
        refill_rate=1500.0  # 90k TPM
    ),
    ResourceType.COST: ResourceLimit(
        capacity=1.0,  # $1 burst
        refill_rate=0.01  # $0.60/min = $36/hour
    )
})
```

**Practical Implications:**

The token bucket approach allows legitimate bursts while preventing sustained overuse. If your application is idle for 30 seconds, it can make a burst of requests when a user arrives, rather than forcing them to wait for rate limit windows to reset.

**Real Constraints:**

- Bucket capacity must be calibrated to actual API limits (too high = rate limit errors, too low = unnecessary waiting)
- Refill rate should account for actual token consumption patterns, not just prompt sizes
- Memory overhead: each bucket requires ~64 bytes, negligible for most applications

### 2. Request Queueing with Priority

Not all requests are equal. User-facing interactive requests should jump the queue ahead of background batch processing.

**Technical Implementation:**

```python
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import heapq
from datetime import datetime

@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    timestamp: float = field(compare=True)
    func: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    future: asyncio.Future = field(default_factory=asyncio.Future, compare=False)
    estimated_tokens: int = field(default=1000, compare=False)

class PriorityRequestQueue:
    """Queue that processes high-priority requests first"""
    
    def __init__(
        self,
        limiter: MultiDimensionalLimiter,
        max_queue_size: int = 1000
    ):
        self.limiter = limiter
        self.max_queue_size = max_queue_size
        self.queue: list[PrioritizedRequest] = []
        self.processing = False
        self.lock = asyncio.Lock()
    
    async def submit(
        self,
        func: Callable,
        priority: int = 5,  # 0 = highest, 10 = lowest
        estimated_tokens: int = 1000,
        *args,
        **kwargs
    ) -> Any:
        """Submit request and wait for result"""
        async with self.lock:
            if len(self.queue) >= self.max_queue_size:
                raise RuntimeError("Queue full")
            
            request = PrioritizedRequest(
                priority=priority,
                timestamp=time.time(),
                func=func,
                args=args,
                kwargs=kwargs,
                estimated_tokens=estimated_tokens
            )
            heapq.heappush(self.queue, request)
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_queue())
        
        return await request.future
    
    async def _process_queue(self):
        """Process queued requests with rate limiting"""
        self.processing = True
        
        while True:
            async with self.lock:
                if not self.queue:
                    self.processing = False
                    return
                request = heapq.heappop(self.queue)
            
            try:
                # Acquire rate limit tokens
                await self.limiter.acquire_async({
                    ResourceType.REQUESTS: 1.0,
                    ResourceType.TOKENS: float(request.estimated_tokens)
                })
                
                # Execute request
                if asyncio.iscoroutinefunction(request.func):
                    result = await request.func(*request.args, **request.kwargs)
                else:
                    result = request.func(*request.args, **request.kwargs)
                
                request.future.set_result(result)
                
            except Exception as e:
                request.future.set_exception(e)
```

**Practical Implications:**

Priority queueing ensures that user-facing requests get processed quickly even when background jobs are running. A user clicking "Generate" shouldn't wait behind 100 batch summary jobs.

**Trade-offs:**

- **Starvation risk:** Low-priority requests can wait indefinitely if high-priority requests keep arriving. Implement aging (gradually increase priority of waiting requests) if needed.
- **Queue memory:** With 1000 queued requests at ~200 bytes each, expect ~200KB memory overhead.
- **Complexity:** Priority queues add latency (heap operations) and can make debugging harder.

### 3. Adaptive Token Estimation

Accurately estimating response tokens prevents over-conservative rate limiting while avoiding rate limit violations.

**Technical Implementation:**

```python
from collections import defaultdict
from typing import Dict, Tuple
import statistics

class AdaptiveTokenEstimator:
    """Learn token consumption patterns to improve estimates"""
    
    def __init__(self, percentile: float = 0.90):
        self.percentile = percentile
        # Store recent observations per endpoint/model
        self.observations: Dict[str, list[Tuple[int, int]]] = defaultdict(list)
        self.max_observations = 100
    
    def estimate(
        self,
        prompt_tokens: int,
        endpoint: str = "default",
        max_response_tokens: Optional[int] = None
    ) -> int:
        """Estimate total tokens (prompt + response)"""
        if endpoint not in self.observations or not self.observations[endpoint]:
            # No data: conservative default
            base_estimate = prompt_tokens + (max_response_tokens or 2000)
            return int(base_estimate * 1.2)  # 20% buffer
        
        # Calculate ratio of response/prompt from observations
        ratios = [
            resp / prompt if prompt > 0 else 1.0
            for prompt, resp in self.observations[endpoint]
        ]
        
        # Use high percentile to avoid underestimation
        if len(ratios) >= 10:
            ratio_estimate = statistics.quantiles(ratios, n=100)[int(self.percentile * 100)]
        else:
            ratio_estimate = max(ratios) * 1.1  # Small sample: use max + buffer
        
        estimated_response = int(prompt_tokens * ratio_estimate)
        
        # Respect max_tokens if set
        if max_response_tokens:
            estimated_response = min(estimated_response, max_response_tokens)
        
        return prompt_tokens + estimated_response
    
    def observe(
        self,
        prompt_tokens: int,
        response_tokens: int,
        endpoint: str = "default"
    ):
        """Record actual token usage"""
        obs = self.observations[endpoint]
        obs.append((prompt_tokens, response_tokens))
        
        # Keep only recent observations
        if len(obs) > self.max_observations:
            obs.pop(0)
    
    def get_stats(self, endpoint: str = "default") -> Dict[str, float]:
        """Get statistics for debugging"""
        if endpoint not in self.observations:
            return {}
        
        ratios = [
            resp / prompt if prompt > 0 else 0
            for prompt, resp in self.observations[endpoint]
        ]
        
        if not ratios:
            return {}
        
        return {
            "mean_ratio": statistics.mean(ratios),
            "median_ratio": statistics.median(ratios),
            "p90_ratio": statistics.quantiles(ratios, n=100)[89] if len(ratios) >= 10 else max(ratios),
            "sample_count": len(ratios)
        }

# Usage
estimator = AdaptiveTokenEstimator(percentile=0.90)

# Initial request (no data)
estimate1 = estimator.estimate(prompt_tokens=100, endpoint="gpt-4")
# Returns ~2120 (conservative)

# After observing actual usage
estimator.observe(prompt_tokens=100, response_tokens=300, endpoint="gpt-4")
estimator.observe(prompt_tokens=150, response_tokens=450, endpoint="gpt-4")

#