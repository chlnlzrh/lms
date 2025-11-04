# Connection Health Monitoring for LLM Applications

## Core Concepts

Connection health monitoring in LLM applications refers to the continuous tracking, validation, and recovery mechanisms for network connections between your application and AI service endpoints. Unlike traditional API monitoring that focuses on basic uptime checks, LLM connection health monitoring must account for token streaming interruptions, partial response handling, rate limit state tracking, and multi-turn conversation context preservation across network failures.

### Traditional vs. Modern Approach

**Traditional API Health Monitoring:**
```python
import requests
from typing import Dict, Any

def traditional_health_check(endpoint: str) -> Dict[str, Any]:
    """Simple ping-style health check"""
    try:
        response = requests.get(f"{endpoint}/health", timeout=5)
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "latency_ms": response.elapsed.total_seconds() * 1000
        }
    except requests.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}

# This misses: streaming failures, rate limits, partial responses,
# token budget exhaustion, context window issues
```

**LLM Connection Health Monitoring:**
```python
import asyncio
import time
from typing import Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import aiohttp

class ConnectionState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RATE_LIMITED = "rate_limited"

@dataclass
class LLMConnectionHealth:
    state: ConnectionState
    latency_p50_ms: float
    latency_p99_ms: float
    stream_completion_rate: float  # % of streams that complete without interruption
    token_throughput: float  # tokens per second
    rate_limit_remaining: Optional[int]
    last_successful_request: float
    consecutive_failures: int
    context_preservation_rate: float  # % of multi-turn conversations maintained
    
    def is_acceptable(self) -> bool:
        """Multi-dimensional health assessment"""
        return (
            self.state in [ConnectionState.HEALTHY, ConnectionState.DEGRADED]
            and self.latency_p99_ms < 10000  # 10s max
            and self.stream_completion_rate > 0.95  # 95% streams complete
            and self.consecutive_failures < 3
        )

class LLMConnectionMonitor:
    def __init__(self, endpoint: str, check_interval: float = 30.0):
        self.endpoint = endpoint
        self.check_interval = check_interval
        self.latencies: list[float] = []
        self.stream_results: list[bool] = []  # True if stream completed
        self.tokens_streamed: list[tuple[int, float]] = []  # (tokens, duration)
        self.consecutive_failures = 0
        
    async def stream_health_check(self) -> Dict[str, Any]:
        """Test actual streaming capability, not just endpoint availability"""
        start = time.time()
        tokens_received = 0
        stream_completed = False
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "messages": [{"role": "user", "content": "Count to 10"}],
                    "stream": True,
                    "max_tokens": 50
                }
                
                async with session.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return self._record_failure(response.status)
                    
                    # Track rate limits from headers
                    rate_limit_remaining = response.headers.get('x-ratelimit-remaining')
                    
                    async for chunk in response.content.iter_any():
                        if chunk:
                            tokens_received += 1
                    
                    stream_completed = True
                    duration = time.time() - start
                    
                    self._record_success(duration, tokens_received, stream_completed)
                    
                    return {
                        "status": "healthy",
                        "latency_ms": duration * 1000,
                        "tokens": tokens_received,
                        "rate_limit_remaining": rate_limit_remaining
                    }
                    
        except asyncio.TimeoutError:
            return self._record_failure("timeout")
        except Exception as e:
            return self._record_failure(str(e))
    
    def _record_success(self, duration: float, tokens: int, completed: bool):
        self.consecutive_failures = 0
        self.latencies.append(duration * 1000)
        self.stream_results.append(completed)
        self.tokens_streamed.append((tokens, duration))
        
        # Keep last 100 measurements
        self.latencies = self.latencies[-100:]
        self.stream_results = self.stream_results[-100:]
        self.tokens_streamed = self.tokens_streamed[-100:]
    
    def _record_failure(self, reason: Any) -> Dict[str, Any]:
        self.consecutive_failures += 1
        self.stream_results.append(False)
        return {"status": "unhealthy", "reason": reason}
    
    def get_health(self) -> LLMConnectionHealth:
        """Calculate comprehensive health metrics"""
        if not self.latencies:
            return LLMConnectionHealth(
                state=ConnectionState.UNHEALTHY,
                latency_p50_ms=0, latency_p99_ms=0,
                stream_completion_rate=0, token_throughput=0,
                rate_limit_remaining=None, last_successful_request=0,
                consecutive_failures=self.consecutive_failures,
                context_preservation_rate=0
            )
        
        sorted_latencies = sorted(self.latencies)
        p50_idx = int(len(sorted_latencies) * 0.5)
        p99_idx = int(len(sorted_latencies) * 0.99)
        
        stream_rate = sum(self.stream_results) / len(self.stream_results)
        
        # Calculate token throughput
        total_tokens = sum(t[0] for t in self.tokens_streamed)
        total_duration = sum(t[1] for t in self.tokens_streamed)
        throughput = total_tokens / total_duration if total_duration > 0 else 0
        
        # Determine state
        if self.consecutive_failures >= 3:
            state = ConnectionState.UNHEALTHY
        elif stream_rate < 0.95 or sorted_latencies[p99_idx] > 10000:
            state = ConnectionState.DEGRADED
        else:
            state = ConnectionState.HEALTHY
        
        return LLMConnectionHealth(
            state=state,
            latency_p50_ms=sorted_latencies[p50_idx],
            latency_p99_ms=sorted_latencies[p99_idx],
            stream_completion_rate=stream_rate,
            token_throughput=throughput,
            rate_limit_remaining=None,  # Would be populated from last response
            last_successful_request=time.time() if self.consecutive_failures == 0 else 0,
            consecutive_failures=self.consecutive_failures,
            context_preservation_rate=1.0  # Simplified for example
        )
```

The critical difference: traditional monitoring tells you if an endpoint responds; LLM health monitoring tells you if your application can successfully complete streaming conversations under real conditions.

### Key Insights

**1. Streaming Failures Are Silent**: A 200 status code doesn't guarantee your stream will complete. Network interruptions, load balancer timeouts, and token budget exhaustion can all cause partial responses. You must monitor stream completion rates, not just HTTP status codes.

**2. Rate Limits Are Stateful**: Unlike traditional APIs where rate limits reset predictably, LLM rate limits often involve multiple dimensions (requests per minute, tokens per minute, concurrent requests). Your monitor must track rate limit state and predict exhaustion before it happens.

**3. Context Corruption Is Invisible**: If a multi-turn conversation loses connection mid-stream, you might lose conversation context. Reconnection doesn't automatically restore state. Health monitoring must verify context preservation across failures.

**4. Latency Distribution Matters More Than Average**: LLM responses have extreme latency variance. A p50 of 1 second with a p99 of 30 seconds will destroy user experience. Monitor percentiles, not means.

### Why This Matters Now

As of 2024, LLM applications are moving from experimental to production-critical. Users expect real-time responsiveness, but LLM endpoints have fundamentally different failure modes than traditional APIs:

- **Token streaming interruptions** are common but not always reported as errors
- **Rate limit exhaustion** can happen mid-conversation without warning
- **Context window overflow** causes silent quality degradation
- **Provider outages** require instant failover to backup providers

Without proper connection health monitoring, you'll discover these failures from user complaints, not your systems. By the time you notice degradation, you've already lost users.

## Technical Components

### 1. Stream Completion Tracking

**Technical Explanation:**  
Stream completion tracking monitors whether token streams from LLM endpoints successfully complete without interruption. Unlike request-response APIs where success is binary (200 or error), streaming responses can fail partially—you receive some tokens, then the connection drops. This requires tracking the entire lifecycle of each stream and maintaining historical completion rates.

**Practical Implications:**  
A stream completion rate below 95% indicates serious problems, but the cause varies:
- Network infrastructure (load balancers timing out long requests)
- Provider issues (service degradation)
- Client-side problems (application timeout configuration)
- Token budget exhaustion (hitting max_tokens mid-sentence)

**Constraints and Trade-offs:**  
Tracking every stream adds memory overhead. For high-volume applications, you must sample streams (e.g., 10% of requests) or use fixed-size circular buffers. The trade-off: sampling can miss brief outages, while full tracking can consume significant memory at scale.

**Concrete Example:**
```python
import asyncio
from collections import deque
from typing import Optional, Callable, Any
from dataclasses import dataclass
import time

@dataclass
class StreamMetrics:
    stream_id: str
    started_at: float
    completed_at: Optional[float]
    tokens_received: int
    expected_tokens: Optional[int]
    interrupted: bool
    error: Optional[str]

class StreamCompletionTracker:
    def __init__(self, max_history: int = 1000):
        self.streams: deque[StreamMetrics] = deque(maxlen=max_history)
        self.active_streams: dict[str, StreamMetrics] = {}
    
    def start_stream(self, stream_id: str, expected_tokens: Optional[int] = None) -> StreamMetrics:
        """Begin tracking a new stream"""
        metrics = StreamMetrics(
            stream_id=stream_id,
            started_at=time.time(),
            completed_at=None,
            tokens_received=0,
            expected_tokens=expected_tokens,
            interrupted=False,
            error=None
        )
        self.active_streams[stream_id] = metrics
        return metrics
    
    def record_token(self, stream_id: str):
        """Record receipt of a token"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id].tokens_received += 1
    
    def complete_stream(self, stream_id: str, success: bool = True, error: Optional[str] = None):
        """Mark stream as completed (successfully or not)"""
        if stream_id not in self.active_streams:
            return
        
        metrics = self.active_streams.pop(stream_id)
        metrics.completed_at = time.time()
        metrics.interrupted = not success
        metrics.error = error
        
        self.streams.append(metrics)
    
    def get_completion_rate(self, window_seconds: Optional[float] = None) -> float:
        """Calculate stream completion rate"""
        now = time.time()
        relevant_streams = [
            s for s in self.streams
            if window_seconds is None or (now - s.started_at) <= window_seconds
        ]
        
        if not relevant_streams:
            return 1.0
        
        completed = sum(1 for s in relevant_streams if not s.interrupted)
        return completed / len(relevant_streams)
    
    def get_interruption_patterns(self) -> dict[str, Any]:
        """Analyze what's causing interruptions"""
        recent_failures = [s for s in self.streams if s.interrupted]
        
        if not recent_failures:
            return {"interruptions": 0}
        
        # Analyze token counts at failure
        tokens_at_failure = [s.tokens_received for s in recent_failures]
        avg_tokens_before_failure = sum(tokens_at_failure) / len(tokens_at_failure)
        
        # Check if failures happen at specific token counts (suggests limits)
        token_clusters = {}
        for tokens in tokens_at_failure:
            bucket = (tokens // 100) * 100  # Group by hundreds
            token_clusters[bucket] = token_clusters.get(bucket, 0) + 1
        
        return {
            "interruptions": len(recent_failures),
            "avg_tokens_before_failure": avg_tokens_before_failure,
            "token_failure_clusters": token_clusters,
            "error_types": {
                err: sum(1 for s in recent_failures if s.error == err)
                for err in set(s.error for s in recent_failures if s.error)
            }
        }
    
    def cleanup_stale_streams(self, timeout_seconds: float = 60):
        """Mark streams that haven't completed as interrupted"""
        now = time.time()
        stale_ids = [
            stream_id for stream_id, metrics in self.active_streams.items()
            if now - metrics.started_at > timeout_seconds
        ]
        
        for stream_id in stale_ids:
            self.complete_stream(stream_id, success=False, error="timeout")

# Usage example
async def monitored_stream_request(tracker: StreamCompletionTracker, stream_id: str):
    """Example of wrapping a stream request with monitoring"""
    metrics = tracker.start_stream(stream_id, expected_tokens=100)
    
    try:
        # Simulate streaming response
        async for token in simulate_llm_stream():
            tracker.record_token(stream_id)
            yield token
        
        tracker.complete_stream(stream_id, success=True)
        
    except asyncio.TimeoutError:
        tracker.complete_stream(stream_id, success=False, error="timeout")
        raise
    except Exception as e:
        tracker.complete_stream(stream_id, success=False, error=type(e).__name__)
        raise

async def simulate_llm_stream():
    """Simulate token stream for testing"""
    for i in range(50):
        await asyncio.sleep(0.1)
        yield f"token_{i}"
```

### 2. Multi-Dimensional Rate Limit Tracking

**Technical Explanation:**  
LLM providers enforce rate limits across multiple dimensions simultaneously: requests per minute (RPM), tokens per minute (TPM), and concurrent requests. Traditional rate limiting tracks one counter; LLM rate limiting requires tracking multiple synchronized counters with different reset windows. Additionally, rate limits are often returned in response headers and can change dynamically based on your usage tier or provider load.

**Practical Implications:**  
You can have remaining RPM capacity but be blocked by TPM exhaustion, or vice versa. Naive retry logic that only checks HTTP 429 status codes will cause cascading failures. You need predictive rate limit tracking that prevents requests before hitting limits, not reactive handling after the fact.

**Constraints and Trade-offs:**  
Accurate rate limit tracking requires storing request history to calculate rolling windows. The trade-off: memory usage grows with request volume. For high-throughput applications, you must either sample requests, use approximation algorithms (like token bucket), or accept some inaccuracy in limit prediction.

**Concrete Example:**
```python
from dataclasses import dataclass
from typing import Dict, Optional
import time
from collections import deque

@dataclass
class RateLimitBucket:
    """Tracks a single dimension of rate limiting"""
    capacity: