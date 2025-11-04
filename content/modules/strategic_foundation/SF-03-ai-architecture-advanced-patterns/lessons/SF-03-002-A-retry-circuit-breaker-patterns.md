# Retry & Circuit Breaker Patterns for LLM Applications

## Core Concepts

LLM APIs are fundamentally different from traditional REST APIs. They're computationally expensive, have variable latency (2-30+ seconds), fail in unpredictable ways (rate limits, token limits, transient errors, model overload), and cost money per token. Naive retry logic that works for database queries will bankrupt you or cascade failures across your system.

**Traditional vs. LLM-Aware Resilience:**

```python
# Traditional: Simple exponential backoff
import time
from typing import TypeVar, Callable

T = TypeVar('T')

def traditional_retry(func: Callable[[], T], max_attempts: int = 3) -> T:
    """Works for databases, fails spectacularly for LLMs"""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise RuntimeError("Unreachable")

# LLM-aware: Selective retry with cost/latency awareness
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONTEXT_LENGTH = "context_length"
    CONTENT_FILTER = "content_filter"
    SERVER_ERROR = "server_error"
    AUTHENTICATION = "authentication"

@dataclass
class RetryDecision:
    should_retry: bool
    wait_seconds: float
    strategy: str
    cost_estimate: float

def llm_retry_decision(
    error: Exception,
    attempt: int,
    tokens_sent: int,
    cost_per_1k_tokens: float
) -> RetryDecision:
    """Context-aware retry logic for LLM calls"""
    error_type = classify_error(error)
    
    # Never retry these - they'll fail again
    if error_type in [ErrorType.CONTEXT_LENGTH, ErrorType.CONTENT_FILTER, 
                      ErrorType.AUTHENTICATION]:
        return RetryDecision(False, 0, "no_retry", 0)
    
    # Rate limits: wait for the specified duration
    if error_type == ErrorType.RATE_LIMIT:
        wait_time = extract_retry_after(error) or (60 * attempt)
        cost = (tokens_sent / 1000) * cost_per_1k_tokens
        return RetryDecision(True, wait_time, "rate_limit_backoff", cost)
    
    # Transient errors: exponential backoff with jitter
    if error_type in [ErrorType.TIMEOUT, ErrorType.SERVER_ERROR]:
        if attempt > 3:  # Don't retry forever - LLM calls are expensive
            return RetryDecision(False, 0, "max_attempts", 0)
        wait_time = min(2 ** attempt + random.uniform(0, 1), 30)
        cost = (tokens_sent / 1000) * cost_per_1k_tokens * attempt
        return RetryDecision(True, wait_time, "exponential_jitter", cost)
    
    return RetryDecision(False, 0, "unknown_error", 0)
```

**Key Insight:** Retry patterns for LLMs must be cost-aware, error-selective, and fast-failing. A single retry of a 100K token context costs real money. Rate limit errors need different handling than transient failures. Context length errors will never succeed on retry.

**Why This Matters Now:**

1. **Cost Explosion:** A bug that retries context-length errors 10 times can cost $1000+ before you notice
2. **Cascade Failures:** LLM services throttle aggressively; naive retries amplify load during outages
3. **User Experience:** 30-second timeouts with 3 retries = 90 seconds of user waiting
4. **Production Reliability:** LLMs have 95-99% uptime (not 99.99% like databases); handling failures is not optional

## Technical Components

### 1. Error Classification and Selective Retry

Not all LLM errors are retryable. Misidentifying error types wastes money and time.

```python
import re
from typing import Optional
from http import HTTPStatus

class LLMError(Exception):
    """Base exception for LLM-related errors"""
    def __init__(self, message: str, error_type: ErrorType, 
                 retry_after: Optional[float] = None):
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after

def classify_error(error: Exception) -> ErrorType:
    """
    Classify errors into actionable categories.
    
    Critical distinction: Permanent vs transient failures.
    """
    error_str = str(error).lower()
    
    # Check HTTP status if available
    if hasattr(error, 'status_code'):
        status = error.status_code
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            return ErrorType.RATE_LIMIT
        elif status == HTTPStatus.REQUEST_TIMEOUT:
            return ErrorType.TIMEOUT
        elif status in [HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN]:
            return ErrorType.AUTHENTICATION
        elif 500 <= status < 600:
            return ErrorType.SERVER_ERROR
    
    # Pattern matching on error messages
    if any(term in error_str for term in ['rate limit', 'quota', 'throttl']):
        return ErrorType.RATE_LIMIT
    elif any(term in error_str for term in ['timeout', 'timed out', 'deadline']):
        return ErrorType.TIMEOUT
    elif any(term in error_str for term in ['context length', 'token limit', 
                                             'maximum context', 'too long']):
        return ErrorType.CONTEXT_LENGTH
    elif any(term in error_str for term in ['content filter', 'content policy', 
                                             'safety', 'inappropriate']):
        return ErrorType.CONTENT_FILTER
    elif any(term in error_str for term in ['internal error', 'server error']):
        return ErrorType.SERVER_ERROR
    
    return ErrorType.SERVER_ERROR  # Conservative default: treat as transient

def extract_retry_after(error: Exception) -> Optional[float]:
    """Extract retry-after hint from error (seconds)"""
    if hasattr(error, 'headers') and 'retry-after' in error.headers:
        try:
            return float(error.headers['retry-after'])
        except (ValueError, TypeError):
            pass
    
    # Parse from error message: "Try again in 20s"
    error_str = str(error)
    match = re.search(r'try again in (\d+)s', error_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    match = re.search(r'retry after (\d+)', error_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    return None
```

**Practical Implications:**
- **Permanent failures** (context length, content filter): Fail immediately, log for debugging
- **Rate limits**: Honor the retry-after header; ignoring it compounds the problem
- **Transient failures**: Retry with backoff, but cap attempts at 3-4 for cost control

**Real Constraints:**
- Error message formats vary between providers; pattern matching is brittle
- Some providers return 500 for rate limits (non-standard)
- Authentication errors might appear as 401 or buried in JSON responses

### 2. Adaptive Backoff Strategies

Static exponential backoff (1s, 2s, 4s...) doesn't account for LLM-specific dynamics: rate limit windows, request correlation, and cost.

```python
import random
from typing import Protocol
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta

class BackoffStrategy(Protocol):
    def next_wait(self, attempt: int, error_type: ErrorType) -> float:
        """Calculate wait time for next retry attempt"""
        ...

@dataclass
class AdaptiveBackoff:
    """
    Backoff that learns from rate limit patterns.
    
    Tracks recent rate limits to predict optimal wait times.
    """
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    
    # Track recent rate limit windows
    _rate_limit_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def next_wait(self, attempt: int, error_type: ErrorType) -> float:
        if error_type == ErrorType.RATE_LIMIT:
            # Learn from history: if we're hitting rate limits frequently,
            # back off more aggressively
            recent_rate_limits = sum(1 for _, err in self._rate_limit_history 
                                    if err == ErrorType.RATE_LIMIT)
            multiplier = 1 + (recent_rate_limits * 0.5)  # 1x, 1.5x, 2x...
            base = self.base_delay * (2 ** attempt) * multiplier
        else:
            # Standard exponential backoff for transient errors
            base = self.base_delay * (2 ** attempt)
        
        wait = min(base, self.max_delay)
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            jitter_amount = wait * 0.3
            wait += random.uniform(-jitter_amount, jitter_amount)
        
        self._rate_limit_history.append((datetime.now(), error_type))
        return max(0, wait)

@dataclass
class TokenAwareBackoff:
    """
    Adjust backoff based on request size - larger contexts = longer waits.
    
    Rationale: Large requests consume more rate limit quota.
    """
    base_delay: float = 1.0
    max_delay: float = 60.0
    tokens_per_second_threshold: int = 100_000
    
    def next_wait(self, attempt: int, error_type: ErrorType, 
                  tokens_sent: int = 0) -> float:
        base = self.base_delay * (2 ** attempt)
        
        # If sending large contexts, wait longer after rate limits
        if error_type == ErrorType.RATE_LIMIT and tokens_sent > 50_000:
            # Estimate: if we sent 100K tokens and limit is 100K/min,
            # we need to wait ~60s
            estimated_wait = (tokens_sent / self.tokens_per_second_threshold) * 60
            base = max(base, estimated_wait)
        
        return min(base + random.uniform(0, 2), self.max_delay)

@dataclass  
class CostAwareBackoff:
    """
    Factor in cumulative cost when deciding retry waits.
    
    After spending $X on retries, fail faster to prevent runaway costs.
    """
    base_delay: float = 1.0
    max_delay: float = 30.0  # Shorter than normal
    cost_threshold: float = 1.0  # Fail faster after $1 in retries
    cumulative_retry_cost: float = 0.0
    
    def next_wait(self, attempt: int, error_type: ErrorType,
                  attempt_cost: float = 0.0) -> float:
        self.cumulative_retry_cost += attempt_cost
        
        # After spending too much, give up quickly
        if self.cumulative_retry_cost > self.cost_threshold:
            return 0.0  # Signal to stop retrying
        
        base = self.base_delay * (2 ** attempt)
        return min(base + random.uniform(0, 1), self.max_delay)
```

**Practical Implications:**
- **Adaptive learning**: If you're hitting rate limits every 5 minutes, your backoff should reflect that pattern
- **Token awareness**: A 100K token request that fails should wait longer than a 1K token request
- **Cost protection**: After $1 in failed retries, something is structurally wrong—fail fast

**Trade-offs:**
- More sophisticated = more state to track
- Learning from history requires persistent storage across requests
- Cost tracking needs accurate token counting (which itself can fail)

### 3. Circuit Breaker Implementation

Circuit breakers prevent cascading failures when an LLM service degrades. Unlike retries (per-request), circuit breakers track system-level health.

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional, Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitMetrics:
    """Track health metrics for circuit breaker decisions"""
    total_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_cost: float = 0.0  # Track cost of failures
    
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

class LLMCircuitBreaker:
    """
    Circuit breaker specifically designed for LLM API calls.
    
    Key differences from standard circuit breakers:
    1. Cost-aware: Track money wasted on failures
    2. Error-selective: Only count certain errors as "failures"
    3. Slower recovery: LLM services take time to recover
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,          # Open after N failures
        failure_rate_threshold: float = 0.5,  # Or 50% failure rate
        recovery_timeout: float = 60.0,       # Wait 60s before testing
        half_open_max_calls: int = 3,         # Test with 3 calls
        cost_threshold: float = 10.0,         # Open if failures cost $10+
    ):
        self.failure_threshold = failure_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.cost_threshold = cost_threshold
        
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self._lock = Lock()
        self._half_open_calls = 0
    
    def call(self, func: Callable[[], Any], call_cost: float = 0.0) -> Any:
        """
        Execute function through circuit breaker.
        
        Raises CircuitBreakerOpen if circuit is open.
        """
        with self._lock:
            # Check if circuit should transition states
            self._update_state()
            
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpen(
                    f"Circuit open: {self.metrics.consecutive_failures} "
                    f"consecutive failures, ${self.metrics.total_cost:.2f} wasted"
                )
            
            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen("Half-open call limit reached")
                self._half_open_calls += 1
        
        # Execute outside lock to prevent blocking
        try:
            result = func()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e, call_cost)
            raise
    
    def _update_state(self) -> None:
        """Transition between circuit states"""
        if self.state == CircuitState.CLOSED:
            # Open if thresholds exceeded
            if (self.metrics.consecutive_failures >= self.failure_threshold or
                self.metrics.failure_rate() >= self.failure_rate_threshold or
                self.metrics.total_cost >= self.