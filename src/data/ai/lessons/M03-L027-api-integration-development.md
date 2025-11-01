# API Integration Development for LLM Systems

## Core Concepts

API integration in LLM systems represents a fundamental shift from traditional deterministic APIs. Instead of simple request-response patterns with predictable outputs, you're working with probabilistic systems that require retry logic, streaming responses, token management, and cost tracking as first-class concerns.

### Traditional vs. Modern API Integration

**Traditional REST API (deterministic):**

```python
import requests
from typing import Dict, Any

def get_user_profile(user_id: int) -> Dict[str, Any]:
    """Traditional API: predictable, fast, fixed response structure"""
    response = requests.get(
        f"https://api.example.com/users/{user_id}",
        timeout=5
    )
    response.raise_for_status()
    return response.json()

# Single call, consistent timing (~100ms), fixed cost
profile = get_user_profile(123)
print(profile["name"])  # Always same format
```

**LLM API (probabilistic):**

```python
import requests
import time
from typing import Optional, Generator
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    tokens_used: int
    latency_ms: float
    cost_usd: float

def call_llm_api(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
    timeout: int = 30
) -> Optional[LLMResponse]:
    """
    LLM API: probabilistic, variable timing, token-based cost,
    potential for streaming, rate limits, and failures
    """
    start = time.time()
    
    try:
        response = requests.post(
            "https://api.llm-provider.com/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=timeout
        )
        response.raise_for_status()
        
        data = response.json()
        latency = (time.time() - start) * 1000
        
        # Cost calculation (example: $0.002 per 1K tokens)
        total_tokens = data["usage"]["total_tokens"]
        cost = (total_tokens / 1000) * 0.002
        
        return LLMResponse(
            content=data["choices"][0]["text"],
            tokens_used=total_tokens,
            latency_ms=latency,
            cost_usd=cost
        )
    except requests.exceptions.Timeout:
        print(f"Request timeout after {timeout}s")
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("Rate limit exceeded")
        return None

# Variable timing (500ms-10s+), unpredictable output, usage-based cost
result = call_llm_api("Summarize the benefits of async programming")
if result:
    print(f"Response: {result.content}")
    print(f"Cost: ${result.cost_usd:.4f}, Latency: {result.latency_ms:.0f}ms")
```

### Key Engineering Shifts

1. **From Deterministic to Probabilistic**: Same input can produce different outputs. Your integration must handle output validation and potential retries with adjusted parameters.

2. **From Fixed to Variable Cost**: Every API call costs money based on token usage. Cost tracking and budget limits become architectural requirements, not afterthoughts.

3. **From Fast to Slow**: LLM APIs can take seconds to respond. Synchronous blocking calls will destroy your application's responsiveness—asynchronous patterns become mandatory.

4. **From Simple to Stateful**: Context windows mean you're managing conversation history, token counts across messages, and deciding what to keep or truncate.

### Why This Matters Now

Traditional integration patterns fail catastrophically with LLM APIs. A naive synchronous implementation can:
- Block your application for 10+ seconds per request
- Exhaust your API budget in hours without monitoring
- Fail silently when rate-limited, losing user requests
- Scale costs linearly (or worse) without optimization

Engineering robust LLM integrations requires treating tokens as a metered resource, implementing defensive retry strategies, and building observability into every request.

## Technical Components

### 1. Token Management & Cost Control

Tokens are the fundamental unit of LLM APIs—not just for billing, but for capacity limits. Every API has a maximum context window (e.g., 8K, 32K, 128K tokens), and you pay for both input and output tokens.

**Technical Explanation:**

Text is encoded into tokens (roughly 0.75 words per token in English). You must track token usage across both the prompt (input) and completion (output) to stay within budget and context limits.

```python
from typing import List, Tuple
import tiktoken

class TokenBudgetManager:
    """Manage token budgets across multiple API calls"""
    
    def __init__(
        self,
        max_tokens_per_request: int = 4000,
        cost_per_1k_input: float = 0.0015,
        cost_per_1k_output: float = 0.002,
        daily_budget_usd: float = 10.0
    ):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens_per_request
        self.input_cost = cost_per_1k_input
        self.output_cost = cost_per_1k_output
        self.daily_budget = daily_budget_usd
        self.spent_today = 0.0
        
    def count_tokens(self, text: str) -> int:
        """Accurate token counting using the actual tokenizer"""
        return len(self.encoder.encode(text))
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost before making API call"""
        input_cost = (input_tokens / 1000) * self.input_cost
        output_cost = (output_tokens / 1000) * self.output_cost
        return input_cost + output_cost
    
    def can_afford(self, estimated_tokens: int) -> Tuple[bool, str]:
        """Check if request fits budget"""
        estimated_cost = self.estimate_cost(estimated_tokens, estimated_tokens)
        
        if self.spent_today + estimated_cost > self.daily_budget:
            return False, f"Would exceed daily budget (${self.spent_today:.2f}/${self.daily_budget:.2f})"
        
        if estimated_tokens > self.max_tokens:
            return False, f"Exceeds max tokens ({estimated_tokens} > {self.max_tokens})"
        
        return True, "OK"
    
    def record_usage(self, input_tokens: int, output_tokens: int) -> float:
        """Track actual usage and cost"""
        cost = self.estimate_cost(input_tokens, output_tokens)
        self.spent_today += cost
        return cost

# Usage example
budget_mgr = TokenBudgetManager(daily_budget_usd=5.0)

prompt = "Explain quantum computing in simple terms"
input_tokens = budget_mgr.count_tokens(prompt)
max_output = 300

can_proceed, reason = budget_mgr.can_afford(input_tokens + max_output)

if can_proceed:
    # Make API call
    result = call_llm_api(prompt, max_tokens=max_output)
    if result:
        cost = budget_mgr.record_usage(input_tokens, result.tokens_used - input_tokens)
        print(f"Cost: ${cost:.4f}, Budget remaining: ${budget_mgr.daily_budget - budget_mgr.spent_today:.2f}")
else:
    print(f"Request blocked: {reason}")
```

**Practical Implications:**
- Pre-calculate token counts before API calls to enforce limits
- Implement per-user or per-session budgets to prevent abuse
- Use different token limits for different request types (quick vs. detailed)

**Trade-offs:**
- Accurate token counting requires the same tokenizer as the API (library dependency)
- Estimating output tokens is imprecise—use conservative limits
- Budget tracking requires persistent storage across application restarts

### 2. Retry Logic with Exponential Backoff

LLM APIs fail for multiple reasons: rate limits (429), temporary errors (503), timeouts, or overload. Unlike traditional APIs, you'll hit rate limits regularly under normal operation.

**Technical Explanation:**

Implement exponential backoff with jitter to handle transient failures and rate limits without overwhelming the API. Different error types require different strategies.

```python
import time
import random
from typing import Optional, Callable, TypeVar, Any
from enum import Enum

T = TypeVar('T')

class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    IMMEDIATE = "immediate"

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        retryable_status_codes: set = {429, 500, 502, 503, 504}
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.retryable_codes = retryable_status_codes

def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with jitter to prevent thundering herd"""
    if config.strategy == RetryStrategy.EXPONENTIAL:
        delay = min(config.base_delay * (2 ** attempt), config.max_delay)
    elif config.strategy == RetryStrategy.LINEAR:
        delay = min(config.base_delay * (attempt + 1), config.max_delay)
    else:  # IMMEDIATE
        delay = 0
    
    # Add jitter: ±25% randomization
    jitter = delay * 0.25 * (random.random() * 2 - 1)
    return max(0, delay + jitter)

def retry_with_backoff(
    func: Callable[..., T],
    config: RetryConfig,
    *args,
    **kwargs
) -> Optional[T]:
    """
    Retry function with exponential backoff and jitter.
    Handles rate limits and transient failures intelligently.
    """
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
            
        except requests.exceptions.HTTPError as e:
            last_exception = e
            status_code = e.response.status_code
            
            # Don't retry client errors (except rate limits)
            if status_code < 500 and status_code != 429:
                print(f"Non-retryable error: {status_code}")
                raise
            
            if status_code not in config.retryable_codes:
                raise
            
            if attempt == config.max_retries:
                print(f"Max retries ({config.max_retries}) exceeded")
                break
            
            # Special handling for rate limits
            if status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    delay = float(retry_after)
                else:
                    delay = calculate_delay(attempt, config)
                print(f"Rate limited. Retrying in {delay:.1f}s...")
            else:
                delay = calculate_delay(attempt, config)
                print(f"Server error {status_code}. Retry {attempt + 1}/{config.max_retries} in {delay:.1f}s")
            
            time.sleep(delay)
            
        except requests.exceptions.Timeout as e:
            last_exception = e
            if attempt == config.max_retries:
                break
            
            delay = calculate_delay(attempt, config)
            print(f"Timeout. Retry {attempt + 1}/{config.max_retries} in {delay:.1f}s")
            time.sleep(delay)
    
    print(f"Request failed after {config.max_retries} retries")
    return None

# Usage
retry_config = RetryConfig(
    max_retries=4,
    base_delay=2.0,
    max_delay=32.0,
    strategy=RetryStrategy.EXPONENTIAL
)

def make_llm_request(prompt: str) -> Optional[str]:
    response = requests.post(
        "https://api.llm-provider.com/v1/completions",
        json={"prompt": prompt, "max_tokens": 100},
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"]

result = retry_with_backoff(
    make_llm_request,
    retry_config,
    "What is machine learning?"
)
```

**Practical Implications:**
- Implement jitter to avoid synchronized retries from multiple clients
- Respect `Retry-After` headers when provided
- Log retry attempts for observability

**Trade-offs:**
- More retries = higher latency but better success rate
- Aggressive backoff = better for API stability but worse user experience
- Must balance between persistence and failing fast

### 3. Streaming Responses

LLM API calls can take 10-30 seconds for long responses. Streaming provides tokens as they're generated, dramatically improving perceived latency and enabling real-time user interfaces.

**Technical Explanation:**

Instead of waiting for the complete response, process Server-Sent Events (SSE) or chunked transfer encoding to receive and display tokens incrementally.

```python
import requests
import json
from typing import Generator, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class StreamMetrics:
    """Track streaming performance metrics"""
    first_token_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_received: int = 0
    chunks_received: int = 0
    started_at: datetime = field(default_factory=datetime.now)

def stream_llm_response(
    prompt: str,
    max_tokens: int = 500,
    on_token: Optional[Callable[[str], None]] = None,
    timeout: int = 60
) -> Generator[str, None, StreamMetrics]:
    """
    Stream LLM response token by token.
    Yields each token as it arrives, returns final metrics.
    """
    metrics = StreamMetrics()
    first_token_received = False
    accumulated_text = ""
    
    try:
        response = requests.post(
            "https://api.llm-provider.com/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True  # Enable streaming
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
            stream=True,  # Critical: don't buffer response
            timeout=timeout
        )
        response.raise_for_status()
        
        # Process Server-Sent Events
        for line in response.iter_lines():
            if not line:
                continue
            
            # SSE format: "data: {json}\n\n"
            line_text = line.decode('utf-8')
            if not line_text.startswith('data: '):
                continue
            
            if line_text.strip() == 'data: [DONE]':
                break
            
            try:
                json_str = line_text[6:]  # Remove "data: " prefix
                chunk = json.loads(json_str)
                
                # Extract token from response
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    
                    if content:
                        if not first_token_received:
                            elapsed = (datetime.now() - metrics.started_at).total_seconds() * 1000
                            metrics.first_token_latency_ms = elapsed
                            first_token_received = True
                        
                        metrics