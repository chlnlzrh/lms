# Error Handling & Resilience in LLM Systems

## Core Concepts

Error handling in LLM systems differs fundamentally from traditional software error handling. In conventional systems, errors are typically deterministic—the same input produces the same error. LLM systems introduce non-deterministic failures, partial successes, and ambiguous failure states that require entirely different resilience strategies.

### Traditional vs. Modern Error Handling

```python
# Traditional API error handling
import requests
from typing import Dict, Any

def fetch_user_data(user_id: int) -> Dict[str, Any]:
    try:
        response = requests.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        # Deterministic: same input = same error
        if e.response.status_code == 404:
            raise UserNotFoundError(f"User {user_id} not found")
        elif e.response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        raise
    except requests.RequestException as e:
        raise NetworkError(f"Network failure: {e}")

# LLM system error handling
import anthropic
from typing import Optional, Tuple
import time
import json

class LLMResponse:
    def __init__(self, content: str, success: bool, 
                 error: Optional[str] = None, metadata: Optional[Dict] = None):
        self.content = content
        self.success = success
        self.error = error
        self.metadata = metadata or {}

def query_llm_resilient(prompt: str, max_retries: int = 3) -> LLMResponse:
    """
    Non-deterministic: same input may produce different errors or successes.
    Must handle: timeouts, rate limits, content policy, malformed output,
    partial responses, quality degradation.
    """
    client = anthropic.Anthropic()
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            # Validate response quality (new concern for LLMs)
            if len(content) < 10:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return LLMResponse("", False, "Response too short", 
                                 {"attempts": attempt + 1})
            
            # Check for refusals (specific to LLMs)
            if "I cannot" in content or "I'm unable" in content:
                return LLMResponse(content, False, "Content policy refusal",
                                 {"attempts": attempt + 1})
            
            return LLMResponse(content, True, None, 
                             {"attempts": attempt + 1})
            
        except anthropic.RateLimitError as e:
            wait_time = 2 ** attempt
            if attempt < max_retries - 1:
                time.sleep(wait_time)
                continue
            return LLMResponse("", False, f"Rate limited after {attempt + 1} attempts",
                             {"attempts": attempt + 1})
        
        except anthropic.APIError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return LLMResponse("", False, f"API error: {str(e)}",
                             {"attempts": attempt + 1})
    
    return LLMResponse("", False, "Max retries exceeded", 
                     {"attempts": max_retries})
```

### Key Engineering Insights

**1. Failure Modes Are Multidimensional:** Traditional systems fail binary (works/doesn't work). LLMs can: succeed technically but fail semantically, produce valid but wrong output, partially complete requests, or degrade quality under load.

**2. Retries Aren't Idempotent:** Retrying the same LLM request may produce different results—both successes and failures. This breaks assumptions about retry safety in distributed systems.

**3. Observability Requires Content Analysis:** HTTP status codes are insufficient. You must analyze response content to detect failures like refusals, hallucinations, or quality degradation.

**4. Timeouts Are Quality Controls:** In traditional systems, timeouts prevent resource exhaustion. In LLM systems, they also prevent low-quality outputs from long-running, confused generation processes.

### Why This Matters Now

Production LLM deployments fail in ways that traditional monitoring doesn't catch. A system can have 99.9% HTTP success rate while delivering garbage to users. Teams that treat LLMs like REST APIs discover failures only through user complaints. Advanced error handling patterns catch these issues before they reach production, reducing incident response time from hours to seconds.

## Technical Components

### 1. Structured Error Classification

LLM errors require taxonomy beyond HTTP status codes. Classification enables appropriate remediation strategies.

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Callable, Any
import re

class ErrorCategory(Enum):
    TRANSIENT_NETWORK = "transient_network"  # Retry immediately
    RATE_LIMIT = "rate_limit"                # Retry with backoff
    CONTENT_POLICY = "content_policy"         # Don't retry, log and alert
    MALFORMED_OUTPUT = "malformed_output"     # Retry with prompt modification
    QUALITY_DEGRADATION = "quality_degradation"  # Retry or fallback
    CONTEXT_LENGTH = "context_length"         # Truncate and retry
    AUTHENTICATION = "authentication"         # Don't retry, alert immediately
    UPSTREAM_FAILURE = "upstream_failure"     # Circuit breaker pattern

@dataclass
class ClassifiedError:
    category: ErrorCategory
    message: str
    recoverable: bool
    retry_strategy: Optional[str]
    metadata: Dict[str, Any]

class ErrorClassifier:
    def __init__(self):
        self.patterns = {
            ErrorCategory.RATE_LIMIT: [
                re.compile(r"rate.*limit", re.IGNORECASE),
                re.compile(r"429", re.IGNORECASE),
                re.compile(r"quota.*exceeded", re.IGNORECASE)
            ],
            ErrorCategory.CONTENT_POLICY: [
                re.compile(r"content.*policy", re.IGNORECASE),
                re.compile(r"cannot.*assist", re.IGNORECASE),
                re.compile(r"unable.*to.*help", re.IGNORECASE)
            ],
            ErrorCategory.CONTEXT_LENGTH: [
                re.compile(r"context.*length", re.IGNORECASE),
                re.compile(r"too.*long", re.IGNORECASE),
                re.compile(r"exceeds.*token", re.IGNORECASE)
            ]
        }
    
    def classify(self, error: Exception, response_content: str = "") -> ClassifiedError:
        error_str = str(error).lower()
        
        # Check exception type first
        if isinstance(error, anthropic.RateLimitError):
            return ClassifiedError(
                category=ErrorCategory.RATE_LIMIT,
                message=str(error),
                recoverable=True,
                retry_strategy="exponential_backoff",
                metadata={"wait_multiplier": 2, "max_wait": 60}
            )
        
        if isinstance(error, anthropic.AuthenticationError):
            return ClassifiedError(
                category=ErrorCategory.AUTHENTICATION,
                message=str(error),
                recoverable=False,
                retry_strategy=None,
                metadata={"alert_severity": "critical"}
            )
        
        # Pattern matching for content analysis
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(error_str) or pattern.search(response_content):
                    return self._create_classified_error(category, error, response_content)
        
        # Default to transient network error
        return ClassifiedError(
            category=ErrorCategory.TRANSIENT_NETWORK,
            message=str(error),
            recoverable=True,
            retry_strategy="immediate",
            metadata={"max_retries": 3}
        )
    
    def _create_classified_error(self, category: ErrorCategory, 
                                 error: Exception, content: str) -> ClassifiedError:
        strategies = {
            ErrorCategory.RATE_LIMIT: ("exponential_backoff", True),
            ErrorCategory.CONTENT_POLICY: (None, False),
            ErrorCategory.CONTEXT_LENGTH: ("truncate_and_retry", True),
            ErrorCategory.MALFORMED_OUTPUT: ("prompt_modification", True)
        }
        
        strategy, recoverable = strategies.get(category, (None, False))
        return ClassifiedError(
            category=category,
            message=str(error),
            recoverable=recoverable,
            retry_strategy=strategy,
            metadata={"original_content": content[:200]}
        )

# Usage example
classifier = ErrorClassifier()

try:
    # LLM call that fails
    response = client.messages.create(...)
except Exception as e:
    classified = classifier.classify(e)
    
    if classified.recoverable:
        if classified.retry_strategy == "exponential_backoff":
            # Implement backoff
            pass
        elif classified.retry_strategy == "truncate_and_retry":
            # Truncate prompt and retry
            pass
    else:
        # Log, alert, fail gracefully
        logger.error(f"Unrecoverable error: {classified.category}")
```

**Practical Implications:** Classification enables automated remediation. A rate limit error triggers exponential backoff, while a content policy error immediately logs and returns a fallback response. Without classification, all errors get the same treatment—usually blind retries that waste time and money.

**Trade-offs:** Pattern matching adds latency (typically 1-5ms) and requires maintenance as error messages evolve. However, the alternative—treating all errors identically—leads to cascading failures and poor user experience.

### 2. Circuit Breakers for LLM Services

Circuit breakers prevent cascading failures when LLM providers degrade. Unlike traditional circuit breakers that track binary failures, LLM circuit breakers must track quality degradation.

```python
from datetime import datetime, timedelta
from collections import deque
from threading import Lock
from typing import Deque, Optional

class LLMCircuitBreaker:
    def __init__(self, 
                 failure_threshold: int = 5,
                 success_threshold: int = 2,
                 timeout_seconds: int = 60,
                 quality_threshold: float = 0.7):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.quality_threshold = quality_threshold
        
        # Track failures and quality scores in sliding window
        self.failure_count = 0
        self.success_count = 0
        self.quality_scores: Deque[float] = deque(maxlen=10)
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.opened_at: Optional[datetime] = None
        self.lock = Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self.lock:
            if self.state == "OPEN":
                if datetime.now() - self.opened_at > timedelta(seconds=self.timeout_seconds):
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker open, retry after {self.timeout_seconds}s"
                    )
        
        try:
            result = func(*args, **kwargs)
            
            # Evaluate response quality
            quality_score = self._evaluate_quality(result)
            
            with self.lock:
                self.quality_scores.append(quality_score)
                
                if quality_score >= self.quality_threshold:
                    self.success_count += 1
                    self.failure_count = 0
                    
                    if self.state == "HALF_OPEN" and self.success_count >= self.success_threshold:
                        self.state = "CLOSED"
                        self.opened_at = None
                else:
                    self.failure_count += 1
                    self.success_count = 0
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                        self.opened_at = datetime.now()
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.success_count = 0
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    self.opened_at = datetime.now()
            
            raise
    
    def _evaluate_quality(self, result: Any) -> float:
        """
        Evaluate response quality. Customize based on your use case.
        Returns score 0.0-1.0.
        """
        if not isinstance(result, LLMResponse):
            return 0.0
        
        if not result.success:
            return 0.0
        
        content = result.content
        score = 1.0
        
        # Length check
        if len(content) < 50:
            score -= 0.3
        
        # Refusal patterns
        refusal_patterns = ["I cannot", "I'm unable", "I don't have access"]
        if any(pattern in content for pattern in refusal_patterns):
            score -= 0.5
        
        # Repetition check
        words = content.split()
        if len(words) > 10 and len(set(words)) / len(words) < 0.5:
            score -= 0.3
        
        return max(0.0, score)
    
    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            avg_quality = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "average_quality": avg_quality,
                "opened_at": self.opened_at.isoformat() if self.opened_at else None
            }

class CircuitBreakerOpenError(Exception):
    pass

# Usage
circuit_breaker = LLMCircuitBreaker(
    failure_threshold=3,
    success_threshold=2,
    timeout_seconds=30,
    quality_threshold=0.7
)

def make_llm_call(prompt: str) -> LLMResponse:
    return query_llm_resilient(prompt)

try:
    result = circuit_breaker.call(make_llm_call, "Analyze this text...")
    print(f"Success: {result.content}")
except CircuitBreakerOpenError as e:
    print(f"Circuit breaker open: {e}")
    # Use fallback or cached response
```

**Practical Implications:** Circuit breakers save cost during provider outages. Without them, systems continue making expensive failed API calls. During a 5-minute provider outage at 100 requests/minute, this saves 500 failed API calls (potentially $5-50 depending on model).

**Constraints:** Quality evaluation adds 2-10ms overhead per request. In high-throughput systems (>1000 req/s), move evaluation to async background thread. Circuit breaker state must be shared across instances—use Redis or similar for distributed systems.

### 3. Structured Output Validation & Recovery

LLM outputs are unreliable. Validation catches failures; recovery strategies fix them without user-visible errors.

```python
from pydantic import BaseModel, ValidationError, Field
from typing import List, Optional, Union, Type
import json

class AnalysisResult(BaseModel):
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_themes: List[str] = Field(..., min_items=1, max_items=10)
    summary: str = Fiel