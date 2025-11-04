# Fallback & Redundancy Patterns for LLM Systems

## Core Concepts

Modern LLM systems fail differently than traditional software. A database query either returns data or throws an exception. An LLM might return malformed JSON, hallucinate facts, hit rate limits, timeout mid-generation, or produce subtly incorrect output that passes validation. These failure modes demand architectural patterns that go beyond simple try-catch blocks.

**Traditional vs. LLM Error Handling:**

```python
# Traditional API with predictable failures
def get_user_data(user_id: int) -> dict:
    try:
        response = db.query("SELECT * FROM users WHERE id = ?", user_id)
        return response
    except DatabaseError:
        # Retry once, then fail
        return db.query("SELECT * FROM users WHERE id = ?", user_id)

# LLM system with unpredictable failure modes
def extract_entities(text: str) -> dict:
    try:
        response = llm.complete(prompt=f"Extract entities: {text}")
        parsed = json.loads(response)
        # What if JSON is valid but semantically wrong?
        # What if it works 95% of the time but fails on edge cases?
        # What if the model is temporarily degraded?
        return parsed
    except (JSONDecodeError, TimeoutError, RateLimitError):
        # Simple retry doesn't solve semantic failures
        pass
```

Fallback and redundancy patterns for LLMs address three critical challenges:

1. **Probabilistic failures**: Outputs may be wrong without throwing errors
2. **Provider instability**: Rate limits, outages, and performance degradation
3. **Cost-quality trade-offs**: Faster/cheaper models fail more often on complex tasks

**Why This Matters Now:**

Production LLM systems regularly face scenarios where a single model or provider becomes unavailable or unreliable. In December 2023, multiple major LLM providers experienced simultaneous outages. Systems with robust fallback patterns maintained 99%+ uptime while single-provider systems went dark. The engineering cost of downtime (lost revenue, degraded user experience, SLA violations) far exceeds the complexity cost of implementing redundancy.

More critically, different models excel at different tasks. A routing architecture that attempts a fast, cheap model first and falls back to a powerful model on failure achieves 60-80% cost reduction while maintaining quality. This isn't premature optimization—it's fundamental system design.

## Technical Components

### 1. Fallback Chains with Quality Validation

A fallback chain attempts multiple strategies in sequence until one produces valid output. Unlike simple retries, each fallback may use different models, prompts, or validation criteria.

```python
from typing import Optional, Callable, List, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import json
import time

T = TypeVar('T')

class FallbackReason(Enum):
    VALIDATION_FAILED = "validation_failed"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    QUALITY_THRESHOLD = "quality_threshold"

@dataclass
class FallbackResult(Generic[T]):
    success: bool
    data: Optional[T]
    attempts: int
    fallback_reasons: List[FallbackReason]
    latency_ms: int
    cost_units: float  # Normalized cost metric

class LLMFallbackChain(Generic[T]):
    def __init__(
        self,
        strategies: List[Callable[[], T]],
        validators: List[Callable[[T], bool]],
        max_attempts: int = 3
    ):
        self.strategies = strategies
        self.validators = validators
        self.max_attempts = max_attempts
    
    def execute(self) -> FallbackResult[T]:
        attempts = 0
        fallback_reasons = []
        start_time = time.time()
        total_cost = 0.0
        
        for strategy_idx, strategy in enumerate(self.strategies):
            for attempt in range(self.max_attempts):
                attempts += 1
                
                try:
                    result = strategy()
                    
                    # Run all validators
                    validation_passed = all(
                        validator(result) for validator in self.validators
                    )
                    
                    if validation_passed:
                        latency = int((time.time() - start_time) * 1000)
                        return FallbackResult(
                            success=True,
                            data=result,
                            attempts=attempts,
                            fallback_reasons=fallback_reasons,
                            latency_ms=latency,
                            cost_units=total_cost
                        )
                    else:
                        fallback_reasons.append(FallbackReason.VALIDATION_FAILED)
                        
                except TimeoutError:
                    fallback_reasons.append(FallbackReason.TIMEOUT)
                except RateLimitError:
                    fallback_reasons.append(FallbackReason.RATE_LIMIT)
                    time.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    fallback_reasons.append(FallbackReason.EXCEPTION)
        
        latency = int((time.time() - start_time) * 1000)
        return FallbackResult(
            success=False,
            data=None,
            attempts=attempts,
            fallback_reasons=fallback_reasons,
            latency_ms=latency,
            cost_units=total_cost
        )

# Example usage with progressive complexity
def extract_structured_data(text: str) -> dict:
    def fast_model_strategy() -> dict:
        # Simulated fast model call
        response = call_llm("fast-model", f"Extract JSON: {text}")
        return json.loads(response)
    
    def powerful_model_strategy() -> dict:
        # Simulated powerful model with better prompt
        response = call_llm(
            "powerful-model",
            f"Extract entities as JSON. Required fields: name, date, amount.\n\n{text}"
        )
        return json.loads(response)
    
    def structured_output_strategy() -> dict:
        # Simulated structured output API
        return call_llm_structured(
            "powerful-model",
            schema={"name": "string", "date": "string", "amount": "number"},
            prompt=text
        )
    
    def validate_required_fields(data: dict) -> bool:
        return all(key in data for key in ["name", "date", "amount"])
    
    def validate_types(data: dict) -> bool:
        try:
            isinstance(data["name"], str)
            isinstance(data["date"], str)
            isinstance(data["amount"], (int, float))
            return True
        except (KeyError, TypeError):
            return False
    
    chain = LLMFallbackChain(
        strategies=[
            fast_model_strategy,
            powerful_model_strategy,
            structured_output_strategy
        ],
        validators=[validate_required_fields, validate_types],
        max_attempts=2
    )
    
    result = chain.execute()
    if result.success:
        return result.data
    else:
        raise ValueError(f"All strategies failed: {result.fallback_reasons}")
```

**Practical Implications:**

- Each validator adds latency (typically 1-5ms). Use only necessary validations.
- Structured output modes (JSON mode) reduce validation failures by 70-90% but aren't available for all models.
- Cost can increase 3-5x if fallbacks trigger frequently. Monitor fallback rates.

**Trade-offs:**

- More fallback layers increase reliability but add complexity and worst-case latency
- Aggressive validation reduces bad outputs but increases fallback frequency
- Each strategy should be meaningfully different (same model with same prompt rarely helps)

### 2. Provider Redundancy with Circuit Breaking

Provider redundancy distributes requests across multiple LLM providers or model endpoints. Circuit breakers prevent cascading failures by temporarily disabling failing providers.

```python
from datetime import datetime, timedelta
from collections import deque
from threading import Lock
from typing import Dict, Deque
import random

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_requests: int = 3
    
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_successes: int = 0
    
    _lock: Lock = Lock()
    
    def record_success(self) -> None:
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_max_requests:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.half_open_successes = 0
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
    
    def can_attempt(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if (self.last_failure_time and 
                    datetime.now() - self.last_failure_time > 
                    timedelta(seconds=self.timeout_seconds)):
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_successes = 0
                    return True
                return False
            else:  # HALF_OPEN
                return self.half_open_successes < self.half_open_max_requests

class ProviderPool:
    def __init__(self, providers: List[str]):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            provider: CircuitBreaker() for provider in providers
        }
        self.latency_history: Dict[str, Deque[float]] = {
            provider: deque(maxlen=100) for provider in providers
        }
        self._lock = Lock()
    
    def select_provider(self) -> Optional[str]:
        """Select provider using weighted random selection based on latency."""
        available_providers = [
            provider for provider, cb in self.circuit_breakers.items()
            if cb.can_attempt()
        ]
        
        if not available_providers:
            return None
        
        # Weight by inverse of average latency
        weights = []
        for provider in available_providers:
            history = self.latency_history[provider]
            if history:
                avg_latency = sum(history) / len(history)
                weight = 1.0 / (avg_latency + 0.1)  # Avoid division by zero
            else:
                weight = 1.0  # Default for new providers
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return random.choices(available_providers, weights=weights)[0]
    
    def record_result(
        self,
        provider: str,
        success: bool,
        latency_ms: float
    ) -> None:
        with self._lock:
            if success:
                self.circuit_breakers[provider].record_success()
                self.latency_history[provider].append(latency_ms)
            else:
                self.circuit_breakers[provider].record_failure()
    
    def get_pool_health(self) -> Dict[str, dict]:
        """Return health status for monitoring."""
        return {
            provider: {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "avg_latency_ms": (
                    sum(self.latency_history[provider]) / 
                    len(self.latency_history[provider])
                    if self.latency_history[provider] else None
                )
            }
            for provider, cb in self.circuit_breakers.items()
        }

class RedundantLLMClient:
    def __init__(self, providers: List[str]):
        self.pool = ProviderPool(providers)
    
    def complete(
        self,
        prompt: str,
        max_retries: int = 3
    ) -> Optional[str]:
        for attempt in range(max_retries):
            provider = self.pool.select_provider()
            
            if provider is None:
                # All circuit breakers open
                time.sleep(2 ** attempt)
                continue
            
            start_time = time.time()
            try:
                response = self._call_provider(provider, prompt)
                latency_ms = (time.time() - start_time) * 1000
                self.pool.record_result(provider, True, latency_ms)
                return response
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                self.pool.record_result(provider, False, latency_ms)
                # Try next provider
                continue
        
        return None
    
    def _call_provider(self, provider: str, prompt: str) -> str:
        # Simulated provider-specific API call
        # In production, this would route to different APIs
        pass
```

**Practical Implications:**

- Circuit breakers prevent wasting time on failing providers (reduces P99 latency by 40-60%)
- Latency-based weighting automatically shifts traffic from degraded providers
- Half-open state allows gradual recovery without traffic spikes

**Constraints:**

- Requires monitoring infrastructure to observe circuit breaker states
- Multiple providers increase operational complexity (API keys, billing, compliance)
- Different providers may have different response formats requiring normalization

### 3. Semantic Validation and Confidence Scoring

Syntactic validation (JSON schema, type checking) catches obvious failures. Semantic validation detects subtle errors like hallucinated facts or off-topic responses.

```python
from typing import Protocol, List
from dataclasses import dataclass

@dataclass
class ValidationResult:
    passed: bool
    confidence: float  # 0.0 to 1.0
    reasons: List[str]

class SemanticValidator(Protocol):
    def validate(self, input: str, output: str) -> ValidationResult:
        ...

class LengthRatioValidator:
    """Detect suspiciously short or long outputs."""
    
    def __init__(self, min_ratio: float = 0.1, max_ratio: float = 5.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def validate(self, input: str, output: str) -> ValidationResult:
        ratio = len(output) / max(len(input), 1)
        reasons = []
        
        if ratio < self.min_ratio:
            reasons.append(f"Output too short: {ratio:.2f}x input length")
            return ValidationResult(False, 0.3, reasons)
        elif ratio > self.max_ratio:
            reasons.append(f"Output too long: {ratio:.2f}x input length")
            return ValidationResult(False, 0.4, reasons)
        
        # Higher confidence for outputs in expected range
        confidence = 1.0 - abs(ratio - 1.0) / 5.0
        return ValidationResult(True, max(confidence, 0.5), reasons)

class KeywordPresenceValidator:
    """Ensure critical keywords appear in output."""
    
    def __init__(self, required_keywords: List[str]):
        self.required_keywords = [kw.lower() for kw in required_keywords]
    
    def validate(self, input: str, output: str) -> ValidationResult:
        output_lower = output.lower()
        present