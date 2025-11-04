# Job Failure Investigation: Systematic Debugging of LLM Pipeline Failures

## Core Concepts

When an LLM-powered application fails in production, traditional debugging approaches often fall short. Unlike deterministic software where identical inputs produce identical outputs, LLM failures exhibit probabilistic behavior, context-dependent errors, and cascading failures across multiple system boundaries.

### Traditional vs. Modern Debugging

```python
# Traditional software debugging
def process_order(order_id: str) -> dict:
    """Deterministic: same input always produces same output or same error"""
    order = db.get_order(order_id)  # Either succeeds or throws specific exception
    if order.status != "pending":
        raise InvalidOrderStateError(f"Expected pending, got {order.status}")
    return {"result": "processed", "order_id": order_id}

# Stack trace points exactly to the problem:
# InvalidOrderStateError at line 4: Expected pending, got shipped


# LLM pipeline debugging
def process_support_ticket(ticket: str) -> dict:
    """Non-deterministic: same input may succeed, fail, or produce invalid output"""
    classification = llm.classify(ticket)  # May return malformed JSON
    sentiment = llm.analyze_sentiment(ticket)  # May timeout
    response = llm.generate_response(
        ticket, 
        classification=classification,  # Downstream failure if classification was wrong
        sentiment=sentiment
    )
    return {"classification": classification, "response": response}

# Error could be:
# - LLM returned invalid JSON (silent failure)
# - Timeout after 30s (infrastructure)
# - Valid format but wrong classification (semantic error)
# - Context limit exceeded (input-dependent)
# - Rate limit hit (system-wide state)
# - Model returned refusal (content policy)
```

### Key Engineering Insight

**Traditional debugging is a depth-first search through a call stack. LLM debugging is a multi-dimensional investigation across:**

1. **Input space:** Token limits, encoding issues, adversarial inputs
2. **Model behavior:** Temperature effects, prompt sensitivity, output format compliance
3. **System state:** Rate limits, API health, network latency
4. **Semantic correctness:** Valid output that's contextually wrong
5. **Cascading failures:** Upstream errors propagating downstream

### Why This Matters Now

Production LLM systems fail differently than you expect:

- **Silent failures:** 200 OK response with semantically incorrect content
- **Intermittent failures:** Same request succeeds 90% of the time
- **Context-dependent failures:** Works in testing, fails with real user data
- **Cost-related failures:** Technically successful but economically unsustainable

Engineers who master systematic LLM failure investigation reduce mean-time-to-resolution (MTTR) from hours to minutes and prevent 60-80% of recurring issues.

## Technical Components

### 1. Observability Layer: Structured Logging for LLM Calls

Traditional logs capture what happened. LLM logs must capture what was intended, what was received, and why it might have failed.

```python
import json
import logging
from typing import Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class LLMCallLog:
    """Comprehensive log entry for LLM interactions"""
    timestamp: str
    request_id: str
    model: str
    prompt_hash: str  # For privacy, log hash not content
    prompt_length: int
    max_tokens: int
    temperature: float
    response_hash: Optional[str]
    response_length: Optional[int]
    latency_ms: Optional[int]
    status: str  # success, timeout, rate_limit, error, invalid_response
    error_type: Optional[str]
    error_message: Optional[str]
    token_count: Optional[int]
    cost_estimate: Optional[float]

class ObservableLLMClient:
    """Wrapper that adds comprehensive observability"""
    
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
    
    def _hash_content(self, content: str) -> str:
        """Hash for identifying unique prompts without storing PII"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def complete(self, prompt: str, **kwargs) -> tuple[Optional[str], LLMCallLog]:
        """Execute completion with full observability"""
        request_id = kwargs.get('request_id', self._generate_id())
        start_time = datetime.now()
        
        log_entry = LLMCallLog(
            timestamp=start_time.isoformat(),
            request_id=request_id,
            model=kwargs.get('model', 'unknown'),
            prompt_hash=self._hash_content(prompt),
            prompt_length=len(prompt),
            max_tokens=kwargs.get('max_tokens', 0),
            temperature=kwargs.get('temperature', 0.0),
            response_hash=None,
            response_length=None,
            latency_ms=None,
            status='initiated',
            error_type=None,
            error_message=None,
            token_count=None,
            cost_estimate=None
        )
        
        try:
            response = self.client.complete(prompt, **kwargs)
            end_time = datetime.now()
            
            # Update log with success metrics
            log_entry.latency_ms = int((end_time - start_time).total_seconds() * 1000)
            log_entry.response_hash = self._hash_content(response)
            log_entry.response_length = len(response)
            log_entry.status = 'success'
            
            # Validate response structure
            if not self._validate_response(response, kwargs.get('expected_format')):
                log_entry.status = 'invalid_response'
                log_entry.error_type = 'format_validation_failed'
            
            self.logger.info(json.dumps(asdict(log_entry)))
            return response, log_entry
            
        except TimeoutError as e:
            log_entry.status = 'timeout'
            log_entry.error_type = 'timeout'
            log_entry.error_message = str(e)
            self.logger.error(json.dumps(asdict(log_entry)))
            return None, log_entry
            
        except Exception as e:
            log_entry.status = 'error'
            log_entry.error_type = type(e).__name__
            log_entry.error_message = str(e)
            self.logger.error(json.dumps(asdict(log_entry)))
            return None, log_entry
    
    def _validate_response(self, response: str, expected_format: Optional[str]) -> bool:
        """Validate response matches expected structure"""
        if not expected_format:
            return True
        if expected_format == 'json':
            try:
                json.loads(response)
                return True
            except json.JSONDecodeError:
                return False
        return True
    
    def _generate_id(self) -> str:
        return f"{datetime.now().timestamp()}-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
```

**Practical Implications:**

- **Prompt hashing** enables identifying duplicate failures without storing sensitive data
- **Latency tracking** separates network issues from model performance issues
- **Status categorization** enables automated alerting on specific failure modes
- **Cost estimation** surfaces economic failures before they become critical

**Trade-offs:**

- Logging adds 5-10ms overhead per call
- Log volume increases by 3-5x compared to minimal logging
- Requires log aggregation infrastructure to be actionable

### 2. Failure Classification System

Not all failures are equal. A systematic taxonomy enables targeted remediation.

```python
from enum import Enum
from typing import Optional

class FailureCategory(Enum):
    """Hierarchical failure classification"""
    # Infrastructure failures - retry usually works
    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    
    # Input failures - fix the input
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    INVALID_INPUT_FORMAT = "invalid_input_format"
    CONTENT_POLICY_VIOLATION = "content_policy_violation"
    
    # Output failures - fix the prompt or validation
    MALFORMED_RESPONSE = "malformed_response"
    INCOMPLETE_RESPONSE = "incomplete_response"
    SEMANTIC_ERROR = "semantic_error"
    
    # System failures - architectural change needed
    COST_THRESHOLD_EXCEEDED = "cost_threshold_exceeded"
    LATENCY_THRESHOLD_EXCEEDED = "latency_threshold_exceeded"
    CASCADING_FAILURE = "cascading_failure"

class FailureAnalyzer:
    """Automatically categorize failures for targeted remediation"""
    
    @staticmethod
    def classify(log_entry: LLMCallLog, response: Optional[str]) -> FailureCategory:
        """Determine root cause category"""
        
        # Infrastructure failures
        if log_entry.status == 'timeout':
            return FailureCategory.NETWORK_TIMEOUT
        
        if 'rate_limit' in str(log_entry.error_message).lower():
            return FailureCategory.RATE_LIMIT
        
        if log_entry.status == 'error' and '503' in str(log_entry.error_message):
            return FailureCategory.SERVICE_UNAVAILABLE
        
        # Input failures
        if 'context_length' in str(log_entry.error_message).lower():
            return FailureCategory.CONTEXT_LIMIT_EXCEEDED
        
        if log_entry.prompt_length == 0 or log_entry.prompt_length > 100000:
            return FailureCategory.INVALID_INPUT_FORMAT
        
        if 'content_policy' in str(log_entry.error_message).lower():
            return FailureCategory.CONTENT_POLICY_VIOLATION
        
        # Output failures
        if log_entry.status == 'invalid_response':
            return FailureCategory.MALFORMED_RESPONSE
        
        if response and log_entry.response_length < (log_entry.max_tokens * 0.1):
            return FailureCategory.INCOMPLETE_RESPONSE
        
        # System failures
        if log_entry.latency_ms and log_entry.latency_ms > 30000:
            return FailureCategory.LATENCY_THRESHOLD_EXCEEDED
        
        return FailureCategory.SEMANTIC_ERROR
    
    @staticmethod
    def get_remediation(category: FailureCategory) -> dict[str, Any]:
        """Return specific remediation strategy"""
        
        strategies = {
            FailureCategory.NETWORK_TIMEOUT: {
                "action": "retry_with_backoff",
                "max_retries": 3,
                "backoff_multiplier": 2,
                "alert": False
            },
            FailureCategory.RATE_LIMIT: {
                "action": "queue_and_retry",
                "delay_seconds": 60,
                "alert": True,
                "alert_threshold": 10  # Alert if happens 10+ times
            },
            FailureCategory.CONTEXT_LIMIT_EXCEEDED: {
                "action": "truncate_input",
                "strategy": "sliding_window",
                "alert": False
            },
            FailureCategory.MALFORMED_RESPONSE: {
                "action": "retry_with_lower_temperature",
                "temperature": 0.0,
                "add_format_instruction": True,
                "alert": True
            },
            FailureCategory.SEMANTIC_ERROR: {
                "action": "manual_review",
                "log_full_context": True,
                "alert": True
            }
        }
        
        return strategies.get(category, {"action": "manual_investigation"})
```

**Real Constraints:**

- Classification requires training data initially (500+ failures for accurate rules)
- Semantic errors require human validation—no automated fix
- Remediation strategies may conflict with SLAs (e.g., retry delays)

### 3. Context Reconstruction for Reproduction

The hardest part of debugging LLM failures: reproducing the exact conditions.

```python
import pickle
from typing import Any
from dataclasses import dataclass

@dataclass
class ReproductionContext:
    """Everything needed to reproduce a failure"""
    prompt: str
    model_config: dict[str, Any]
    system_state: dict[str, Any]
    timestamp: str
    request_id: str
    
    def save(self, filepath: str):
        """Persist for later reproduction"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'ReproductionContext':
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class ReproducibleLLMClient:
    """Client that captures full context for every call"""
    
    def __init__(self, client, storage_path: str = "./reproductions"):
        self.client = client
        self.storage_path = storage_path
    
    def complete_with_capture(
        self, 
        prompt: str,
        capture_on_failure: bool = True,
        **kwargs
    ) -> tuple[Optional[str], Optional[str]]:
        """Execute and capture reproduction context on failure"""
        
        context = ReproductionContext(
            prompt=prompt,
            model_config={
                'model': kwargs.get('model'),
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000),
                'top_p': kwargs.get('top_p', 1.0)
            },
            system_state={
                'timestamp': datetime.now().isoformat(),
                'client_version': getattr(self.client, 'version', 'unknown')
            },
            timestamp=datetime.now().isoformat(),
            request_id=kwargs.get('request_id', 'unknown')
        )
        
        try:
            response = self.client.complete(prompt, **kwargs)
            return response, None
            
        except Exception as e:
            if capture_on_failure:
                filepath = f"{self.storage_path}/{context.request_id}.pkl"
                context.save(filepath)
                return None, filepath
            raise
    
    def reproduce_failure(self, filepath: str) -> tuple[Optional[str], Exception]:
        """Attempt to reproduce a captured failure"""
        context = ReproductionContext.load(filepath)
        
        print(f"Reproducing failure from {context.timestamp}")
        print(f"Model config: {context.model_config}")
        print(f"Prompt length: {len(context.prompt)}")
        
        try:
            response = self.client.complete(
                context.prompt,
                **context.model_config
            )
            return response, None
        except Exception as e:
            return None, e
```

**Concrete Example:**

```python
# In production, capture failure
client = ReproducibleLLMClient(llm_client)
response, repro_file = client.complete_with_capture(
    prompt=user_input,
    model='gpt-4',
    temperature=0.7
)

if repro_file:
    print(f"Failure captured: {repro_file}")
    # Later, in development environment:
    # response, error = client.reproduce_failure(repro_file)
```

### 4. Differential Analysis: Isolating Variables

When the same prompt sometimes works and sometimes fails, isolate which variable matters.

```python
from typing import Callable
import statistics

class DifferentialAnalyzer:
    """Isolate which variables affect failure rates"""
    
    def __init__(self, client):
        self.client = client
    
    def test_variable_impact(
        self,
        prompt: str,
        variable_name: str,
        values: list[Any],
        trials_per_value: int = 10,
        **base_config
    ) -> dict[Any, dict]:
        """Test how changing one variable affects outcomes"""
        
        results = {}
        
        for value in values:
            config = base_config.copy()
            config[variable_name] = value
            
            successes = 0
            latencies =