# API Design & OpenAPI Specification for AI Systems

## Core Concepts

APIs for AI systems differ fundamentally from traditional REST APIs. While a traditional API might return structured data from a database query, an AI API returns probabilistic outputs from compute-intensive operations. This distinction changes everything—from how you design endpoints to how you handle errors and measure performance.

**Traditional vs. AI API Design:**

```python
# Traditional API: Deterministic, fast, stateless
from flask import Flask, jsonify
from typing import Dict, List

app = Flask(__name__)

@app.route('/api/users/<int:user_id>')
def get_user(user_id: int) -> Dict:
    # Database lookup: ~10ms, deterministic result
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    return jsonify(user)
```

```python
# AI API: Probabilistic, slow, stateful
from flask import Flask, request, jsonify
from typing import Dict, Any
import asyncio

app = Flask(__name__)

@app.route('/api/analyze', methods=['POST'])
async def analyze_text(data: Dict[str, Any]) -> Dict:
    # Model inference: ~2000ms, non-deterministic
    # Requires: streaming, timeouts, cost tracking, rate limiting
    text = request.json.get('text')
    context = request.json.get('context', [])  # Stateful
    
    result = await llm_client.complete(
        prompt=text,
        context=context,
        max_tokens=500,
        temperature=0.7,  # Affects determinism
        timeout=30.0
    )
    
    return jsonify({
        'result': result.text,
        'tokens_used': result.usage.total_tokens,
        'latency_ms': result.latency,
        'model_version': result.model
    })
```

**Key Engineering Insights:**

1. **Cost is a first-class concern**: Unlike database queries that cost fractions of a cent, each AI API call can cost $0.01-$1.00. Your API design must expose cost controls to clients.

2. **Streaming is not optional**: With 30-second response times being common, users need progressive results. This affects every layer of your API stack.

3. **Versioning is complex**: Model updates change behavior unpredictably. You need explicit model version control in your API contract.

4. **State management is critical**: Conversation history, context windows, and few-shot examples turn "stateless" REST APIs into stateful sessions.

**Why This Matters Now:**

The explosion of AI capabilities has created a new API design pattern. Engineers trained on traditional REST principles make predictable mistakes: they underestimate timeout needs, ignore token limits, and fail to expose model parameters. OpenAPI specification becomes your contract not just for data shapes, but for computational constraints. Teams shipping AI features in 2024 need API designs that handle 10x latency variability, explicit cost budgets, and model version drift—concerns that didn't exist in traditional API design.

## Technical Components

### 1. Request/Response Schema Design

AI APIs require richer schemas than traditional REST. You're not just describing JSON shapes—you're describing computational constraints, streaming protocols, and probabilistic outputs.

**Technical Explanation:**

OpenAPI 3.1 provides the schema definition language, but AI APIs need extensions for token limits, streaming modes, and model parameters. The schema becomes your executable documentation and client SDK generator.

**Practical Implementation:**

```python
# openapi_schema.py
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, validator

class CompletionRequest(BaseModel):
    """Request schema with AI-specific constraints."""
    
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=100000,  # Character limit before tokenization
        description="Input text for completion"
    )
    
    max_tokens: int = Field(
        default=500,
        ge=1,
        le=4000,  # Explicit computational limit
        description="Maximum tokens to generate"
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic)"
    )
    
    stream: bool = Field(
        default=False,
        description="Enable Server-Sent Events streaming"
    )
    
    context: Optional[List[dict]] = Field(
        default=None,
        max_items=50,  # Prevent context overflow
        description="Conversation history"
    )
    
    model_version: Literal["fast", "accurate", "balanced"] = Field(
        default="balanced",
        description="Model variant to use"
    )
    
    @validator('context')
    def validate_context_tokens(cls, v):
        """Ensure context doesn't exceed token budget."""
        if v:
            # Rough estimation: 4 chars = 1 token
            total_chars = sum(len(str(msg)) for msg in v)
            estimated_tokens = total_chars // 4
            if estimated_tokens > 8000:
                raise ValueError(f"Context too large: ~{estimated_tokens} tokens")
        return v

class CompletionResponse(BaseModel):
    """Response schema with observability fields."""
    
    text: str = Field(..., description="Generated completion")
    
    usage: dict = Field(
        ...,
        description="Token usage breakdown",
        example={
            "prompt_tokens": 150,
            "completion_tokens": 300,
            "total_tokens": 450
        }
    )
    
    latency_ms: int = Field(..., description="Request duration")
    
    model_version: str = Field(..., description="Actual model used")
    
    finish_reason: Literal["completed", "length", "content_filter"] = Field(
        ...,
        description="Why generation stopped"
    )
```

**Constraints & Trade-offs:**

- Schema validation adds ~5ms latency but prevents 90% of invalid requests
- Strict token limits prevent runaway costs but may truncate valid inputs
- Context validation requires tokenization preview, adding computational overhead

### 2. Streaming Protocol Design

Long-running AI operations require streaming to maintain acceptable UX. This fundamentally changes your API from request-response to bi-directional communication.

**Technical Explanation:**

Server-Sent Events (SSE) provides one-way streaming over HTTP/1.1 without WebSocket complexity. For AI APIs, you stream partial results as tokens generate, allowing progressive rendering.

**Implementation:**

```python
# streaming_api.py
from flask import Flask, Response, request
from typing import Generator, Dict, Any
import json
import time

app = Flask(__name__)

def generate_completion_stream(prompt: str, max_tokens: int) -> Generator[str, None, None]:
    """Stream completion tokens as they generate."""
    
    # Send initial metadata
    yield f"data: {json.dumps({'type': 'start', 'timestamp': time.time()})}\n\n"
    
    tokens_used = 0
    current_text = ""
    
    # Simulate streaming token generation
    for token in llm_client.stream_complete(prompt, max_tokens):
        tokens_used += 1
        current_text += token
        
        # Stream each token immediately
        chunk = {
            'type': 'token',
            'token': token,
            'cumulative_text': current_text,
            'tokens_used': tokens_used
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
        # Check client disconnect
        if tokens_used % 10 == 0:
            time.sleep(0.01)  # Allow client to process
    
    # Send completion metadata
    final_chunk = {
        'type': 'done',
        'total_tokens': tokens_used,
        'finish_reason': 'completed',
        'latency_ms': int((time.time() - start_time) * 1000)
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"

@app.route('/api/complete/stream', methods=['POST'])
def stream_completion():
    """Streaming completion endpoint."""
    data = request.json
    
    return Response(
        generate_completion_stream(
            prompt=data['prompt'],
            max_tokens=data.get('max_tokens', 500)
        ),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
            'Connection': 'keep-alive'
        }
    )
```

**Client Implementation:**

```python
# streaming_client.py
import requests
import json
from typing import Callable

def consume_stream(url: str, payload: dict, on_token: Callable[[str], None]):
    """Client-side streaming consumer."""
    
    response = requests.post(
        url,
        json=payload,
        stream=True,
        timeout=(5, 60)  # (connect, read) timeouts
    )
    
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data = json.loads(line_str[6:])
                
                if data['type'] == 'token':
                    on_token(data['token'])
                elif data['type'] == 'done':
                    print(f"\nCompleted: {data['total_tokens']} tokens")
                    break

# Usage
def print_token(token: str):
    print(token, end='', flush=True)

consume_stream(
    'http://localhost:5000/api/complete/stream',
    {'prompt': 'Explain APIs', 'max_tokens': 200},
    on_token=print_token
)
```

**Real Constraints:**

- SSE requires HTTP/1.1 or HTTP/2 with proper server configuration
- Proxies/load balancers often buffer responses; requires `X-Accel-Buffering: no`
- Client reconnection logic needed for network interruptions (not shown)
- Memory usage: buffering 500-token stream = ~2KB per request

### 3. Error Handling & Rate Limiting

AI APIs fail differently than traditional APIs. Timeouts are common, rate limits are token-based not request-based, and "soft failures" (partial completions) are valid responses.

**Technical Explanation:**

HTTP status codes must distinguish between client errors (400s), server errors (500s), and AI-specific issues (model unavailable, content filtered, token limit exceeded).

**Implementation:**

```python
# error_handling.py
from flask import Flask, jsonify, request
from functools import wraps
from typing import Callable
import time
from collections import defaultdict

app = Flask(__name__)

class AIAPIError(Exception):
    """Base exception for AI API errors."""
    def __init__(self, message: str, status_code: int, error_type: str):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type

class RateLimiter:
    """Token-based rate limiter."""
    
    def __init__(self, tokens_per_minute: int):
        self.tokens_per_minute = tokens_per_minute
        self.usage: Dict[str, List[tuple]] = defaultdict(list)
    
    def check_limit(self, api_key: str, estimated_tokens: int) -> bool:
        """Check if request would exceed rate limit."""
        now = time.time()
        cutoff = now - 60  # 1 minute window
        
        # Remove old entries
        self.usage[api_key] = [
            (ts, tokens) for ts, tokens in self.usage[api_key]
            if ts > cutoff
        ]
        
        # Calculate current usage
        current_usage = sum(tokens for _, tokens in self.usage[api_key])
        
        if current_usage + estimated_tokens > self.tokens_per_minute:
            return False
        
        # Record this request
        self.usage[api_key].append((now, estimated_tokens))
        return True
    
    def get_reset_time(self, api_key: str) -> float:
        """Get seconds until rate limit resets."""
        if not self.usage[api_key]:
            return 0.0
        oldest = self.usage[api_key][0][0]
        return max(0, 60 - (time.time() - oldest))

rate_limiter = RateLimiter(tokens_per_minute=100000)

def handle_ai_errors(f: Callable):
    """Decorator for consistent error handling."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        
        except AIAPIError as e:
            return jsonify({
                'error': {
                    'type': e.error_type,
                    'message': e.message,
                    'status': e.status_code
                }
            }), e.status_code
        
        except TimeoutError:
            return jsonify({
                'error': {
                    'type': 'timeout',
                    'message': 'Request exceeded 30s timeout',
                    'status': 504
                }
            }), 504
        
        except Exception as e:
            # Log actual error securely
            app.logger.error(f"Unexpected error: {str(e)}")
            return jsonify({
                'error': {
                    'type': 'internal_error',
                    'message': 'Internal server error',
                    'status': 500
                }
            }), 500
    
    return decorated_function

@app.route('/api/complete', methods=['POST'])
@handle_ai_errors
def complete():
    """Completion endpoint with rate limiting and error handling."""
    data = request.json
    api_key = request.headers.get('X-API-Key')
    
    if not api_key:
        raise AIAPIError(
            'API key required',
            401,
            'authentication_error'
        )
    
    # Estimate tokens (rough: 4 chars = 1 token)
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 500)
    estimated_tokens = (len(prompt) // 4) + max_tokens
    
    # Check rate limit
    if not rate_limiter.check_limit(api_key, estimated_tokens):
        reset_time = rate_limiter.get_reset_time(api_key)
        raise AIAPIError(
            f'Rate limit exceeded. Reset in {reset_time:.0f}s',
            429,
            'rate_limit_error'
        )
    
    try:
        result = llm_client.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=30.0
        )
        
        return jsonify({
            'text': result.text,
            'usage': result.usage,
            'model_version': result.model
        })
    
    except ContentFilterException:
        raise AIAPIError(
            'Content filtered by safety system',
            400,
            'content_filter'
        )
    
    except ContextLengthException:
        raise AIAPIError(
            'Prompt exceeds maximum context length',
            400,
            'context_length_exceeded'
        )
```

**Practical Implications:**

- Token-based rate limiting prevents cost abuse more effectively than request counting
- Specific error types enable intelligent client retry logic
- 429 responses should include `Retry-After` header (shown in reset_time)

### 4. OpenAPI Specification Generation

Hand-writing OpenAPI specs is error-prone. Generate them from code to ensure accuracy and enable automatic client SDK generation.

**Implementation:**

```python
# openapi_generation.py
from flask import Flask
from flask_openapi3 import OpenAPI, Info, Tag
from pydantic import BaseModel, Field
from typing import Optional, List

# Define OpenAPI metadata
info = Info(
    title="AI Completion API",
    version="1.0.0",
    description="Production AI text completion service"
)

app = OpenAPI(__name__, info=info)

completion_tag = Tag(name="Completion", description="Text completion operations")

# Request/Response models (from earlier)
class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100000)
    max_tokens: int = Field(default=500, ge