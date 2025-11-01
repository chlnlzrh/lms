# API-Hosted Models: Engineering Production LLM Integration

## Core Concepts

API-hosted models are large language models accessed through HTTP endpoints, where the computational infrastructure, model weights, and inference optimization are managed by the provider. You send text, receive text—the billions of parameters and specialized hardware remain abstracted behind a REST API.

### Traditional vs. API-Hosted Architecture

```python
# Traditional ML model integration (pre-LLM era)
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TraditionalSentimentAnalyzer:
    def __init__(self, model_path: str):
        # Load model file (~10MB), runs on application server
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
    def analyze(self, text: str) -> str:
        # Feature engineering required
        features = self.vectorizer.transform([text])
        prediction = self.model.predict(features)[0]
        return "positive" if prediction == 1 else "negative"

# API-hosted LLM integration (modern)
import requests
from typing import Dict, Any

class APIHostedAnalyzer:
    def __init__(self, api_key: str, endpoint: str):
        # No model files, no local GPU required
        self.api_key = api_key
        self.endpoint = endpoint
        
    def analyze(self, text: str) -> Dict[str, Any]:
        # Natural language in, structured data out
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze sentiment of: {text}\n"
                                   f"Return JSON with: sentiment, confidence, reasoning"
                    }
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            },
            timeout=30
        )
        return response.json()
```

**Key differences engineers notice immediately:**

1. **No local model artifacts**: No 10GB weight files, no CUDA version conflicts, no GPU memory management
2. **Zero feature engineering**: Send raw text, get structured responses without preprocessing pipelines
3. **Pay-per-token economics**: Costs scale with usage, not infrastructure provisioning
4. **Stateless by default**: Each request is independent; context must be explicitly sent

### Why This Architecture Matters Now

The shift to API-hosted models changes the engineering bottleneck from **"How do we train/deploy this model?"** to **"How do we design prompts and manage API constraints?"**

Three critical insights:

1. **Latency is network-bound, not compute-bound**: A 70B parameter model responds in 2-5 seconds over the network. Your bottleneck is now HTTP round-trips and token generation speed, not matrix multiplication time.

2. **Context is currency**: You pay for every token sent and received. A chatbot that includes full conversation history in every request burns budget exponentially.

3. **Reliability requires defensive engineering**: API rate limits, transient failures, and timeout handling become first-class architectural concerns, not afterthoughts.

## Technical Components

### 1. Request/Response Protocol

API-hosted models use a message-based protocol where you send conversational turns and receive generated completions.

```python
from dataclasses import dataclass
from typing import List, Optional, Literal
from enum import Enum

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class CompletionRequest:
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def estimate_prompt_tokens(self) -> int:
        """Rough estimate: 1 token ≈ 4 characters for English"""
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4

@dataclass
class CompletionResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: Literal["stop", "length", "content_filter"]
    
    @property
    def cost_usd(self) -> float:
        """Example pricing: $0.03/1K prompt, $0.06/1K completion"""
        prompt_cost = (self.prompt_tokens / 1000) * 0.03
        completion_cost = (self.completion_tokens / 1000) * 0.06
        return prompt_cost + completion_cost
```

**Practical implications:**

- **System messages** set persistent behavior without users seeing them: "You are a code review assistant. Be concise and cite line numbers."
- **Temperature 0.0-0.3** for deterministic tasks (data extraction, code generation); **0.7-1.0** for creative tasks (brainstorming, storytelling)
- **max_tokens** acts as a safety limit—set to prevent runaway costs if the model generates unexpectedly long outputs

**Real constraint example:**

```python
def safe_completion_request(
    messages: List[Message],
    max_budget_usd: float = 0.10
) -> CompletionRequest:
    """Prevent single request from exceeding budget"""
    estimated_prompt_tokens = sum(len(m.content) for m in messages) // 4
    
    # At $0.03/1K prompt + $0.06/1K completion
    # If prompt is 2000 tokens, completion must be < 1000 tokens to stay under $0.10
    max_completion_tokens = int(
        ((max_budget_usd - (estimated_prompt_tokens / 1000 * 0.03)) / 0.06) * 1000
    )
    
    return CompletionRequest(
        messages=messages,
        max_tokens=max(100, max_completion_tokens),  # Minimum 100 tokens
        temperature=0.2
    )
```

### 2. Streaming vs. Buffered Responses

Streaming returns tokens as they're generated; buffering waits for complete response.

```python
import json
from typing import Iterator, Optional

class StreamingClient:
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint
    
    def stream_completion(
        self, 
        messages: List[Message]
    ) -> Iterator[str]:
        """Yield tokens as they arrive"""
        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream"
            },
            json={
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": True
            },
            stream=True,
            timeout=60
        )
        
        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    data = json.loads(line[6:])
                    if content := data.get("choices", [{}])[0].get("delta", {}).get("content"):
                        yield content
    
    def buffer_completion(
        self, 
        messages: List[Message]
    ) -> str:
        """Wait for complete response"""
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False
            },
            timeout=60
        )
        return response.json()["choices"][0]["message"]["content"]

# Usage comparison
client = StreamingClient(api_key="...", endpoint="...")

# Streaming: User sees first token in ~500ms
print("Streaming response:")
for token in client.stream_completion([Message(Role.USER, "Explain quicksort")]):
    print(token, end="", flush=True)
print("\n")

# Buffered: User waits 5 seconds, sees everything at once
print("Buffered response:")
result = client.buffer_completion([Message(Role.USER, "Explain quicksort")])
print(result)
```

**When to use each:**

- **Streaming**: User-facing chatbots, long-form content generation (users perceive 3-5x faster response)
- **Buffered**: Background jobs, structured data extraction where you parse the complete JSON response

### 3. Rate Limits and Retry Logic

API providers enforce rate limits (requests per minute, tokens per day) to manage capacity. Production code must handle 429 (rate limited) and 5xx (server error) responses.

```python
import time
from typing import Callable, TypeVar, Optional
import logging

T = TypeVar('T')

class RateLimitError(Exception):
    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after

class APIClient:
    def __init__(
        self, 
        api_key: str, 
        endpoint: str,
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger(__name__)
    
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate delay: 1s, 2s, 4s, 8s..."""
        return self.base_delay * (2 ** attempt)
    
    def _should_retry(self, status_code: int) -> bool:
        """Retry on rate limits and server errors, not client errors"""
        return status_code in {429, 500, 502, 503, 504}
    
    def request_with_retry(
        self,
        messages: List[Message],
        timeout: int = 30
    ) -> CompletionResponse:
        """Robust API request with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "messages": [
                            {"role": m.role, "content": m.content} 
                            for m in messages
                        ]
                    },
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    choice = data["choices"][0]
                    usage = data["usage"]
                    
                    return CompletionResponse(
                        content=choice["message"]["content"],
                        prompt_tokens=usage["prompt_tokens"],
                        completion_tokens=usage["completion_tokens"],
                        total_tokens=usage["total_tokens"],
                        finish_reason=choice["finish_reason"]
                    )
                
                if self._should_retry(response.status_code):
                    retry_after = int(response.headers.get("Retry-After", 0))
                    delay = max(retry_after, self._exponential_backoff(attempt))
                    
                    self.logger.warning(
                        f"Status {response.status_code}, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue
                
                # Client error (4xx) - don't retry
                response.raise_for_status()
                
            except requests.Timeout:
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    self.logger.warning(f"Timeout, retrying in {delay}s")
                    time.sleep(delay)
                else:
                    raise
        
        raise RateLimitError("Max retries exceeded")
```

**Critical trade-off**: Aggressive retries improve reliability but increase latency. For user-facing features, set `max_retries=2` with `base_delay=0.5`. For background jobs, use `max_retries=5` with `base_delay=2.0`.

### 4. Context Window Management

Context window is the maximum tokens (prompt + completion) the model can process. Exceeding it causes 400 errors.

```python
from collections import deque
from typing import Deque

class ConversationManager:
    def __init__(
        self, 
        max_context_tokens: int = 4096,
        system_message: Optional[str] = None
    ):
        self.max_context_tokens = max_context_tokens
        self.system_message = system_message
        self.messages: Deque[Message] = deque()
        
        if system_message:
            self.messages.append(Message(Role.SYSTEM, system_message))
    
    def _estimate_tokens(self, text: str) -> int:
        """Conservative estimate: 1 token per 3 characters"""
        return len(text) // 3
    
    def _total_tokens(self) -> int:
        return sum(self._estimate_tokens(m.content) for m in self.messages)
    
    def add_message(self, role: Role, content: str) -> None:
        """Add message, evicting old messages if needed"""
        new_message = Message(role, content)
        self.messages.append(new_message)
        
        # Evict oldest non-system messages to stay under limit
        # Reserve 1000 tokens for completion
        while self._total_tokens() > (self.max_context_tokens - 1000):
            if len(self.messages) <= 1:  # Keep at least system message
                break
            
            # Remove oldest message that isn't system message
            for i, msg in enumerate(self.messages):
                if msg.role != Role.SYSTEM:
                    removed = self.messages[i]
                    del self.messages[i]
                    print(f"Evicted message ({self._estimate_tokens(removed.content)} tokens)")
                    break
    
    def get_messages(self) -> List[Message]:
        return list(self.messages)

# Usage example
conversation = ConversationManager(
    max_context_tokens=4096,
    system_message="You are a helpful Python tutor."
)

conversation.add_message(Role.USER, "Explain list comprehensions")
conversation.add_message(Role.ASSISTANT, "[Long explanation...]")
conversation.add_message(Role.USER, "Show me an example")
conversation.add_message(Role.ASSISTANT, "[Code example...]")

# After 10 turns, oldest messages are automatically evicted
for i in range(10):
    conversation.add_message(Role.USER, f"Follow-up question {i}")
    conversation.add_message(Role.ASSISTANT, "Response...")
```

**Practical pattern**: For chatbots, maintain a sliding window of the last 5-10 turns. For summarization tasks, periodically condense conversation history into a summary and reset context.

### 5. Structured Output Parsing

LLMs generate text; your application needs JSON, enums, or validated data structures.

```python
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
import json
import re

class CodeReview(BaseModel):
    severity: Literal["critical", "major", "minor", "suggestion"]
    line_number: int = Field(ge=1)
    issue: str = Field(min_length=10)
    suggestion: str = Field(min_length=10)
    code_snippet: str

class CodeReviewResponse(BaseModel):
    reviews: List[CodeReview]
    overall_quality: Literal["poor", "fair", "good", "excellent"]

def extract_structured_output(
    messages: List[Message],
    client: APIClient,
    response_model: type[BaseModel],
    max_attempts: int = 3
) -> BaseModel:
    """Request structured output with validation and retry"""
    
    # Add JSON schema instruction to prompt
    schema