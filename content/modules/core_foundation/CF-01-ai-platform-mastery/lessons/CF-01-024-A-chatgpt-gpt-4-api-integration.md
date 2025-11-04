# ChatGPT & GPT-4 API Integration: Building Production-Ready LLM Applications

## Core Concepts

### Technical Definition

API integration with large language models means sending structured HTTP requests to hosted inference endpoints that execute transformer models trained on vast text corpora. You send prompts (text instructions) and receive completions (generated text responses) without managing model weights, GPU infrastructure, or inference optimization.

The fundamental shift: instead of importing a library and calling functions with structured parameters, you're communicating with an AI system through natural language over REST APIs. Your "function calls" are now prose instructions, and the "return values" are text that requires parsing and validation.

### Engineering Analogy: Traditional vs. LLM Approach

**Traditional API Integration:**
```python
# Traditional structured API
import translation_lib

result = translation_lib.translate(
    text="Hello, world",
    source_lang="en",
    target_lang="es",
    formality="informal"
)
# result: {"translated": "Hola, mundo", "confidence": 0.98}
```

**LLM API Integration:**
```python
# LLM-based approach
import openai
from typing import Dict, Any

def translate_with_llm(text: str, target_lang: str) -> Dict[str, Any]:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": f"Translate to {target_lang}: {text}"}
        ],
        temperature=0.3
    )
    
    return {
        "translated": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens,
        "cost_usd": response.usage.total_tokens * 0.00003  # Approximate
    }
```

The second approach is more flexible (handles any language pair without retraining) but introduces new challenges: non-deterministic outputs, token-based pricing, latency variability, and the need for output parsing.

### Key Insights That Change Engineering Perspective

1. **Interfaces become conversations**: Instead of rigid function signatures, you define behavior through natural language instructions. This is powerful but requires treating the API like a probabilistic system, not a deterministic function.

2. **Costs are variable and usage-based**: Every API call consumes tokens (roughly 0.75 words per token). A simple translation might cost $0.0003, but a document analysis could cost $0.50. Cost optimization becomes a first-class engineering concern.

3. **Latency is unpredictable**: Response times range from 500ms to 30+ seconds depending on output length and server load. You must architect for asynchronous patterns and implement timeouts.

4. **Prompt is code**: The text you send directly determines system behavior. Prompt engineering is real engineering—it requires versioning, testing, and systematic optimization.

### Why This Matters Now

LLM APIs reached production-readiness in 2023-2024 with:
- **Sub-second response times** for typical queries (previously 5-10s)
- **Context windows of 128K+ tokens** enabling full document processing
- **Structured output modes** (JSON mode, function calling) reducing parsing brittleness
- **Price drops of 10-100x** making large-scale deployment economically viable

Engineers who master this integration pattern can build features in hours that previously required months of ML engineering and infrastructure work.

## Technical Components

### Component 1: Authentication and Client Initialization

**Technical Explanation:**

API access requires an authentication key sent in HTTP headers. The official client libraries handle this automatically, managing connection pooling, retries, and request formatting.

```python
import os
from openai import OpenAI
from typing import Optional

class LLMClient:
    """Wrapper for OpenAI API with configuration management."""
    
    def __init__(self, api_key: Optional[str] = None):
        # Prefer explicit key, fallback to environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required: pass directly or set OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=self.api_key)
        
    def test_connection(self) -> bool:
        """Verify API connectivity and authentication."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

# Usage
client = LLMClient()
if client.test_connection():
    print("API ready")
```

**Practical Implications:**

- Store API keys in environment variables or secret management systems—never commit to version control
- Initialize clients once at application startup (they maintain connection pools)
- Test connectivity during deployment health checks

**Real Constraints:**

- API keys have rate limits (requests per minute) tied to your account tier
- Keys can be revoked; implement graceful degradation
- Network failures require exponential backoff retry logic

### Component 2: Message Structure and Role System

**Technical Explanation:**

The Chat Completions API uses a message array where each message has a `role` (system/user/assistant) and `content` (text). The model generates the next assistant message based on conversation history.

```python
from typing import List, Dict
from openai import OpenAI

def create_chat_completion(
    client: OpenAI,
    system_prompt: str,
    user_message: str,
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """
    Generate completion with explicit role separation.
    
    Args:
        system_prompt: Instructions defining assistant behavior
        user_message: Current user input
        conversation_history: Previous messages for context
    
    Returns:
        Generated assistant response
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Example: Code review assistant
client = OpenAI()

system_prompt = """You are a senior engineer conducting code reviews. 
Focus on: security vulnerabilities, performance issues, and maintainability. 
Provide specific, actionable feedback."""

code = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
"""

review = create_chat_completion(
    client=client,
    system_prompt=system_prompt,
    user_message=f"Review this Python function:\n\n{code}"
)

print(review)
# Output highlights SQL injection vulnerability, suggests parameterized queries
```

**Practical Implications:**

- **System messages** set behavior/persona—these rarely change per request
- **User messages** contain the actual task/question
- **Assistant messages** in history allow multi-turn conversations with context
- Order matters: system → conversation history → current user message

**Real Constraints:**

- Total tokens (all messages combined) must fit within model's context window
- Longer conversations require truncation strategies (keep recent + system, drop middle)
- System prompts consume tokens on every request—keep concise

### Component 3: Model Selection and Parameters

**Technical Explanation:**

Different models offer trade-offs between capability, speed, and cost. Parameters control output randomness (temperature), length (max_tokens), and diversity (top_p).

```python
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI

class ModelTier(Enum):
    FAST = "gpt-3.5-turbo"      # $0.0005/1K input tokens
    CAPABLE = "gpt-4"           # $0.03/1K input tokens
    ADVANCED = "gpt-4-turbo"    # $0.01/1K input tokens

@dataclass
class CompletionConfig:
    """Configuration for completion requests."""
    model: str
    temperature: float = 0.7  # 0.0-2.0, lower = more deterministic
    max_tokens: int = 1000     # Max output length
    top_p: float = 1.0         # Nucleus sampling (alternative to temperature)
    frequency_penalty: float = 0.0  # -2.0 to 2.0, reduces repetition
    presence_penalty: float = 0.0   # -2.0 to 2.0, encourages topic diversity

def completion_with_config(
    client: OpenAI,
    messages: List[Dict[str, str]],
    config: CompletionConfig
) -> Dict[str, any]:
    """Execute completion with detailed configuration."""
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty
    )
    
    return {
        "content": response.choices[0].message.content,
        "finish_reason": response.choices[0].finish_reason,
        "tokens": {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }
    }

# Example: Deterministic extraction vs creative generation
client = OpenAI()
messages = [{"role": "user", "content": "Write a product tagline for AI code review"}]

# Deterministic mode (data extraction, classification)
deterministic_config = CompletionConfig(
    model=ModelTier.FAST.value,
    temperature=0.0,  # Completely deterministic
    max_tokens=50
)

# Creative mode (brainstorming, content generation)
creative_config = CompletionConfig(
    model=ModelTier.CAPABLE.value,
    temperature=1.2,  # High randomness
    max_tokens=100,
    presence_penalty=0.6  # Encourage topic diversity
)

result_deterministic = completion_with_config(client, messages, deterministic_config)
result_creative = completion_with_config(client, messages, creative_config)

print(f"Deterministic: {result_deterministic['content']}")
print(f"Creative: {result_creative['content']}")
```

**Practical Implications:**

- **Classification/extraction tasks**: Use temperature=0.0 and fast models
- **Creative tasks**: Use temperature=0.8-1.2 and capable models
- **max_tokens** prevents runaway costs; calculate based on expected output length
- Monitor `finish_reason`: "length" means output was truncated

**Real Constraints:**

- Temperature=0.0 isn't truly deterministic (~95% consistency due to floating point)
- Higher temperature increases cost (generates more tokens to find completion)
- Different models have different context windows (4K-128K tokens)

### Component 4: Error Handling and Rate Limits

**Technical Explanation:**

API calls fail for multiple reasons: rate limits, timeouts, invalid requests, server errors. Production code must handle these gracefully with retries and exponential backoff.

```python
import time
from typing import Optional, Callable
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

class RobustLLMClient:
    """Production-ready client with error handling."""
    
    def __init__(self, api_key: str, max_retries: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.max_retries = max_retries
    
    def completion_with_retry(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        **kwargs
    ) -> Optional[str]:
        """
        Execute completion with exponential backoff retry.
        
        Handles: rate limits, transient server errors, timeouts.
        Does not retry: authentication errors, invalid requests.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=30.0,  # 30 second timeout
                    **kwargs
                )
                return response.choices[0].message.content
                
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise
                # Rate limit hit - wait with exponential backoff
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"Rate limited. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except APITimeoutError as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Timeout. Retry {attempt + 1}/{self.max_retries}")
                time.sleep(1)
                
            except APIError as e:
                # Server error (5xx) - retry
                if 500 <= e.status_code < 600:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
                else:
                    # Client error (4xx) - don't retry
                    raise
        
        return None

# Usage with error handling
client = RobustLLMClient(api_key=os.getenv("OPENAI_API_KEY"))

try:
    result = client.completion_with_retry(
        messages=[{"role": "user", "content": "Analyze this error log..."}],
        temperature=0.3,
        max_tokens=500
    )
    print(result)
except RateLimitError:
    print("Rate limit exceeded after retries - implement queue")
except APIError as e:
    print(f"API error: {e}")
```

**Practical Implications:**

- Always set timeouts to prevent hanging requests
- Implement exponential backoff for rate limits (don't spam retry)
- Log errors with request IDs for debugging
- Consider request queuing for high-volume applications

**Real Constraints:**

- Rate limits vary by model and account tier (default: ~3,500 RPM for GPT-4)
- Retry logic increases latency; set max retry limits
- Some errors (authentication, invalid input) should never retry

### Component 5: Structured Output and Parsing

**Technical Explanation:**

LLM outputs are strings requiring parsing into structured data. JSON mode forces valid JSON output; function calling enables schema-validated structured responses.

```python
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError
from openai import OpenAI

# Define structured output schema
class ExtractedEntity(BaseModel):
    """Schema for extracted entities."""
    name: str
    entity_type: str  # PERSON, ORGANIZATION, LOCATION, etc.
    context: str  # Sentence where entity appears

class ExtractionResult(BaseModel):
    """Complete extraction result."""
    entities: List[ExtractedEntity]
    summary: str

def extract_entities_json_mode(
    client: OpenAI,
    text: str
) -> Optional[ExtractionResult]:
    """
    Extract entities using JSON mode for guaranteed valid JSON.
    
    JSON mode requires explicitly asking for JSON in prompt.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "Extract entities from text. Output valid JSON only."
            },
            {
                "role": "user",
                "content": f"""Extract all named entities from this text:

{text}

Output JSON format:
{{
  "entities": [
    {{"name": "entity name", "entity_type": "PERSON|ORG|LOCATION", "context": "sentence"}}
  ],
  "summary": "brief summary"
}}"""
            }
        ],
        response_format={"type": "json_object"},  # Enforce JSON mode
        temperature=0.3
    )
    
    try:
        # Parse and validate against schema
        data = json.loads(response.choices[0