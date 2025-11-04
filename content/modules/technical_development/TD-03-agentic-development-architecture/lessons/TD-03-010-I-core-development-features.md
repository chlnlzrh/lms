# Core Development Features in LLM Applications

## Core Concepts

Large Language Models fundamentally change how we build software features. Instead of writing explicit logic for every edge case, we describe what we want in natural language and let the model handle the complexity. This isn't about replacing programming—it's about adding a new computational primitive to your toolkit.

### Traditional vs. LLM-Based Approach

Consider a content moderation system that flags inappropriate comments:

**Traditional approach:**
```python
import re
from typing import List, Dict

class TraditionalModerator:
    def __init__(self):
        # Hundreds of patterns to maintain
        self.banned_words = {'profanity1', 'profanity2', 'slur1'}
        self.suspicious_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email
        ]
        self.spam_indicators = {
            'click here', 'limited time', 'act now'
        }
    
    def moderate(self, text: str) -> Dict[str, any]:
        text_lower = text.lower()
        flags = []
        
        # Check banned words
        for word in self.banned_words:
            if word in text_lower:
                flags.append(f"banned_word: {word}")
        
        # Check patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(f"suspicious_pattern: {pattern}")
        
        # Check spam
        spam_count = sum(1 for indicator in self.spam_indicators 
                        if indicator in text_lower)
        if spam_count >= 2:
            flags.append("likely_spam")
        
        return {
            'approved': len(flags) == 0,
            'flags': flags,
            'confidence': 1.0 if flags else 0.8
        }

# Problem: Misses context, sarcasm, evolving language
moderator = TraditionalModerator()
print(moderator.moderate("You're such a genius! 🙄"))  
# {'approved': True, 'flags': [], 'confidence': 0.8}
# Missed the sarcastic insult
```

**LLM-based approach:**
```python
from typing import Dict, List
import json
import anthropic  # or openai, or any LLM client

class LLMModerator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = """You are a content moderator. Analyze text for:
- Harassment, insults, or personal attacks
- Hate speech or discrimination
- Personal information (PII)
- Spam or manipulation
- Sexual or violent content

Return JSON: {"approved": bool, "flags": [str], "reasoning": str, "confidence": float}"""
    
    def moderate(self, text: str) -> Dict[str, any]:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            temperature=0,  # Deterministic for moderation
            system=self.system_prompt,
            messages=[{"role": "user", "content": text}]
        )
        
        return json.loads(response.content[0].text)

# Handles context, tone, evolving language
moderator = LLMModerator(api_key="your-key")
result = moderator.moderate("You're such a genius! 🙄")
# {'approved': False, 'flags': ['sarcastic_insult'], 
#  'reasoning': 'Sarcastic tone indicates mockery', 'confidence': 0.85}
```

### Key Engineering Insights

**1. Instructions as Code**: Your prompts are executable specifications. Version control them, test them, and iterate on them like any other code artifact.

**2. Probabilistic Outputs**: Unlike deterministic functions, LLM responses vary. Design for this—use temperature=0 for consistency, structured outputs for reliability, and validation layers.

**3. Context as Memory**: LLMs don't maintain state between calls. Every request must include all necessary context, making context management a first-class engineering concern.

**4. Latency Trade-offs**: LLM calls take 500ms-5s. This fundamentally changes UX patterns—streaming responses, optimistic UI updates, and async processing become essential.

### Why This Matters Now

The cost-performance curve for LLMs crossed a critical threshold in 2024. Tasks that cost $1 per operation two years ago now cost $0.001. This makes entire categories of features economically viable:

- **Content generation**: Product descriptions, email drafts, code documentation
- **Data extraction**: Parsing unstructured text, form filling, invoice processing
- **Classification**: Sentiment analysis, topic categorization, intent detection
- **Transformation**: Translation, summarization, format conversion, style adaptation

These aren't future possibilities—they're production-ready now, and your competitors are shipping them.

## Technical Components

### 1. Structured Output Generation

**Technical Explanation**: LLMs naturally produce free-form text, but applications need predictable data structures. Structured output forces the model to return valid JSON, XML, or other formats that integrate cleanly with typed systems.

**Practical Implementation**:
```python
from typing import List, Literal
from pydantic import BaseModel, Field
import anthropic

class ProductReview(BaseModel):
    """Structured product review analysis"""
    sentiment: Literal['positive', 'negative', 'neutral']
    rating: int = Field(ge=1, le=5, description="Star rating 1-5")
    key_points: List[str] = Field(max_items=5)
    would_recommend: bool
    category: Literal['electronics', 'clothing', 'food', 'other']

def analyze_review(review_text: str, api_key: str) -> ProductReview:
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        temperature=0,
        system=f"""Analyze product reviews. Return JSON matching this schema:
{ProductReview.model_json_schema()}""",
        messages=[{
            "role": "user",
            "content": f"Analyze this review:\n\n{review_text}"
        }]
    )
    
    # Parse and validate
    import json
    data = json.loads(response.content[0].text)
    return ProductReview(**data)

# Usage
review = """I bought these headphones last week and I'm blown away! 
The noise cancellation is incredible on flights. Battery lasts all day. 
Only complaint is they're a bit heavy for long sessions. 
Still, would definitely buy again."""

result = analyze_review(review, "your-key")
print(f"Sentiment: {result.sentiment}, Rating: {result.rating}/5")
print(f"Key points: {', '.join(result.key_points)}")
# Sentiment: positive, Rating: 4/5
# Key points: excellent noise cancellation, long battery life, slightly heavy
```

**Constraints**:
- Schema complexity impacts accuracy—keep under 10 fields
- Deeply nested structures (>3 levels) increase failure rates
- Always validate output; even with structured prompts, models can hallucinate invalid JSON

### 2. Few-Shot Learning Patterns

**Technical Explanation**: Few-shot learning provides examples within your prompt to establish patterns. The model learns your specific requirements from examples rather than lengthy descriptions.

**Practical Implementation**:
```python
from typing import List, Tuple

def create_fewshot_extractor(
    examples: List[Tuple[str, str]], 
    api_key: str
):
    """Factory for creating domain-specific extractors via few-shot learning"""
    
    def extract(text: str) -> str:
        # Build prompt with examples
        prompt_parts = ["Extract key information from text:\n"]
        
        for input_text, expected_output in examples:
            prompt_parts.append(f"\nInput: {input_text}")
            prompt_parts.append(f"Output: {expected_output}\n")
        
        prompt_parts.append(f"\nInput: {text}")
        prompt_parts.append("Output:")
        
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            temperature=0,
            messages=[{
                "role": "user",
                "content": "".join(prompt_parts)
            }]
        )
        
        return response.content[0].text.strip()
    
    return extract

# Training examples for invoice extraction
invoice_examples = [
    (
        "INVOICE #12345 | Date: 2024-01-15 | Amount: $1,234.56 | Acme Corp",
        "invoice_number: 12345, date: 2024-01-15, amount: 1234.56, vendor: Acme Corp"
    ),
    (
        "Bill dated 03/22/2024 from XYZ Ltd - Total Due: $450.00 (Ref: INV-999)",
        "invoice_number: INV-999, date: 2024-03-22, amount: 450.00, vendor: XYZ Ltd"
    ),
]

extractor = create_fewshot_extractor(invoice_examples, "your-key")

# Test with new format
new_invoice = "Purchase Order PO-2024-567 | 2024-02-10 | Beta Industries | $3,200.75"
print(extractor(new_invoice))
# invoice_number: PO-2024-567, date: 2024-02-10, amount: 3200.75, vendor: Beta Industries
```

**Real-World Constraints**:
- Optimal example count: 3-7 examples (more doesn't always help)
- Examples consume context window—long examples limit query space
- Example selection matters: diverse edge cases outperform similar examples

### 3. Streaming Responses

**Technical Explanation**: Streaming delivers tokens as they're generated rather than waiting for completion. This reduces perceived latency from 3000ms to 300ms for first-token, transforming UX for long-form generation.

**Practical Implementation**:
```python
import anthropic
from typing import Iterator, Dict
import time

class StreamingGenerator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_stream(
        self, 
        prompt: str,
        system: str = ""
    ) -> Iterator[Dict[str, any]]:
        """
        Yields chunks with metadata for real-time processing
        """
        start_time = time.time()
        token_count = 0
        
        with self.client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.7,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text_chunk in stream.text_stream:
                token_count += 1
                yield {
                    'chunk': text_chunk,
                    'token_count': token_count,
                    'elapsed_ms': int((time.time() - start_time) * 1000),
                    'done': False
                }
            
            # Final metadata
            yield {
                'chunk': '',
                'token_count': token_count,
                'elapsed_ms': int((time.time() - start_time) * 1000),
                'done': True
            }

# Usage: Real-time display
generator = StreamingGenerator("your-key")
prompt = "Write a technical explanation of binary search in 100 words."

print("Response: ", end='', flush=True)
for event in generator.generate_stream(prompt):
    if event['chunk']:
        print(event['chunk'], end='', flush=True)
    
    if event['done']:
        print(f"\n\n[Completed in {event['elapsed_ms']}ms, "
              f"{event['token_count']} tokens]")

# Response: Binary search is an efficient algorithm for finding...
# [Completed in 2341ms, 98 tokens]
```

**Performance Implications**:
- First token typically arrives in 200-500ms
- Subsequent tokens stream at ~50-100 tokens/second
- User engagement increases 40-60% with streaming vs. waiting for complete response
- Error handling becomes complex—partial responses may need rollback

### 4. Context Window Management

**Technical Explanation**: LLMs have fixed context windows (typically 8k-200k tokens). Every request consumes: system prompt + conversation history + current input + expected output. Exceeding the limit truncates or errors.

**Practical Implementation**:
```python
from typing import List, Dict
import tiktoken  # OpenAI's tokenizer, works as approximation

class ContextManager:
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")
        # Reserve space for system prompt and output
        self.usable_tokens = max_tokens - 1000
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def trim_conversation(
        self, 
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Keep most recent messages that fit in context"""
        total_tokens = 0
        kept_messages = []
        
        # Iterate from newest to oldest
        for message in reversed(messages):
            msg_tokens = self.count_tokens(message['content'])
            
            if total_tokens + msg_tokens > self.usable_tokens:
                break
            
            kept_messages.insert(0, message)
            total_tokens += msg_tokens
        
        return kept_messages
    
    def create_summary_context(
        self, 
        old_messages: List[Dict[str, str]],
        api_key: str
    ) -> str:
        """Compress old messages into summary"""
        if not old_messages:
            return ""
        
        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in old_messages
        ])
        
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation in 100 words:\n\n{conversation}"
            }]
        )
        
        return response.content[0].text

# Usage in chatbot
manager = ContextManager(max_tokens=8000)
conversation_history = [
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What's its population?"},
    {"role": "assistant", "content": "Paris has approximately 2.2 million..."},
    # ... 50 more messages
]

# Before sending new message, trim context
trimmed = manager.trim_conversation(conversation_history)
print(f"Kept {len(trimmed)} of {len(conversation_history)} messages")
print(f"Token usage: {sum(manager.count_tokens(m['content']) for m in trimmed)}")
```

**Trade-offs**:
- **Truncation**: Fast, lossy, may lose critical context
- **Summarization**: Preserves intent, adds latency and cost
- **Sliding window**: Simple, but loses distant context
- **Hierarchical summarization**: Best quality, highest complexity

### 5. Error Handling and Fallbacks

**Technical Explanation**: LLM calls fail for multiple reasons: rate limits, timeouts, malformed outputs, content policy violations. Production systems need graceful degradation, not crashes.

**Practical Implementation**:
```python
from typing import Optional, Callable, Any
import time