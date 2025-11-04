# Memory & State Management in LLM Applications

## Core Concepts

Large Language Models are stateless by design. Each API call is independent—the model doesn't "remember" previous interactions unless you explicitly provide that context. This architectural decision, while computationally efficient for model providers, shifts the entire burden of memory and state management to application developers.

### Traditional vs. Modern State Management

In traditional stateful applications, session state persists server-side:

```python
# Traditional web application (simplified)
class UserSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history: list[dict] = []
        
    def add_message(self, message: str) -> str:
        # State automatically persists server-side
        self.conversation_history.append({"user": message})
        response = self.process(message)
        self.conversation_history.append({"bot": response})
        return response
```

With LLMs, you must explicitly pass all relevant context with every request:

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

@dataclass
class Message:
    role: str  # "system", "user", or "assistant"
    content: str
    timestamp: Optional[float] = None

class LLMConversation:
    def __init__(self, system_prompt: str, max_context_tokens: int = 4000):
        self.messages: List[Message] = [
            Message(role="system", content=system_prompt)
        ]
        self.max_context_tokens = max_context_tokens
        
    def add_message(self, role: str, content: str) -> List[Dict[str, str]]:
        """Add message and return full context for next API call."""
        import time
        self.messages.append(
            Message(role=role, content=content, timestamp=time.time())
        )
        
        # Must manually manage what gets sent to the model
        return self._prepare_context()
    
    def _prepare_context(self) -> List[Dict[str, str]]:
        """Convert messages to API format."""
        return [
            {"role": msg.role, "content": msg.content} 
            for msg in self.messages
        ]
```

### Key Insights

**1. Context is Currency**: Every token in your conversation history costs money and latency. A 10-message conversation with 200 tokens per message consumes 2,000 tokens of context on every subsequent request—even if only the last 2 messages are relevant.

**2. Memory Architecture Defines User Experience**: Poor memory management creates conversations where the model "forgets" critical information or, conversely, becomes slower and more expensive as conversations grow.

**3. State Management is Application-Specific**: There's no universal solution. A customer service chatbot, a code review assistant, and a creative writing tool require fundamentally different memory strategies.

### Why This Matters Now

Production LLM applications fail most often not from poor prompts, but from poor state management. As conversations extend beyond 10-20 turns, naive implementations either:
- Exceed context windows and crash
- Become prohibitively expensive (10x cost increases are common)
- Degrade in quality as relevant context gets diluted
- Experience significant latency increases (2-5 second delays)

Effective memory management is the difference between a prototype and a production-ready application.

## Technical Components

### 1. Conversation Buffer Management

The most basic form of memory: store all messages and send them with each request.

```python
from typing import List, Dict
import tiktoken  # tokenizer library

class BufferMemory:
    def __init__(self, model: str = "gpt-4"):
        self.messages: List[Dict[str, str]] = []
        self.encoding = tiktoken.encoding_for_model(model)
        
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
    
    def get_context(self) -> List[Dict[str, str]]:
        return self.messages.copy()
    
    def count_tokens(self) -> int:
        """Estimate token count for current context."""
        # Rough approximation - actual tokenization is more complex
        total = 0
        for msg in self.messages:
            total += len(self.encoding.encode(msg["content"]))
            total += 4  # Overhead per message
        return total
    
    def is_within_limit(self, max_tokens: int) -> bool:
        return self.count_tokens() <= max_tokens

# Usage
memory = BufferMemory()
memory.add_message("system", "You are a helpful assistant.")
memory.add_message("user", "What is Python?")
memory.add_message("assistant", "Python is a high-level programming language...")

print(f"Current context: {memory.count_tokens()} tokens")
print(f"Within 4K limit: {memory.is_within_limit(4000)}")
```

**Practical Implications**: 
- Simple to implement and debug
- Works well for short conversations (< 10 turns)
- Predictable behavior—no information loss

**Constraints**:
- Context window limits (4K-128K tokens depending on model)
- Linear cost scaling: 20 turns = 20x the tokens of 1 turn
- No prioritization—all context treated equally

### 2. Sliding Window Memory

Keep only the N most recent messages, discarding older ones.

```python
from collections import deque
from typing import List, Dict, Optional

class SlidingWindowMemory:
    def __init__(self, window_size: int = 10, preserve_system: bool = True):
        """
        Args:
            window_size: Number of recent messages to retain
            preserve_system: Always keep system message regardless of window
        """
        self.window_size = window_size
        self.preserve_system = preserve_system
        self.system_message: Optional[Dict[str, str]] = None
        self.messages: deque = deque(maxlen=window_size)
    
    def add_message(self, role: str, content: str) -> None:
        if role == "system" and self.preserve_system:
            self.system_message = {"role": role, "content": content}
        else:
            self.messages.append({"role": role, "content": content})
    
    def get_context(self) -> List[Dict[str, str]]:
        context = []
        if self.system_message:
            context.append(self.system_message)
        context.extend(list(self.messages))
        return context

# Comparative example
buffer = BufferMemory()
sliding = SlidingWindowMemory(window_size=6)

for i in range(20):
    buffer.add_message("user", f"Message {i}")
    sliding.add_message("user", f"Message {i}")

print(f"Buffer size: {len(buffer.messages)} messages")  # 20
print(f"Sliding window: {len(sliding.messages)} messages")  # 6
print(f"Buffer tokens: {buffer.count_tokens()}")  # ~400 tokens
print(f"Memory savings: {(1 - 6/20) * 100:.0f}%")  # 70% reduction
```

**Practical Implications**:
- Constant memory and cost regardless of conversation length
- Predictable performance characteristics
- Good for task-focused conversations with limited context needs

**Constraints**:
- Information loss: model "forgets" older context
- No intelligent selection—recent ≠ relevant
- Can cause confusion in long, multi-topic conversations

### 3. Summarization-Based Memory

Periodically compress older messages into summaries, preserving essential information while reducing tokens.

```python
from typing import List, Dict, Optional
import json

class SummarizationMemory:
    def __init__(self, 
                 summarization_threshold: int = 10,
                 keep_recent: int = 4):
        """
        Args:
            summarization_threshold: Trigger summary after N messages
            keep_recent: Always keep this many recent messages unsummarized
        """
        self.threshold = summarization_threshold
        self.keep_recent = keep_recent
        self.summary: Optional[str] = None
        self.messages: List[Dict[str, str]] = []
        
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        
        # Check if summarization needed (excluding system message)
        non_system = [m for m in self.messages if m["role"] != "system"]
        if len(non_system) >= self.threshold:
            self._trigger_summarization()
    
    def _trigger_summarization(self) -> None:
        """
        In production, this would call the LLM to generate a summary.
        Here we show the structure.
        """
        # Separate system message, old messages, and recent messages
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        non_system = [m for m in self.messages if m["role"] != "system"]
        
        to_summarize = non_system[:-self.keep_recent]
        keep_messages = non_system[-self.keep_recent:]
        
        # Create summary prompt (in production, call LLM here)
        summary_prompt = self._create_summary_prompt(to_summarize)
        
        # Simulated summary (in production: llm_response = call_llm(summary_prompt))
        new_summary = f"[Summary of {len(to_summarize)} messages covering: ...]"
        
        # Update state
        if self.summary:
            self.summary = f"{self.summary}\n\n{new_summary}"
        else:
            self.summary = new_summary
            
        self.messages = system_msgs + keep_messages
    
    def _create_summary_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Generate prompt for summarization."""
        conversation = "\n".join([
            f"{m['role']}: {m['content']}" for m in messages
        ])
        return f"""Summarize the following conversation, preserving:
- Key facts and decisions
- User preferences or requirements
- Important context for future messages

Conversation:
{conversation}

Summary:"""
    
    def get_context(self) -> List[Dict[str, str]]:
        """Return summary + recent messages."""
        if not self.summary:
            return self.messages
        
        # Insert summary as a system-level context message
        context = []
        for msg in self.messages:
            if msg["role"] == "system":
                # Append summary to system message
                enhanced = msg.copy()
                enhanced["content"] = f"{msg['content']}\n\nPrevious conversation summary:\n{self.summary}"
                context.append(enhanced)
            else:
                context.append(msg)
        
        # If no system message, add summary as one
        if not any(m["role"] == "system" for m in self.messages):
            context.insert(0, {
                "role": "system",
                "content": f"Previous conversation summary:\n{self.summary}"
            })
            context.extend(self.messages)
            
        return context

# Usage comparison
buffer = BufferMemory()
summary_mem = SummarizationMemory(summarization_threshold=8, keep_recent=3)

# Simulate 15-message conversation
for i in range(15):
    user_msg = f"User question {i} about topic {i // 3}"
    asst_msg = f"Assistant response to question {i}"
    
    buffer.add_message("user", user_msg)
    buffer.add_message("assistant", asst_msg)
    
    summary_mem.add_message("user", user_msg)
    summary_mem.add_message("assistant", asst_msg)

print(f"Buffer messages: {len(buffer.messages)}")  # 30
print(f"Summary memory messages: {len(summary_mem.messages)}")  # 6
print(f"Has summary: {summary_mem.summary is not None}")  # True
```

**Practical Implications**:
- Scales to very long conversations (100+ turns)
- Maintains bounded context size
- Preserves important historical information

**Constraints**:
- Additional LLM call for summarization (cost + latency)
- Lossy compression—details may be lost
- Quality depends on summarization prompt and model capability
- Complexity in implementation and debugging

### 4. Semantic Memory with Vector Storage

Store conversation history in a vector database and retrieve only relevant messages based on semantic similarity to the current query.

```python
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class StoredMessage:
    role: str
    content: str
    embedding: np.ndarray
    timestamp: float
    message_id: int

class SemanticMemory:
    def __init__(self, top_k: int = 5):
        """
        Args:
            top_k: Number of most relevant messages to retrieve
        """
        self.top_k = top_k
        self.stored_messages: List[StoredMessage] = []
        self.message_counter = 0
        
    def add_message(self, role: str, content: str) -> None:
        """
        In production, embedding would come from an embedding model.
        """
        import time
        
        # Simulated embedding (in production: embedding = embed_model(content))
        embedding = self._create_embedding(content)
        
        msg = StoredMessage(
            role=role,
            content=content,
            embedding=embedding,
            timestamp=time.time(),
            message_id=self.message_counter
        )
        self.stored_messages.append(msg)
        self.message_counter += 1
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Placeholder for actual embedding model.
        In production, use OpenAI embeddings, sentence-transformers, etc.
        """
        # Simulated: random embedding for demonstration
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384)  # 384-dimensional embedding
    
    def get_relevant_context(self, query: str, max_messages: int = None) -> List[Dict[str, str]]:
        """Retrieve semantically similar messages."""
        if not self.stored_messages:
            return []
        
        query_embedding = self._create_embedding(query)
        
        # Calculate similarity scores
        similarities: List[Tuple[float, StoredMessage]] = []
        for msg in self.stored_messages:
            similarity = self._cosine_similarity(query_embedding, msg.embedding)
            similarities.append((similarity, msg))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Get top-k
        k = max_messages or self.top_k
        relevant = similarities[:k]
        
        # Convert to message format, preserving chronological order
        messages = [msg for _, msg in relevant]
        messages.sort(key=lambda x: x.message_id)
        
        return [{"role": m.role, "content": m.content} for m in messages]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Usage example
semantic_mem = SemanticMemory(top_k=3)

# Simulate conversation with multiple topics
conversations = [
    ("user", "What's the capital of France?"),
    ("assistant", "The capital of France is Paris."),
    ("user", "Tell me about Python programming."),
    ("assistant", "Python is a versatile programming language..."),
    ("user", "What's the weather like in Paris?"),
    ("assistant", "I don't have real-time weather data..."),
    ("user", "How do I install Python?"),
    ("assistant", "You can download Python from python.org..."),
]

for role, content in conversations:
    semantic_mem.add_message(role, content)

# Query for Python-related context
query = "How do I debug Python code?"
relevant = semantic_mem.get_relevant_context(query)

print(f"Total stored messages: