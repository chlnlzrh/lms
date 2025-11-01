# Context Window & Memory Architecture

## Core Concepts

### Technical Definition

A context window is the fixed-size buffer of tokens that a language model can process in a single forward pass. Unlike traditional stateful systems that maintain persistent memory, LLMs are stateless—they only "see" what's in the current context window. Every API call starts fresh; the model has no inherent memory of previous interactions.

Think of it like this: traditional databases keep state and can query historical data on demand. LLMs work more like pure functions—they receive all necessary information as input parameters and produce output without side effects. There's no session state, no database connection, no persistent memory between calls.

```python
# Traditional stateful system (database/session)
class TraditionalChatbot:
    def __init__(self):
        self.conversation_history = []  # Persistent state
    
    def respond(self, user_message: str) -> str:
        self.conversation_history.append(user_message)
        # Can reference ANY past interaction
        context = self.get_relevant_history()
        return self.generate_response(context)

# LLM approach (stateless)
def llm_chat(messages: list[dict]) -> str:
    # Must explicitly pass ALL context every time
    # No access to anything outside this function parameter
    response = api_call(messages)  # Fresh call, no memory
    return response

# Every call requires complete context
call_1 = llm_chat([{"role": "user", "content": "My name is Alice"}])
call_2 = llm_chat([{"role": "user", "content": "What's my name?"}])
# Returns: "I don't have that information" - previous call is invisible
```

### Engineering Analogy: HTTP vs. State Management

The context window challenge mirrors the transition from CGI scripts to modern web applications:

```python
# CGI-era: No state, every request standalone (like LLM calls)
def handle_request(request_data: str) -> str:
    # No access to previous requests
    return process(request_data)

# Modern web: Explicit state management (what you must do with LLMs)
def handle_request_with_session(request_data: str, session_id: str) -> str:
    session = load_session(session_id)  # Explicitly retrieve context
    response = process(request_data, session)
    save_session(session_id, session)  # Explicitly persist
    return response
```

Just as web developers learned to manage state through cookies, sessions, and databases, LLM engineers must explicitly manage context. The difference: your "session storage" has a hard size limit (the context window), and every access costs money and latency.

### Key Insights

1. **Memory is an illusion**: When a chatbot "remembers" your name, it's because your name is copied into every subsequent request, not because the model retained it.

2. **Token budgets are your primary constraint**: A 4K token context window isn't "4000 words of conversation"—it's 4000 tokens *total* for system instructions, conversation history, retrieved documents, and the model's response combined.

3. **Context is expensive**: Every token in your context window costs money and increases latency. A 100K token context costs 25x more than a 4K token context *per request*.

4. **You're doing garbage collection**: Unlike traditional memory management where the system handles cleanup, you must decide what stays in context and what gets discarded.

### Why This Matters Now

Context window architecture directly impacts:

- **Cost**: Your monthly API bill is roughly proportional to total tokens processed
- **Latency**: Time-to-first-token scales with context size (roughly linear)
- **Capability**: Poor context management causes the model to "forget" crucial information or include irrelevant data
- **Scale**: Inefficient context usage limits how many users you can serve

A typical mistake: including entire conversation histories in every call, burning 10K tokens for context when 1K would suffice, resulting in 10x higher costs and 3-5x higher latency.

## Technical Components

### 1. Token Accounting

**Technical Explanation**

Tokens are the atomic units of text processing for LLMs. They're not characters or words—they're subword units determined by the model's tokenizer. English text averages ~4 characters per token, but this varies significantly:

```python
from typing import List

def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters for English"""
    return len(text) // 4

# Accurate token counting requires the actual tokenizer
def count_tokens_accurate(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Accurate token counting using tiktoken.
    cl100k_base is used by GPT-4 and GPT-3.5-turbo.
    """
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

# Examples showing variance
examples = [
    "Hello world",           # 2 tokens
    "Hello, world!",         # 4 tokens (punctuation matters)
    "HelloWorld",            # 3 tokens (no space = different tokenization)
    "你好世界",               # 4 tokens (non-English varies more)
    "API" * 100,             # 100 tokens (repeated patterns)
]

for text in examples:
    estimated = estimate_tokens(text)
    actual = count_tokens_accurate(text)
    print(f"Text: '{text[:30]}...' | Estimated: {estimated} | Actual: {actual}")
```

**Practical Implications**

Your context budget includes:
- System prompts (50-500 tokens)
- User message (10-5000+ tokens)
- Conversation history (0-10000+ tokens)
- Retrieved documents/context (0-50000+ tokens)
- Model's response (reserved, typically 500-4000 tokens)

**Real Constraints**

Token limits are hard boundaries. Exceeding them causes:
1. Request rejection (400 error)
2. Automatic truncation (some providers silently truncate)
3. Degraded quality (model sees partial context)

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TokenBudget:
    """Track token allocation across context components"""
    max_tokens: int
    system_prompt: int
    reserved_for_response: int
    
    def available_for_context(self) -> int:
        """Tokens available for history + retrieved docs"""
        return self.max_tokens - self.system_prompt - self.reserved_for_response
    
    def is_within_budget(self, context_tokens: int) -> bool:
        """Check if context fits in budget"""
        return (self.system_prompt + context_tokens + 
                self.reserved_for_response) <= self.max_tokens

# Example: 4K context window
budget = TokenBudget(
    max_tokens=4096,
    system_prompt=200,
    reserved_for_response=800
)

print(f"Available for conversation/docs: {budget.available_for_context()}")  # 3096
print(f"Can fit 5K token document? {budget.is_within_budget(5000)}")  # False
```

### 2. Context Window Sliding & Truncation

**Technical Explanation**

When conversation history exceeds your token budget, you must decide what to keep. Common strategies:

1. **Head truncation**: Drop oldest messages
2. **Tail truncation**: Drop most recent messages (rarely useful)
3. **Sliding window**: Keep most recent N messages
4. **Selective retention**: Keep system prompt + recent messages + important past messages

```python
from typing import List, Dict, Optional

def sliding_window_truncate(
    messages: List[Dict[str, str]],
    max_tokens: int,
    system_prompt_tokens: int,
    tokenizer_func
) -> List[Dict[str, str]]:
    """
    Keep most recent messages that fit within token budget.
    Always preserve system prompt (first message).
    """
    if not messages:
        return []
    
    # Preserve system prompt
    result = [messages[0]]
    available_tokens = max_tokens - system_prompt_tokens
    
    # Add messages from most recent backwards
    for message in reversed(messages[1:]):
        message_tokens = tokenizer_func(message['content'])
        if message_tokens <= available_tokens:
            result.insert(1, message)  # Insert after system prompt
            available_tokens -= message_tokens
        else:
            break  # Stop when budget exhausted
    
    return result

def selective_truncate(
    messages: List[Dict[str, str]],
    max_tokens: int,
    system_prompt_tokens: int,
    keep_recent: int,
    tokenizer_func
) -> List[Dict[str, str]]:
    """
    Keep system prompt + last N messages + summarize the rest.
    More sophisticated than pure sliding window.
    """
    if len(messages) <= keep_recent + 1:  # +1 for system prompt
        return messages
    
    system_prompt = [messages[0]]
    recent_messages = messages[-keep_recent:]
    middle_messages = messages[1:-keep_recent]
    
    # Calculate tokens
    recent_tokens = sum(tokenizer_func(m['content']) for m in recent_messages)
    available_for_summary = max_tokens - system_prompt_tokens - recent_tokens
    
    # Create summary of middle section
    if middle_messages and available_for_summary > 100:
        summary = f"[Previous conversation with {len(middle_messages)} messages summarized]"
        summary_msg = {"role": "system", "content": summary}
        return system_prompt + [summary_msg] + recent_messages
    
    return system_prompt + recent_messages
```

**Practical Implications**

- **Sliding window**: Simple but model "forgets" early context completely
- **Selective retention**: Better coherence but requires summarization strategy
- **Cost trade-off**: Summarization requires extra API calls but reduces per-request tokens

**Real Constraints**

Each strategy has failure modes:
- Sliding window loses critical context from early conversation
- Summarization adds latency (extra API call) and can lose nuance
- Keeping too much history wastes tokens on irrelevant information

### 3. Message Role Architecture

**Technical Explanation**

LLMs use structured message formats with specific roles. The most common pattern:

```python
from typing import List, Dict, Literal
from dataclasses import dataclass

Role = Literal["system", "user", "assistant"]

@dataclass
class Message:
    role: Role
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

class ConversationBuilder:
    """Type-safe conversation construction"""
    
    def __init__(self, system_prompt: str):
        self.messages: List[Message] = [
            Message(role="system", content=system_prompt)
        ]
    
    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))
    
    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))
    
    def get_messages(self) -> List[Dict[str, str]]:
        return [m.to_dict() for m in self.messages]
    
    def validate(self) -> bool:
        """Ensure valid message structure"""
        if not self.messages or self.messages[0].role != "system":
            return False
        
        # Check for valid alternation (user/assistant should alternate)
        for i in range(1, len(self.messages) - 1):
            if self.messages[i].role == self.messages[i + 1].role:
                if self.messages[i].role in ["user", "assistant"]:
                    return False  # User and assistant shouldn't repeat
        
        return True

# Usage
conv = ConversationBuilder("You are a helpful coding assistant.")
conv.add_user_message("How do I reverse a list in Python?")
conv.add_assistant_message("Use list.reverse() or reversed(list)")
conv.add_user_message("What's the time complexity?")

messages = conv.get_messages()
```

**Practical Implications**

- **System role**: Sets behavioral context, always processed first, typically not repeated
- **User role**: Represents human input, drives model behavior
- **Assistant role**: Previous model outputs, necessary for multi-turn coherence

**Real Constraints**

Role ordering matters:
- System messages must come first (some APIs enforce this)
- User and assistant should alternate for best results
- Multiple consecutive user messages can confuse context
- Missing assistant messages break conversation flow

### 4. Context Stuffing & Retrieval

**Technical Explanation**

For tasks requiring external knowledge, you must inject relevant information into the context window. This is "context stuffing" or retrieval-augmented generation (RAG):

```python
from typing import List, Dict, Tuple

class ContextManager:
    """Manage context budget with retrieved documents"""
    
    def __init__(self, max_tokens: int, tokenizer_func):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer_func
    
    def build_context_with_docs(
        self,
        system_prompt: str,
        user_query: str,
        conversation_history: List[Dict[str, str]],
        retrieved_docs: List[str],
        max_response_tokens: int = 800
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Prioritize token budget: system > query > docs > history
        Returns (messages, total_tokens)
        """
        # Calculate base requirements
        system_tokens = self.tokenizer(system_prompt)
        query_tokens = self.tokenizer(user_query)
        
        budget = self.max_tokens - system_tokens - query_tokens - max_response_tokens
        
        # Allocate tokens to retrieved docs first
        doc_context = self._select_docs(retrieved_docs, budget // 2)
        doc_tokens = self.tokenizer(doc_context)
        
        # Remaining budget for conversation history
        remaining_budget = budget - doc_tokens
        history_context = self._truncate_history(
            conversation_history, remaining_budget
        )
        
        # Build final message structure
        enhanced_system_prompt = (
            f"{system_prompt}\n\n"
            f"Relevant information:\n{doc_context}"
        )
        
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            *history_context,
            {"role": "user", "content": user_query}
        ]
        
        total_tokens = sum(self.tokenizer(m['content']) for m in messages)
        return messages, total_tokens
    
    def _select_docs(self, docs: List[str], token_budget: int) -> str:
        """Select docs that fit in budget, most relevant first"""
        selected = []
        used_tokens = 0
        
        for doc in docs:
            doc_tokens = self.tokenizer(doc)
            if used_tokens + doc_tokens <= token_budget:
                selected.append(doc)
                used_tokens += doc_tokens
            else:
                break
        
        return "\n\n".join(selected)
    
    def _truncate_history(
        self, history: List[Dict[str, str]], token_budget: int
    ) -> List[Dict[str, str]]:
        """Keep most recent history that fits budget"""
        result = []
        used_tokens = 0
        
        for message in reversed(history):
            msg_tokens = self.tokenizer(message['content'])
            if used_tokens + msg_tokens <= token_budget:
                result.insert(0, message)
                used_tokens += msg_tokens
            else:
                break
        
        return result

# Example usage
def tokenizer_stub(text: str) -> int:
    """Stub tokenizer for example"""
    return len(text) // 4

manager = ContextManager(max_tokens=4096, tokenizer_func=tokenizer_stub)

retrieved_docs = [
    "Python's list.reverse() modifies the list in-place, O(n) time complexity.",
    "The reversed() function returns an iterator, doesn't modify original.",
    "List slicing [::-1] creates a new reversed list, uses O(n) extra space."
]

messages, total = manager.build_context_with_docs(
    system_prompt