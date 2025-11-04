# Context Window Management: Engineering for Token Limits

## Core Concepts

A context window is the maximum number of tokens—roughly 0.75 words per token—that a language model can process in a single request, including both input (prompt + conversation history + retrieved documents) and output (generated response). Unlike traditional databases where you query against unlimited storage, every interaction with an LLM operates within this fixed memory boundary.

**Traditional vs. Modern Approach:**

```python
# Traditional database query - no memory constraints
def get_customer_history(customer_id: str) -> List[Dict]:
    """Retrieve unlimited transaction history"""
    query = "SELECT * FROM transactions WHERE customer_id = ?"
    return database.execute(query, customer_id)  # Can return millions of rows

# LLM interaction - hard token limit
def analyze_customer_with_llm(
    customer_id: str,
    max_context_tokens: int = 8000  # Fixed constraint
) -> str:
    """Analyze customer with token budget"""
    history = get_customer_history(customer_id)
    
    # Must fit: system prompt + history + response space
    system_tokens = count_tokens("Analyze this customer data...")  # ~50
    response_budget = 1000  # Reserve for output
    available_for_history = max_context_tokens - system_tokens - response_budget
    
    # Critical: Must select/compress history to fit
    condensed_history = compress_to_token_limit(history, available_for_history)
    
    return llm.generate(
        system="Analyze this customer data...",
        user=condensed_history,
        max_tokens=response_budget
    )
```

**Key Engineering Insights:**

1. **Context windows are your working memory, not your database.** Just as RAM constrains in-memory operations, context windows constrain per-request reasoning. You must architect around this limitation.

2. **Every token has opportunity cost.** Using 2000 tokens for boilerplate instructions means 2000 fewer tokens for actual data, examples, or reasoning space.

3. **Compression is not optional—it's architectural.** Production systems need systematic approaches to select, summarize, and prioritize information within token budgets.

**Why This Matters Now:**

Modern applications—RAG systems, chatbots, code analysis tools—routinely exceed context limits. A customer support bot with 50-message conversations, a code reviewer analyzing 20 files, or a research assistant processing 100-page documents all face the same constraint: fitting unbounded data into bounded context. Poor context management results in truncated inputs, lost information, degraded quality, and failed requests. Engineers who master context window management build systems that scale gracefully as data grows.

## Technical Components

### 1. Token Counting and Budgeting

Tokens are the atomic units LLMs process. Understanding tokenization is essential for accurate budget management.

```python
from typing import List, Dict
import tiktoken  # OpenAI's tokenizer library

class TokenBudget:
    """Manage token allocation across prompt components"""
    
    def __init__(self, model: str = "gpt-4", total_limit: int = 8192):
        self.encoder = tiktoken.encoding_for_model(model)
        self.total_limit = total_limit
        self.allocations: Dict[str, int] = {}
    
    def count_tokens(self, text: str) -> int:
        """Accurate token count for given text"""
        return len(self.encoder.encode(text))
    
    def allocate(self, component: str, tokens: int) -> None:
        """Reserve tokens for a component"""
        if sum(self.allocations.values()) + tokens > self.total_limit:
            raise ValueError(f"Allocation exceeds limit: {self.total_limit}")
        self.allocations[component] = tokens
    
    def remaining(self) -> int:
        """Available tokens after allocations"""
        return self.total_limit - sum(self.allocations.values())
    
    def fits(self, text: str) -> bool:
        """Check if text fits in remaining budget"""
        return self.count_tokens(text) <= self.remaining()


# Example usage
budget = TokenBudget(model="gpt-4", total_limit=8192)

# Fixed allocations
system_prompt = "You are a helpful assistant analyzing code quality."
budget.allocate("system", budget.count_tokens(system_prompt))
budget.allocate("response", 1500)  # Reserve for output

# Dynamic content
code_files = load_code_files()
available = budget.remaining()
print(f"Available for code content: {available} tokens")
# Output: Available for code content: 6672 tokens
```

**Practical Implications:**

- Different models use different tokenizers (GPT-4, Claude, Llama). Always count tokens using the correct tokenizer for your model.
- Special characters, code, and non-English text often tokenize inefficiently (more tokens per character).
- Always reserve tokens for the response; requesting 8192 input tokens with an 8192-token limit will fail.

**Real Constraints:**

Token counting adds latency (typically 1-5ms for 10K tokens). For high-throughput systems, cache token counts for static content or estimate using character ratios (avg 4 characters ≈ 1 token for English).

### 2. Content Truncation Strategies

When content exceeds limits, you must decide what to keep. Naive truncation loses critical information.

```python
from enum import Enum
from typing import List, Callable

class TruncationStrategy(Enum):
    HEAD = "keep_beginning"
    TAIL = "keep_end"
    HEAD_TAIL = "keep_both_ends"
    SLIDING = "sliding_window"

def truncate_to_limit(
    text: str,
    token_limit: int,
    strategy: TruncationStrategy,
    encoder: tiktoken.Encoding
) -> str:
    """Truncate text using specified strategy"""
    tokens = encoder.encode(text)
    
    if len(tokens) <= token_limit:
        return text
    
    if strategy == TruncationStrategy.HEAD:
        # Keep beginning - good for following instructions
        truncated = tokens[:token_limit]
    
    elif strategy == TruncationStrategy.TAIL:
        # Keep end - good for recent context
        truncated = tokens[-token_limit:]
    
    elif strategy == TruncationStrategy.HEAD_TAIL:
        # Keep both ends - preserve context and recent info
        head_size = token_limit // 2
        tail_size = token_limit - head_size
        truncated = tokens[:head_size] + tokens[-tail_size:]
    
    elif strategy == TruncationStrategy.SLIDING:
        # Most recent window - for conversations
        truncated = tokens[-token_limit:]
    
    return encoder.decode(truncated)


# Practical example: Chat history management
encoder = tiktoken.encoding_for_model("gpt-4")

def manage_conversation_context(
    messages: List[Dict[str, str]],
    max_tokens: int = 6000
) -> List[Dict[str, str]]:
    """Keep conversation within token limit"""
    
    # Always preserve system message and last user message
    system_msg = messages[0]
    last_user_msg = messages[-1]
    middle_messages = messages[1:-1]
    
    # Calculate required space
    system_tokens = len(encoder.encode(system_msg["content"]))
    last_tokens = len(encoder.encode(last_user_msg["content"]))
    available = max_tokens - system_tokens - last_tokens
    
    # Fit as many recent messages as possible
    kept_messages = [system_msg]
    current_tokens = 0
    
    for msg in reversed(middle_messages):
        msg_tokens = len(encoder.encode(msg["content"]))
        if current_tokens + msg_tokens <= available:
            kept_messages.insert(1, msg)  # Insert after system
            current_tokens += msg_tokens
        else:
            break
    
    kept_messages.append(last_user_msg)
    return kept_messages
```

**Trade-offs:**

- **HEAD**: Preserves instructions but loses recent context. Use for task-focused prompts.
- **TAIL**: Keeps recent info but may lose task definition. Use for ongoing conversations.
- **HEAD_TAIL**: Balanced but creates discontinuity. Use when both context and recency matter.
- **SLIDING**: Natural for conversations but gradually forgets original context.

### 3. Semantic Compression

Intelligent compression preserves meaning while reducing tokens. More sophisticated than simple truncation.

```python
from typing import List, Dict, Optional
import numpy as np

class SemanticCompressor:
    """Compress content while preserving semantic value"""
    
    def __init__(self, encoder: tiktoken.Encoding):
        self.encoder = encoder
    
    def extract_key_sentences(
        self,
        text: str,
        target_tokens: int,
        ranking_fn: Optional[Callable] = None
    ) -> str:
        """Extract most important sentences to fit budget"""
        sentences = text.split('. ')
        
        if ranking_fn is None:
            # Default: prioritize sentences with key terms
            ranking_fn = self._default_sentence_ranker
        
        # Score each sentence
        scored = [(ranking_fn(s), s) for s in sentences]
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Add highest-ranked sentences until token limit
        selected = []
        current_tokens = 0
        
        for score, sentence in scored:
            sentence_tokens = len(self.encoder.encode(sentence))
            if current_tokens + sentence_tokens <= target_tokens:
                selected.append((score, sentence))
                current_tokens += sentence_tokens
            else:
                break
        
        # Restore original order
        selected.sort(key=lambda x: sentences.index(x[1]))
        return '. '.join([s for _, s in selected]) + '.'
    
    def _default_sentence_ranker(self, sentence: str) -> float:
        """Simple heuristic: prioritize sentences with numbers, questions, imperatives"""
        score = 0.0
        if any(char.isdigit() for char in sentence):
            score += 1.0  # Contains data
        if '?' in sentence:
            score += 0.5  # Question - likely important
        if sentence.lower().startswith(('must', 'should', 'will', 'note')):
            score += 0.75  # Imperative/important marker
        return score
    
    def summarize_list_items(self, items: List[str], target_tokens: int) -> str:
        """Compress list to fit budget"""
        total_tokens = sum(len(self.encoder.encode(item)) for item in items)
        
        if total_tokens <= target_tokens:
            return '\n'.join(f"- {item}" for item in items)
        
        # Compression strategy: keep count + sample
        count = len(items)
        samples = min(5, target_tokens // 50)  # Approx 50 tokens per item
        
        compressed = [
            f"Total items: {count}",
            "Sample entries:",
            *[f"- {items[i]}" for i in range(min(samples, count))]
        ]
        
        if count > samples:
            compressed.append(f"... and {count - samples} more")
        
        return '\n'.join(compressed)


# Example: Compress code review context
compressor = SemanticCompressor(encoder)

code_review_context = """
The authentication module implements JWT token validation. 
It uses RS256 algorithm for signing. 
The module was refactored last month.
Token expiration is set to 3600 seconds.
Error handling needs improvement in the refresh flow.
The code follows PEP 8 style guidelines.
Unit test coverage is 78%.
Performance benchmarks show 1200 requests per second.
"""

# Compress to 50 tokens while preserving key info
compressed = compressor.extract_key_sentences(
    code_review_context,
    target_tokens=50
)
print(compressed)
# Output focuses on technical specifics: algorithm, performance, coverage
```

**Practical Implications:**

Semantic compression trades processing time (1-10ms) for better context utilization. Use when input significantly exceeds limits and quality matters more than latency.

### 4. Hierarchical Context Management

Structure information by importance, loading detail only when budget allows.

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ContextLayer:
    """Hierarchical context with priority"""
    priority: int  # Lower = more important
    name: str
    content: str
    required: bool = False

class HierarchicalContext:
    """Manage context with priority-based loading"""
    
    def __init__(self, token_budget: TokenBudget):
        self.budget = token_budget
        self.layers: List[ContextLayer] = []
    
    def add_layer(
        self,
        name: str,
        content: str,
        priority: int = 5,
        required: bool = False
    ) -> None:
        """Add context layer with priority"""
        layer = ContextLayer(priority, name, content, required)
        self.layers.append(layer)
    
    def build_context(self) -> str:
        """Build context fitting within budget"""
        # Sort by priority (required first, then by priority number)
        sorted_layers = sorted(
            self.layers,
            key=lambda l: (not l.required, l.priority)
        )
        
        context_parts = []
        
        for layer in sorted_layers:
            tokens = self.budget.count_tokens(layer.content)
            
            if layer.required:
                # Must include, regardless of budget
                context_parts.append(f"[{layer.name}]\n{layer.content}")
            elif tokens <= self.budget.remaining():
                # Include if it fits
                context_parts.append(f"[{layer.name}]\n{layer.content}")
                self.budget.allocate(layer.name, tokens)
            else:
                # Skip this layer
                print(f"Skipping layer '{layer.name}' - insufficient budget")
        
        return "\n\n".join(context_parts)


# Example: Build context for code analysis
budget = TokenBudget(total_limit=4000)
budget.allocate("response", 1000)

ctx = HierarchicalContext(budget)

# Core context - always included
ctx.add_layer(
    "task",
    "Review this pull request for security vulnerabilities and performance issues.",
    priority=1,
    required=True
)

# Important context
ctx.add_layer(
    "current_code",
    open("auth_module.py").read(),
    priority=2,
    required=True
)

# Helpful but optional context
ctx.add_layer(
    "previous_reviews",
    "Past reviews flagged: SQL injection risk, missing input validation",
    priority=3
)

ctx.add_layer(
    "style_guide",
    "Follow Google Python Style Guide. Max line length 100 chars.",
    priority=4
)

ctx.add_layer(
    "test_results",
    "All 47 unit tests passing. Coverage: 82%",
    priority=5
)

final_context = ctx.build_context()
# Includes task + current_code + as much optional context as fits
```

**Real Constraints:**

Hierarchical management adds complexity. Use when you have clearly prioritizable context (instructions > data > examples > metadata) and budget varies significantly by request.

### 5. Streaming and Chunking for Long Content

Process content in chunks when analysis doesn't require full context simultaneously.

```python
from typing import Iterator, List, Dict
import json

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    encoder: tiktoken.Encoding
) -> Iterator[str]:
    """Split text into overlapping chunks"""
    tokens = encoder.encode(text)
    
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        yield encoder.decode(chunk_tokens)
        start = end - overlap  # Overlap to maintain context


def analyze_long_document(
    document: str,
    model_context_limit: int = 8000,
    analysis_prompt: str = "Summarize key points:"
) -> Dict[str, any]:
    """Analyze document exceeding context window"""
    encoder = tiktoken.