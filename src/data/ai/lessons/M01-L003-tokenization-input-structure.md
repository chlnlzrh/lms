# Tokenization & Input Structure: The Foundation of LLM Engineering

## Core Concepts

### What Tokenization Actually Is

Tokenization is the process of converting text into numerical sequences that language models can process. Unlike traditional string parsing that splits on whitespace or delimiters, LLM tokenization uses learned subword units that balance vocabulary size with semantic coverage.

Think of it like this: traditional parsing treats text as an array of characters or words, while tokenization treats text as a stream of meaning-carrying chunks that may or may not align with linguistic boundaries.

```python
# Traditional string parsing
text = "The AI can't process this easily!"
words = text.split()  # ['The', 'AI', "can't", 'process', 'this', 'easily!']
# Problem: Can't handle unknown words, massive vocabulary needed

# LLM tokenization (conceptual representation)
# Actual: ['The', ' AI', ' can', "'t", ' process', ' this', ' easily', '!']
# Each token maps to an integer: [464, 15592, 460, 470, 1920, 428, 6768, 0]
```

The key insight: **Tokenization directly impacts cost, context limits, model behavior, and output quality.** Every API call charges per token. Every model has a maximum token count. Poor tokenization understanding leads to unexpected truncations, higher costs, and degraded performance.

### Engineering Analogy: Database Indexing

Tokenization is similar to how databases index data:

```python
# Bad indexing: Store entire strings, compare character-by-character
def search_naive(database: list[str], query: str) -> list[str]:
    """O(n*m) - slow, memory-intensive"""
    return [item for item in database if query in item]

# Good indexing: Pre-process into searchable chunks
def search_indexed(index: dict[str, list[int]], query: str) -> list[int]:
    """O(1) average - fast, memory-efficient"""
    return index.get(query, [])
```

Like database indexing, tokenization pre-processes text into an optimized format that makes computation tractable. The model never "sees" your text—it only processes token IDs.

### Why This Matters NOW

1. **Cost Control**: GPT-4 costs $0.03/1K input tokens. A poorly structured prompt might use 2x-3x more tokens than necessary.
2. **Context Window Management**: With 128K token limits, understanding token usage is critical for long-document processing.
3. **Multilingual Applications**: Different languages have wildly different token densities (Chinese: ~1.5 chars/token, English: ~4 chars/token).
4. **Prompt Engineering**: Token boundaries affect model interpretation—spaces, capitalization, and formatting all change token sequences.

## Technical Components

### 1. Tokenization Algorithms

Most modern LLMs use **Byte-Pair Encoding (BPE)** or **WordPiece** variants. These algorithms build vocabularies by iteratively merging the most frequent character pairs.

```python
from typing import Dict, List

def simple_bpe_example(text: str, num_merges: int = 10) -> Dict[str, int]:
    """
    Simplified BPE to demonstrate the concept.
    Real implementations are more sophisticated.
    """
    # Start with character-level tokens
    vocab = {char: idx for idx, char in enumerate(set(text))}
    tokens = list(text)
    
    for merge_step in range(num_merges):
        # Count adjacent pairs
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        
        if not pairs:
            break
            
        # Merge most frequent pair
        best_pair = max(pairs, key=pairs.get)
        new_token = ''.join(best_pair)
        vocab[new_token] = len(vocab)
        
        # Apply merge
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    
    return vocab

# Example usage
text = "the the the theater"
vocab = simple_bpe_example(text, num_merges=3)
print(f"Vocabulary size: {len(vocab)}")
# Likely merges: 'th', 'the', 'the ' (space matters!)
```

**Practical Implications:**
- Common subwords become single tokens (e.g., "ing", "tion", "pre-")
- Rare words split into multiple tokens
- Spaces are often attached to the following word (" hello" vs "hello")
- Special characters may be separate tokens

**Real Constraints:**
- Vocabulary size typically 50K-100K tokens (balancing granularity vs. memory)
- Training on English-heavy data means non-English text uses more tokens
- Once a model is trained, its tokenizer is fixed—you can't modify it

### 2. Token Encoding and Decoding

```python
import tiktoken  # OpenAI's tokenizer library

def analyze_tokenization(text: str, model: str = "gpt-4") -> None:
    """
    Analyze how text gets tokenized and the implications.
    """
    encoding = tiktoken.encoding_for_model(model)
    
    # Encode to tokens
    tokens = encoding.encode(text)
    token_count = len(tokens)
    
    # Decode back (with token boundaries visible)
    decoded_tokens = [encoding.decode([token]) for token in tokens]
    
    print(f"Original text: '{text}'")
    print(f"Token count: {token_count}")
    print(f"Characters: {len(text)}")
    print(f"Chars per token: {len(text) / token_count:.2f}")
    print(f"\nToken breakdown:")
    for idx, token_text in enumerate(decoded_tokens):
        print(f"  {idx}: '{token_text}' (ID: {tokens[idx]})")
    print()

# Install: pip install tiktoken

# Example 1: English text
analyze_tokenization("The quick brown fox jumps over the lazy dog.")

# Example 2: Code
analyze_tokenization("def hello_world():\n    print('Hello!')")

# Example 3: Non-English (requires different token density)
analyze_tokenization("你好世界")  # Chinese: "Hello World"

# Example 4: Special characters
analyze_tokenization("Email: user@example.com, Cost: $99.99")
```

**Expected Output Insights:**
- English prose: ~4 characters per token
- Code: Often fewer chars/token (keywords are common)
- Chinese/Japanese: ~1.5-2 characters per token (3x more expensive!)
- Special characters often split awkwardly

**Key Learning:**
```python
def calculate_api_cost(text: str, 
                       input_cost_per_1k: float = 0.03,
                       model: str = "gpt-4") -> float:
    """Calculate actual API cost for given text."""
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(text))
    return (token_count / 1000) * input_cost_per_1k

# Compare costs
english_text = "This is a test message." * 100
chinese_text = "这是一个测试消息。" * 100

print(f"English cost: ${calculate_api_cost(english_text):.4f}")
print(f"Chinese cost: ${calculate_api_cost(chinese_text):.4f}")
# Chinese is typically 2-3x more expensive for same semantic content!
```

### 3. Special Tokens and Control Sequences

LLMs use special tokens to mark structure: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, etc.

```python
from typing import List, Dict

def format_chat_messages(messages: List[Dict[str, str]]) -> str:
    """
    Demonstrate how chat messages get structured internally.
    Actual format varies by model, but concept is universal.
    """
    formatted = ""
    
    # Special tokens (conceptual—actual tokens are model-specific)
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Pattern: <|im_start|>role\ncontent<|im_end|>
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Add generation prompt
    formatted += "<|im_start|>assistant\n"
    
    return formatted

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]

formatted = format_chat_messages(messages)
print(formatted)

# Calculate actual token usage including special tokens
encoding = tiktoken.encoding_for_model("gpt-4")
content_only = "You are a helpful assistant.What is 2+2?"
full_formatted = formatted

print(f"\nContent-only characters: {len(content_only)}")
print(f"With structure: {len(full_formatted)}")
print(f"Overhead: {len(full_formatted) - len(content_only)} characters")
```

**Practical Implications:**
- Chat formatting adds ~10-20 tokens per message (role markers, newlines)
- System messages consume tokens from your context window
- Multi-turn conversations accumulate overhead quickly

**Trade-off Example:**
```python
def compare_message_strategies(user_query: str) -> None:
    """Compare token usage of different message strategies."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    
    # Strategy 1: Detailed system message
    strategy1 = [
        {"role": "system", "content": "You are an expert Python developer with 10 years of experience. Always provide detailed explanations, include error handling, and follow PEP 8 style guidelines."},
        {"role": "user", "content": user_query}
    ]
    
    # Strategy 2: Minimal system message
    strategy2 = [
        {"role": "system", "content": "Python expert."},
        {"role": "user", "content": user_query}
    ]
    
    # Strategy 3: Instructions in user message
    strategy3 = [
        {"role": "user", "content": f"As a Python expert: {user_query}"}
    ]
    
    for idx, strategy in enumerate([strategy1, strategy2, strategy3], 1):
        full_text = "\n".join(m["content"] for m in strategy)
        tokens = len(encoding.encode(full_text))
        print(f"Strategy {idx}: {tokens} tokens")

compare_message_strategies("How do I read a CSV file?")
# Shows concrete token differences between approaches
```

### 4. Context Window and Truncation

Context windows are measured in tokens, not characters. Exceeding limits causes silent truncation or errors.

```python
from typing import List, Optional

def safe_context_window(
    text: str,
    max_tokens: int = 4096,
    model: str = "gpt-4",
    truncate_from: str = "end"
) -> tuple[str, bool]:
    """
    Safely handle text that might exceed context window.
    Returns: (truncated_text, was_truncated)
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text, False
    
    # Truncate intelligently
    if truncate_from == "end":
        truncated_tokens = tokens[:max_tokens]
    elif truncate_from == "start":
        truncated_tokens = tokens[-max_tokens:]
    elif truncate_from == "middle":
        # Keep beginning and end, remove middle
        keep_tokens = max_tokens // 2
        truncated_tokens = tokens[:keep_tokens] + tokens[-keep_tokens:]
    else:
        raise ValueError(f"Invalid truncate_from: {truncate_from}")
    
    truncated_text = encoding.decode(truncated_tokens)
    return truncated_text, True

# Test with a long document
long_text = "This is sentence number {}. " * 10000
long_text = long_text.format(*range(10000))

result, was_truncated = safe_context_window(long_text, max_tokens=100)
print(f"Was truncated: {was_truncated}")
print(f"Result length: {len(result)} characters")
print(f"Result: {result[:200]}...")
```

**Critical Insight:**
Token limits are hard constraints. A document that's 50,000 words in English might be 75,000 tokens—won't fit in a 32K context window even though it "should."

### 5. Token Boundaries and Semantic Meaning

Token boundaries affect model interpretation in subtle but important ways.

```python
def demonstrate_token_boundary_effects() -> None:
    """Show how token boundaries affect model behavior."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    
    # Example 1: Whitespace matters
    examples = [
        "python",      # Different tokens than below
        " python",     # Leading space changes tokenization
        "Python",      # Capitalization changes tokenization
        "PYTHON",      # All caps might be different again
    ]
    
    print("Whitespace and capitalization effects:")
    for example in examples:
        tokens = encoding.encode(example)
        print(f"  '{example}' -> {tokens} ({len(tokens)} token(s))")
    
    # Example 2: Code formatting
    code_compact = "def hello():print('hi')"
    code_spaced = "def hello():\n    print('hi')"
    
    print(f"\nCode formatting:")
    print(f"  Compact: {len(encoding.encode(code_compact))} tokens")
    print(f"  Spaced:  {len(encoding.encode(code_spaced))} tokens")
    
    # Example 3: Repeated patterns
    repeated_chars = "a" * 100
    repeated_word = "hello " * 20
    
    print(f"\nRepetition:")
    print(f"  100 'a's: {len(encoding.encode(repeated_chars))} tokens")
    print(f"  'hello ' x20: {len(encoding.encode(repeated_word))} tokens")

demonstrate_token_boundary_effects()
```

## Hands-On Exercises

### Exercise 1: Build a Token Budget Analyzer

**Objective:** Create a tool to analyze and optimize prompt token usage before API calls.

**Time:** 10 minutes

**Instructions:**

```python
import tiktoken
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TokenBudget:
    system_tokens: int
    user_tokens: int
    max_output_tokens: int
    total_input_tokens: int
    remaining_context: int
    estimated_cost_usd: float

def analyze_token_budget(
    system_message: Optional[str],
    user_message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    max_context_tokens: int = 4096,
    max_output_tokens: int = 500,
    input_cost_per_1k: float = 0.03,
    output_cost_per_1k: float = 0.06,
    model: str = "gpt-4"
) -> TokenBudget:
    """
    Analyze token budget for a prompt before sending to API.
    """
    encoding = tiktoken.encoding_for_model(model)
    
    # Calculate token counts
    system_tokens = len(encoding.encode(system_message)) if system_message else 0
    user_tokens = len(encoding.encode(user_message))
    
    # Add conversation history
    history_tokens = 0
    if conversation_history:
        for msg in conversation_history:
            history_tokens += len(encoding.encode(msg["content"]))
            history_tokens += 4  # Approximate overhead per message
    
    total_input = system_tokens +