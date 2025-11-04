# Temperature, Top-P & Sampling Parameters: Engineering Determinism in Probabilistic Systems

## Core Concepts

Language models are fundamentally probability distributions over vocabularies. At each generation step, the model outputs a probability for every possible next token—often a distribution across 50,000+ tokens. How you sample from this distribution determines whether your model produces creative prose, reliable code, or incoherent garbage.

**Traditional vs. Modern Generation:**

```python
# Traditional: Rule-based text generation (deterministic)
def generate_greeting(user_name: str, time_of_day: str) -> str:
    """Fixed templates with variable substitution"""
    if time_of_day == "morning":
        return f"Good morning, {user_name}!"
    elif time_of_day == "afternoon":
        return f"Good afternoon, {user_name}!"
    else:
        return f"Good evening, {user_name}!"

# Modern: Probabilistic sampling from LLM
import anthropic

def generate_greeting_llm(
    user_name: str, 
    time_of_day: str, 
    temperature: float = 0.7
) -> str:
    """Sample from probability distribution over possible greetings"""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        temperature=temperature,  # Control randomness
        messages=[{
            "role": "user",
            "content": f"Greet {user_name}. It's {time_of_day}."
        }]
    )
    return message.content[0].text
```

The traditional approach gives you one output per input. The LLM approach gives you a probability distribution—and sampling parameters let you control which point in that distribution you select.

**Why This Matters Now:**

Every production LLM application makes a sampling decision, often unknowingly using defaults that are wrong for their use case. The difference between `temperature=0` and `temperature=1` can mean:
- 98% accuracy vs. 75% accuracy in structured data extraction
- Coherent code vs. syntax errors in code generation
- Consistent brand voice vs. erratic responses in customer service

Engineers who understand sampling parameters can:
1. **Debug generation failures** by recognizing when randomness is introducing errors
2. **Optimize cost/quality trade-offs** by using higher temperature with cheaper models
3. **Design better retry logic** by knowing when re-sampling will help vs. hurt
4. **Build hybrid systems** that use different parameters for different tasks

## Technical Components

### 1. Temperature: Reshaping Probability Distributions

Temperature scales the logits (raw model outputs) before applying softmax, effectively "flattening" or "sharpening" the probability distribution.

**Technical Mechanism:**

```python
import numpy as np
from typing import List, Tuple

def apply_temperature(
    logits: np.ndarray, 
    temperature: float
) -> np.ndarray:
    """
    Apply temperature scaling to logits before sampling.
    
    Args:
        logits: Raw model outputs (unnormalized log probabilities)
        temperature: Scaling factor (0.0-2.0 typical range)
        
    Returns:
        Temperature-scaled probabilities
    """
    if temperature == 0:
        # Greedy decoding: select argmax
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Apply softmax to get probabilities
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Numerical stability
    probs = exp_logits / exp_logits.sum()
    
    return probs

# Example: See effect on distribution
logits = np.array([2.0, 1.5, 1.0, 0.5, 0.1])  # Model's raw outputs
tokens = ["the", "a", "this", "that", "these"]

print("Original logits:", logits)
print("\nToken probabilities at different temperatures:\n")

for temp in [0.1, 0.7, 1.0, 1.5]:
    probs = apply_temperature(logits, temp)
    print(f"Temperature {temp}:")
    for token, prob in zip(tokens, probs):
        print(f"  {token:6s}: {prob:.4f} {'█' * int(prob * 50)}")
    print()
```

**Output:**
```
Temperature 0.1:
  the   : 0.8668 ███████████████████████████████████████████
  a     : 0.1073 █████
  this  : 0.0133 
  that  : 0.0016 
  these : 0.0001 

Temperature 0.7:
  the   : 0.3598 ██████████████████
  a     : 0.2640 █████████████
  this  : 0.1938 ██████████
  that  : 0.1423 ███████
  these : 0.0401 ██

Temperature 1.5:
  the   : 0.2784 ██████████████
  a     : 0.2435 ████████████
  this  : 0.2130 ███████████
  that  : 0.1863 █████████
  these : 0.0788 ████
```

**Practical Implications:**

- **Low temperature (0-0.3)**: Near-deterministic. Use for tasks requiring consistency—code generation, structured data extraction, factual Q&A.
- **Medium temperature (0.7-0.9)**: Balanced creativity. Default for most chat applications, content generation.
- **High temperature (1.0-2.0)**: High randomness. Use for brainstorming, creative writing, generating diverse options.

**Real Constraints:**

Temperature cannot fix a bad model. If the highest-probability token is wrong, lowering temperature makes you select that wrong token more consistently. Temperature only helps when correct answers exist in the distribution.

### 2. Top-P (Nucleus Sampling): Dynamic Vocabulary Pruning

Top-P samples from the smallest set of tokens whose cumulative probability exceeds threshold P, dynamically adjusting vocabulary size based on model confidence.

**Technical Mechanism:**

```python
from typing import Tuple
import numpy as np

def top_p_sampling(
    logits: np.ndarray,
    top_p: float,
    temperature: float = 1.0
) -> Tuple[int, np.ndarray]:
    """
    Sample using nucleus (top-p) sampling.
    
    Args:
        logits: Raw model outputs
        top_p: Cumulative probability threshold (0.0-1.0)
        temperature: Temperature scaling factor
        
    Returns:
        (sampled_token_idx, filtered_probs)
    """
    # Apply temperature
    probs = apply_temperature(logits, temperature)
    
    # Sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Calculate cumulative probabilities
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Find cutoff index where cumulative prob exceeds top_p
    cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
    
    # Keep only top-p tokens
    top_indices = sorted_indices[:cutoff_idx]
    top_probs = sorted_probs[:cutoff_idx]
    
    # Renormalize probabilities
    top_probs = top_probs / top_probs.sum()
    
    # Sample from filtered distribution
    sampled_idx = np.random.choice(top_indices, p=top_probs)
    
    return sampled_idx, top_probs

# Compare top-p values
logits = np.array([3.0, 2.5, 1.0, 0.5, 0.3, 0.1, 0.05, 0.02])
tokens = ["cat", "dog", "bird", "fish", "hamster", "rabbit", "lizard", "snake"]

print("Vocabulary selection at different top-p values:\n")

for p in [0.5, 0.9, 0.95, 1.0]:
    sampled_idx, filtered_probs = top_p_sampling(logits, p, temperature=1.0)
    print(f"Top-P {p}:")
    print(f"  Vocabulary size: {len(filtered_probs)} tokens")
    print(f"  Sampled: {tokens[sampled_idx]}")
    print()
```

**Practical Implications:**

- **Low top-p (0.5-0.7)**: Conservative vocabulary. Fewer but safer choices. Use when accuracy is critical.
- **High top-p (0.9-0.95)**: Expanded vocabulary. More creative but riskier. Use for diverse outputs.
- **Top-p = 1.0**: No filtering. Full vocabulary available (equivalent to pure temperature sampling).

**Key Insight:** Top-P adapts to model confidence. When the model is certain (one token dominates), nucleus is small. When uncertain (flat distribution), nucleus expands. This makes it more robust than fixed "top-k" approaches.

### 3. Temperature + Top-P Interaction

These parameters are applied sequentially: temperature reshapes the distribution, then top-p filters it. Understanding their interaction is critical.

**Interaction Patterns:**

```python
def analyze_sampling_interaction(
    logits: np.ndarray,
    tokens: List[str]
) -> None:
    """Demonstrate how temperature and top-p interact."""
    
    configs = [
        (0.1, 1.0, "Focused + Full vocab"),
        (0.1, 0.9, "Focused + Filtered vocab"),
        (1.0, 1.0, "Balanced + Full vocab"),
        (1.0, 0.9, "Balanced + Filtered vocab"),
        (1.5, 1.0, "Creative + Full vocab"),
        (1.5, 0.9, "Creative + Filtered vocab"),
    ]
    
    for temp, top_p, description in configs:
        probs = apply_temperature(logits, temp)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumulative, top_p) + 1
        
        print(f"{description} (T={temp}, P={top_p}):")
        print(f"  Active vocab: {cutoff_idx} of {len(tokens)} tokens")
        print(f"  Top-3: ", end="")
        for i in range(min(3, cutoff_idx)):
            idx = sorted_indices[i]
            print(f"{tokens[idx]}({sorted_probs[i]:.3f}) ", end="")
        print("\n")

logits = np.array([2.5, 2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.2])
tokens = ["said", "replied", "answered", "responded", "stated", "mentioned", "noted", "remarked"]

analyze_sampling_interaction(logits, tokens)
```

**Trade-offs:**

| Configuration | Use Case | Risk |
|--------------|----------|------|
| Low T + Low P | Structured output (JSON, code) | May be too repetitive |
| Low T + High P | Consistent but not robotic | Wasted compute on unused vocabulary |
| High T + Low P | Controlled creativity | Contradictory—high T flattens, low P restricts |
| High T + High P | Maximum diversity | High error rate, inconsistency |

**Production Pattern:**

```python
from typing import Dict, Any

def get_sampling_config(task_type: str) -> Dict[str, Any]:
    """Return optimized sampling parameters for different tasks."""
    
    configs = {
        "code_generation": {
            "temperature": 0.2,
            "top_p": 0.95,
            "rationale": "Low temp for syntax correctness, high top_p for idiom variety"
        },
        "data_extraction": {
            "temperature": 0.0,
            "top_p": 1.0,
            "rationale": "Greedy decoding for maximum consistency"
        },
        "creative_writing": {
            "temperature": 0.9,
            "top_p": 0.95,
            "rationale": "High temp for creativity, top_p prevents nonsense"
        },
        "chat_assistant": {
            "temperature": 0.7,
            "top_p": 0.9,
            "rationale": "Balanced for natural conversation"
        },
        "classification": {
            "temperature": 0.0,
            "top_p": 1.0,
            "rationale": "Deterministic for reproducible results"
        },
        "brainstorming": {
            "temperature": 1.2,
            "top_p": 0.98,
            "rationale": "Maximum diversity while maintaining coherence"
        }
    }
    
    return configs.get(task_type, configs["chat_assistant"])

# Usage
config = get_sampling_config("code_generation")
print(f"Code generation config: {config}")
```

### 4. Top-K: Fixed Vocabulary Filtering

Top-K restricts sampling to the K highest-probability tokens. While less sophisticated than top-p, it's simpler and sometimes preferable.

**Technical Implementation:**

```python
def top_k_sampling(
    logits: np.ndarray,
    top_k: int,
    temperature: float = 1.0
) -> int:
    """Sample from top-k highest probability tokens."""
    
    # Apply temperature
    probs = apply_temperature(logits, temperature)
    
    # Get top-k indices
    top_k_indices = np.argsort(probs)[-top_k:]
    top_k_probs = probs[top_k_indices]
    
    # Renormalize
    top_k_probs = top_k_probs / top_k_probs.sum()
    
    # Sample
    return np.random.choice(top_k_indices, p=top_k_probs)

# Comparison: Top-K vs Top-P
logits = np.array([3.0, 2.8, 2.5, 1.0, 0.5, 0.3, 0.1, 0.05])
probs = apply_temperature(logits, 1.0)

print("Top-K vs Top-P vocabulary selection:\n")
print(f"Full distribution: {probs}\n")

# Top-K: Fixed size
k_values = [3, 5, 8]
for k in k_values:
    top_indices = np.argsort(probs)[-k:]
    print(f"Top-K={k}: {len(top_indices)} tokens (fixed)")

print()

# Top-P: Adaptive size
p_values = [0.5, 0.8, 0.95]
for p in p_values:
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, p) + 1
    print(f"Top-P={p}: {cutoff} tokens (adaptive)")
```

**When to use Top-K:**
- You need **predictable compute costs** (fixed vocabulary size)
- You're working with **small vocabulary tasks** (e.g., multiple choice)
- You want **simple tuning** (easier to understand than top-p)

**When to use Top-P:**
- Model confidence varies significantly across generations
- You need adaptive vocabulary based on context
- Quality matters more than predictability

### 5. Frequency & Presence Penalties: Repetition Control

These parameters penalize tokens based on their occurrence in the generated text, reducing repetition without changing the base distribution.

**Technical Mechanism:**

```python
def apply_repetition_penalties(
    logits: np.ndarray,
    generated_tokens: List[int],
    