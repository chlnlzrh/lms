# Generative AI Fundamentals

## Core Concepts

### What Generative AI Actually Is

Generative AI refers to machine learning systems that create new content—text, images, code, audio—by learning patterns from existing data and generating novel outputs that statistically resemble their training data. Unlike traditional AI systems that classify, predict, or optimize based on fixed rules, generative models learn probability distributions and sample from them to produce original content.

The fundamental shift: traditional software executes predetermined logic paths, while generative AI synthesizes responses by computing probability distributions over possible outputs.

### Engineering Analogy: Template Systems vs. Generative Systems

**Traditional Approach (Template-Based):**

```python
from typing import Dict, List

class TraditionalEmailGenerator:
    """Rule-based content generation using templates"""
    
    def __init__(self):
        self.templates: Dict[str, str] = {
            'welcome': 'Hello {name}, welcome to {service}!',
            'reminder': 'Hi {name}, you have {count} pending items.',
            'error': 'Dear {name}, an error occurred: {error_code}'
        }
    
    def generate(self, template_type: str, **kwargs) -> str:
        """Generate content by filling template slots"""
        if template_type not in self.templates:
            raise ValueError(f"Unknown template: {template_type}")
        
        template = self.templates[template_type]
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")

# Usage
traditional = TraditionalEmailGenerator()
print(traditional.generate('welcome', name='Alice', service='CloudApp'))
# Output: "Hello Alice, welcome to CloudApp!"

# Limitations: 
# - Only produces predefined patterns
# - Cannot adapt tone or style
# - Requires explicit template for every scenario
# - No understanding of context or intent
```

**Generative AI Approach:**

```python
from typing import Optional
import json

class GenerativeEmailSystem:
    """Simulates generative AI behavior for email creation"""
    
    def __init__(self, model_endpoint: str):
        self.model_endpoint = model_endpoint
    
    def generate(
        self, 
        intent: str, 
        context: Dict[str, any],
        style: Optional[str] = None,
        constraints: Optional[Dict[str, any]] = None
    ) -> str:
        """
        Generate content based on intent and context.
        Model learns patterns and creates appropriate response.
        """
        prompt = self._construct_prompt(intent, context, style, constraints)
        
        # In reality, this calls a language model API
        # Model has learned email patterns from millions of examples
        response = self._call_model(prompt)
        return response
    
    def _construct_prompt(
        self, 
        intent: str, 
        context: Dict[str, any],
        style: Optional[str],
        constraints: Optional[Dict[str, any]]
    ) -> str:
        """Build prompt that guides generation"""
        prompt_parts = [f"Intent: {intent}"]
        
        for key, value in context.items():
            prompt_parts.append(f"{key}: {value}")
        
        if style:
            prompt_parts.append(f"Style: {style}")
        
        if constraints:
            prompt_parts.append(f"Constraints: {json.dumps(constraints)}")
        
        return "\n".join(prompt_parts)
    
    def _call_model(self, prompt: str) -> str:
        """Placeholder for actual model inference"""
        # Real implementation would call API endpoint
        # Model generates token-by-token based on learned probabilities
        pass

# Usage
generative = GenerativeEmailSystem(model_endpoint="https://api.example/v1")

# Same intent, different context - generates appropriate variations
email1 = generative.generate(
    intent="welcome new user",
    context={
        'name': 'Alice',
        'service': 'CloudApp',
        'signup_method': 'referral',
        'referrer': 'Bob'
    },
    style="friendly and professional"
)

# Model can handle scenarios never explicitly programmed
email2 = generative.generate(
    intent="apologize for service disruption",
    context={
        'name': 'Alice',
        'issue': 'database timeout',
        'downtime_hours': 3,
        'compensation': 'one month credit'
    },
    style="empathetic and technical",
    constraints={'max_length': 150}
)

# Capabilities:
# - Adapts to novel scenarios
# - Varies tone and style naturally
# - Understands context and relationships
# - Generates human-like, coherent content
```

### Key Insights That Change Engineering Thinking

**1. From Deterministic to Probabilistic Systems**

Traditional code: `if condition: return result_a else: return result_b`

Generative AI: Returns a distribution over possible outputs, sampled based on learned patterns. Same input can yield different outputs.

**2. From Explicit Programming to Learned Behavior**

You don't program logic—you provide examples (training data) and objectives (loss functions). The model discovers patterns you didn't explicitly encode.

**3. From Perfect Reliability to Statistical Reliability**

Traditional systems: 100% predictable behavior within specified domains.

Generative AI: High probability of appropriate responses, but outputs require validation. Trade determinism for flexibility.

**4. From Local to Global Pattern Matching**

Traditional regex: Matches exact patterns.

Generative models: Learn semantic relationships across vast context, understanding that "car" relates to "vehicle," "transportation," "wheels" through learned representations.

### Why This Matters Now

**Capability Threshold Crossed:** Models now generate production-quality content across domains—code, technical documentation, customer service responses, data transformation scripts—at scale and cost previously impossible.

**Economic Shift:** Tasks requiring human-level language understanding that cost $50-100/hour can now be automated at $0.01-1.00 per task with 80-95% quality thresholds.

**Engineering Bottleneck Removal:** Generative AI eliminates the need to manually code every variation, edge case, and scenario. Instead of maintaining thousands of templates, you maintain prompts and validation logic.

**New Capability Class:** Systems can now handle tasks impossible with traditional programming: semantic search, contextual summarization, creative problem-solving, natural language interfaces to structured systems.

## Technical Components

### 1. Tokens: The Atomic Unit of Generative AI

**Technical Explanation:**

Generative models don't process text as characters or words—they process tokens. A token represents a chunk of text (typically 3-4 characters for English) encoded as an integer. Models learn relationships between token IDs through training.

Tokenization splits input text into these subword units using algorithms like Byte-Pair Encoding (BPE) or WordPiece. The vocabulary size (typically 32k-100k tokens) represents all possible token IDs the model can process.

**Practical Implications:**

```python
from typing import List
import re

class SimpleTokenizer:
    """Simplified tokenizer demonstrating core concepts"""
    
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.pattern = re.compile(r'\w+|[^\w\s]')
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        words = self.pattern.findall(text.lower())
        tokens = []
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Unknown word handling: break into subwords
                tokens.extend(self._handle_unknown(word))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        tokens = [self.inverse_vocab.get(tid, '<unk>') for tid in token_ids]
        return ' '.join(tokens)
    
    def _handle_unknown(self, word: str) -> List[int]:
        """Break unknown words into character-level tokens"""
        return [self.vocab.get(c, self.vocab['<unk>']) for c in word]
    
    def count_tokens(self, text: str) -> int:
        """Essential for cost calculation and context management"""
        return len(self.encode(text))

# Example vocabulary (real vocabularies have 32k-100k tokens)
vocab = {
    'hello': 1, 'world': 2, 'the': 3, 'a': 4, 'is': 5,
    'artificial': 6, 'intelligence': 7, 'ai': 8,
    '<unk>': 0, '<pad>': 9, '<start>': 10, '<end>': 11
}

tokenizer = SimpleTokenizer(vocab)

# Token counting affects costs and limits
text1 = "Hello world"
tokens1 = tokenizer.encode(text1)
print(f"Text: '{text1}'")
print(f"Tokens: {tokens1}")
print(f"Token count: {tokenizer.count_tokens(text1)}")
# Output: Tokens: [1, 2], Token count: 2

text2 = "Artificial Intelligence is transforming software"
tokens2 = tokenizer.encode(text2)
print(f"\nText: '{text2}'")
print(f"Token count: {tokenizer.count_tokens(text2)}")
# Token count affects API costs (e.g., $0.03 per 1k tokens)
```

**Real Constraints:**

- **Context limits:** Models have maximum token windows (4k, 8k, 32k, 128k). Exceeding this truncates input or causes errors.
- **Cost scaling:** API pricing is per-token. A 10k token request costs 10x a 1k token request.
- **Token efficiency varies by language:** English: ~4 chars/token. Python code: ~3 chars/token. Non-Latin languages: less efficient.

**Concrete Example:**

```python
def estimate_api_cost(text: str, price_per_1k_tokens: float = 0.03) -> float:
    """Estimate cost for processing text"""
    # Real tokenizers more sophisticated, but approximation:
    estimated_tokens = len(text) / 4  # ~4 chars per token for English
    cost = (estimated_tokens / 1000) * price_per_1k_tokens
    return cost

document = "A" * 50000  # 50k character document
print(f"Estimated cost: ${estimate_api_cost(document):.2f}")
# Output: ~$0.38 for 12.5k tokens

# Optimization: summarize first, then process
summary = document[:4000]  # Reduce to 1k tokens
print(f"Optimized cost: ${estimate_api_cost(summary):.2f}")
# Output: ~$0.03 (12x cheaper)
```

### 2. Temperature: Controlling Randomness vs. Determinism

**Technical Explanation:**

Temperature is a parameter (typically 0.0-2.0) that controls randomness in output generation. During generation, the model computes probability distributions over possible next tokens. Temperature scales these probabilities before sampling:

- **Temperature = 0:** Always selects highest probability token (deterministic)
- **Temperature = 1:** Samples from unmodified probability distribution (balanced)
- **Temperature > 1:** Flattens distribution, increasing randomness
- **Temperature < 1:** Sharpens distribution, favoring high-probability tokens

**Practical Implications:**

```python
import numpy as np
from typing import List, Tuple

class TemperatureSampler:
    """Demonstrates temperature's effect on token selection"""
    
    def __init__(self, vocabulary: List[str]):
        self.vocabulary = vocabulary
    
    def sample_next_token(
        self, 
        logits: np.ndarray,
        temperature: float = 1.0
    ) -> Tuple[str, float]:
        """
        Sample next token given model logits and temperature.
        
        Args:
            logits: Raw model outputs (unnormalized log probabilities)
            temperature: Controls randomness
        
        Returns:
            (selected_token, probability)
        """
        if temperature == 0:
            # Greedy selection - always pick highest probability
            token_idx = np.argmax(logits)
            probabilities = self._softmax(logits / 1.0)
            return self.vocabulary[token_idx], probabilities[token_idx]
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        probabilities = self._softmax(scaled_logits)
        
        # Sample from distribution
        token_idx = np.random.choice(len(self.vocabulary), p=probabilities)
        return self.vocabulary[token_idx], probabilities[token_idx]
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities"""
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / exp_logits.sum()
    
    def demonstrate_temperature_effect(self, logits: np.ndarray):
        """Show how temperature affects selection"""
        temperatures = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        print("Original logits:", logits)
        print("\nTemperature effects:")
        
        for temp in temperatures:
            samples = []
            for _ in range(5):
                token, prob = self.sample_next_token(logits, temperature=temp)
                samples.append(f"{token}({prob:.2f})")
            
            print(f"  T={temp}: {' '.join(samples)}")

# Example: Model deciding between code completion options
vocabulary = ['return', 'if', 'for', 'def', 'print']
logits = np.array([3.0, 2.1, 1.5, 0.8, 0.5])  # Model's raw outputs

sampler = TemperatureSampler(vocabulary)
sampler.demonstrate_temperature_effect(logits)

# Output (example):
# T=0.0: return(0.58) return(0.58) return(0.58) return(0.58) return(0.58)
# T=0.5: return(0.74) return(0.74) if(0.18) return(0.74) return(0.74)
# T=1.0: return(0.58) if(0.24) return(0.58) for(0.13) if(0.24)
# T=1.5: if(0.32) return(0.41) for(0.16) if(0.32) for(0.16)
# T=2.0: for(0.25) if(0.29) return(0.34) for(0.25) def(0.08)
```

**Real Constraints:**

- **Factual tasks:** Use temperature 0-0.3 (minimize hallucination)
- **Creative tasks:** Use temperature 0.7-1.2 (increase variety)
- **Very high temperature (>1.5):** Often produces incoherent output

**Concrete Example:**

```python
def generate_with_temperature(prompt: str, temperature: float) -> str:
    """
    Simulates generation with different temperatures.
    In practice, this would call an actual model API.
    """
    # This is a placeholder - real implementation calls model
    pass

# Factual extraction - use low temperature
prompt_factual = "Extract the date from: 'Meeting scheduled for March 15, 2024'"
# response = generate_with_temperature(prompt_factual, temperature=0.0)
# Expected: Deterministic, accurate: "March 15, 2024"

# Creative writing - use higher temperature
prompt_creative = "Write a tagline for a cloud storage product"
# response = generate_with_temperature(prompt_creative, temperature=0.9)
# Expected: Varied, creative outputs across runs
```

### 3. Embeddings: Semantic Representation in Vector Space

**Technical Explanation:**

Embeddings convert discrete tokens into continuous vector representations (typically 768-4096 dimensions) where semantic similarity corresponds to geometric proximity. Words with similar meanings cluster together in this high-dimensional space.

Models learn these representations during training, encoding semantic, syntactic, and contextual relationships into numerical form that enables mathematical operations on meaning.

**Practical Implications:**