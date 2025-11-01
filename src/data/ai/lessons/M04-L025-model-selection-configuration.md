# Model Selection & Configuration

## Core Concepts

Model selection in LLM engineering is the process of choosing and configuring language models based on technical requirements, performance constraints, and task characteristics. Unlike traditional machine learning where you train models from scratch, LLM engineering involves selecting from pre-trained foundation models and configuring their behavior through parameters.

### Engineering Analogy: Database Selection vs. Model Selection

```python
# Traditional ML: Building from scratch (like writing a custom database)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Train custom model for each task
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(training_texts)
model = LogisticRegression()
model.fit(X_train, y_train)

# New task = new model, new training data, new deployment
```

```python
# Modern LLM: Selecting pre-trained models (like choosing PostgreSQL vs. Redis)
from anthropic import Anthropic
from openai import OpenAI

# Choose model based on task requirements
client = Anthropic()

def classify_text(text: str, temperature: float = 0.0) -> str:
    """Use pre-trained model with configuration"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",  # Model selection
        max_tokens=100,                       # Configuration
        temperature=temperature,              # Behavior tuning
        messages=[{
            "role": "user",
            "content": f"Classify sentiment: {text}"
        }]
    )
    return response.content[0].text

# Same model, different configs for different tasks
```

The paradigm shift: You're no longer training models—you're selecting and configuring pre-existing intelligence. This is analogous to choosing between PostgreSQL, MongoDB, and Redis based on your data access patterns, rather than building a database engine from scratch.

### Key Insights That Change Your Engineering Approach

**1. Model capabilities exist on a capability-cost-speed spectrum.** There's no single "best" model—only optimal choices for specific constraints. A model that's perfect for real-time chatbots may be wasteful for batch processing.

**2. Configuration parameters fundamentally alter model behavior.** Temperature, top-p, and max tokens aren't just knobs—they're architectural decisions that affect reliability, creativity, and cost.

**3. Context window size is a hard architectural constraint.** Unlike traditional APIs where you can paginate or chunk freely, context windows impose fundamental limits on problem-solving approaches.

### Why This Matters Now

The LLM landscape currently features dozens of capable models with overlapping capabilities but different trade-offs. Engineers who understand selection criteria and configuration can achieve 5-10x cost reductions while improving latency and quality. Poor choices lead to unnecessary expenses (using frontier models for simple tasks), failed user experiences (exceeding context windows), or unreliable outputs (wrong temperature settings).

## Technical Components

### 1. Model Capability Tiers

LLMs exist in distinct capability tiers that determine their appropriate use cases:

**Technical Explanation:**  
Models are categorized by parameter count, training data quality, and post-training optimization. Frontier models (100B+ parameters) excel at reasoning and complex tasks. Mid-tier models (10-50B parameters) handle structured tasks efficiently. Small models (<10B parameters) optimize for speed and cost.

**Practical Implications:**  
Your choice affects response time (50ms to 5s), cost per request ($0.0001 to $0.10), and output quality. Using frontier models for simple tasks wastes money; using small models for complex reasoning produces poor results.

**Real Constraints:**  
- Frontier models: 2-5s latency, $0.01-0.10 per request, best reasoning
- Mid-tier models: 500ms-2s latency, $0.001-0.01 per request, good for structured tasks
- Small models: 50-500ms latency, $0.0001-0.001 per request, best for simple classification

**Concrete Example:**

```python
from typing import Literal
import time
import json

ModelTier = Literal["frontier", "mid-tier", "small"]

class ModelSelector:
    """Select models based on task complexity"""
    
    MODELS = {
        "frontier": {
            "name": "claude-3-5-sonnet-20241022",
            "cost_per_1k_tokens": 0.015,
            "avg_latency_ms": 2000,
            "use_cases": ["reasoning", "complex_analysis", "code_generation"]
        },
        "mid-tier": {
            "name": "claude-3-haiku-20240307",
            "cost_per_1k_tokens": 0.0025,
            "avg_latency_ms": 800,
            "use_cases": ["summarization", "translation", "structured_extraction"]
        },
        "small": {
            "name": "gpt-3.5-turbo",
            "cost_per_1k_tokens": 0.0005,
            "avg_latency_ms": 300,
            "use_cases": ["classification", "simple_qa", "keyword_extraction"]
        }
    }
    
    @classmethod
    def select_for_task(cls, task_type: str, latency_budget_ms: int = 5000,
                       cost_budget_per_request: float = 0.10) -> dict:
        """Select optimal model based on task and constraints"""
        
        for tier in ["small", "mid-tier", "frontier"]:
            model = cls.MODELS[tier]
            
            # Estimate tokens (rough heuristic: 500 tokens for avg request)
            estimated_cost = (500 / 1000) * model["cost_per_1k_tokens"]
            
            if (task_type in model["use_cases"] and 
                model["avg_latency_ms"] <= latency_budget_ms and
                estimated_cost <= cost_budget_per_request):
                return {
                    "tier": tier,
                    "model": model["name"],
                    "estimated_cost": estimated_cost,
                    "estimated_latency_ms": model["avg_latency_ms"]
                }
        
        return cls.MODELS["frontier"]  # Fallback to most capable

# Usage examples
selector = ModelSelector()

# Simple classification - uses small model
simple_task = selector.select_for_task("classification")
print(f"Classification: {simple_task['model']} (${simple_task['estimated_cost']:.4f})")

# Complex reasoning - uses frontier model
complex_task = selector.select_for_task("reasoning")
print(f"Reasoning: {complex_task['model']} (${complex_task['estimated_cost']:.4f})")

# Latency-constrained task
fast_task = selector.select_for_task("summarization", latency_budget_ms=500)
print(f"Fast summarization: {fast_task['model']} ({fast_task['estimated_latency_ms']}ms)")
```

### 2. Temperature and Sampling Parameters

Temperature controls randomness in token selection, fundamentally affecting output determinism and creativity.

**Technical Explanation:**  
Temperature scales the logit distribution before sampling. At temperature=0, the model selects the highest probability token (argmax). At temperature=1.0, tokens are sampled according to their raw probability distribution. Higher temperatures flatten the distribution, increasing randomness.

**Practical Implications:**  
- Temperature=0: Deterministic, repeatable outputs for production systems
- Temperature=0.3-0.7: Balanced for creative tasks with some consistency
- Temperature=0.8-1.0: High variability for brainstorming, creative writing

Top-p (nucleus sampling) provides an alternative: sample from the smallest set of tokens whose cumulative probability exceeds p.

**Real Constraints:**  
Temperature doesn't improve factual accuracy—it only affects token selection randomness. For factual tasks, use temperature=0. For creative tasks where multiple valid answers exist, use temperature=0.7-1.0.

**Concrete Example:**

```python
from anthropic import Anthropic
from typing import List
import statistics

client = Anthropic(api_key="your-api-key")

def test_temperature_consistency(prompt: str, temperature: float, 
                                runs: int = 5) -> dict:
    """Test output consistency at different temperatures"""
    responses = []
    
    for _ in range(runs):
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        responses.append(response.content[0].text)
    
    # Calculate diversity (unique responses / total responses)
    unique_responses = len(set(responses))
    diversity = unique_responses / runs
    
    return {
        "temperature": temperature,
        "diversity": diversity,
        "sample_responses": responses[:3]
    }

# Test factual task at different temperatures
factual_prompt = "What is 15 * 24? Provide only the number."

print("Factual Task (arithmetic):")
for temp in [0.0, 0.5, 1.0]:
    result = test_temperature_consistency(factual_prompt, temp)
    print(f"Temp {temp}: Diversity={result['diversity']:.2f}")
    print(f"Sample: {result['sample_responses'][0]}\n")

# Test creative task
creative_prompt = "Suggest a creative name for a coffee shop."

print("\nCreative Task (naming):")
for temp in [0.0, 0.5, 1.0]:
    result = test_temperature_consistency(creative_prompt, temp)
    print(f"Temp {temp}: Diversity={result['diversity']:.2f}")
    print(f"Samples: {result['sample_responses']}\n")
```

**Expected output patterns:**
- Factual task: Low diversity at all temperatures (correct answer is deterministic)
- Creative task: Diversity increases with temperature (0.0→ 0.2, 0.5→ 0.6, 1.0→ 0.9)

### 3. Context Window Architecture

Context windows define the maximum tokens (input + output) a model can process in a single request.

**Technical Explanation:**  
Context windows are architectural limits determined by the model's positional encoding mechanism. Current models range from 4K to 200K+ tokens. One token ≈ 0.75 words for English text. The window includes system prompts, conversation history, user input, and generated output.

**Practical Implications:**  
Context window size determines:
- Maximum document size for analysis
- Conversation length before memory loss
- Amount of examples you can include (few-shot learning)
- Cost (many models charge per token in context)

**Real Constraints:**  
Exceeding context windows causes hard failures (errors or truncation). Models also experience "lost in the middle" effects—information in the middle of long contexts is accessed less reliably than information at the beginning or end.

**Concrete Example:**

```python
from typing import List, Optional
import tiktoken

class ContextWindowManager:
    """Manage context window constraints"""
    
    def __init__(self, model: str = "gpt-4", max_tokens: int = 8192):
        self.model = model
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def fit_documents_in_context(self, documents: List[str], 
                                 system_prompt: str,
                                 max_output_tokens: int = 1000) -> dict:
        """Determine how many documents fit in context window"""
        
        system_tokens = self.count_tokens(system_prompt)
        available_tokens = self.max_tokens - system_tokens - max_output_tokens
        
        fitted_docs = []
        total_tokens = 0
        
        for doc in documents:
            doc_tokens = self.count_tokens(doc)
            if total_tokens + doc_tokens <= available_tokens:
                fitted_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break
        
        return {
            "fitted_count": len(fitted_docs),
            "total_input_tokens": system_tokens + total_tokens,
            "reserved_output_tokens": max_output_tokens,
            "total_tokens": system_tokens + total_tokens + max_output_tokens,
            "remaining_capacity": available_tokens - total_tokens
        }
    
    def truncate_to_fit(self, text: str, reserve_output: int = 500) -> str:
        """Truncate text to fit in context window"""
        max_input_tokens = self.max_tokens - reserve_output
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_input_tokens:
            return text
        
        # Truncate and add indicator
        truncated_tokens = tokens[:max_input_tokens - 10]
        truncated_text = self.encoding.decode(truncated_tokens)
        return truncated_text + "\n\n[...truncated...]"

# Usage example
manager = ContextWindowManager(model="gpt-4", max_tokens=8192)

documents = [
    "Document 1: " + "word " * 1000,  # ~1000 tokens
    "Document 2: " + "word " * 1000,
    "Document 3: " + "word " * 1000,
    "Document 4: " + "word " * 1000,
    "Document 5: " + "word " * 1000,
]

system_prompt = "Analyze the following documents and provide insights."

result = manager.fit_documents_in_context(documents, system_prompt)
print(f"Context Window Analysis:")
print(f"  Model: {manager.model} ({manager.max_tokens} tokens)")
print(f"  Fitted documents: {result['fitted_count']}/{len(documents)}")
print(f"  Total tokens: {result['total_tokens']}")
print(f"  Remaining capacity: {result['remaining_capacity']} tokens")

# Handle truncation
long_text = "word " * 10000
truncated = manager.truncate_to_fit(long_text, reserve_output=500)
print(f"\nTruncation: {manager.count_tokens(long_text)} → {manager.count_tokens(truncated)} tokens")
```

### 4. Max Tokens and Stop Sequences

Max tokens limits output length; stop sequences halt generation at specific patterns.

**Technical Explanation:**  
Max tokens sets a hard limit on generated tokens. The model generates tokens sequentially until hitting max_tokens, a stop sequence, or natural completion. Stop sequences are literal strings that trigger immediate generation halt.

**Practical Implications:**  
- Set max_tokens based on expected output length + buffer
- Use stop sequences for structured output (JSON, code blocks)
- Insufficient max_tokens causes truncated responses
- Excessive max_tokens wastes money and latency

**Concrete Example:**

```python
from anthropic import Anthropic
from typing import List, Optional
import json

client = Anthropic(api_key="your-api-key")

def extract_structured_data(text: str, max_tokens: int = 500,
                           stop_sequences: Optional[List[str]] = None) -> dict:
    """Extract structured data with controlled output length"""
    
    prompt = f"""Extract key information from this text as JSON:
{text}

Return ONLY valid JSON with keys: "name", "email", "phone", "summary".
JSON:"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=max_tokens,
        temperature=0,
        stop_sequences=stop_sequences or ["\n\n", "---"],
        messages=[{"role": "user", "content": prompt}]
    )
    
    output = response.content[0].text
    
    return {
        "raw_output": output,
        "tokens_used": response.usage.output_tokens,
        "stop_reason": response.stop_reason,
        "parsed_json": json.loads(output) if output.strip().startswith("{") else None
    }

# Test with different max_tokens
sample_text = """
John Smith is a software engineer at a tech company. 
Contact: john.smith@email.com, +1-555-0123.
He specializes in distributed systems and has 10 years of experience