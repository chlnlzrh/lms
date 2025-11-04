# Zero-Shot, Few-Shot & Chain-of-Thought Prompting

## Core Concepts

In traditional software engineering, you write explicit instructions for every task. If you want to classify text sentiment, you write rules or train a model on labeled data. If you want to extract entities, you define patterns or build a supervised learning pipeline. The logic is explicit, hardcoded, and specific to each use case.

Large Language Models fundamentally change this paradigm. Instead of writing code that implements logic, you describe what you want in natural language. The model already contains compressed knowledge from its training data—your job is to activate the right patterns through your prompts.

**Traditional Approach:**
```python
import re
from typing import List, Dict

class SentimentClassifier:
    def __init__(self):
        self.positive_words = {'good', 'great', 'excellent', 'amazing'}
        self.negative_words = {'bad', 'terrible', 'awful', 'poor'}
    
    def classify(self, text: str) -> str:
        words = set(text.lower().split())
        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

# Requires explicit rules for every domain
classifier = SentimentClassifier()
result = classifier.classify("The product quality is excellent")
```

**LLM Approach:**
```python
from anthropic import Anthropic
from typing import Literal

client = Anthropic(api_key="your-api-key")

def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": f"Classify the sentiment as positive, negative, or neutral: {text}"
        }]
    )
    return response.content[0].text.strip().lower()

# Works across domains without code changes
result = classify_sentiment("The product quality is excellent")
```

The three prompting techniques we'll explore represent different ways to activate an LLM's capabilities:

- **Zero-shot**: Task description only—no examples
- **Few-shot**: Task description plus 2-5 examples
- **Chain-of-Thought**: Guide the model to show reasoning steps

### Key Insights That Change Engineering Thinking

**1. Examples Are Code Now**  
In traditional ML, examples are training data processed offline. With LLMs, examples in your prompt are executable instructions that immediately change behavior. Your prompt is your program.

**2. Reasoning Can Be Scaffolded**  
You can't see inside a trained neural network's reasoning. But you can ask an LLM to externalize its reasoning as text, which improves accuracy and provides debuggability.

**3. The Cold Start Problem Disappears**  
Traditional ML requires thousands of labeled examples before any functionality. LLMs work immediately (zero-shot) and improve rapidly with just 2-5 examples (few-shot).

### Why This Matters Now

Modern engineering increasingly involves unstructured data: user feedback, support tickets, legal documents, research papers. Building custom models for each task is expensive and slow. These prompting techniques let you deploy production-quality text processing in hours instead of months, with zero training infrastructure.

The trade-off: you exchange training time and infrastructure for per-request costs and latency. For most applications processing <10M requests/month, this is economically favorable.

## Technical Components

### 1. Zero-Shot Prompting: Task Description as Specification

Zero-shot means the model receives only the task description—no examples of correct outputs. The model must infer the task from its pre-trained knowledge.

**Technical Explanation:**  
During pre-training, the model saw countless examples of tasks described then executed in text. When you write "Translate X to Y" or "Summarize this text," you're matching patterns the model learned. Success depends on: (1) task clarity, (2) whether similar tasks appeared in training data, (3) task complexity.

**Practical Implications:**  
Zero-shot works well for common tasks (translation, summarization, simple classification) but struggles with domain-specific tasks or complex reasoning. It's fastest to implement but has the lowest accuracy ceiling.

**Real Constraints:**  
- Output format reliability: ~85-95% for common formats (JSON, lists)
- Domain-specific accuracy: drops 20-40% vs few-shot for specialized tasks
- No correction mechanism if the model misunderstands the task

**Concrete Example:**
```python
from anthropic import Anthropic
import json

client = Anthropic(api_key="your-api-key")

def extract_entities_zero_shot(text: str) -> dict:
    """Extract people, organizations, and locations from text."""
    
    prompt = f"""Extract all people, organizations, and locations from this text.
Return as JSON with keys: people, organizations, locations (each a list of strings).

Text: {text}

JSON:"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)

# Test it
text = """Apple CEO Tim Cook announced a partnership with Microsoft. 
The deal was signed in Seattle and will impact operations in London."""

entities = extract_entities_zero_shot(text)
print(json.dumps(entities, indent=2))

# Expected output:
# {
#   "people": ["Tim Cook"],
#   "organizations": ["Apple", "Microsoft"],
#   "locations": ["Seattle", "London"]
# }
```

### 2. Few-Shot Prompting: Examples as In-Context Learning

Few-shot prompting provides 2-5 examples of input-output pairs before the actual task. The model learns the pattern from these examples without parameter updates.

**Technical Explanation:**  
This exploits the attention mechanism in transformers. When processing your actual input, the model's attention heads reference the example pairs in context. The examples effectively reprogram the model's behavior for that specific request. This is called "in-context learning."

**Practical Implications:**  
Few-shot dramatically improves accuracy for domain-specific tasks, unusual formats, or nuanced classification. The cost: increased token usage (examples consume context) and prompt engineering time (choosing good examples matters).

**Real Constraints:**  
- Optimal example count: usually 3-5 (more doesn't always help)
- Example selection matters: diverse, representative examples outperform random selection
- Token overhead: 3 examples ≈ 200-500 tokens (costs and latency)

**Concrete Example:**
```python
from anthropic import Anthropic
from typing import List, Dict

client = Anthropic(api_key="your-api-key")

def classify_support_ticket(
    ticket: str,
    examples: List[Dict[str, str]] = None
) -> str:
    """Classify support tickets using few-shot learning."""
    
    # Default examples for ticket classification
    if examples is None:
        examples = [
            {
                "ticket": "I can't log in, getting error 403",
                "category": "authentication"
            },
            {
                "ticket": "Payment was charged twice on my card",
                "category": "billing"
            },
            {
                "ticket": "The dashboard loads slowly, takes 30+ seconds",
                "category": "performance"
            },
            {
                "ticket": "How do I export my data to CSV format?",
                "category": "feature_question"
            }
        ]
    
    # Build few-shot prompt
    prompt = "Classify support tickets into categories.\n\n"
    
    for ex in examples:
        prompt += f"Ticket: {ex['ticket']}\n"
        prompt += f"Category: {ex['category']}\n\n"
    
    prompt += f"Ticket: {ticket}\n"
    prompt += "Category:"
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text.strip()

# Test with a new ticket
new_ticket = "API returns 500 errors during peak hours"
category = classify_support_ticket(new_ticket)
print(f"Category: {category}")  # Expected: "performance" or "technical"

# Compare with zero-shot (worse at domain-specific categories)
def classify_zero_shot(ticket: str) -> str:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=20,
        messages=[{
            "role": "user",
            "content": f"Classify this support ticket into one category: {ticket}\nCategory:"
        }]
    )
    return response.content[0].text.strip()

zero_category = classify_zero_shot(new_ticket)
print(f"Zero-shot category: {zero_category}")
```

### 3. Chain-of-Thought Prompting: Externalizing Reasoning

Chain-of-Thought (CoT) prompting asks the model to show its reasoning steps before providing the final answer. This improves accuracy on complex tasks requiring multi-step reasoning.

**Technical Explanation:**  
Language models generate text token-by-token. Each token is predicted based on all previous tokens. By generating reasoning steps as text, the model creates intermediate tokens that serve as "scratchpad" for the final answer. This gives the model more computation (more forward passes) to solve the problem.

**Practical Implications:**  
CoT significantly improves accuracy on math, logic, multi-step analysis, and debugging tasks. Trade-off: 3-5x more output tokens (higher cost and latency). Use when accuracy is critical and tasks require reasoning.

**Real Constraints:**  
- Cost multiplier: 3-5x due to reasoning tokens
- Latency increase: proportional to output length
- Reasoning quality varies: model can still make logical errors
- Works best with larger models (smaller models produce worse reasoning)

**Concrete Example:**
```python
from anthropic import Anthropic
import re

client = Anthropic(api_key="your-api-key")

def analyze_with_chain_of_thought(
    problem: str,
    include_reasoning: bool = True
) -> dict:
    """Solve a problem with optional chain-of-thought reasoning."""
    
    if include_reasoning:
        prompt = f"""{problem}

Think through this step-by-step:
1. First, identify the key information
2. Then, work through the logic
3. Finally, provide your answer

Show your reasoning, then provide the final answer after "ANSWER:"."""
    else:
        prompt = f"{problem}\n\nProvide your answer:"
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.content[0].text
    
    # Extract reasoning and answer
    if "ANSWER:" in text:
        parts = text.split("ANSWER:")
        reasoning = parts[0].strip()
        answer = parts[1].strip()
    else:
        reasoning = ""
        answer = text.strip()
    
    return {
        "reasoning": reasoning,
        "answer": answer,
        "tokens_used": response.usage.output_tokens
    }

# Test with a multi-step problem
problem = """A database has 1000 users. 40% are on the free plan, 35% on basic, 
and 25% on premium. If we increase premium price by 20% and lose 10% of premium users, 
but convert 5% of basic users to premium, what's the net change in premium users?"""

# With chain-of-thought
cot_result = analyze_with_chain_of_thought(problem, include_reasoning=True)
print("WITH REASONING:")
print(cot_result["reasoning"])
print(f"\nAnswer: {cot_result['answer']}")
print(f"Tokens: {cot_result['tokens_used']}")

# Without chain-of-thought (often less accurate)
direct_result = analyze_with_chain_of_thought(problem, include_reasoning=False)
print(f"\n\nDIRECT ANSWER: {direct_result['answer']}")
print(f"Tokens: {direct_result['tokens_used']}")
```

### 4. Example Selection Strategy for Few-Shot

Not all examples are equally valuable. Strategic example selection can improve accuracy by 15-30% compared to random selection.

**Technical Explanation:**  
The model's attention mechanism weights examples by similarity to the current input. Diverse examples that cover the decision boundary are more effective than redundant examples. For classification, include edge cases and common confusions.

**Practical Implementation:**
```python
from anthropic import Anthropic
from typing import List, Dict, Callable
import random

client = Anthropic(api_key="your-api-key")

class FewShotClassifier:
    """Reusable few-shot classification with example selection strategies."""
    
    def __init__(self, examples: List[Dict[str, str]], model: str = "claude-3-5-sonnet-20241022"):
        self.examples = examples
        self.model = model
        self.client = Anthropic(api_key="your-api-key")
    
    def select_examples(
        self, 
        strategy: str = "diverse",
        n: int = 4
    ) -> List[Dict[str, str]]:
        """Select examples using different strategies."""
        
        if strategy == "random":
            return random.sample(self.examples, min(n, len(self.examples)))
        
        elif strategy == "diverse":
            # Select examples with different labels
            selected = []
            seen_labels = set()
            
            for ex in self.examples:
                label = ex.get("category") or ex.get("label")
                if label not in seen_labels:
                    selected.append(ex)
                    seen_labels.add(label)
                    if len(selected) >= n:
                        break
            
            # Fill remaining slots randomly if needed
            if len(selected) < n:
                remaining = [ex for ex in self.examples if ex not in selected]
                selected.extend(random.sample(remaining, min(n - len(selected), len(remaining))))
            
            return selected
        
        elif strategy == "balanced":
            # Equal examples per category
            by_category = {}
            for ex in self.examples:
                label = ex.get("category") or ex.get("label")
                if label not in by_category:
                    by_category[label] = []
                by_category[label].append(ex)
            
            per_category = max(1, n // len(by_category))
            selected = []
            
            for category_examples in by_category.values():
                selected.extend(random.sample(
                    category_examples, 
                    min(per_category, len(category_examples))
                ))
            
            return selected[:n]
        
        return self.examples[:n]
    
    def classify(self, text: str, strategy: str = "diverse", n_examples: int = 4) -> str:
        """Classify text using selected examples."""
        
        examples = self.select_examples(strategy, n_examples)
        
        prompt = "Classify the following text.\n\n"
        for ex in examples:
            prompt += f"Text: {ex['text']}\n"
            prompt += f"Category: {ex.get('category') or ex.get('label')}\n\n"
        
        prompt += f"Text: {text}\nCategory:"
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()

# Usage example
training_examples = [
    {"text": "Stock prices surged after earnings report", "category": "finance"},
    {"text": "New vaccine shows promising trial results", "category": "health"},
    {"text": "Championship game ends in overtime thriller", "category": "sports"},
    {"text": "Interest rates held steady by central bank", "category": "finance"},
    {"text": "Breakthrough in quantum computing announced", "category": "technology"},
    {"text": "Severe weather warnings issued for region", "category": "weather"},
    {"text": "Major