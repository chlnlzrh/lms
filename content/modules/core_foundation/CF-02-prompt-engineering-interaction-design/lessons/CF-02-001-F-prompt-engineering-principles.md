# Prompt Engineering Principles: A Technical Foundation

Prompt engineering is the practice of designing inputs to language models that reliably produce desired outputs. Unlike traditional programming where you write explicit instructions in formal syntax, prompt engineering involves crafting natural language specifications that guide probabilistic text generation systems.

This isn't about clever tricks or magic words. It's about understanding how language models process input and applying systematic techniques to achieve consistent, measurable results.

## The Paradigm Shift: From Imperative to Declarative

Traditional software engineering uses imperative programming—you specify exactly *how* to accomplish a task:

```python
def extract_email(text: str) -> str | None:
    """Extract email using explicit rules."""
    import re
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(pattern, text)
    return match.group(0) if match else None

# Requires anticipating every edge case
text = "Contact us at support@example.com for help"
email = extract_email(text)
```

With language models, you shift to declarative programming—you specify *what* you want, and the model figures out how:

```python
from anthropic import Anthropic

def extract_email_llm(text: str) -> str | None:
    """Extract email using language model."""
    client = Anthropic(api_key="your-api-key")
    
    prompt = f"""Extract the email address from this text.
Return only the email address, nothing else.
If no email exists, return "NONE".

Text: {text}"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.content[0].text.strip()
    return None if result == "NONE" else result
```

The language model approach handles variations you didn't explicitly program: emails with unusual TLDs, internationalized domains, or embedded in complex formatting. But it introduces new challenges: non-deterministic outputs, higher latency, and dependency on model capabilities.

**Why this matters now:** Language models have crossed a capability threshold. They can reliably perform complex tasks that would require thousands of lines of traditional code—translation, summarization, analysis, structured data extraction. But only if you prompt them correctly. Poor prompting produces unreliable outputs that make LLMs seem useless. Good prompting unlocks transformative capabilities.

## Core Technical Components

### 1. Instruction Clarity and Specificity

Language models interpret prompts probabilistically based on patterns from training data. Vague instructions activate broader probability distributions, increasing output variance.

**Technical explanation:** When you provide a clear, specific instruction, you constrain the model's output space. Think of it as narrowing a search beam—the more specific your prompt, the more concentrated the probability mass on desired outputs.

**Practical implications:**

```python
# Vague prompt - high output variance
vague_prompt = "Analyze this data"

# Specific prompt - constrained output space
specific_prompt = """Analyze this sales data and provide:
1. Total revenue (sum of all transactions)
2. Average transaction value
3. Number of unique customers
4. Top 3 products by revenue

Format as JSON with keys: total_revenue, avg_transaction, unique_customers, top_products"""
```

**Real constraints:** Specificity has costs. Longer prompts consume more tokens (increasing latency and cost) and can paradoxically confuse models if overly detailed. Balance specificity with conciseness.

**Concrete example:**

```python
import json
from anthropic import Anthropic

def analyze_sales_data(data: str) -> dict:
    """Analyze sales with specific structured output."""
    client = Anthropic(api_key="your-api-key")
    
    prompt = f"""Analyze this sales data and return JSON with these exact keys:
- total_revenue: sum of all sale amounts as float
- avg_transaction: mean transaction value as float
- unique_customers: count of distinct customer IDs as int
- top_products: array of top 3 product names by revenue

Data:
{data}

Return only valid JSON, no other text."""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)

# Test data
sales_data = """
transaction_id,customer_id,product,amount
1,C001,Widget,29.99
2,C002,Gadget,49.99
3,C001,Widget,29.99
4,C003,Doohickey,15.99
5,C002,Widget,29.99
"""

result = analyze_sales_data(sales_data)
print(json.dumps(result, indent=2))
# {
#   "total_revenue": 155.95,
#   "avg_transaction": 31.19,
#   "unique_customers": 3,
#   "top_products": ["Widget", "Gadget", "Doohickey"]
# }
```

### 2. Context and Examples (Few-Shot Learning)

Language models learn task patterns from examples in the prompt. This "few-shot learning" is fundamentally different from traditional function calls—you're showing the model the input-output mapping you want.

**Technical explanation:** Examples bias the model's generation toward similar patterns. Each example updates the model's internal attention weights, making outputs that match the demonstrated pattern more likely.

**Practical implications:**

```python
# Zero-shot (no examples)
zero_shot = "Extract the sentiment from: 'This product is okay but overpriced'"

# Few-shot (with examples)
few_shot = """Extract sentiment as positive, negative, or neutral.

Examples:
Text: "I love this product! Best purchase ever."
Sentiment: positive

Text: "Terrible quality. Complete waste of money."
Sentiment: negative

Text: "It works as expected, nothing special."
Sentiment: neutral

Text: "This product is okay but overpriced"
Sentiment:"""
```

**Real constraints:** Each example consumes tokens. Three examples might use 100-200 tokens. For high-volume applications, this multiplies costs. Choose examples that cover edge cases, not obvious patterns.

**Concrete example:**

```python
from typing import Literal

def classify_sentiment(
    text: str,
    use_examples: bool = True
) -> Literal["positive", "negative", "neutral"]:
    """Classify sentiment with optional few-shot examples."""
    client = Anthropic(api_key="your-api-key")
    
    if use_examples:
        prompt = """Classify the sentiment as: positive, negative, or neutral

Examples:
"Exceeded expectations! Would buy again." → positive
"Useless. Returned immediately." → negative  
"Does what it says, nothing more." → neutral
"Great price but flimsy build quality." → neutral

Classify: "{text}"
Return only one word: positive, negative, or neutral"""
    else:
        prompt = f"""Classify the sentiment as: positive, negative, or neutral

Text: {text}
Return only one word: positive, negative, or neutral"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt.format(text=text)}]
    )
    
    return response.content[0].text.strip().lower()

# Test
test_cases = [
    "Amazing product, highly recommend!",
    "Poor quality, very disappointed.",
    "It's fine, does the job.",
    "Love the features but terrible customer service"
]

for text in test_cases:
    sentiment = classify_sentiment(text)
    print(f"{text[:40]:40} → {sentiment}")
```

### 3. Output Format Constraints

Language models generate free-form text by default. Constraining output format requires explicit specification and often validation.

**Technical explanation:** Format constraints work by priming the model to continue in a specific pattern. Starting with `{"name":` strongly biases the model toward completing valid JSON. But there's no guarantee—you must validate and handle malformed outputs.

**Practical implications:**

```python
# Weak format guidance
weak = "List the key points"

# Strong format constraint
strong = """List exactly 3 key points in this JSON format:
{
  "points": [
    "First point here",
    "Second point here", 
    "Third point here"
  ]
}"""
```

**Real constraints:** Even with strict format instructions, models occasionally deviate. Production code must include parsing error handling and retry logic.

**Concrete example:**

```python
import json
from typing import List, Optional

def extract_key_points(
    text: str,
    num_points: int = 3,
    max_retries: int = 2
) -> Optional[List[str]]:
    """Extract key points with format validation and retries."""
    client = Anthropic(api_key="your-api-key")
    
    prompt = f"""Extract exactly {num_points} key points from this text.

Return ONLY valid JSON in this exact format:
{{
  "points": ["point 1", "point 2", "point 3"]
}}

Text: {text}"""
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON if wrapped in markdown code blocks
            content = response.content[0].text.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            data = json.loads(content.strip())
            
            # Validate structure
            if "points" in data and isinstance(data["points"], list):
                if len(data["points"]) == num_points:
                    return data["points"]
                    
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                print(f"Failed to parse JSON after {max_retries} attempts")
                return None
            continue
    
    return None

# Test
article = """
Artificial intelligence is transforming healthcare through improved diagnostics,
personalized treatment plans, and drug discovery acceleration. Machine learning
models can now detect diseases from medical images with accuracy matching human
experts. AI systems analyze patient data to recommend optimal treatments based
on individual genetics and history. In pharmaceutical research, AI predicts
molecular interactions, reducing drug development time from years to months.
"""

points = extract_key_points(article, num_points=3)
if points:
    for i, point in enumerate(points, 1):
        print(f"{i}. {point}")
```

### 4. Role and Persona Specification

Assigning a role to the model influences its output style, knowledge prioritization, and problem-solving approach.

**Technical explanation:** Roles activate different patterns in the model's training data. Saying "You are a data engineer" increases the probability of technical, implementation-focused responses compared to "You are a business analyst."

**Practical implications:**

```python
# Generic role
generic = "Explain database indexing"

# Specific technical role
technical = """You are a senior database engineer explaining to a junior developer.
Explain database indexing with:
- Technical definition
- Performance implications (Big O notation)
- When to use vs. avoid
- Concrete example with table schema"""
```

**Real constraints:** Role specification is not magic. The model's actual knowledge is fixed. A role can't make the model know facts it wasn't trained on, but it can change how it presents known information.

**Concrete example:**

```python
def generate_explanation(
    topic: str,
    role: str = "senior software engineer",
    audience: str = "junior developer"
) -> str:
    """Generate explanations with role-based framing."""
    client = Anthropic(api_key="your-api-key")
    
    prompt = f"""You are a {role} explaining {topic} to a {audience}.

Provide:
1. Clear technical definition (2-3 sentences)
2. Practical implications (why it matters)
3. Code example demonstrating the concept
4. Common mistake to avoid

Be concrete and technical. Use code, not analogies."""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

explanation = generate_explanation(
    topic="database connection pooling",
    role="backend infrastructure engineer",
    audience="developer building a high-traffic API"
)
print(explanation)
```

### 5. Chain of Thought and Reasoning

Complex tasks benefit from explicitly instructing the model to show its reasoning process before providing an answer.

**Technical explanation:** By prompting the model to "think step by step," you force it to generate intermediate reasoning tokens. This literally gives the model more computation steps (more forward passes through its layers) to process the problem.

**Practical implications:**

```python
# Direct answer (may be incorrect)
direct = "What is 15% of 847?"

# Chain of thought
cot = """Calculate 15% of 847. Think step by step:
1. Convert percentage to decimal
2. Multiply
3. Provide final answer"""
```

**Real constraints:** Chain of thought increases token usage significantly—sometimes 3-5x more tokens. Use for complex reasoning tasks, not simple lookups.

**Concrete example:**

```python
from typing import Dict, Any

def debug_code_with_reasoning(code: str, error: str) -> Dict[str, Any]:
    """Debug code by making model reason through the problem."""
    client = Anthropic(api_key="your-api-key")
    
    prompt = f"""Debug this code that produces an error.

Code:
{code}

Error:
{error}

Think through this step by step:
1. What is the code trying to do?
2. What does the error indicate?
3. What line is the problem on?
4. Why is this happening?
5. How to fix it?

After your reasoning, provide the corrected code.

Format your response as:
REASONING:
[your step-by-step analysis]

CORRECTED_CODE:
```python
[corrected code here]
```
"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.content[0].text
    
    # Parse response
    parts = content.split("CORRECTED_CODE:")
    reasoning = parts[0].replace("REASONING:", "").strip()
    
    # Extract code from markdown
    code_section = parts[1] if len(parts) > 1 else ""
    if "```python" in code_section:
        corrected = code_section.split("```python")[1].split("```")[0].strip()
    else:
        corrected = code_section.strip()
    
    return {
        "reasoning": reasoning,
        "corrected_code": corrected
    }

# Test
buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

data = [10, 20, 30, None, 40]
print(calculate_average(data))
"""

error_msg = "TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'"

result = debug_code_with_reasoning(buggy_code, error_msg)
print("REASONING:")
print(result["reasoning"])
print("\nCORRECTED CODE:")
print(result["corrected_code"])
```

## Hands-On Exercises

### Exercise 1: Prompt Refinement for Structured Data Extraction

**Objective:** Transform a vague prompt into a specific, reliable prompt that extracts structured data from unstructured text.

**Instructions:**

1. Start with this buggy code that uses a vague prompt:

```python
from anthropic import Anthropic
import json

client = Anthropic(api_key="your-api-key")

def extract_meeting_info_vague(text: str) -> dict:
    """Vague prompt - unreliable extraction."""
    prompt = f"Extract meeting information from: {text}"
    
    response = client.messages.create(