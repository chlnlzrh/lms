# Context Engineering & Instruction Clarity

## Core Concepts

Context engineering is the practice of structuring and optimizing the information you provide to a language model to maximize output quality, consistency, and relevance. Unlike traditional programming where functions operate on precisely defined inputs with deterministic outputs, LLMs operate probabilistically on natural language context—making the *how* of information presentation as critical as the *what*.

### Traditional vs. Context-Based Paradigm

```python
# Traditional deterministic programming
def extract_date(text: str) -> datetime:
    """Clear interface, predictable behavior"""
    pattern = r'\d{4}-\d{2}-\d{2}'
    match = re.search(pattern, text)
    if match:
        return datetime.strptime(match.group(), '%Y-%m-%d')
    raise ValueError("No date found")

# Result: Always returns same output for same input
extract_date("Meeting on 2024-03-15")  # Always: 2024-03-15

# Context-based LLM programming
def extract_date_llm(text: str, context: str = "") -> str:
    """Behavior depends heavily on context structure"""
    prompt = f"{context}\n\nExtract the date: {text}"
    return llm.complete(prompt)

# Different contexts produce different behaviors
extract_date_llm("Meeting on 2024-03-15", 
                 context="Return ISO format")  # "2024-03-15"

extract_date_llm("Meeting on 2024-03-15",
                 context="Return human-readable format")  # "March 15, 2024"

extract_date_llm("Meeting on 2024-03-15",
                 context="You are a JSON API")  # {"date": "2024-03-15"}
```

The critical insight: **Your context is your interface specification, runtime configuration, and documentation combined.** Poor context engineering is equivalent to calling functions with ambiguous parameters and expecting consistent results.

### Why This Matters Now

Three engineering realities make context engineering essential:

1. **Token costs are real**: Context inefficiency directly impacts operational costs. A 2000-token prompt costs 10x more than a 200-token prompt with GPT-4.
2. **Latency compounds**: Every unnecessary token adds generation time. In production systems processing millions of requests, context bloat creates measurable user experience degradation.
3. **Quality degrades non-linearly**: LLMs don't fail gracefully with poor context—they confidently produce plausible-looking garbage, creating silent failures in production systems.

## Technical Components

### 1. Context Hierarchy and Information Positioning

LLMs exhibit **recency bias** and **primacy effects**—information placement significantly affects how it influences outputs. Models weight recent context more heavily and pay special attention to the beginning of prompts.

**Technical explanation**: Attention mechanisms in transformers create position-dependent weights. While modern models use positional encodings that theoretically treat all positions equally, empirical testing shows consistent bias toward prompt ends (recency) and beginnings (primacy).

**Practical implications**:

```python
from typing import List, Dict
import anthropic

def analyze_with_context_ordering(
    query: str,
    context_items: List[str],
    position: str = "end"  # "start", "end", or "both"
) -> str:
    """Demonstrate impact of context positioning"""
    
    if position == "start":
        # Critical info at start
        prompt = f"""{context_items[0]}

Query: {query}

Additional context: {' '.join(context_items[1:])}"""
    
    elif position == "end":
        # Critical info at end (often more effective)
        prompt = f"""Additional context: {' '.join(context_items[:-1])}

{context_items[-1]}

Query: {query}"""
    
    else:  # both
        # Sandwich important info
        prompt = f"""{context_items[0]}

Additional context: {' '.join(context_items[1:-1])}

{context_items[-1]}

Query: {query}"""
    
    return prompt

# Example: Most important constraint should be positioned strategically
critical_constraint = "Output must be valid JSON with no markdown formatting."
background_info = ["User is a software engineer", "Context is API development"]

# Poor positioning (buried in middle)
poor_prompt = analyze_with_context_ordering(
    "Generate an API response",
    background_info + [critical_constraint],
    position="start"
)

# Better positioning (at end, near output)
better_prompt = analyze_with_context_ordering(
    "Generate an API response",
    background_info + [critical_constraint],
    position="end"
)
```

**Real constraints**: Recency bias varies by model architecture and size. Smaller models show stronger positional effects. Testing with your specific model is essential.

### 2. Instruction Decomposition and Specificity

Complex instructions must be decomposed into explicit, sequenced steps. LLMs perform significantly better with step-by-step instructions than with implied or complex multi-part requirements.

**Technical explanation**: Each instruction competes for attention weight in the model's computation. Compound instructions ("do X and Y considering Z") create ambiguous attention patterns. Decomposed instructions create clear computational paths.

```python
from enum import Enum
from dataclasses import dataclass

class InstructionStyle(Enum):
    COMPOUND = "compound"
    DECOMPOSED = "decomposed"

@dataclass
class TaskResult:
    output: str
    instruction_tokens: int
    follows_all_requirements: bool

def create_extraction_prompt(style: InstructionStyle, text: str) -> str:
    """Compare compound vs decomposed instructions"""
    
    if style == InstructionStyle.COMPOUND:
        # Compound: Everything in one complex sentence
        return f"""Extract all email addresses, phone numbers, and dates from 
the text, format them as JSON with proper validation, remove duplicates, 
and sort chronologically where applicable: {text}"""
    
    else:
        # Decomposed: Clear sequential steps
        return f"""Analyze the following text and extract information step-by-step:

1. Identify all email addresses
2. Identify all phone numbers
3. Identify all dates
4. Validate each extracted item for correct format
5. Remove any duplicates within each category
6. Sort dates chronologically
7. Output as JSON with keys: "emails", "phones", "dates"

Text: {text}

Output:"""

# Example usage
sample_text = """
Contact John at john@example.com or 555-0100.
Meeting on 2024-03-15. Also reach out to john@example.com
or call 555-0101. Follow-up on 2024-03-10.
"""

# Decomposed instructions typically yield 30-50% higher accuracy
# on multi-requirement tasks in empirical testing
compound_prompt = create_extraction_prompt(InstructionStyle.COMPOUND, sample_text)
decomposed_prompt = create_extraction_prompt(InstructionStyle.DECOMPOSED, sample_text)

print(f"Compound length: {len(compound_prompt.split())} words")
print(f"Decomposed length: {len(decomposed_prompt.split())} words")
print(f"Token overhead: ~{len(decomposed_prompt.split()) - len(compound_prompt.split())} words")
```

**Trade-offs**: Decomposed instructions use 20-40% more tokens but improve task completion rates by 30-60% for complex multi-step operations. Cost vs. quality trade-off depends on your application's error tolerance.

### 3. Role and Perspective Framing

Assigning a specific role or expertise level to the model creates consistent behavioral patterns by activating relevant training distributions. This is not anthropomorphization—it's statistical distribution targeting.

**Technical explanation**: Training data contains clear distributional differences between how different "roles" communicate. Medical literature uses different vocabulary, structure, and reasoning patterns than legal documents. Role framing biases the model toward those specific distributions.

```python
from typing import Optional
from dataclasses import dataclass

@dataclass
class RoleContext:
    role: str
    expertise_level: str
    communication_style: str
    
    def to_prompt_prefix(self) -> str:
        return f"""You are a {self.role} with {self.expertise_level} expertise.
Communication style: {self.communication_style}
"""

def create_code_review(
    code: str,
    role_context: Optional[RoleContext] = None
) -> str:
    """Demonstrate impact of role framing"""
    
    base_prompt = f"Review this code:\n\n```python\n{code}\n```\n\nReview:"
    
    if role_context is None:
        return base_prompt
    
    return f"{role_context.to_prompt_prefix()}\n{base_prompt}"

# Example code to review
buggy_code = """
def process_data(items):
    result = []
    for i in range(len(items)):
        result.append(items[i] * 2)
    return result
"""

# Without role framing: Generic response
generic_prompt = create_code_review(buggy_code)

# With senior engineer role: Focus on design patterns, performance
senior_eng_context = RoleContext(
    role="senior software engineer",
    expertise_level="10+ years",
    communication_style="direct, focuses on architectural implications and performance"
)
senior_prompt = create_code_review(buggy_code, senior_eng_context)

# With security specialist role: Focus on security implications
security_context = RoleContext(
    role="security engineer",
    expertise_level="specialized in secure coding",
    communication_style="emphasizes security risks and mitigation"
)
security_prompt = create_code_review(buggy_code, security_context)

print("Generic prompt output emphasis: General improvements")
print("Senior engineer output emphasis: List comprehensions, pythonic patterns")
print("Security engineer output emphasis: Input validation, injection risks")
```

**Real constraints**: Role framing works best when the role exists clearly in training data. Highly specialized or fictional roles produce unpredictable results. Test effectiveness for your domain.

### 4. Output Format Specification

Explicit format constraints dramatically improve parsing reliability and downstream integration. Structure specifications should be provided as both description and example.

**Technical explanation**: LLMs are trained to continue patterns. Providing concrete examples of desired output format creates stronger statistical priors than descriptive instructions alone.

```python
from typing import Dict, Any, List
import json
import re

class OutputFormatter:
    """Utility for creating robust format specifications"""
    
    @staticmethod
    def create_json_spec(
        schema: Dict[str, str],
        example: Dict[str, Any],
        strict: bool = True
    ) -> str:
        """Generate format specification with schema and example"""
        
        spec = "Output must be valid JSON matching this structure:\n\n"
        
        # Add schema description
        spec += "Schema:\n"
        for key, type_desc in schema.items():
            spec += f"  - {key}: {type_desc}\n"
        
        # Add concrete example
        spec += f"\nExample format:\n```json\n{json.dumps(example, indent=2)}\n```\n"
        
        if strict:
            spec += "\nIMPORTANT: Output ONLY the JSON object. No explanatory text before or after."
        
        return spec
    
    @staticmethod
    def extract_json(response: str) -> Dict[str, Any]:
        """Robust JSON extraction from potentially messy output"""
        
        # Try direct parse first
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find first { to last }
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                pass
        
        raise ValueError("No valid JSON found in response")

# Example usage
schema = {
    "sentiment": "string, one of: positive, negative, neutral",
    "confidence": "float between 0 and 1",
    "key_phrases": "array of strings"
}

example = {
    "sentiment": "positive",
    "confidence": 0.87,
    "key_phrases": ["excellent service", "highly recommend"]
}

format_spec = OutputFormatter.create_json_spec(schema, example, strict=True)

# Use in prompt
def analyze_sentiment_structured(text: str) -> str:
    return f"""Analyze the sentiment of this text:

"{text}"

{format_spec}"""

prompt = analyze_sentiment_structured("The product exceeded expectations!")
print(prompt)

# The format spec + extraction makes parsing 90%+ reliable
# vs 60-70% with description alone
```

**Practical implications**: Always provide both schema and example. Examples should show edge cases (empty arrays, null values, etc.). Budget 50-100 extra tokens for robust format specs—worth it for reliable parsing.

### 5. Constraint and Boundary Definition

Explicit negative constraints ("do NOT do X") are as important as positive instructions. Models often produce unwanted behaviors unless explicitly forbidden.

```python
from typing import Set
from dataclasses import dataclass

@dataclass
class ConstraintSet:
    must_include: Set[str]
    must_not_include: Set[str]
    length_max: Optional[int] = None
    length_min: Optional[int] = None
    
    def to_prompt_section(self) -> str:
        """Convert constraints to clear prompt section"""
        sections = []
        
        if self.must_include:
            sections.append("REQUIRED elements:")
            for item in self.must_include:
                sections.append(f"  - {item}")
        
        if self.must_not_include:
            sections.append("\nFORBIDDEN elements:")
            for item in self.must_not_include:
                sections.append(f"  - {item}")
        
        if self.length_min or self.length_max:
            length_constraint = "Length: "
            if self.length_min:
                length_constraint += f"minimum {self.length_min} words"
            if self.length_max:
                if self.length_min:
                    length_constraint += f", maximum {self.length_max} words"
                else:
                    length_constraint += f"maximum {self.length_max} words"
            sections.append(f"\n{length_constraint}")
        
        return "\n".join(sections)

# Example: Generate product description with tight constraints
def generate_product_description(
    product_name: str,
    features: List[str],
    constraints: ConstraintSet
) -> str:
    
    return f"""Generate a product description for: {product_name}

Features: {', '.join(features)}

{constraints.to_prompt_section()}

Description:"""

# Without negative constraints, models often add unwanted elements
minimal_constraints = ConstraintSet(
    must_include={"product name", "key features"},
    must_not_include=set(),
    length_max=100
)

# With negative constraints, output is more controlled
strict_constraints = ConstraintSet(
    must_include={"product name", "key features"},
    must_not_include={
        "pricing information",
        "superlatives like 'best' or 'revolutionary'",
        "comparisons to competitors",
        "guarantees or warranties"
    },
    length_max=100
)

prompt_minimal = generate_product_description(
    "CloudSync Pro",
    ["real-time sync", "end-to-end encryption", "cross-platform"],
    minimal_constraints
)

prompt_strict = generate_product_description(
    "CloudSync Pro",
    ["real-time sync", "end-to-end encryption", "cross-platform"],
    strict_constraints
)

# Empirical testing shows strict constraints reduce unwanted content
# by 70-80% while only adding 20-30 tokens
```

## Hands-On Exercises

### Exercise 1: Context Position Optimization

**Objective**: Measure how context positioning affects output quality for a multi-constraint task.

**Instructions**:

1. Set up test environment:

```python
from typing import List, Tuple
import time

def test_context_positions(llm_client, query: str, contexts: List[