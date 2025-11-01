# Custom Rules & Style Enforcement: Advanced Implementation for LLM Systems

## Core Concepts

Custom rules and style enforcement represent the systematic control of LLM output through explicit constraints, formatting requirements, and behavioral guidelines embedded in prompts or system configurations. Unlike general prompting, which guides content direction, rule enforcement creates deterministic boundaries that the model must respect regardless of input variations.

### Traditional vs. Modern Constraint Systems

```python
# Traditional approach: Post-processing validation
def generate_response_v1(user_input: str, api_client) -> dict:
    """Generate response, then validate and retry if needed."""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        response = api_client.complete(
            prompt=f"Answer this question: {user_input}"
        )
        
        # Post-generation validation
        if len(response.split()) <= 100:
            if not contains_banned_words(response):
                if response.endswith(('.', '!', '?')):
                    return {"status": "success", "text": response}
    
    return {"status": "failed", "text": "Could not generate valid response"}


# Modern approach: Embedded enforcement
def generate_response_v2(user_input: str, api_client) -> dict:
    """Embed constraints directly in system prompt."""
    system_rules = """
You must follow these rules EXACTLY:
1. Response length: 50-100 words maximum
2. End every response with proper punctuation (. ! ?)
3. Never use: profanity, medical advice, financial recommendations
4. Format: Use markdown headers for sections
5. Tone: Professional but conversational

Violating these rules makes your response invalid.
"""
    
    response = api_client.complete(
        system=system_rules,
        prompt=user_input
    )
    
    # Validation as safety net, not primary mechanism
    if not validate_rules(response):
        raise RuleViolationError(f"Response violated rules: {response[:100]}")
    
    return {"status": "success", "text": response}
```

The modern approach reduces token waste by 60-80% (eliminating retry loops) and achieves 95%+ first-attempt compliance versus 40-60% with post-processing.

### Engineering Paradigm Shift

Traditional software validation assumes deterministic functions: `validate(output) -> bool`. LLMs require **constraint injection** where rules become part of the computation itself. Think of it like moving from runtime type checking to static typing—the constraints shape generation, not just filter results.

**Critical insights:**

1. **Rules are context, not filters**: The model interprets and applies rules during generation, making them part of the inference process rather than post-hoc validation.

2. **Specificity compounds**: Adding a fifth rule doesn't reduce compliance by 20%. Each rule interacts with others; five rules might reduce overall compliance by 50-70% without proper structuring.

3. **Rules compete for attention**: In a 4K context window, 500 tokens of rules leave 3.5K for actual task completion. Rule density has a cognitive cost on model performance.

### Why This Matters Now

Production LLM systems face regulatory requirements (GDPR, HIPAA), brand consistency mandates, and user safety obligations that cannot rely on probabilistic compliance. A financial services chatbot that occasionally suggests unregistered investment advice creates legal liability. A customer service bot that randomly drops into casual slang damages brand trust.

Early LLM deployments treated style as a "nice to have." Current production systems require enforcement mechanisms with 99%+ compliance rates, verifiable audit trails, and graceful degradation when rules conflict with user requests.

## Technical Components

### 1. Rule Hierarchy & Priority Systems

Rule conflicts emerge when multiple constraints cannot simultaneously be satisfied. A system rule demanding "concise responses under 50 words" conflicts with a user request for "detailed explanation with examples."

**Technical implementation:**

```python
from enum import IntEnum
from typing import List, Dict, Optional
from dataclasses import dataclass

class RulePriority(IntEnum):
    """Priority levels for rule enforcement."""
    SAFETY_CRITICAL = 100      # Never violate (PII, harmful content)
    COMPLIANCE_REQUIRED = 80   # Legal/regulatory requirements
    BRAND_STANDARDS = 60       # Style guide, tone requirements
    QUALITY_PREFERENCES = 40   # Formatting, structure preferences
    OPTIMIZATION_HINTS = 20    # Performance suggestions

@dataclass
class EnforcementRule:
    """Structured rule with priority and validation."""
    name: str
    description: str
    priority: RulePriority
    validation_pattern: Optional[str] = None
    examples_compliant: List[str] = None
    examples_violation: List[str] = None

def build_rule_system(rules: List[EnforcementRule]) -> str:
    """Generate prioritized rule text for system prompt."""
    # Sort by priority, group by level
    sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)
    
    prompt_sections = []
    prompt_sections.append("# ENFORCEMENT RULES (Priority Order)\n")
    
    current_priority = None
    for rule in sorted_rules:
        if rule.priority != current_priority:
            current_priority = rule.priority
            prompt_sections.append(f"\n## {current_priority.name} (Priority: {current_priority.value})")
            prompt_sections.append("These rules OVERRIDE lower priority requirements.\n")
        
        prompt_sections.append(f"### {rule.name}")
        prompt_sections.append(f"{rule.description}")
        
        if rule.examples_compliant:
            prompt_sections.append("\n✓ COMPLIANT:")
            for example in rule.examples_compliant:
                prompt_sections.append(f"  - {example}")
        
        if rule.examples_violation:
            prompt_sections.append("\n✗ VIOLATION:")
            for example in rule.examples_violation:
                prompt_sections.append(f"  - {example}")
        
        prompt_sections.append("")  # Blank line separator
    
    return "\n".join(prompt_sections)

# Example usage
rules = [
    EnforcementRule(
        name="PII Protection",
        description="Never output social security numbers, credit card numbers, or passwords. Redact if present in input.",
        priority=RulePriority.SAFETY_CRITICAL,
        examples_compliant=["The account ending in ****1234 was updated."],
        examples_violation=["The account 5555-1234-5678-9012 was updated."]
    ),
    EnforcementRule(
        name="Response Length",
        description="Aim for 100-200 words unless user explicitly requests longer.",
        priority=RulePriority.QUALITY_PREFERENCES
    ),
    EnforcementRule(
        name="Professional Tone",
        description="Use business-appropriate language. No slang or casual expressions.",
        priority=RulePriority.BRAND_STANDARDS,
        examples_compliant=["I understand your concern.", "Let me clarify that point."],
        examples_violation=["Yeah, totally get it!", "That's gonna be tough."]
    )
]

system_prompt = build_rule_system(rules)
print(system_prompt[:500])
```

**Practical implications:**

- Priority systems prevent rule paralysis where the model spends tokens trying to satisfy incompatible constraints
- Explicit hierarchies reduce ambiguity-related hallucination by 30-40%
- Validation patterns enable automated compliance checking without human review

**Real constraints:**

Priority systems add 200-500 tokens to system prompts. For applications with small context windows (4K), this represents 5-12% overhead. Test whether structured priorities improve compliance enough to justify the cost versus simple bullet lists.

### 2. Style Specification Languages

Natural language rules like "be professional" yield inconsistent interpretation. Style specification languages provide structured, parseable rule definitions.

```python
from typing import Literal, Union
from pydantic import BaseModel, Field, validator

class ToneSpecification(BaseModel):
    """Structured tone definition."""
    formality: Literal["casual", "conversational", "professional", "formal", "academic"]
    emotion: Literal["neutral", "empathetic", "enthusiastic", "serious"]
    technical_level: Literal["layperson", "informed", "specialist", "expert"]
    
class FormatSpecification(BaseModel):
    """Structured format requirements."""
    structure: Literal["paragraph", "bullet_points", "numbered_list", "sections_with_headers"]
    markdown_enabled: bool = True
    max_words: int = Field(ge=10, le=5000)
    include_examples: bool = False
    include_citations: bool = False

class ContentConstraints(BaseModel):
    """What content is allowed/forbidden."""
    forbidden_topics: List[str] = Field(default_factory=list)
    required_disclaimers: List[str] = Field(default_factory=list)
    allowed_domains: Optional[List[str]] = None  # For URL references
    
    @validator('forbidden_topics')
    def topics_lowercase(cls, v):
        return [topic.lower() for topic in v]

class StyleEnforcementSpec(BaseModel):
    """Complete style specification."""
    tone: ToneSpecification
    format: FormatSpecification
    content: ContentConstraints
    
    def to_prompt_rules(self) -> str:
        """Convert specification to natural language rules."""
        rules = []
        
        # Tone rules
        rules.append(f"TONE: Write at a {self.tone.formality} level with {self.tone.emotion} emotional tone.")
        rules.append(f"Assume reader is a {self.tone.technical_level}.")
        
        # Format rules
        rules.append(f"\nFORMAT: Use {self.format.structure} structure.")
        rules.append(f"Length: Maximum {self.format.max_words} words.")
        if self.format.include_examples:
            rules.append("Include at least one concrete example.")
        
        # Content rules
        if self.content.forbidden_topics:
            topics_str = ", ".join(self.content.forbidden_topics)
            rules.append(f"\nFORBIDDEN: Never discuss {topics_str}.")
        
        if self.content.required_disclaimers:
            rules.append("\nDISCLAIMERS: Include these disclaimers:")
            for disclaimer in self.content.required_disclaimers:
                rules.append(f"  - {disclaimer}")
        
        return "\n".join(rules)

# Example: Customer support bot specification
support_spec = StyleEnforcementSpec(
    tone=ToneSpecification(
        formality="professional",
        emotion="empathetic",
        technical_level="informed"
    ),
    format=FormatSpecification(
        structure="sections_with_headers",
        max_words=250,
        include_examples=True
    ),
    content=ContentConstraints(
        forbidden_topics=["pricing for non-customers", "unreleased features"],
        required_disclaimers=["This guidance is based on current product capabilities."]
    )
)

print(support_spec.to_prompt_rules())
```

**Practical implications:**

- Structured specifications enable A/B testing of individual rule components (e.g., test 3 formality levels while holding other parameters constant)
- Machine-readable specs integrate with validation pipelines and analytics
- Version control and rollback become trivial (store specs as JSON)

**Trade-offs:**

Structured specs require upfront engineering investment. For simple use cases (single bot with stable rules), direct natural language prompts are faster to implement. Use structured specs when managing 5+ bots, frequently updating rules, or requiring compliance auditing.

### 3. Few-Shot Rule Demonstration

Abstract rules like "maintain consistent formatting" are ambiguous. Few-shot examples demonstrate exact compliance.

```python
from typing import List, Tuple

def create_few_shot_enforcement(
    rules: str,
    examples: List[Tuple[str, str, str]]  # (input, bad_output, good_output)
) -> str:
    """Generate few-shot prompt with explicit rule demonstrations."""
    
    prompt_parts = [rules, "\n# EXAMPLES\n"]
    
    for i, (user_input, bad_output, good_output) in enumerate(examples, 1):
        prompt_parts.append(f"\n## Example {i}")
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append(f"\n✗ INCORRECT (violates rules):")
        prompt_parts.append(f"{bad_output}")
        prompt_parts.append(f"\n✓ CORRECT (follows rules):")
        prompt_parts.append(f"{good_output}")
        prompt_parts.append("\n---")
    
    return "\n".join(prompt_parts)

# Example: Enforce structured technical explanations
rule_text = """
RULES:
1. Start with one-sentence definition
2. Follow with "Why it matters:" section
3. Include code example with comments
4. End with "Key takeaway:" one-liner
"""

examples = [
    (
        "Explain API rate limiting",
        # Bad output (violates structure)
        "API rate limiting controls request frequency. It prevents abuse and ensures fair resource allocation. You might implement it with a token bucket algorithm.",
        # Good output (follows structure)
        """API rate limiting restricts the number of requests a client can make within a time window.

Why it matters: Prevents service degradation from traffic spikes and ensures equitable resource distribution across clients.

```python
# Simple rate limiter using token bucket
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = []  # Timestamps of recent requests
```

Key takeaway: Rate limiting trades off immediate availability for long-term system stability."""
    ),
    (
        "Explain database indexing",
        # Bad output
        "Indexes speed up queries by creating sorted data structures. They're like a book index that helps you find information quickly without reading every page.",
        # Good output
        """Database indexing creates auxiliary data structures to accelerate query performance.

Why it matters: Transforms O(n) table scans into O(log n) lookups, critical for production databases serving millions of rows.

```python
# Illustrative comparison
# Without index: scan all rows
result = [row for row in table if row['user_id'] == target]  # O(n)

# With index: binary search on sorted structure
result = index_lookup(user_id_index, target)  # O(log n)
```

Key takeaway: Indexes trade increased write cost and storage for dramatically faster reads."""
    )
]

enforcement_prompt = create_few_shot_enforcement(rule_text, examples)

# Use in actual API call
def generate_with_enforcement(user_question: str, api_client) -> str:
    """Generate response with few-shot rule enforcement."""
    full_prompt = f"{enforcement_prompt}\n\n# YOUR TURN\nUser: {user_question}\n\nAssistant:"
    
    response = api_client.complete(
        system="You are a technical educator. Follow the structure rules exactly.",
        prompt=full_prompt
    )
    
    return response
```

**Practical implications:**

- Few-shot examples increase rule compliance by 40-60% compared to abstract rules alone
- Particularly effective for format-heavy rules (JSON structure, specific markdown patterns)
- Examples serve as implicit test cases; if model can't reproduce example quality, rules are too complex

**Constraints:**

Each example pair consumes 100-300 tokens. Context window limits typically allow 3-5 examples maximum. Prioritize examples that demonstrate the most frequently violated or most critical rules.

### 4. Validation & Enforcement Pipelines

Even with strong prompting, probabilistic models occasionally violate rules. Production systems require validation layers with graceful handling strategies.

```python
import re
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum

class ViolationType(Enum):
    """Severity levels for rule violations."""
    CRITICAL = "critical"      # Must block output
    WARNING = "warning"        # Log but may allow
    INFO = "info"             # Track for metrics only

@dataclass
class ValidationResult:
    """Result of rule validation check."""
    passed: bool
    violation_type: Optional[ViolationType]
    rule_name: str
    message: str
    suggestion: Optional[str] = None

class RuleValidator:
    """Validates LLM outputs against defined rules."""
    
    def __init__(self):
        self.validators: Dict[str, Callable] = {}
        
    def register(
        