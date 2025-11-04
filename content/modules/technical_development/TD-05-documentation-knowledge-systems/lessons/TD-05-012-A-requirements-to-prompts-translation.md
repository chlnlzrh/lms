# Requirements-to-Prompts Translation

## Core Concepts

### Technical Definition

Requirements-to-prompts translation is the systematic process of converting structured business or technical requirements into executable natural language instructions that guide LLM behavior. Unlike traditional API specifications where parameters are typed and validated at compile time, prompts encode requirements as semantic constraints that models interpret at inference time.

This represents a fundamental shift in how we specify system behavior:

```python
# Traditional API approach: Typed, validated parameters
def generate_report(
    data: pd.DataFrame,
    format: Literal["csv", "json", "html"],
    include_summary: bool = True,
    max_rows: int = 1000
) -> str:
    """Parameter validation happens at call time."""
    if len(data) > max_rows:
        raise ValueError(f"Exceeds max_rows: {max_rows}")
    # Implementation with explicit control flow
    return formatted_report

# LLM approach: Requirements encoded as natural language
def generate_report_with_llm(data: pd.DataFrame, requirements: str) -> str:
    """
    requirements example:
    'Generate a summary report in HTML format. Include up to 1000 rows.
    Add statistical summary at the top. Use responsive tables.'
    """
    prompt = f"""
    Data: {data.to_json()}
    
    Requirements: {requirements}
    
    Generate the report following all requirements exactly.
    """
    return llm_call(prompt)
```

The traditional approach gives you compile-time safety and explicit error handling. The LLM approach trades that certainty for flexibility—you can specify requirements that would require extensive code changes in minutes. But this flexibility comes with a cost: ambiguity, unpredictability, and the need for robust validation.

### Key Engineering Insights

**1. Prompts are executable specifications, not documentation.** When you write requirements for a human developer, vagueness is acceptable—developers ask clarifying questions. When you translate requirements to prompts, ambiguity directly becomes runtime behavior variance. Every underspecified detail becomes a probability distribution over possible interpretations.

**2. The translation process is lossy by design.** Natural language cannot encode all the precision of typed systems. You're not just changing formats; you're moving from a deterministic execution model to a probabilistic one. The skill is knowing which precision to preserve and which to let the model infer.

**3. Validation shifts from compile-time to runtime.** In traditional systems, type checkers and linters catch specification errors before execution. With prompts, every execution is a validation event. This means your translation must include explicit verification instructions and structured output formats that enable automated checking.

### Why This Matters Now

The gap between business requirements and working code has historically been the most expensive bottleneck in software delivery. Requirements documents are ambiguous; code is precise. Translators (human developers) spend weeks resolving this impedance mismatch.

LLMs change the economics: they execute ambiguous specifications directly, but only if you can systematically translate requirements into prompts that preserve critical constraints while leveraging model capabilities. Teams that master this translation can prototype in hours instead of weeks, but teams that do it poorly ship unpredictable, unreliable systems.

The skill isn't optional anymore—it's becoming as fundamental as writing SQL queries or REST API specifications.

## Technical Components

### 1. Requirement Decomposition

**Technical Explanation:** Breaking complex, multi-faceted requirements into atomic, independently verifiable constraints that can be encoded as separate prompt components or validation rules.

**Practical Implications:** LLMs have limited ability to simultaneously satisfy many constraints. A requirement like "generate creative but factual marketing copy that's SEO-optimized and brand-aligned" contains at least four potentially conflicting objectives. Decomposition lets you prioritize, sequence, or separate these into multi-step workflows.

```python
from typing import TypedDict, List
from enum import Enum

class ConstraintType(Enum):
    MUST_HAVE = "must"      # Hard constraint, validate strictly
    SHOULD_HAVE = "should"  # Soft constraint, encourage but don't fail
    NICE_TO_HAVE = "could"  # Optional, mention if space permits

class AtomicConstraint(TypedDict):
    id: str
    type: ConstraintType
    description: str
    validation_fn: callable  # How to check if satisfied

def decompose_requirement(requirement: str) -> List[AtomicConstraint]:
    """
    Example requirement: 
    "Generate a product description that highlights key features,
    stays under 100 words, maintains professional tone, and includes
    a call-to-action."
    """
    return [
        {
            "id": "content",
            "type": ConstraintType.MUST_HAVE,
            "description": "Highlight key product features",
            "validation_fn": lambda text, features: all(
                f.lower() in text.lower() for f in features
            )
        },
        {
            "id": "length",
            "type": ConstraintType.MUST_HAVE,
            "description": "Stay under 100 words",
            "validation_fn": lambda text: len(text.split()) <= 100
        },
        {
            "id": "tone",
            "type": ConstraintType.SHOULD_HAVE,
            "description": "Maintain professional tone",
            "validation_fn": lambda text: not any(
                word in text.lower() 
                for word in ["awesome", "amazing", "incredible"]
            )
        },
        {
            "id": "cta",
            "type": ConstraintType.MUST_HAVE,
            "description": "Include call-to-action",
            "validation_fn": lambda text: any(
                phrase in text.lower() 
                for phrase in ["buy now", "learn more", "get started", "try"]
            )
        }
    ]
```

**Real Constraints:** Decomposition has overhead. Each atomic constraint adds tokens to your prompt and increases cognitive load on the model. Empirically, models handle 3-5 explicit constraints well; beyond 7-8, performance degrades. For complex requirements, consider chain-of-thought decomposition or multi-stage generation.

### 2. Constraint Encoding Strategies

**Technical Explanation:** Different requirements need different encoding approaches in prompts. Format constraints work well as explicit instructions; semantic constraints often need examples; creative constraints benefit from temperature and sampling parameters rather than prompt engineering.

```python
from typing import Literal

class ConstraintEncoder:
    """Translates different constraint types into prompt components."""
    
    @staticmethod
    def encode_format_constraint(
        format_type: Literal["json", "xml", "markdown", "csv"]
    ) -> str:
        """Format constraints: Explicit, with schema."""
        schemas = {
            "json": """
Output must be valid JSON matching this schema:
{
  "title": "string",
  "summary": "string",
  "key_points": ["string"],
  "confidence": 0.0-1.0
}
""",
            "markdown": """
Output must be valid Markdown with this structure:
# Title
## Summary
- Key point 1
- Key point 2
""",
        }
        return schemas.get(format_type, f"Output as {format_type}")
    
    @staticmethod
    def encode_semantic_constraint(examples: List[tuple[str, str]]) -> str:
        """Semantic constraints: Few-shot examples work better than rules."""
        encoded = "Follow these examples:\n\n"
        for input_ex, output_ex in examples:
            encoded += f"Input: {input_ex}\nOutput: {output_ex}\n\n"
        return encoded
    
    @staticmethod
    def encode_style_constraint(style_guide: dict) -> str:
        """Style constraints: Explicit rules with negative examples."""
        rules = []
        for aspect, guideline in style_guide.items():
            rules.append(f"- {aspect}: {guideline['rule']}")
            if "avoid" in guideline:
                rules.append(f"  ❌ Don't: {guideline['avoid']}")
            if "prefer" in guideline:
                rules.append(f"  ✅ Do: {guideline['prefer']}")
        return "Style Guide:\n" + "\n".join(rules)

# Usage example
encoder = ConstraintEncoder()

# Format: Explicit schema
format_prompt = encoder.encode_format_constraint("json")

# Semantics: Examples
semantic_prompt = encoder.encode_semantic_constraint([
    ("Analyze user churn", "Customer retention analysis showing 23% quarterly attrition"),
    ("Review Q4 sales", "Q4 revenue performance breakdown with YoY comparison")
])

# Style: Rules with examples
style_prompt = encoder.encode_style_constraint({
    "tone": {
        "rule": "Professional but conversational",
        "avoid": "Hey there! This is super awesome!",
        "prefer": "This represents a significant improvement in performance."
    },
    "length": {
        "rule": "Concise sentences, 15-20 words average",
        "avoid": "This is a really long sentence that goes on and on...",
        "prefer": "Clear, direct communication improves comprehension."
    }
})
```

**Real Constraints:** Different model families respond differently to encoding strategies. Instruction-tuned models follow explicit rules better; base models need more examples. Test your encoding strategy with your specific model and adjust based on failure modes.

### 3. Context and Scope Management

**Technical Explanation:** Requirements often assume context that's obvious to humans but missing for models. Translating requirements means making implicit context explicit while managing token budgets.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RequirementContext:
    """Explicit context that requirements assume."""
    domain: str                    # "healthcare", "finance", "e-commerce"
    user_expertise: str            # "expert", "intermediate", "novice"
    output_usage: str              # "internal_report", "customer_facing", "api_response"
    constraints: dict              # Regulatory, performance, brand constraints
    assumed_knowledge: List[str]   # What can be referenced without explanation

def translate_with_context(
    requirement: str,
    context: RequirementContext
) -> str:
    """Add necessary context to requirement before converting to prompt."""
    
    prompt_parts = []
    
    # Add domain context
    if context.domain:
        prompt_parts.append(f"Domain: {context.domain}")
        prompt_parts.append(f"Use {context.domain}-specific terminology appropriately.")
    
    # Add audience context
    audience_map = {
        "expert": "Use technical terminology without explanation.",
        "intermediate": "Balance technical accuracy with clarity.",
        "novice": "Explain technical terms when first used."
    }
    prompt_parts.append(audience_map.get(context.user_expertise, ""))
    
    # Add usage context (affects tone, precision, disclaimers)
    if context.output_usage == "customer_facing":
        prompt_parts.append("Output will be shown to customers. Be clear and avoid jargon.")
    elif context.output_usage == "internal_report":
        prompt_parts.append("Internal technical audience. Prioritize accuracy over simplicity.")
    
    # Add hard constraints
    if "regulatory" in context.constraints:
        prompt_parts.append(f"Regulatory requirements: {context.constraints['regulatory']}")
    
    # Add assumed knowledge as explicit references
    if context.assumed_knowledge:
        prompt_parts.append("You may reference: " + ", ".join(context.assumed_knowledge))
    
    # Add the actual requirement
    prompt_parts.append(f"\nTask: {requirement}")
    
    return "\n".join(prompt_parts)

# Example usage
context = RequirementContext(
    domain="healthcare",
    user_expertise="intermediate",
    output_usage="customer_facing",
    constraints={"regulatory": "HIPAA-compliant, no PII in examples"},
    assumed_knowledge=["common medical abbreviations", "insurance terminology"]
)

requirement = "Summarize the patient's treatment plan"

prompt = translate_with_context(requirement, context)
print(prompt)
# Output includes domain context, audience level, HIPAA compliance note, etc.
```

**Trade-offs:** More context improves relevance but consumes tokens. For tight token budgets, prioritize constraints that directly affect correctness over those that affect style. Profile your prompts: if the model already handles certain context well (e.g., professional tone), don't waste tokens restating it.

### 4. Output Structure Specification

**Technical Explanation:** Requirements that produce structured outputs (reports, API responses, data transformations) need explicit format specifications that enable programmatic validation and downstream processing.

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import json

class StructuredOutput(BaseModel):
    """Define expected output structure for validation."""
    
    class Config:
        # Enable strict validation
        extra = "forbid"

class SummaryOutput(StructuredOutput):
    title: str = Field(..., min_length=10, max_length=100)
    summary: str = Field(..., min_length=50, max_length=500)
    key_points: List[str] = Field(..., min_items=3, max_items=7)
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    @validator("key_points")
    def validate_key_points(cls, points):
        for point in points:
            if len(point) < 10:
                raise ValueError(f"Key point too short: {point}")
        return points

def create_structured_prompt(
    requirement: str,
    output_model: type[BaseModel]
) -> str:
    """Generate prompt that specifies exact output structure."""
    
    # Get JSON schema from Pydantic model
    schema = output_model.schema()
    
    prompt = f"""
{requirement}

Output must be valid JSON matching this exact schema:

{json.dumps(schema, indent=2)}

Example valid output:
{json.dumps(output_model.Config.schema_extra.get("example", {}), indent=2)}

Rules:
1. Return ONLY valid JSON, no explanatory text
2. All required fields must be present
3. Values must match specified types and constraints
4. Do not include fields not in the schema
"""
    return prompt

def validate_output(response: str, expected_model: type[BaseModel]) -> tuple[bool, Optional[BaseModel], Optional[str]]:
    """Validate LLM output against expected structure."""
    try:
        # Parse JSON
        parsed = json.loads(response)
        # Validate against Pydantic model
        validated = expected_model(**parsed)
        return True, validated, None
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, None, f"Validation failed: {str(e)}"

# Example usage
SummaryOutput.Config.schema_extra = {
    "example": {
        "title": "Q4 Performance Analysis",
        "summary": "Revenue increased 15% YoY driven by enterprise segment growth...",
        "key_points": [
            "Enterprise segment grew 28% QoQ",
            "Customer acquisition cost decreased 12%",
            "Churn rate remained stable at 3.2%"
        ],
        "confidence": 0.87
    }
}

prompt = create_structured_prompt(
    "Analyze the quarterly business metrics and provide key insights",
    SummaryOutput
)

# After getting LLM response
response = """{"title": "Q4 Analysis", "summary": "...", ...}"""
is_valid, parsed_output, error = validate_output(response, SummaryOutput)

if is_valid:
    print(f"Validated output: {parsed_output.title}")
else:
    print(f"Validation error: {error}")
```

**Real Constraints:** Structured output specifications increase prompt size significantly. For simple structures, explicit format instructions work well. For complex nested structures, consider using models that support function calling or JSON mode, which handles schema validation more reliably than prompt engineering alone.

### 5. Verification Criteria Translation

**Technical Explanation:** Requirements often include acceptance criteria ("should be accurate", "must be comprehensive"). These need translation into concrete, programmatically checkable conditions.

```python
from typing import Callable, Dict, Any
import re

class VerificationCriteria:
    """Translate vague acceptance criteria into testable checks."""
    
    @staticmethod
    def translate_accuracy_criterion(
        reference_data: Any,
        similarity_threshold: float = 0.85
    ) -> Callable[[str], bool]:
        """'Must be accurate' → Check against reference data."""
        def check(output: str) -> bool:
            # Implementation depends on domain
            # Example: