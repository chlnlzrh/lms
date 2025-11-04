# Pattern Library Development for LLM Applications

## Core Concepts

### Technical Definition

A pattern library in the context of LLM applications is a systematically organized collection of reusable prompt structures, response handling strategies, and interaction sequences that have been validated through testing and production use. Unlike traditional software pattern libraries that encapsulate code structures (like design patterns), LLM pattern libraries encapsulate successful approaches to communication with language models, including prompt templates, few-shot examples, chain-of-thought sequences, and error recovery strategies.

### Engineering Analogy: Traditional vs. Modern Approaches

**Traditional Approach: Ad-Hoc Prompting**

```python
def analyze_sentiment(text: str) -> str:
    """Naive approach - every call is a new experiment"""
    prompt = f"What's the sentiment of: {text}"
    response = llm_call(prompt)
    return response

# Every developer writes their own variation
def check_sentiment(comment: str) -> str:
    prompt = f"Tell me if this is positive or negative: {comment}"
    return llm_call(prompt)

# No consistency, no learning, no improvement
```

**Modern Approach: Pattern Library**

```python
from typing import Literal, TypedDict, List
from dataclasses import dataclass
from enum import Enum

class SentimentLabel(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

@dataclass
class SentimentPattern:
    system_context: str
    user_template: str
    few_shot_examples: List[tuple[str, str]]
    output_format: str
    
    def build_prompt(self, text: str) -> dict:
        """Consistently structured prompt with validated components"""
        examples = "\n".join([
            f"Text: {ex[0]}\nSentiment: {ex[1]}" 
            for ex in self.few_shot_examples
        ])
        
        return {
            "system": self.system_context,
            "user": f"{examples}\n\n{self.user_template.format(text=text)}"
        }

# Reusable, tested pattern
SENTIMENT_ANALYSIS_PATTERN = SentimentPattern(
    system_context=(
        "You are a sentiment analyzer. Return only one word: "
        "positive, negative, neutral, or mixed."
    ),
    user_template="Text: {text}\nSentiment:",
    few_shot_examples=[
        ("This product exceeded my expectations!", "positive"),
        ("Terrible service, never again.", "negative"),
        ("The package arrived on time.", "neutral"),
    ],
    output_format="single_word"
)

def analyze_sentiment(text: str) -> SentimentLabel:
    """Uses validated pattern with consistent results"""
    prompt = SENTIMENT_ANALYSIS_PATTERN.build_prompt(text)
    response = llm_call(prompt)
    return SentimentLabel(response.strip().lower())
```

The pattern library approach provides version control for prompts, enables A/B testing of variations, ensures consistent behavior across teams, and creates a foundation for measuring improvements over time.

### Key Insights That Change How Engineers Think

**1. Prompts Are Code, Not Configuration**

The prompt engineering phase reveals that prompts have the same characteristics as code: they require testing, debugging, versioning, and optimization. Treating them as throwaway strings leads to unmaintainable systems.

**2. Context Is Your Working Memory**

Unlike traditional APIs where state lives in databases, LLM state lives entirely in the context window. Pattern libraries must explicitly manage what information persists across interactions.

**3. Determinism Through Structure, Not Temperature**

Setting `temperature=0` doesn't guarantee consistent outputs. True consistency comes from constrained output formats, explicit instructions, and validated patterns that guide the model toward predictable responses.

### Why This Matters NOW

The LLM ecosystem has matured beyond experimentation. Production systems require:

- **Reproducibility**: Debugging requires knowing exactly what prompt generated a problematic output
- **Collaboration**: Teams need shared vocabularies and tested approaches
- **Iteration Speed**: Pattern libraries let you improve one component without breaking others
- **Cost Control**: Validated patterns use tokens efficiently and reduce wasted API calls
- **Quality Assurance**: You can't test what you can't standardize

## Technical Components

### 1. Prompt Templates with Variable Binding

**Technical Explanation**

Prompt templates separate static instruction text from dynamic input data, enabling reuse while maintaining type safety and validation. Unlike simple string interpolation, proper templating includes input sanitization, length constraints, and format validation.

**Practical Implications**

Templates prevent injection attacks, ensure consistent formatting, and enable automatic validation before expensive API calls. They also create natural boundaries for testing individual components.

**Real Constraints and Trade-offs**

- **Verbosity vs. Flexibility**: Highly structured templates reduce flexibility but increase reliability
- **Token Usage**: Template overhead consumes tokens on every call—optimize shared instructions
- **Maintenance Burden**: Each template variation creates another component to maintain

**Concrete Example**

```python
from typing import Protocol, Optional
import re

class PromptTemplate(Protocol):
    def render(self, **kwargs) -> str: ...
    def validate(self, **kwargs) -> bool: ...

class ExtractionTemplate:
    """Extract structured data from unstructured text"""
    
    def __init__(
        self,
        max_input_length: int = 2000,
        required_fields: Optional[List[str]] = None
    ):
        self.max_input_length = max_input_length
        self.required_fields = required_fields or []
        
        self.system = (
            "Extract information into JSON format. "
            "Use null for missing values. "
            "Return only valid JSON, no explanation."
        )
        
        self.user_template = """
Text: {text}

Extract these fields:
{field_schema}

JSON:"""
    
    def validate(self, text: str, fields: dict) -> tuple[bool, str]:
        """Validate inputs before API call"""
        if len(text) > self.max_input_length:
            return False, f"Text exceeds {self.max_input_length} chars"
        
        if not text.strip():
            return False, "Empty text provided"
        
        for field in self.required_fields:
            if field not in fields:
                return False, f"Missing required field: {field}"
        
        return True, ""
    
    def render(self, text: str, fields: dict) -> dict:
        """Render validated prompt"""
        is_valid, error = self.validate(text, fields)
        if not is_valid:
            raise ValueError(f"Invalid input: {error}")
        
        # Sanitize input - remove potential injection patterns
        clean_text = re.sub(r'```json\s*\{', '{', text)
        
        field_schema = "\n".join([
            f"- {name}: {spec}" 
            for name, spec in fields.items()
        ])
        
        return {
            "system": self.system,
            "user": self.user_template.format(
                text=clean_text,
                field_schema=field_schema
            )
        }

# Usage
template = ExtractionTemplate(
    max_input_length=1000,
    required_fields=["name", "email"]
)

prompt = template.render(
    text="Contact John Doe at john@example.com for details",
    fields={
        "name": "string - person's full name",
        "email": "string - email address",
        "phone": "string - phone number or null"
    }
)
```

### 2. Few-Shot Example Management

**Technical Explanation**

Few-shot learning provides the model with example input-output pairs to establish the desired behavior pattern. Effective example management involves selecting representative cases, ordering them strategically, and dynamically adjusting based on context.

**Practical Implications**

Well-chosen examples can reduce prompt complexity and improve accuracy more than verbose instructions. Poor examples waste tokens and can bias outputs in unexpected ways.

**Real Constraints and Trade-offs**

- **Token Cost**: Each example consumes context—typically 3-5 examples is optimal
- **Example Selection**: Diverse examples improve generalization; similar examples improve consistency
- **Dynamic vs. Static**: Static examples are predictable; dynamic selection (based on input similarity) improves relevance but adds complexity

**Concrete Example**

```python
from typing import List, Tuple, Callable
import json

class FewShotExampleStore:
    """Manage and select few-shot examples strategically"""
    
    def __init__(self):
        self.examples: List[Tuple[str, str, dict]] = []
        # (input, output, metadata) tuples
    
    def add_example(
        self,
        input_text: str,
        output: str,
        metadata: Optional[dict] = None
    ) -> None:
        """Add validated example to store"""
        meta = metadata or {}
        meta['added_at'] = datetime.utcnow().isoformat()
        self.examples.append((input_text, output, meta))
    
    def get_examples(
        self,
        n: int = 3,
        filter_fn: Optional[Callable] = None,
        shuffle: bool = False
    ) -> List[Tuple[str, str]]:
        """Retrieve examples with optional filtering"""
        candidates = self.examples
        
        if filter_fn:
            candidates = [
                ex for ex in candidates 
                if filter_fn(ex[2])  # Filter by metadata
            ]
        
        if shuffle:
            import random
            candidates = random.sample(
                candidates, 
                min(n, len(candidates))
            )
        else:
            candidates = candidates[:n]
        
        return [(ex[0], ex[1]) for ex in candidates]
    
    def format_examples(
        self,
        examples: List[Tuple[str, str]],
        format_style: str = "markdown"
    ) -> str:
        """Format examples for prompt inclusion"""
        if format_style == "markdown":
            return "\n\n".join([
                f"Input:\n{inp}\n\nOutput:\n{out}"
                for inp, out in examples
            ])
        elif format_style == "json":
            return json.dumps([
                {"input": inp, "output": out}
                for inp, out in examples
            ], indent=2)
        else:
            raise ValueError(f"Unknown format: {format_style}")

# Usage
store = FewShotExampleStore()

# Add examples with metadata for selective retrieval
store.add_example(
    input_text="Analyze: Revenue increased 15% YoY",
    output='{"metric": "revenue", "change": 15, "direction": "increase", "period": "year-over-year"}',
    metadata={"domain": "financial", "complexity": "simple"}
)

store.add_example(
    input_text="Analyze: Customer churn dropped from 8% to 5% this quarter",
    output='{"metric": "churn", "change": -37.5, "direction": "decrease", "period": "quarter"}',
    metadata={"domain": "customer", "complexity": "simple"}
)

store.add_example(
    input_text="Analyze: EBITDA margins compressed despite top-line growth",
    output='{"metric": "ebitda_margin", "change": null, "direction": "decrease", "period": null, "note": "qualitative"}',
    metadata={"domain": "financial", "complexity": "complex"}
)

# Retrieve examples filtered by domain
financial_examples = store.get_examples(
    n=2,
    filter_fn=lambda m: m.get("domain") == "financial"
)

formatted = store.format_examples(financial_examples, format_style="markdown")
```

### 3. Output Parsing and Validation

**Technical Explanation**

LLM outputs are probabilistic text generation, not structured API responses. Robust patterns include explicit output format specifications, parsing logic with fallbacks, and validation to ensure outputs meet business requirements.

**Practical Implications**

Even with perfect prompts, models occasionally produce malformed output. Production patterns must handle parsing failures gracefully, log failures for pattern improvement, and provide fallback behaviors.

**Real Constraints and Trade-offs**

- **Strictness vs. Robustness**: Strict parsing catches errors early; lenient parsing improves success rates
- **Retry Logic**: Retrying failed parses costs money and time but may recover from transient issues
- **Schema Evolution**: Changing output schemas breaks existing parsers—version your patterns

**Concrete Example**

```python
from typing import Union, Optional, TypeVar, Generic
from pydantic import BaseModel, ValidationError
import json

T = TypeVar('T', bound=BaseModel)

class ParseResult(Generic[T]):
    """Container for parse results with error handling"""
    
    def __init__(
        self,
        success: bool,
        data: Optional[T] = None,
        error: Optional[str] = None,
        raw_output: str = ""
    ):
        self.success = success
        self.data = data
        self.error = error
        self.raw_output = raw_output
    
    def unwrap_or_raise(self) -> T:
        """Get data or raise exception"""
        if not self.success:
            raise ValueError(f"Parse failed: {self.error}")
        return self.data
    
    def unwrap_or(self, default: T) -> T:
        """Get data or return default"""
        return self.data if self.success else default

class OutputParser(Generic[T]):
    """Parse and validate LLM outputs against schemas"""
    
    def __init__(self, schema: type[T], max_retries: int = 2):
        self.schema = schema
        self.max_retries = max_retries
    
    def parse(self, raw_output: str) -> ParseResult[T]:
        """Parse with fallback strategies"""
        
        # Strategy 1: Direct JSON parse
        result = self._try_json_parse(raw_output)
        if result.success:
            return result
        
        # Strategy 2: Extract JSON from markdown code blocks
        result = self._try_extract_json(raw_output)
        if result.success:
            return result
        
        # Strategy 3: Clean common issues and retry
        result = self._try_cleaned_parse(raw_output)
        if result.success:
            return result
        
        return ParseResult(
            success=False,
            error="All parsing strategies failed",
            raw_output=raw_output
        )
    
    def _try_json_parse(self, text: str) -> ParseResult[T]:
        """Attempt direct JSON parsing"""
        try:
            data = json.loads(text)
            validated = self.schema(**data)
            return ParseResult(success=True, data=validated, raw_output=text)
        except (json.JSONDecodeError, ValidationError) as e:
            return ParseResult(success=False, error=str(e), raw_output=text)
    
    def _try_extract_json(self, text: str) -> ParseResult[T]:
        """Extract JSON from markdown code blocks"""
        import re
        pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return self._try_json_parse(match.group(1))
        
        return ParseResult(success=False, error="No JSON found in text")
    
    def _try_cleaned_parse(self, text: str) -> ParseResult[T]:
        """Clean common issues and retry"""
        # Remove leading/trailing text
        cleaned = text.strip()
        
        # Find first { and last }
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]
            return self._try_json_parse(cleaned)
        
        return ParseResult(success=False, error="Could not extract JSON structure")

# Define output schema
class SentimentAnalysis(BaseModel):
    label: Literal["positive", "negative", "neutral", "mixed"]
    confidence: float
    key_phrases: List[str]

# Usage
parser = OutputParser(SentimentAnalysis)

# Case 1: Clean output
result = parser.parse('{"label": "positive", "confidence": 0.92, "key_phrases": ["great", "exceeded expectations"]}')
assert result.success
data = result.unwrap_or_raise()

# Case 2: Output with markdown
messy_output = """
Here's the analysis:

```