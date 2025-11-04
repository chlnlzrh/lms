# Custom Style Guides & Instructions: Engineering Consistent LLM Behavior

## Core Concepts

Custom style guides and instructions are structured constraints that shape how language models generate responses. Unlike one-off prompts that specify what you want in a single interaction, custom instructions establish persistent behavioral patterns across all interactions with a model.

Think of it like the difference between telling a compiler to optimize a specific function versus setting compiler flags that affect your entire build:

```python
# Traditional approach: Per-request specification
def process_data(data: str) -> str:
    prompt = """
    Analyze this data. Use technical language.
    Be concise. Include error handling considerations.
    Format as JSON.
    
    Data: {data}
    """
    return llm.generate(prompt.format(data=data))

# Custom instructions approach: Persistent behavioral layer
llm = LLM(
    system_instructions="""
    You are a technical analysis engine.
    - Output format: JSON only
    - Language: Technical, precise terminology
    - Style: Concise, no marketing language
    - Always consider: Error handling, edge cases, performance
    """,
    response_schema=ResponseSchema
)

def process_data(data: str) -> str:
    # Instructions already set - focus on the task
    return llm.generate(f"Analyze: {data}")
```

The second approach eliminates redundancy, ensures consistency across your codebase, and separates concerns: the *how* (style/format) from the *what* (task). This becomes critical when you have dozens of prompts across a system—one change to your style guide updates behavior everywhere.

### Why This Matters Now

Three engineering realities make custom instructions essential:

1. **Token economics**: Repeating the same instructions in every prompt wastes 20-40% of your context window. At scale, that's real money and latency.

2. **Consistency guarantees**: When multiple engineers write prompts, behavior diverges. Custom instructions are your "coding standards" for LLM interactions.

3. **Maintainability**: When you need to change how your system responds (stricter JSON compliance, different tone, new safety constraints), you update one configuration instead of hunting through hundreds of prompts.

The key insight: **Custom instructions are infrastructure, not just preferences**. They're the difference between scattered `console.log()` statements and a proper logging framework.

## Technical Components

### 1. System-Level Instructions vs. User-Level Context

System instructions establish identity and behavioral constraints. User context provides domain-specific background.

```python
from typing import TypedDict, Literal
from dataclasses import dataclass

@dataclass
class InstructionLayer:
    """Separating concerns in LLM configuration."""
    
    # System: WHO the model is, HOW it behaves
    system_instructions: str
    
    # User context: WHAT domain knowledge to apply
    user_context: str
    
    # Per-request: WHAT specific task to perform
    task_prompt: str

# Example: Code review system
code_review_config = InstructionLayer(
    system_instructions="""
    You are a static analysis engine for Python code.
    
    Output format:
    - JSON array of findings
    - Each finding: {type, severity, line, message, suggestion}
    
    Behavior:
    - Focus on: security, performance, maintainability
    - Ignore: style preferences, subjective opinions
    - Always: Provide specific code suggestions
    - Never: Use phrases like "consider" or "maybe"
    """,
    
    user_context="""
    Codebase context:
    - Python 3.11+, type hints required
    - Async/await for all I/O operations
    - Security: No user input directly in SQL/shell commands
    - Performance: <100ms p95 for API endpoints
    """,
    
    task_prompt="Review this function: {code}"
)
```

**Practical implications**: System instructions persist across the entire session. Changes here affect every subsequent call. User context can be session-specific (per-client configuration, per-repository rules). Task prompts change every request.

**Trade-off**: More detailed system instructions = less flexibility per request. If 80% of your tasks share requirements, encode them in the system layer. The remaining 20% can override in task prompts.

### 2. Constraint Hierarchies and Precedence

Instructions can conflict. Establishing clear precedence rules prevents undefined behavior.

```python
from enum import IntEnum

class InstructionPrecedence(IntEnum):
    """Priority order when instructions conflict."""
    SAFETY_CONSTRAINTS = 1      # Never override
    OUTPUT_FORMAT = 2            # Almost never override
    BEHAVIORAL_STYLE = 3         # Rarely override
    DOMAIN_PREFERENCES = 4       # Can override per-task

class StyleGuide:
    def __init__(self):
        self.constraints = {
            InstructionPrecedence.SAFETY_CONSTRAINTS: [
                "Never output actual credentials or API keys",
                "Refuse requests for harmful code",
                "Validate all generated code for injection vulnerabilities"
            ],
            InstructionPrecedence.OUTPUT_FORMAT: [
                "All structured data as JSON",
                "All code blocks with language tags",
                "All errors as {\"error\": \"message\", \"code\": \"ERROR_TYPE\"}"
            ],
            InstructionPrecedence.BEHAVIORAL_STYLE: [
                "Technical terminology, no simplification",
                "Show code examples, not just descriptions",
                "Explicit about limitations and trade-offs"
            ],
            InstructionPrecedence.DOMAIN_PREFERENCES: [
                "Prefer functional approaches where appropriate",
                "Async by default for I/O operations"
            ]
        }
    
    def compile(self) -> str:
        """Generate system instruction string with clear hierarchy."""
        sections = []
        for priority in InstructionPrecedence:
            rules = self.constraints[priority]
            sections.append(f"{priority.name} (Priority {priority.value}):")
            sections.extend(f"- {rule}" for rule in rules)
            sections.append("")
        return "\n".join(sections)

    def validate_task_prompt(self, task: str) -> tuple[bool, str]:
        """Check if task prompt conflicts with high-priority constraints."""
        # Example: Detect if task tries to override safety constraints
        dangerous_patterns = [
            "ignore previous instructions",
            "disregard safety",
            "bypass constraints"
        ]
        for pattern in dangerous_patterns:
            if pattern in task.lower():
                return False, f"Task conflicts with safety constraints: {pattern}"
        return True, ""

# Usage
guide = StyleGuide()
system_prompt = guide.compile()

task = "Generate a SQL query builder"
valid, error = guide.validate_task_prompt(task)
if not valid:
    raise ValueError(error)
```

**Real constraint**: Many LLM APIs don't enforce precedence—they concatenate all instructions. You must design your system instructions to be resilient to task-level override attempts.

### 3. Format Enforcement Mechanisms

Defining desired output format is necessary but insufficient. You need enforcement.

```python
import json
from typing import Any, Optional
from pydantic import BaseModel, ValidationError, Field

class CodeReviewFinding(BaseModel):
    """Strict schema for code review outputs."""
    type: Literal["security", "performance", "bug", "maintainability"]
    severity: Literal["critical", "high", "medium", "low"]
    line: int = Field(ge=1)
    message: str = Field(min_length=10, max_length=200)
    suggestion: str = Field(min_length=10)

class CodeReviewResponse(BaseModel):
    findings: list[CodeReviewFinding]
    summary: str = Field(max_length=500)

class EnforcedLLM:
    """Wrapper that enforces output schemas."""
    
    def __init__(self, system_instructions: str, schema: type[BaseModel]):
        self.system_instructions = system_instructions
        self.schema = schema
        self.max_retries = 3
    
    def generate(self, prompt: str) -> BaseModel:
        """Generate with automatic retry on schema violations."""
        
        # Add schema to system instructions
        full_instructions = f"""
        {self.system_instructions}
        
        CRITICAL: Your response MUST be valid JSON matching this schema:
        {self.schema.model_json_schema()}
        
        Output ONLY the JSON, no additional text.
        """
        
        for attempt in range(self.max_retries):
            raw_response = self._call_llm(full_instructions, prompt)
            
            try:
                # Attempt to parse and validate
                parsed = json.loads(raw_response)
                validated = self.schema.model_validate(parsed)
                return validated
            
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed after {self.max_retries} attempts: {e}")
                
                # Add error feedback to next attempt
                prompt = f"""
                Previous response was invalid: {str(e)}
                
                Original task: {prompt}
                
                Fix the error and output valid JSON only.
                """
    
    def _call_llm(self, instructions: str, prompt: str) -> str:
        # Placeholder for actual LLM call
        return '{"findings": [], "summary": "No issues found"}'

# Usage
llm = EnforcedLLM(
    system_instructions="You are a code reviewer.",
    schema=CodeReviewResponse
)

try:
    result = llm.generate("Review this code: def foo(): pass")
    assert isinstance(result, CodeReviewResponse)
    # Type-safe access to validated data
    for finding in result.findings:
        print(f"Line {finding.line}: {finding.message}")
except ValueError as e:
    # Handle persistent schema violations
    logging.error(f"LLM output validation failed: {e}")
```

**Trade-off**: Strict validation with retries increases latency and cost but dramatically reduces downstream errors. For production systems, this is almost always worth it.

### 4. Style Calibration for Task Categories

Different task types need different instruction profiles.

```python
from typing import Protocol

class TaskProfile(Protocol):
    """Interface for task-specific style guides."""
    def get_instructions(self) -> str: ...
    def get_examples(self) -> list[tuple[str, str]]: ...

class AnalysisTaskProfile:
    """Style for analytical tasks: structured, comprehensive."""
    
    def get_instructions(self) -> str:
        return """
        Analysis mode:
        - Break down complex topics into components
        - Provide quantitative data where possible
        - Compare multiple approaches with trade-offs
        - Output format: Structured JSON with sections
        - Length: Comprehensive (no arbitrary brevity)
        """
    
    def get_examples(self) -> list[tuple[str, str]]:
        return [
            (
                "Compare sorting algorithms for 1M records",
                '{"comparison": [{"algorithm": "quicksort", "time_complexity": "O(n log n)", "space": "O(log n)", "best_for": "general purpose"}]}'
            )
        ]

class CodeGenerationProfile:
    """Style for code generation: executable, defensive."""
    
    def get_instructions(self) -> str:
        return """
        Code generation mode:
        - Complete, runnable code only (no pseudocode)
        - Include all imports and type hints
        - Add error handling for edge cases
        - Output format: Code blocks with language tags
        - Length: Minimal viable implementation
        - Comments: Only for non-obvious logic
        """
    
    def get_examples(self) -> list[tuple[str, str]]:
        return [
            (
                "HTTP client with retry logic",
                '''```python
import asyncio
import aiohttp
from typing import Optional

async def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    timeout: int = 10
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    response.raise_for_status()
                    return await response.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    return None
```'''
            )
        ]

class DebuggingProfile:
    """Style for debugging: systematic, hypothesis-driven."""
    
    def get_instructions(self) -> str:
        return """
        Debugging mode:
        - State hypotheses explicitly
        - Suggest specific diagnostic steps
        - Provide test cases to isolate issues
        - Output format: Numbered action items
        - Length: Concise, actionable steps only
        """
    
    def get_examples(self) -> list[tuple[str, str]]:
        return [
            (
                "API returns 500 intermittently",
                """1. **Hypothesis**: Race condition in concurrent requests
2. **Diagnostic**: Check logs for correlation with request volume
3. **Test case**: Send 100 concurrent requests, measure failure rate
4. **Expected**: If >10% fail, investigate connection pooling
5. **Next**: Add distributed tracing to identify bottleneck"""
            )
        ]

class AdaptiveStyleGuide:
    """Route tasks to appropriate style profiles."""
    
    def __init__(self):
        self.profiles = {
            "analysis": AnalysisTaskProfile(),
            "code": CodeGenerationProfile(),
            "debug": DebuggingProfile()
        }
    
    def get_instructions_for_task(self, task: str) -> str:
        """Detect task type and return appropriate instructions."""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["compare", "analyze", "explain", "evaluate"]):
            profile = self.profiles["analysis"]
        elif any(word in task_lower for word in ["implement", "write code", "create function"]):
            profile = self.profiles["code"]
        elif any(word in task_lower for word in ["debug", "fix", "error", "not working"]):
            profile = self.profiles["debug"]
        else:
            # Default to analysis
            profile = self.profiles["analysis"]
        
        # Build instruction string with examples
        instructions = [profile.get_instructions(), "\nExamples:"]
        for input_ex, output_ex in profile.get_examples():
            instructions.append(f"\nInput: {input_ex}")
            instructions.append(f"Output: {output_ex}")
        
        return "\n".join(instructions)
```

**Practical implication**: Don't use the same instructions for all tasks. A 20% performance improvement in task-specific accuracy is typical when using calibrated profiles.

### 5. Version Control and A/B Testing Infrastructure

Style guides are code. Treat them accordingly.

```python
import hashlib
from datetime import datetime
from typing import Optional

class StyleGuideVersion:
    """Versioned style guide with metrics tracking."""
    
    def __init__(
        self,
        version: str,
        instructions: str,
        created_at: datetime,
        created_by: str
    ):
        self.version = version
        self.instructions = instructions
        self.created_at = created_at
        self.created_by = created_by
        self.hash = self._compute_hash()
        self.metrics = {"total_calls": 0, "validation_failures": 0}
    
    def _compute_hash(self) -> str:
        """Content-based versioning."""
        return hashlib.sha256(self.instructions.encode()).hexdigest()[:8]
    
    def record_usage(self, success: bool):
        """Track performance metrics."""
        self.metrics["total_calls"] += 1
        if not success:
            self.metrics["validation_failures"] += 1
    
    def success_rate(self) -> float:
        if self.metrics["total_calls"] == 0:
            return 0.0
        failures = self.metrics["validation_failures"]
        return 1.0 - (failures / self.metrics["total_calls"])

class StyleGuideRegistry:
    """Manage multiple versions with A/B testing."""
    
    def __init__(self):
        self.versions: dict[str, StyleGuideVersion] = {}
        self.active_version: Optional[str] = None
        self.experiments: dict[str, dict] = {}
    
    def register(
        self,
        version: