# Team Prompt Libraries: Engineering Reusable AI Interactions

## Core Concepts

A **team prompt library** is a version-controlled, systematically organized collection of validated prompt patterns that encapsulates institutional knowledge about effective LLM interactions. Think of it as your team's API specification layer for AI systems—defining standardized interfaces, parameters, and expected behaviors for common AI operations.

### Traditional vs. Modern Approach

**Traditional ad-hoc prompting:**

```python
# Each developer writes prompts inline, inconsistent patterns
def analyze_code(code: str) -> str:
    prompt = f"Look at this code and tell me if it's good:\n{code}"
    return llm_call(prompt)

def review_pr(diff: str) -> str:
    prompt = f"Review this: {diff}"
    return llm_call(prompt)

# Problems:
# - No consistency across team
# - No versioning or testing
# - Knowledge locked in individual developers
# - Quality varies wildly
# - No measurable improvement over time
```

**Prompt library approach:**

```python
from typing import TypedDict, Literal
from datetime import datetime

class PromptTemplate:
    def __init__(
        self, 
        template: str, 
        version: str,
        validated_at: datetime,
        success_rate: float
    ):
        self.template = template
        self.version = version
        self.validated_at = validated_at
        self.success_rate = success_rate
    
    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs)

# Centralized, versioned, validated prompts
CODE_ANALYSIS_V2 = PromptTemplate(
    template="""Analyze the following {language} code for:
1. Correctness: logical errors, edge cases
2. Security: potential vulnerabilities
3. Performance: algorithmic complexity

Code:
```{language}
{code}
```

Provide specific line numbers and actionable recommendations.""",
    version="2.1.0",
    validated_at=datetime(2024, 1, 15),
    success_rate=0.87  # 87% of outputs required no revision
)

def analyze_code(code: str, language: str) -> str:
    prompt = CODE_ANALYSIS_V2.render(code=code, language=language)
    return llm_call(prompt)

# Benefits:
# - Consistent quality across team
# - A/B testable (compare versions)
# - Onboard new engineers faster
# - Continuous improvement with metrics
# - Reviewable and auditable
```

### Key Engineering Insights

**1. Prompts are infrastructure, not code comments.** They deserve the same rigor as API design: versioning, testing, documentation, and deprecation policies. A poorly designed prompt can cascade failures across your entire system.

**2. Specificity compounds.** A prompt library with 20 well-crafted, specific templates outperforms 5 generic ones by orders of magnitude. Generic prompts produce generic outputs that require extensive post-processing.

**3. Observability is non-negotiable.** Without instrumentation (success rates, token usage, latency), you're optimizing blind. Every prompt execution should generate metrics.

### Why This Matters Now

The gap between teams with prompt libraries and those without is widening rapidly. Consider:

- **Cost control:** Teams report 40-60% token reduction by eliminating redundant context and optimizing for conciseness
- **Quality consistency:** Standard deviation in output quality drops by ~70% when using validated templates
- **Velocity:** New features requiring LLM integration go from days to hours when building on proven patterns
- **Compliance:** Regulated industries need audit trails—prompt versioning provides it

Without systematic prompt management, you're accumulating technical debt that becomes harder to refactor as your AI surface area expands.

## Technical Components

### 1. Template Structure and Parameterization

A robust prompt template balances flexibility with constraints. Too rigid and it's not reusable; too flexible and quality degrades.

**Technical implementation:**

```python
from typing import Optional, List, Dict, Any
from enum import Enum
import json

class OutputFormat(Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    PLAIN = "plain"

class PromptTemplate:
    """
    Immutable prompt template with type-safe parameters.
    """
    def __init__(
        self,
        name: str,
        system_message: str,
        user_template: str,
        required_params: List[str],
        optional_params: Optional[Dict[str, Any]] = None,
        output_format: OutputFormat = OutputFormat.PLAIN,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ):
        self.name = name
        self.system_message = system_message
        self.user_template = user_template
        self.required_params = required_params
        self.optional_params = optional_params or {}
        self.output_format = output_format
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    def validate_params(self, params: Dict[str, Any]) -> None:
        """Fail fast on missing required parameters."""
        missing = set(self.required_params) - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
    
    def render(self, **kwargs) -> Dict[str, Any]:
        """Render prompt with parameters, applying defaults."""
        # Merge with optional defaults
        final_params = {**self.optional_params, **kwargs}
        self.validate_params(final_params)
        
        user_message = self.user_template.format(**final_params)
        
        # Add output format instruction
        if self.output_format == OutputFormat.JSON:
            user_message += "\n\nProvide output as valid JSON only."
        
        return {
            "system": self.system_message,
            "user": user_message,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

# Example: SQL query generation template
SQL_GENERATOR = PromptTemplate(
    name="sql_generator_v3",
    system_message="""You are a SQL expert. Generate safe, optimized queries.
Always use parameterized queries. Explain your indexing strategy.""",
    user_template="""Database schema:
{schema}

User request: {request}

Constraints:
- Read-only operations
- Maximum 1000 rows
- Execution timeout: 30s
{additional_constraints}""",
    required_params=["schema", "request"],
    optional_params={"additional_constraints": ""},
    output_format=OutputFormat.JSON,
    temperature=0.1,  # Low temperature for deterministic SQL
    max_tokens=500
)
```

**Practical implications:** Parameter validation catches errors before expensive LLM calls. Type safety prevents runtime failures in production. Default values encode best practices (e.g., low temperature for code generation).

**Trade-offs:** More structure means more upfront work. For experimental prompts, start simple and formalize as patterns emerge. Over-parameterization creates maintenance burden.

### 2. Versioning and Migration Strategy

Prompt evolution is inevitable. Breaking changes need migration paths, just like database schemas.

```python
from dataclasses import dataclass
from typing import Callable, Optional
import hashlib

@dataclass
class PromptVersion:
    version: str
    template: PromptTemplate
    deprecated: bool = False
    successor: Optional[str] = None
    migration_guide: Optional[str] = None
    
    @property
    def content_hash(self) -> str:
        """Deterministic hash for detecting unintended changes."""
        content = f"{self.template.system_message}{self.template.user_template}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]

class PromptLibrary:
    def __init__(self):
        self._prompts: Dict[str, List[PromptVersion]] = {}
    
    def register(
        self, 
        name: str, 
        version: PromptVersion
    ) -> None:
        if name not in self._prompts:
            self._prompts[name] = []
        self._prompts[name].append(version)
    
    def get(
        self, 
        name: str, 
        version: Optional[str] = None
    ) -> PromptTemplate:
        """Get specific version or latest non-deprecated."""
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found")
        
        versions = self._prompts[name]
        
        if version:
            for v in versions:
                if v.version == version:
                    if v.deprecated:
                        print(f"Warning: {name}@{version} is deprecated. "
                              f"Migrate to {v.successor}. {v.migration_guide}")
                    return v.template
            raise ValueError(f"Version {version} not found")
        
        # Return latest non-deprecated
        active = [v for v in versions if not v.deprecated]
        if not active:
            raise ValueError(f"No active versions for {name}")
        
        return sorted(active, key=lambda v: v.version)[-1].template
    
    def deprecate(
        self, 
        name: str, 
        version: str, 
        successor: str,
        guide: str
    ) -> None:
        """Mark version as deprecated with migration path."""
        for v in self._prompts[name]:
            if v.version == version:
                v.deprecated = True
                v.successor = successor
                v.migration_guide = guide
                break

# Usage example
library = PromptLibrary()

# Register v1
library.register("code_review", PromptVersion(
    version="1.0.0",
    template=PromptTemplate(
        name="code_review_v1",
        system_message="Review code for issues.",
        user_template="Code: {code}",
        required_params=["code"]
    )
))

# Register improved v2
library.register("code_review", PromptVersion(
    version="2.0.0",
    template=PromptTemplate(
        name="code_review_v2",
        system_message="Review code for correctness, security, and performance.",
        user_template="""Language: {language}
Code:
{code}

Focus areas: {focus_areas}""",
        required_params=["code", "language"],
        optional_params={"focus_areas": "all"}
    )
))

# Deprecate v1
library.deprecate(
    "code_review", 
    "1.0.0",
    successor="2.0.0",
    guide="Add 'language' parameter. Use 'focus_areas' for targeted reviews."
)

# This triggers deprecation warning
review_prompt = library.get("code_review", version="1.0.0")
```

**Real constraints:** Versioning adds complexity. Only version prompts used in multiple places. For one-off tasks, inline prompts are fine. Hash-based change detection prevents accidental modifications to production prompts.

### 3. Testing and Validation Framework

Prompts need tests like code needs tests. Golden datasets, regression suites, and quality metrics.

```python
from typing import List, Callable
from dataclasses import dataclass
import re

@dataclass
class TestCase:
    name: str
    inputs: Dict[str, Any]
    validation: Callable[[str], bool]
    expected_characteristics: List[str]

class PromptTestSuite:
    def __init__(self, template: PromptTemplate):
        self.template = template
        self.test_cases: List[TestCase] = []
    
    def add_test(self, test: TestCase) -> None:
        self.test_cases.append(test)
    
    def run(self, llm_call: Callable) -> Dict[str, Any]:
        """Execute all tests and return results."""
        results = {
            "passed": 0,
            "failed": 0,
            "failures": []
        }
        
        for test in self.test_cases:
            try:
                prompt_data = self.template.render(**test.inputs)
                output = llm_call(prompt_data)
                
                if test.validation(output):
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["failures"].append({
                        "test": test.name,
                        "reason": "Validation failed",
                        "output": output[:200]
                    })
            except Exception as e:
                results["failed"] += 1
                results["failures"].append({
                    "test": test.name,
                    "reason": str(e)
                })
        
        results["pass_rate"] = results["passed"] / len(self.test_cases)
        return results

# Example: Testing SQL generator
sql_tests = PromptTestSuite(SQL_GENERATOR)

def validate_sql_output(output: str) -> bool:
    """Check if output is valid JSON with SQL and explanation."""
    try:
        data = json.loads(output)
        return (
            "query" in data and
            "explanation" in data and
            "SELECT" in data["query"].upper() and
            len(data["explanation"]) > 20
        )
    except:
        return False

sql_tests.add_test(TestCase(
    name="simple_select",
    inputs={
        "schema": "users(id, email, created_at)",
        "request": "Get all users created in the last week"
    },
    validation=validate_sql_output,
    expected_characteristics=[
        "Uses WHERE clause with date comparison",
        "Includes index recommendation",
        "Parameterized for injection safety"
    ]
))

sql_tests.add_test(TestCase(
    name="complex_join",
    inputs={
        "schema": "users(id, email), orders(id, user_id, total)",
        "request": "Find users with orders over $100"
    },
    validation=validate_sql_output,
    expected_characteristics=[
        "Uses JOIN",
        "Filters on total",
        "Suggests index on total column"
    ]
))

# Run tests (in CI/CD)
# results = sql_tests.run(your_llm_function)
# assert results["pass_rate"] > 0.9, "Prompt quality regression detected"
```

**Concrete benefits:** Catch regressions when updating prompts. Establish quality baselines. Automated testing in CI/CD pipelines. Teams report 85%+ reduction in production prompt issues after implementing test suites.

### 4. Observability and Metrics Collection

Instrument every prompt execution to track performance and iterate effectively.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import time

@dataclass
class PromptExecution:
    prompt_name: str
    prompt_version: str
    inputs: Dict[str, Any]
    output: str
    duration_ms: float
    tokens_used: int
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None
    user_feedback: Optional[int] = None  # 1-5 rating

class PromptObserver:
    """Collect metrics on prompt executions."""
    def __init__(self):
        self.executions: List[PromptExecution] = []
    
    def log(self, execution: PromptExecution) -> None:
        self.executions.append(execution)
    
    def get_metrics(self, prompt_name: str) -> Dict[str, Any]:
        """Calculate aggregate metrics for a prompt."""
        relevant = [e for e in self.executions if e.prompt_name == prompt_name]
        
        if not relevant:
            return {}
        
        total_cost = sum(e.cost_usd for e in relevant)
        avg_duration = sum(e.duration_ms for e in relevant) / len(relevant)
        success_rate = sum(1 for e in relevant if e.success) / len(relevant)
        
        with_feedback = [e for e in relevant if e.user_feedback]
        avg_rating = (
            sum(e.user_feedback for e in with_feedback) / len(with_feedback)
            if with_feedback else None
        )
        
        return {
            "total_executions": len(relevant),
            "success_rate": success_rate,
            "avg_duration_ms": avg_duration,