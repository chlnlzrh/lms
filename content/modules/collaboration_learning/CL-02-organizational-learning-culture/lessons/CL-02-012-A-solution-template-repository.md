# Solution Template Repository: Engineering Reusable AI Patterns

## Core Concepts

A solution template repository is a structured collection of validated, parameterized AI implementation patterns that encapsulate both code and decision logic for recurring AI problems. Unlike traditional code libraries that provide functions, solution templates combine architecture, prompts, evaluation criteria, configuration, and orchestration logic into composable units.

### Traditional vs. Template-Based Approach

**Traditional Approach:**
```python
# Each developer implements from scratch
def analyze_sentiment(text: str) -> str:
    # Custom prompt every time
    prompt = f"What's the sentiment of: {text}"
    response = llm_call(prompt)
    return response

def extract_entities(text: str) -> list:
    # Reinventing the wheel
    prompt = f"Find entities in: {text}"
    response = llm_call(prompt)
    return parse_response(response)

# No consistency, no learning across implementations
```

**Template-Based Approach:**
```python
from typing import TypedDict, Generic, TypeVar, Callable
import json

T = TypeVar('T')

class TemplateConfig(TypedDict):
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str

class SolutionTemplate(Generic[T]):
    """Base template with validation and instrumentation."""
    
    def __init__(
        self,
        name: str,
        config: TemplateConfig,
        output_parser: Callable[[str], T],
        validator: Callable[[T], bool]
    ):
        self.name = name
        self.config = config
        self.output_parser = output_parser
        self.validator = validator
        self.metrics: dict = {"calls": 0, "failures": 0}
    
    def execute(self, user_prompt: str, **kwargs) -> T:
        """Execute with built-in validation and monitoring."""
        self.metrics["calls"] += 1
        
        try:
            response = self._call_llm(user_prompt, **kwargs)
            parsed = self.output_parser(response)
            
            if not self.validator(parsed):
                raise ValueError(f"Validation failed for {self.name}")
            
            return parsed
        except Exception as e:
            self.metrics["failures"] += 1
            raise

# Reusable, tested, monitored template
sentiment_template = SolutionTemplate(
    name="sentiment_analysis_v2",
    config={
        "model": "claude-sonnet-4",
        "temperature": 0.1,
        "max_tokens": 50,
        "system_prompt": "You are a sentiment classifier. Return only: positive, negative, or neutral."
    },
    output_parser=lambda x: x.strip().lower(),
    validator=lambda x: x in ["positive", "negative", "neutral"]
)
```

### Key Engineering Insights

**1. Templates Encode Organizational Learning**

Every AI implementation teaches lessons about what works. Without templates, this knowledge evaporates. A template that's been run 10,000 times and refined embodies far more wisdom than documentation.

**2. Parameterization is the Core Design Challenge**

The hard part isn't writing code—it's determining what should be configurable vs. fixed. Too rigid: templates aren't reusable. Too flexible: they're just wrapper functions with no opinionation. The sweet spot captures the 80% case while allowing 20% customization.

**3. Templates are Living Artifacts**

Unlike static libraries, templates evolve through usage metrics. A template with 80% failure rate gets deprecated. One with 99% success becomes the standard. The repository becomes a performance-ranked marketplace of solutions.

### Why This Matters Now

AI development has reached the "design patterns" phase. In the 1990s, every team implemented singletons and observers differently until GoF patterns standardized approaches. AI is at that inflection point. Without templates:

- Teams solve the same problems repeatedly (wasted effort)
- Quality varies wildly across implementations (production risk)
- Knowledge doesn't accumulate (organizational amnesia)
- Onboarding takes months (productivity drain)

Template repositories compress expertise into reusable form, accelerating teams from months to days.

## Technical Components

### 1. Template Structure Schema

A template must capture everything needed to reproduce a solution reliably.

**Core Structure:**
```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

class TemplateCategory(Enum):
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    TRANSFORMATION = "transformation"
    REASONING = "reasoning"

@dataclass
class PromptTemplate:
    """The actual prompt with variable interpolation."""
    system: str
    user: str
    variables: List[str]
    few_shot_examples: Optional[List[Dict[str, str]]] = None
    
    def render(self, **kwargs) -> Dict[str, str]:
        """Render template with provided variables."""
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        
        return {
            "system": self.system.format(**kwargs),
            "user": self.user.format(**kwargs)
        }

@dataclass
class EvaluationCriteria:
    """How to measure template success."""
    success_metrics: List[str]  # ["accuracy", "latency", "cost"]
    validation_function: str  # Name of validator function
    test_cases: List[Dict[str, Any]]
    minimum_thresholds: Dict[str, float]  # {"accuracy": 0.95}

@dataclass
class TemplateMetadata:
    """Documentation and usage guidance."""
    name: str
    version: str
    category: TemplateCategory
    description: str
    use_cases: List[str]
    limitations: List[str]
    cost_estimate: str  # "~$0.001 per call"
    author: str
    last_updated: str

@dataclass
class SolutionTemplateDefinition:
    """Complete template specification."""
    metadata: TemplateMetadata
    prompt: PromptTemplate
    config: TemplateConfig
    evaluation: EvaluationCriteria
    dependencies: List[str] = field(default_factory=list)
    
    def to_json(self) -> str:
        """Serialize for storage."""
        # Implementation would handle enum conversion, etc.
        pass
```

**Practical Implications:**

This structure enforces completeness. A template without evaluation criteria is just a code snippet. One without test cases can't be validated. The schema makes quality non-negotiable.

**Trade-offs:**

Rich structure means higher initial effort (20-30 minutes vs. 5 to create) but prevents the "quick hack that becomes production code" problem. You're forced to think about validation upfront.

**Example:**
```python
email_extraction = SolutionTemplateDefinition(
    metadata=TemplateMetadata(
        name="email_entity_extractor",
        version="2.1.0",
        category=TemplateCategory.EXTRACTION,
        description="Extract structured entities from customer emails",
        use_cases=["support ticket routing", "CRM enrichment"],
        limitations=["Requires English text", "Max 2000 tokens"],
        cost_estimate="~$0.002 per email",
        author="platform-team",
        last_updated="2024-01-15"
    ),
    prompt=PromptTemplate(
        system="Extract entities from customer emails. Return valid JSON only.",
        user="Email: {email_text}\n\nExtract: customer_name, issue_type, product_mentioned, urgency_level",
        variables=["email_text"],
        few_shot_examples=[
            {
                "input": "Hi, my Premium account login isn't working. Please help ASAP!",
                "output": '{"customer_name": null, "issue_type": "login", "product_mentioned": "Premium", "urgency_level": "high"}'
            }
        ]
    ),
    config={
        "model": "claude-sonnet-4",
        "temperature": 0.0,
        "max_tokens": 200
    },
    evaluation=EvaluationCriteria(
        success_metrics=["json_validity", "field_accuracy", "latency"],
        validation_function="validate_email_entities",
        test_cases=[],  # Would contain 20+ real examples
        minimum_thresholds={"json_validity": 0.99, "field_accuracy": 0.90}
    )
)
```

### 2. Template Registry and Discovery

Templates are useless if engineers can't find them. The registry provides searchable, versioned access.

**Implementation:**
```python
from typing import Optional, List
import sqlite3
from datetime import datetime

class TemplateRegistry:
    """Persistent storage and discovery for templates."""
    
    def __init__(self, db_path: str = "templates.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create schema if not exists."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS templates (
                name TEXT,
                version TEXT,
                category TEXT,
                definition TEXT,
                created_at TEXT,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                PRIMARY KEY (name, version)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON templates(category)
        """)
        conn.commit()
        conn.close()
    
    def register(self, template: SolutionTemplateDefinition) -> None:
        """Add or update template."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO templates 
            (name, version, category, definition, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            template.metadata.name,
            template.metadata.version,
            template.metadata.category.value,
            template.to_json(),
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def find(
        self, 
        category: Optional[TemplateCategory] = None,
        min_success_rate: float = 0.0,
        search_term: Optional[str] = None
    ) -> List[SolutionTemplateDefinition]:
        """Search templates with filters."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT definition FROM templates WHERE success_rate >= ?"
        params = [min_success_rate]
        
        if category:
            query += " AND category = ?"
            params.append(category.value)
        
        if search_term:
            query += " AND (name LIKE ? OR definition LIKE ?)"
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        query += " ORDER BY success_rate DESC, usage_count DESC"
        
        cursor = conn.execute(query, params)
        results = [
            SolutionTemplateDefinition.from_json(row[0]) 
            for row in cursor.fetchall()
        ]
        conn.close()
        
        return results
    
    def record_usage(
        self, 
        name: str, 
        version: str, 
        success: bool
    ) -> None:
        """Update template metrics."""
        conn = sqlite3.connect(self.db_path)
        
        # Atomic update of running average
        conn.execute("""
            UPDATE templates 
            SET 
                usage_count = usage_count + 1,
                success_rate = (
                    (success_rate * usage_count + ?) / (usage_count + 1)
                )
            WHERE name = ? AND version = ?
        """, (1.0 if success else 0.0, name, version))
        
        conn.commit()
        conn.close()
```

**Practical Implications:**

The registry transforms templates from files into data. You can now ask: "Show me all extraction templates with >95% success rate" or "What's the most-used classification template?" This enables data-driven template selection.

**Constraints:**

Local SQLite works for teams up to ~50 engineers. Beyond that, consider PostgreSQL for concurrent access. The schema is intentionally simple—premature optimization (like full-text search) adds complexity for minimal gain early on.

### 3. Template Execution Engine

Templates need consistent execution with observability and error handling.

```python
from typing import Any, Optional, Dict
import time
import logging
from contextlib import contextmanager

class TemplateExecutor:
    """Executes templates with monitoring and error handling."""
    
    def __init__(
        self, 
        registry: TemplateRegistry,
        llm_client: Any  # Your LLM client
    ):
        self.registry = registry
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def _monitor_execution(self, template_name: str, template_version: str):
        """Track execution metrics."""
        start_time = time.time()
        success = False
        
        try:
            yield
            success = True
        finally:
            duration = time.time() - start_time
            self.registry.record_usage(template_name, template_version, success)
            
            self.logger.info(
                f"Template: {template_name} v{template_version} | "
                f"Success: {success} | Duration: {duration:.2f}s"
            )
    
    def execute(
        self,
        template: SolutionTemplateDefinition,
        variables: Dict[str, Any],
        validate: bool = True
    ) -> Any:
        """Execute template with full lifecycle management."""
        
        with self._monitor_execution(
            template.metadata.name, 
            template.metadata.version
        ):
            # 1. Render prompt
            rendered = template.prompt.render(**variables)
            
            # 2. Call LLM
            response = self.llm_client.generate(
                system=rendered["system"],
                user=rendered["user"],
                **template.config
            )
            
            # 3. Parse output (template-specific logic)
            parsed = self._parse_response(response, template)
            
            # 4. Validate if requested
            if validate:
                if not self._validate(parsed, template):
                    raise ValueError(
                        f"Output failed validation: {parsed}"
                    )
            
            return parsed
    
    def _parse_response(
        self, 
        response: str, 
        template: SolutionTemplateDefinition
    ) -> Any:
        """Apply template-specific parsing."""
        # For structured outputs
        if template.metadata.category == TemplateCategory.EXTRACTION:
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                    return json.loads(json_str)
                raise
        
        # For classifications
        elif template.metadata.category == TemplateCategory.CLASSIFICATION:
            return response.strip().lower()
        
        # Default: return raw
        return response
    
    def _validate(
        self, 
        output: Any, 
        template: SolutionTemplateDefinition
    ) -> bool:
        """Run template's validation logic."""
        validator_name = template.evaluation.validation_function
        
        # In production, load validator from registry
        # For now, basic validation
        if template.metadata.category == TemplateCategory.EXTRACTION:
            return isinstance(output, dict)
        elif template.metadata.category == TemplateCategory.CLASSIFICATION:
            return isinstance(output, str) and len(output) > 0
        
        return True
```

**Trade-offs:**

Execution engines add latency (typically <10ms) but provide critical observability. Without monitoring, you can't tell if a template's success rate is dropping. The cost is negligible compared to LLM calls (100-500ms).

### 4. Template Versioning Strategy

Templates evolve. Version management prevents breaking changes.

```python
from dataclasses import dataclass
from typing import List, Optional
import semver

@dataclass
class TemplateVersion:
    """Semantic versioning for templates."""