# Version Control Integration for AI/LLM Systems

## Core Concepts

Version control for AI systems differs fundamentally from traditional software version control. While conventional applications version code that produces deterministic outputs, AI systems require versioning of multiple interdependent components—prompts, model configurations, training data lineage, evaluation datasets, and system outputs—where changes to any component can produce dramatically different behaviors.

**Traditional vs. AI Version Control:**

```python
# Traditional Software: Deterministic output from versioned code
# app.py - v1.2.3
def calculate_discount(price: float, customer_tier: str) -> float:
    """Same inputs always produce same outputs"""
    if customer_tier == "premium":
        return price * 0.15
    return price * 0.05

# Result: git commit captures everything needed to reproduce behavior

# ============================================================

# AI System: Non-deterministic outputs from multiple versioned components
# recommendation_system.py
import hashlib
from datetime import datetime
from typing import Dict, Any

def generate_recommendation(
    user_context: Dict[str, Any],
    prompt_template: str,  # Version: prompt_v2.3
    model_config: Dict,     # Version: config_20240115
    model_id: str,          # Version: gpt-4-turbo-2024-01
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Same inputs can produce different outputs - versioning requires tracking:
    - Prompt template version
    - Model version/ID
    - Configuration parameters
    - Input data characteristics
    - Output for reproducibility
    - Timestamp and execution context
    """
    
    # Generate content hash of all inputs for reproducibility tracking
    input_signature = hashlib.sha256(
        f"{prompt_template}{model_config}{model_id}{user_context}".encode()
    ).hexdigest()[:12]
    
    # In production: call LLM API
    response = call_llm(prompt_template, user_context, model_config)
    
    # Version control metadata embedded in response
    return {
        "recommendation": response,
        "version_metadata": {
            "prompt_version": "prompt_v2.3",
            "model_id": model_id,
            "config_version": "config_20240115",
            "input_signature": input_signature,
            "timestamp": datetime.utcnow().isoformat(),
            "temperature": temperature
        }
    }

# Result: Need to version multiple artifacts + capture non-deterministic outputs
```

**Key Engineering Insights:**

1. **Prompt-Code Coupling**: Prompts are executable logic, not configuration. A prompt change can have the same impact as a code refactor, but traditional diff tools provide minimal insight into behavioral changes.

2. **Reproducibility vs. Replicability**: You cannot reproduce identical outputs (non-deterministic), but you can replicate the conditions that produced an output (deterministic inputs). Version control must capture all conditions.

3. **Evaluation as Testing**: Traditional CI/CD runs deterministic tests. AI systems require versioned evaluation datasets and acceptance criteria that acknowledge statistical variance.

4. **Multi-Artifact Dependencies**: A single "version" of an AI system comprises: application code, prompt templates, model identifiers, configuration parameters, and often evaluation datasets. Changes to any component necessitate re-validation.

**Why This Matters Now:**

Production AI systems fail silently. A prompt change that improves 80% of cases but catastrophically fails 5% won't trigger compiler errors or obvious test failures. Rigorous version control is the only defense against undetected regressions, enabling:

- Rollback to last-known-good configurations when quality degrades
- A/B testing with precise control groups
- Audit trails for compliance and debugging
- Collaborative development without stepping on each other's prompts

## Technical Components

### 1. Prompt Versioning Strategy

**Technical Explanation:**

Prompts must be version-controlled as first-class code artifacts, not configuration strings. This means structured storage (separate files), semantic versioning, and explicit dependency declarations.

**Implementation Pattern:**

```python
# prompts/customer_support/escalation_v2_1_0.py
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class PromptVersion:
    """Structured prompt with metadata"""
    version: str
    template: str
    required_variables: list[str]
    model_constraints: Dict[str, Any]
    changelog: str

ESCALATION_PROMPT = PromptVersion(
    version="2.1.0",
    template="""You are a customer support specialist. Analyze this conversation and determine if escalation is needed.

Conversation history:
{conversation_history}

Customer sentiment: {sentiment_score}
Issue category: {issue_category}

Respond with JSON:
{
  "escalate": boolean,
  "reasoning": string,
  "suggested_action": string
}""",
    required_variables=["conversation_history", "sentiment_score", "issue_category"],
    model_constraints={
        "max_tokens": 500,
        "temperature": 0.3,
        "response_format": "json"
    },
    changelog="""
    2.1.0: Added sentiment_score parameter for better accuracy
    2.0.0: Changed to JSON output format (breaking change)
    1.2.1: Fixed issue with multi-turn conversations
    """
)

def load_prompt(version: str = "latest") -> PromptVersion:
    """Load specific prompt version - enables A/B testing and rollback"""
    if version == "latest":
        return ESCALATION_PROMPT
    # In production: load from version control or database
    raise ValueError(f"Version {version} not found")
```

**Practical Implications:**

- **Searchability**: Prompts in separate files enable full-text search across all versions
- **Code Review**: Changes appear in pull requests with full context
- **Testing**: Each prompt version can have dedicated test suites

**Real Constraints:**

- **Overhead**: More files to manage than inline strings
- **Migration**: Legacy systems with hardcoded prompts require refactoring
- **Trade-off**: Accept structural overhead for operational safety

### 2. Configuration and Model Tracking

**Technical Explanation:**

Model parameters (temperature, max_tokens, top_p) and model identifiers dramatically affect outputs. These must be versioned alongside prompts with explicit defaults and validation.

**Implementation Pattern:**

```python
# config/model_configs.py
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import json
from pathlib import Path

class ModelConfig(BaseModel):
    """Type-safe model configuration with validation"""
    config_id: str = Field(..., description="Unique configuration identifier")
    model_id: str = Field(..., description="Model version (e.g., gpt-4-turbo-2024-04-09)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0, le=8000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    
    @validator('temperature')
    def validate_temperature(cls, v, values):
        """Warn about common misconfiguration"""
        if v > 1.5:
            print(f"Warning: High temperature ({v}) may produce inconsistent outputs")
        return v
    
    def to_file(self, path: Path) -> None:
        """Persist configuration for version control"""
        path.write_text(json.dumps(self.dict(), indent=2))
    
    @classmethod
    def from_file(cls, path: Path) -> 'ModelConfig':
        """Load versioned configuration"""
        return cls(**json.loads(path.read_text()))

# Example configurations
PRODUCTION_CONFIG = ModelConfig(
    config_id="prod_v1.3",
    model_id="gpt-4-turbo-2024-04-09",
    temperature=0.3,  # Low temperature for consistency
    max_tokens=2000
)

CREATIVE_CONFIG = ModelConfig(
    config_id="creative_v1.0",
    model_id="gpt-4-turbo-2024-04-09",
    temperature=1.2,  # Higher temperature for variety
    max_tokens=3000
)

# Save to version control
# PRODUCTION_CONFIG.to_file(Path("config/prod_v1.3.json"))
```

**Practical Implications:**

- **Type Safety**: Pydantic catches configuration errors before API calls
- **Auditability**: Every request can be traced to specific configuration version
- **Experimentation**: Easy to create and compare configuration variants

**Real Constraints:**

- **Boilerplate**: Requires more setup than dictionary configurations
- **Learning Curve**: Team needs to understand validation framework
- **Trade-off**: Accept initial complexity for runtime safety

### 3. Execution Logging and Reproducibility

**Technical Explanation:**

Since LLM outputs are non-deterministic, capturing actual execution results is essential. Each invocation should log inputs, configuration, outputs, and metadata for debugging and auditing.

**Implementation Pattern:**

```python
# utils/execution_logger.py
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import uuid

@dataclass
class ExecutionLog:
    """Complete record of LLM invocation"""
    execution_id: str
    timestamp: str
    prompt_version: str
    config_id: str
    model_id: str
    input_hash: str  # SHA256 of input for deduplication
    input_data: Dict[str, Any]
    output: str
    latency_ms: float
    tokens_used: Optional[int]
    error: Optional[str] = None
    
    def save(self, log_dir: Path) -> None:
        """Persist execution log for version control and debugging"""
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.execution_id}.json"
        log_file.write_text(json.dumps(asdict(self), indent=2))

class VersionedExecutor:
    """Execute LLM calls with full versioning and logging"""
    
    def __init__(self, log_dir: Path = Path("logs/executions")):
        self.log_dir = log_dir
    
    def execute(
        self,
        prompt_version: str,
        config: ModelConfig,
        input_data: Dict[str, Any],
        llm_client: Any  # Your LLM client
    ) -> ExecutionLog:
        """Execute with complete tracking"""
        
        execution_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat()
        
        # Hash inputs for deduplication
        input_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Execute LLM call
        start_time = datetime.utcnow()
        try:
            # In production: actual API call
            # response = llm_client.generate(
            #     prompt=format_prompt(prompt_version, input_data),
            #     **config.dict()
            # )
            # Simulated for example:
            output = f"[Simulated response for {input_data}]"
            tokens_used = 150
            error = None
        except Exception as e:
            output = ""
            tokens_used = None
            error = str(e)
        
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Create execution log
        log = ExecutionLog(
            execution_id=execution_id,
            timestamp=timestamp,
            prompt_version=prompt_version,
            config_id=config.config_id,
            model_id=config.model_id,
            input_hash=input_hash,
            input_data=input_data,
            output=output,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            error=error
        )
        
        log.save(self.log_dir)
        return log

# Usage
executor = VersionedExecutor()
log = executor.execute(
    prompt_version="escalation_v2.1.0",
    config=PRODUCTION_CONFIG,
    input_data={"conversation_history": "...", "sentiment_score": -0.3},
    llm_client=None  # Your client
)
print(f"Execution {log.execution_id} logged: {log.latency_ms:.2f}ms")
```

**Practical Implications:**

- **Debugging**: Exact reproduction of issues with full input/output context
- **Performance Tracking**: Latency trends across versions
- **Cost Management**: Token usage per configuration

**Real Constraints:**

- **Storage**: Logs accumulate quickly (plan retention policies)
- **PII**: Input/output may contain sensitive data (implement scrubbing)
- **Trade-off**: Storage costs vs. debugging capability

### 4. Evaluation Dataset Versioning

**Technical Explanation:**

Evaluation datasets are test suites for AI systems. They must be versioned separately from code because dataset updates require full re-evaluation of existing models/prompts.

**Implementation Pattern:**

```python
# evaluation/dataset_manager.py
from typing import List, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime

@dataclass
class EvaluationCase:
    """Single test case with expected behavior"""
    case_id: str
    input_data: Dict[str, Any]
    expected_output: Optional[Any]  # May be None for human eval
    acceptance_criteria: str  # Description of success condition
    category: str  # For segmented analysis

@dataclass
class EvaluationDataset:
    """Versioned collection of test cases"""
    dataset_id: str
    version: str
    created_at: str
    description: str
    cases: List[EvaluationCase]
    
    def save(self, path: Path) -> None:
        """Persist dataset for version control"""
        data = {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "created_at": self.created_at,
            "description": self.description,
            "cases": [
                {
                    "case_id": c.case_id,
                    "input_data": c.input_data,
                    "expected_output": c.expected_output,
                    "acceptance_criteria": c.acceptance_criteria,
                    "category": c.category
                }
                for c in self.cases
            ]
        }
        path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def from_file(cls, path: Path) -> 'EvaluationDataset':
        """Load versioned dataset"""
        data = json.loads(path.read_text())
        cases = [EvaluationCase(**c) for c in data["cases"]]
        return cls(
            dataset_id=data["dataset_id"],
            version=data["version"],
            created_at=data["created_at"],
            description=data["description"],
            cases=cases
        )

# Create evaluation dataset
escalation_eval = EvaluationDataset(
    dataset_id="escalation_eval",
    version="1.2.0",
    created_at=datetime.utcnow().isoformat(),
    description="Test cases for escalation detection system",
    cases=[
        EvaluationCase(
            case_id="clear_escalation_001",
            input_data={
                "conversation_history": "I've been waiting 3 weeks! This is unacceptable!",
                "sentiment_score": -0.8,
                "issue_category": "delivery_delay"
            },
            expected_output={"escalate": True},
            acceptance_criteria="Should escalate for high-negative sentiment + delay",
            category="clear_positive"
        ),
        EvaluationCase(
            case_id="edge_case_002",
            input_data={
                "conversation_history": "I'm a bit disappointed but I understand delays happen.",
                "sentiment_score": -0.2,
                "issue_category": "delivery_delay"
            },