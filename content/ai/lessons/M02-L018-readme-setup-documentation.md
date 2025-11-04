# README & Setup Documentation for AI/LLM Projects

## Core Concepts

### Technical Definition

README and setup documentation for AI/LLM projects is the structured collection of configuration files, dependency specifications, environment setup instructions, and API key management that enables reproducible execution across different development and production environments. Unlike traditional software projects where dependencies are relatively stable, LLM projects introduce volatile model versions, API endpoint changes, token limit variations, and provider-specific authentication patterns that require explicit documentation and environment isolation.

### Engineering Analogy: Traditional vs. LLM Project Setup

**Traditional Web Application Setup:**

```python
# requirements.txt
flask==2.3.0
sqlalchemy==2.0.0
pytest==7.4.0

# .env (simple, stable credentials)
DATABASE_URL=postgresql://localhost/myapp
SECRET_KEY=your-secret-key

# README.md
## Setup
1. pip install -r requirements.txt
2. flask run
```

**LLM Application Setup:**

```python
# requirements.txt
openai==1.12.0  # API changes frequently, breaking changes common
anthropic==0.18.1  # Different SDK patterns than OpenAI
tiktoken==0.6.0  # For accurate token counting
python-dotenv==1.0.0
pydantic==2.6.0  # For response validation

# .env (multiple providers, complex credential management)
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
MODEL_NAME=gpt-4-turbo-preview  # Model names change, affect pricing
MAX_TOKENS=4096  # Critical for cost control
TEMPERATURE=0.7  # Affects reproducibility
FALLBACK_MODEL=gpt-3.5-turbo  # Redundancy planning

# config.py - Environment-aware configuration
from typing import Literal
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str | None = None
    model_name: str = "gpt-4-turbo-preview"
    max_tokens: int = 4096
    temperature: float = 0.7
    environment: Literal["development", "staging", "production"] = "development"
    
    # Cost controls
    max_requests_per_minute: int = 60
    daily_token_budget: int = 1_000_000
    
    class Config:
        env_file = ".env"
        extra = "forbid"  # Catch typos in .env files

settings = Settings()
```

The critical difference: LLM projects require **runtime configuration as code** because model behavior, costs, and availability change frequently. A model name isn't just a string—it's a contract specifying capabilities, pricing, context window size, and availability.

### Key Insights That Change Engineering Thinking

**1. API Keys Are Executable Spending Authority**

In traditional apps, leaked API keys might expose data. In LLM apps, leaked keys can burn thousands of dollars in hours through automated attacks or runaway loops. Documentation must treat credentials as financial instruments, not just authentication tokens.

**2. Model Versions Aren't Semantic Versions**

When you specify `flask==2.3.0`, you get predictable behavior. When you specify `gpt-4-turbo`, you might get different models over time as providers update them. Documentation must capture model behavior snapshots, not just names.

**3. Setup Documentation Is a Runtime Contract**

Your README isn't just onboarding—it's a specification of the operational constraints your code assumes. Token limits, rate limits, and model capabilities are implicit dependencies that must be explicit.

### Why This Matters NOW

As of 2024, the LLM ecosystem is in rapid flux:
- Major providers release new models monthly with different pricing and capabilities
- SDK breaking changes occur every 2-3 months across major libraries
- Regional availability and data residency requirements are evolving
- Context window sizes jumped from 4K to 128K+ in 18 months, enabling entirely new architectures

Projects without rigorous setup documentation become unmaintainable within weeks. Engineers waste hours debugging environment mismatches, rate limit errors, or unexpected costs that proper documentation would prevent.

## Technical Components

### 1. Dependency Specification with Version Pinning

**Technical Explanation:**

LLM SDKs evolve rapidly with breaking changes. The OpenAI SDK moved from v0.x to v1.x with a complete API redesign. Anthropic's SDK changed response structure multiple times. Pinning exact versions ensures reproducibility.

**Practical Implementation:**

```python
# requirements.txt - BAD (loose pinning)
openai
tiktoken
langchain

# requirements.txt - GOOD (exact pinning with justification)
openai==1.12.0  # v1.x has streaming API we depend on
anthropic==0.18.1  # Last version before response format change
tiktoken==0.6.0  # Matches OpenAI's internal tokenizer
python-dotenv==1.0.0
pydantic==2.6.0  # v2 required for discriminated unions

# requirements-dev.txt - Development tools
pytest==8.0.0
pytest-asyncio==0.23.0  # For async LLM call testing
black==24.1.0
mypy==1.8.0

# For reproducibility, generate lockfile
# pip freeze > requirements.lock
```

**Real Constraints:**

- **Trade-off**: Strict pinning improves reproducibility but delays security updates
- **Solution**: Document update schedule (e.g., "Review dependencies monthly, update quarterly unless CVE")
- **Constraint**: Different SDKs may require conflicting dependency versions (e.g., both want different `requests` versions)

**Concrete Example:**

```python
# version_check.py - Verify critical dependency versions at startup
import sys
from importlib.metadata import version

REQUIRED_VERSIONS = {
    "openai": "1.12.0",
    "tiktoken": "0.6.0",
}

def verify_dependencies():
    """Fail fast if dependency versions don't match requirements."""
    errors = []
    for package, required in REQUIRED_VERSIONS.items():
        actual = version(package)
        if actual != required:
            errors.append(f"{package}: expected {required}, got {actual}")
    
    if errors:
        print("ERROR: Dependency version mismatch:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    verify_dependencies()
```

### 2. Environment Variable Management

**Technical Explanation:**

LLM applications need multiple provider keys, model configurations, and runtime parameters. Environment variables separate secrets from code, but require structured validation to catch configuration errors before runtime.

**Practical Implementation:**

```python
# .env.example - Template committed to repo
OPENAI_API_KEY=sk-proj-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Model configuration
MODEL_NAME=gpt-4-turbo-preview
FALLBACK_MODEL=gpt-3.5-turbo
MAX_TOKENS=4096
TEMPERATURE=0.7

# Rate limiting
MAX_REQUESTS_PER_MINUTE=60
DAILY_TOKEN_BUDGET=1000000

# Environment
ENVIRONMENT=development

# .env - Actual secrets (in .gitignore)
# Copy from .env.example and fill in real values

# config.py - Validated configuration
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # Required secrets
    openai_api_key: str = Field(..., min_length=20)
    anthropic_api_key: str | None = Field(None, min_length=20)
    
    # Model configuration
    model_name: str = "gpt-4-turbo-preview"
    fallback_model: str = "gpt-3.5-turbo"
    max_tokens: int = Field(4096, ge=1, le=128000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    
    # Operational constraints
    max_requests_per_minute: int = Field(60, ge=1)
    daily_token_budget: int = Field(1_000_000, ge=1000)
    
    environment: Literal["development", "staging", "production"] = "development"
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
    
    @field_validator("model_name", "fallback_model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        valid_models = {
            "gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229"
        }
        if v not in valid_models:
            raise ValueError(f"Model {v} not in approved list: {valid_models}")
        return v
    
    class Config:
        env_file = ".env"
        extra = "forbid"  # Reject unknown environment variables

# Load and validate on import
settings = Settings()
```

**Real Constraints:**

- **Security**: Environment variables visible to all processes in same environment
- **Solution**: Use secret management for production (AWS Secrets Manager, HashiCorp Vault)
- **Trade-off**: Validation at startup vs. lazy loading (fail fast vs. partial functionality)

### 3. Model Configuration Documentation

**Technical Explanation:**

Model names encode implicit contracts about capabilities, pricing, context windows, and availability. Documentation must make these contracts explicit to prevent runtime surprises.

**Practical Implementation:**

```python
# models.py - Model capabilities as code
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ModelSpec:
    """Specification of model capabilities and constraints."""
    name: str
    provider: Literal["openai", "anthropic"]
    context_window: int  # tokens
    max_output_tokens: int
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD
    supports_streaming: bool
    supports_function_calling: bool
    region_restrictions: list[str]  # ISO country codes
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for token usage."""
        return (
            (input_tokens / 1000) * self.cost_per_1k_input +
            (output_tokens / 1000) * self.cost_per_1k_output
        )

# Model registry - Single source of truth
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gpt-4-turbo-preview": ModelSpec(
        name="gpt-4-turbo-preview",
        provider="openai",
        context_window=128_000,
        max_output_tokens=4_096,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        supports_streaming=True,
        supports_function_calling=True,
        region_restrictions=[]
    ),
    "gpt-3.5-turbo": ModelSpec(
        name="gpt-3.5-turbo",
        provider="openai",
        context_window=16_385,
        max_output_tokens=4_096,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        supports_streaming=True,
        supports_function_calling=True,
        region_restrictions=[]
    ),
    "claude-3-sonnet-20240229": ModelSpec(
        name="claude-3-sonnet-20240229",
        provider="anthropic",
        context_window=200_000,
        max_output_tokens=4_096,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supports_streaming=True,
        supports_function_calling=False,
        region_restrictions=[]
    )
}

def get_model_spec(model_name: str) -> ModelSpec:
    """Get model specification with validation."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]

# Usage in README
```

**README.md Model Configuration Section:**

```markdown
## Model Configuration

### Supported Models

| Model | Context | Cost (Input/Output per 1K) | Streaming | Function Calling |
|-------|---------|----------------------------|-----------|------------------|
| gpt-4-turbo-preview | 128K | $0.01/$0.03 | ✓ | ✓ |
| gpt-3.5-turbo | 16K | $0.0005/$0.0015 | ✓ | ✓ |
| claude-3-sonnet | 200K | $0.003/$0.015 | ✓ | ✗ |

### Model Selection Strategy

```python
# Default: GPT-4 Turbo for quality
MODEL_NAME=gpt-4-turbo-preview

# Cost-sensitive: GPT-3.5 Turbo (30x cheaper input)
MODEL_NAME=gpt-3.5-turbo

# Large context: Claude 3 Sonnet (200K context)
MODEL_NAME=claude-3-sonnet-20240229
```

**Real Constraints:**

- **Version Drift**: Providers update models behind stable names (e.g., "gpt-4" points to different models over time)
- **Solution**: Timestamp when ModelSpec was verified, document update process
- **Cost Visibility**: Without explicit documentation, engineers don't know they're burning 30x cost on unnecessary GPT-4 calls

### 4. Token Counting and Cost Estimation

**Technical Explanation:**

LLM costs scale with token usage, not request count. Accurate token counting prevents budget overruns and enables cost-based optimization decisions.

**Practical Implementation:**

```python
# token_utils.py
import tiktoken
from typing import Protocol

class TokenCounter(Protocol):
    """Protocol for token counting across different providers."""
    def count_tokens(self, text: str) -> int: ...

class OpenAITokenCounter:
    """Token counter for OpenAI models."""
    
    def __init__(self, model: str = "gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

class AnthropicTokenCounter:
    """Approximate token counter for Anthropic models."""
    
    def count_tokens(self, text: str) -> int:
        # Anthropic uses similar tokenization, ~4 chars per token
        return len(text) // 4

def get_token_counter(provider: str) -> TokenCounter:
    """Factory for provider-specific token counters."""
    if provider == "openai":
        return OpenAITokenCounter()
    elif provider == "anthropic":
        return AnthropicTokenCounter()
    else:
        raise ValueError(f"Unknown provider: {provider}")

# cost_tracker.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class UsageRecord:
    """Track individual LLM call costs."""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    request_id: Optional[str] = None

class CostTracker:
    """Track and limit LLM spending."""
    
    def __init__(self, daily_budget_usd: float):
        self.daily_budget = daily_budget_usd
        self.usage_history: list[UsageRecord] = []
    
    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> UsageRecord:
        """Record usage and calculate cost."""
        spec = get_model_spec(model)
        cost = spec.estimate_cost(input_tokens, output_tokens)
        
        record = UsageRecord(
            timestamp=datetime.