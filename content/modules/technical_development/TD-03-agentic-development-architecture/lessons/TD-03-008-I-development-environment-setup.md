# Development Environment Setup for LLM Development

## Core Concepts

### Technical Definition

A development environment for LLM applications is a configured workspace containing API clients, dependency management tools, credential handling systems, and testing frameworks optimized for iterative experimentation with language model APIs. Unlike traditional software development environments that primarily manage compilation toolchains and debuggers, LLM development environments prioritize API version control, token usage tracking, prompt versioning, and response caching to manage the non-deterministic and cost-per-request nature of language model interactions.

### Engineering Analogy: Traditional vs. LLM Development

**Traditional API Development:**
```python
# Traditional REST API client setup
import requests
from typing import Dict, Any

class TraditionalAPIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def get_data(self, endpoint: str) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/{endpoint}")
        response.raise_for_status()
        return response.json()

# Deterministic, cacheable, version-stable
client = TraditionalAPIClient("https://api.example.com", "key123")
result = client.get_data("users/42")  # Same input = same output
```

**LLM API Development:**
```python
# LLM API client with environment-specific requirements
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

class LLMClient:
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        log_requests: bool = True
    ):
        # Environment-aware credential loading
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("API key required: set LLM_API_KEY or pass api_key")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.log_requests = log_requests
        
        # Track costs and usage
        self.request_log: List[Dict[str, Any]] = []
        self.total_tokens = 0
        
    def generate(
        self, 
        prompt: str,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Non-deterministic generation with tracking.
        Same prompt can yield different outputs.
        """
        request_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature or self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Simulate API call
        tokens_used = len(prompt.split()) + 100  # Simplified
        self.total_tokens += tokens_used
        
        if self.log_requests:
            self.request_log.append({
                **request_data,
                "tokens_used": tokens_used,
                "cost_estimate": tokens_used * 0.00003  # Example pricing
            })
        
        return {
            "response": f"Generated response for: {prompt[:50]}...",
            "tokens": tokens_used,
            "model": self.model
        }
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Critical for cost management in LLM development."""
        return {
            "total_requests": len(self.request_log),
            "total_tokens": self.total_tokens,
            "estimated_cost": self.total_tokens * 0.00003,
            "by_model": {self.model: self.total_tokens}
        }

# Non-deterministic, cost-per-call, requires logging
client = LLMClient(model="gpt-4", log_requests=True)
result1 = client.generate("Explain recursion")
result2 = client.generate("Explain recursion")  # May differ from result1
print(client.get_usage_summary())
```

The fundamental difference: traditional APIs are deterministic services where the same input produces the same output, while LLM APIs are probabilistic services where outputs vary, costs accumulate per request, and the development environment must support experimentation tracking, version management for prompts, and cost monitoring as first-class concerns.

### Key Insights

**1. Credential Security Changes from Binary to Continuous Risk**

Traditional development often treats credentials as either "secure" or "compromised." LLM API keys represent continuous financial risk—a leaked key means ongoing charges until revoked. Your environment must treat credential exposure with the same severity as exposing a live payment processing token.

**2. Reproducibility Requires Explicit Configuration Capture**

Because LLM responses vary with model version, temperature, and other parameters, reproducibility demands capturing complete configuration state. Unlike traditional software where code + data = reproducible results, LLM applications require code + data + model parameters + API version + random seed (if available) for reproducibility.

**3. Local Development Must Account for Cloud Costs**

Traditional local development is "free" after setup. Every LLM API call during development costs money. Your environment needs cost tracking from day one, not as an afterthought. A single expensive debugging session can cost more than your entire traditional infrastructure.

### Why This Matters Now

The LLM ecosystem is in rapid flux. API providers change pricing, deprecate models, and modify behavior. An environment setup from six months ago may reference deprecated models, use outdated client libraries with breaking changes, or lack cost controls that weren't necessary when token prices were 10x higher. Current best practices emphasize:

- **Isolation**: Each project with independent dependencies to avoid version conflicts
- **Observability**: Built-in logging and cost tracking from first API call
- **Flexibility**: Easy switching between providers as pricing and capabilities evolve
- **Safety**: Guardrails preventing accidental expensive operations during development

## Technical Components

### 1. Virtual Environment and Dependency Isolation

**Technical Explanation:**

Python virtual environments create isolated Python installations, preventing dependency conflicts when different LLM libraries require incompatible versions of shared dependencies. LLM libraries update frequently—often monthly—making dependency isolation critical.

**Practical Implementation:**

```python
# setup_project.py - Automated environment setup
import subprocess
import sys
from pathlib import Path
from typing import List

def create_llm_environment(project_name: str, python_version: str = "3.11") -> None:
    """
    Create isolated environment for LLM development.
    
    Args:
        project_name: Project directory name
        python_version: Python version to use (3.10+ recommended for type hints)
    """
    project_path = Path(project_name)
    project_path.mkdir(exist_ok=True)
    
    # Create virtual environment
    venv_path = project_path / "venv"
    subprocess.run([
        sys.executable, "-m", "venv", str(venv_path)
    ], check=True)
    
    # Determine pip path based on OS
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    # Core LLM development dependencies
    dependencies = [
        "openai>=1.0.0",  # OpenAI API client
        "anthropic>=0.18.0",  # Anthropic Claude client
        "python-dotenv>=1.0.0",  # Environment variable management
        "tiktoken>=0.6.0",  # Token counting
        "requests>=2.31.0",  # HTTP client
        "tenacity>=8.2.0",  # Retry logic
        "pytest>=7.4.0",  # Testing framework
        "pytest-asyncio>=0.23.0",  # Async testing
    ]
    
    # Install dependencies
    subprocess.run([
        str(pip_path), "install", *dependencies
    ], check=True)
    
    # Create requirements.txt for version locking
    requirements_path = project_path / "requirements.txt"
    subprocess.run([
        str(pip_path), "freeze"
    ], stdout=open(requirements_path, "w"), check=True)
    
    # Create .gitignore
    gitignore_path = project_path / ".gitignore"
    gitignore_path.write_text("""
venv/
.env
__pycache__/
*.pyc
.pytest_cache/
logs/
*.log
""".strip())
    
    print(f"✓ Environment created at {project_path}")
    print(f"✓ Activate with: source {venv_path}/bin/activate")

if __name__ == "__main__":
    create_llm_environment("my_llm_project")
```

**Real Constraints:**

- Virtual environments add 50-200MB per project (trade-off for isolation)
- Activation required before each session (shell state, not system-wide)
- Cross-platform activation commands differ (Windows: `venv\Scripts\activate`, Unix: `source venv/bin/activate`)

**Concrete Trade-offs:**

Use virtual environments for: Multiple projects, production deployments, team collaboration.
Skip for: One-off experiments, Jupyter notebooks with kernel isolation, Docker-based development.

### 2. Secure Credential Management

**Technical Explanation:**

LLM API keys are bearer tokens—possession equals authorization. Environment variables provide process-level isolation, preventing credential hardcoding in source control while remaining accessible to application code. The `.env` file pattern separates configuration from code, enabling different credentials per environment (development, staging, production) without code changes.

**Practical Implementation:**

```python
# config.py - Centralized configuration management
import os
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class LLMConfig:
    """Type-safe configuration for LLM clients."""
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    timeout_seconds: int
    max_retries: int
    
    @classmethod
    def from_env(cls, provider: str = "openai") -> "LLMConfig":
        """
        Load configuration from environment variables.
        
        Raises:
            ValueError: If required environment variables missing
        """
        # Load .env file if present
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
        
        # Provider-specific environment variable names
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        
        api_key = os.getenv(env_vars.get(provider, "LLM_API_KEY"))
        if not api_key:
            raise ValueError(
                f"Missing API key. Set {env_vars.get(provider)} "
                f"in environment or .env file"
            )
        
        return cls(
            api_key=api_key,
            model=os.getenv("LLM_MODEL", "gpt-4"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT", "30")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3"))
        )
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Serialize config for logging.
        
        Args:
            include_secrets: If False, masks API key
        """
        return {
            "api_key": self.api_key if include_secrets else "***MASKED***",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries
        }


# Example .env file structure (not committed to git)
def create_env_template(output_path: Path = Path(".env.example")) -> None:
    """Create template .env file for team members."""
    template = """
# OpenAI Configuration
OPENAI_API_KEY=sk-your-key-here

# Anthropic Configuration  
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Model Configuration
LLM_MODEL=gpt-4
LLM_MAX_TOKENS=1000
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=30
LLM_MAX_RETRIES=3

# Development Settings
LOG_LEVEL=INFO
ENABLE_CACHING=true
""".strip()
    
    output_path.write_text(template)
    print(f"✓ Created {output_path}")
    print("✓ Copy to .env and add your actual API keys")


# Usage example
if __name__ == "__main__":
    try:
        config = LLMConfig.from_env("openai")
        print("Configuration loaded:")
        print(config.to_dict(include_secrets=False))
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Creating .env template...")
        create_env_template()
```

**Real Constraints:**

- Environment variables are process-scoped—child processes inherit but can't modify parent's environment
- `.env` files are plaintext—still vulnerable if disk access compromised
- Some platforms (cloud functions, containers) provide alternative secret management systems that should be preferred in production

**Concrete Example:**

Development cost: A leaked API key used for 24 hours before detection, running a chatbot with 10,000 queries averaging 500 tokens each = 5M tokens = $150-300 depending on model. Proper credential management prevents this scenario.

### 3. Request Logging and Cost Tracking

**Technical Explanation:**

LLM APIs charge per token processed. Without logging, development costs are invisible until the bill arrives. Request logging captures input/output tokens, model used, timestamp, and calculated cost, enabling real-time budget monitoring and post-hoc analysis of expensive operations.

**Practical Implementation:**

```python
# logger.py - Request logging with cost tracking
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import contextmanager

@dataclass
class LLMRequest:
    """Structured log entry for LLM API call."""
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    duration_seconds: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMRequestLogger:
    """
    Thread-safe logger for LLM API requests with cost aggregation.
    """
    
    # Pricing per 1K tokens (example rates, update frequently)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    }
    
    def __init__(self, log_dir: Path = Path("logs")):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"llm_requests_{date_str}.jsonl"
        
        self.requests: List[LLMRequest] = []
    
    def calculate_cost(
        self, 
        model: str, 
        prompt_tokens: int, 
        completion_tokens: int
    ) -> float:
        """
        Calculate cost in USD for request.
        