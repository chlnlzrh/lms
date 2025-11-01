# Project DNA Rollout Pattern

## Core Concepts

The **Project DNA Rollout Pattern** is a structured approach to encoding and propagating project-specific context, constraints, and conventions across AI-assisted development workflows. Rather than treating each LLM interaction as a stateless conversation, this pattern establishes a persistent, version-controlled "genetic code" for your project that shapes every AI interaction—from code generation to architecture decisions to documentation.

### Engineering Analogy: Configuration Drift vs. Infrastructure as Code

```python
# TRADITIONAL APPROACH: Ad-hoc LLM interactions
# Each developer maintains their own context in their head

def get_user_data(user_id):
    # Developer asks LLM: "Write a function to get user data"
    # LLM generates generic code, developer manually adjusts
    conn = connect_db()  # Which DB? What connection params?
    result = conn.query(f"SELECT * FROM users WHERE id = {user_id}")  # SQL injection!
    return result

# Each interaction starts from zero. No project conventions applied.
# Result: Inconsistent code, security issues,架构 drift
```

```python
# PROJECT DNA PATTERN: Context-aware, convention-enforced interactions
# DNA file establishes project genome that informs every interaction

"""
Project DNA (dna.md):
- Database: PostgreSQL with asyncpg, connection pooling via connection_manager
- Security: Always use parameterized queries, validate all inputs with pydantic
- Error handling: Raise custom exceptions from exceptions.py, log via structlog
- Code style: Return pydantic models, use type hints, async/await for I/O
"""

from typing import Optional
from pydantic import UUID4, ValidationError
from asyncpg import Pool
from structlog import get_logger

from .models import User
from .exceptions import UserNotFoundError
from .database import get_connection_pool

logger = get_logger()

async def get_user_data(
    user_id: UUID4,
    pool: Pool
) -> User:
    """
    Retrieve user data by ID.
    
    Args:
        user_id: UUID of the user to retrieve
        pool: asyncpg connection pool
        
    Returns:
        User model with validated data
        
    Raises:
        UserNotFoundError: If user doesn't exist
        ValidationError: If retrieved data fails validation
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE id = $1",
                user_id
            )
            
        if not row:
            logger.warning("user_not_found", user_id=str(user_id))
            raise UserNotFoundError(f"User {user_id} not found")
            
        return User(**dict(row))
        
    except ValidationError as e:
        logger.error("user_data_validation_failed", user_id=str(user_id), error=str(e))
        raise
```

The difference is architectural: instead of manually correcting every LLM output, the Project DNA pattern front-loads project context so the AI generates code that already conforms to your conventions, security requirements, and technical stack.

### Key Insights

1. **Context compounds**: Each time you manually correct AI-generated code, you're paying a "context tax." With 50 LLM interactions per day across a team of 10 developers, that's 500 manual corrections daily. Project DNA eliminates 70-90% of these.

2. **DNA is executable documentation**: Unlike traditional documentation that becomes stale, Project DNA is actively used by both humans and AI systems, creating natural incentive to keep it current.

3. **Propagation beats correction**: It's 10x more efficient to encode conventions once in DNA than to correct violations in every generated artifact.

### Why This Matters Now

As of 2024, most engineering teams use LLMs for code generation, but treat each interaction as isolated. This creates three critical problems:

- **Context re-entry overhead**: Developers spend 30-40% of their LLM interaction time re-explaining project constraints
- **Inconsistent outputs**: Different developers get different code styles, patterns, and architecture decisions from the same LLM
- **Knowledge fragmentation**: Project conventions live in tribal knowledge, Slack threads, and outdated wiki pages

The Project DNA pattern solves these by treating project context as a first-class artifact that's versioned, reviewed, and automatically injected into AI workflows.

## Technical Components

### 1. DNA Document Structure

The DNA document is a hierarchical, machine-parseable specification of project essentials.

```python
# dna_parser.py
from typing import Dict, List, Any
from pathlib import Path
import yaml
from dataclasses import dataclass

@dataclass
class TechStack:
    language: str
    version: str
    framework: str
    framework_version: str
    database: str
    orm: str
    
@dataclass
class SecurityPolicy:
    authentication: str
    authorization: str
    input_validation: str
    secrets_management: str
    
@dataclass
class CodeConventions:
    style_guide: str
    linting: List[str]
    type_checking: bool
    error_handling_pattern: str
    logging_framework: str
    
@dataclass
class ArchitecturePatterns:
    structure: str  # e.g., "layered", "hexagonal", "microservices"
    dependency_injection: bool
    async_style: str
    testing_strategy: str

class ProjectDNA:
    """Parser and accessor for project DNA configuration."""
    
    def __init__(self, dna_path: Path = Path("dna.yaml")):
        self.path = dna_path
        self._raw = self._load()
        
        self.tech_stack = TechStack(**self._raw["tech_stack"])
        self.security = SecurityPolicy(**self._raw["security"])
        self.conventions = CodeConventions(**self._raw["conventions"])
        self.architecture = ArchitecturePatterns(**self._raw["architecture"])
        
    def _load(self) -> Dict[str, Any]:
        """Load and validate DNA configuration."""
        if not self.path.exists():
            raise FileNotFoundError(f"DNA file not found: {self.path}")
            
        with open(self.path) as f:
            data = yaml.safe_load(f)
            
        required_sections = ["tech_stack", "security", "conventions", "architecture"]
        missing = [s for s in required_sections if s not in data]
        if missing:
            raise ValueError(f"DNA missing required sections: {missing}")
            
        return data
    
    def to_prompt_context(self) -> str:
        """Generate formatted context string for LLM prompts."""
        return f"""
# Project Technical Context

## Technology Stack
- Language: {self.tech_stack.language} {self.tech_stack.version}
- Framework: {self.tech_stack.framework} {self.tech_stack.framework_version}
- Database: {self.tech_stack.database}
- ORM: {self.tech_stack.orm}

## Security Requirements
- Authentication: {self.security.authentication}
- Authorization: {self.security.authorization}
- Input Validation: {self.security.input_validation}
- Secrets: {self.security.secrets_management}

## Code Conventions
- Style: {self.conventions.style_guide}
- Linting: {', '.join(self.conventions.linting)}
- Type Checking: {'Required' if self.conventions.type_checking else 'Optional'}
- Error Handling: {self.conventions.error_handling_pattern}
- Logging: {self.conventions.logging_framework}

## Architecture
- Pattern: {self.architecture.structure}
- DI: {'Yes' if self.architecture.dependency_injection else 'No'}
- Async: {self.architecture.async_style}
- Testing: {self.architecture.testing_strategy}
"""

# Example dna.yaml
dna_config = """
tech_stack:
  language: Python
  version: "3.11"
  framework: FastAPI
  framework_version: "0.104"
  database: PostgreSQL 15
  orm: SQLAlchemy 2.0

security:
  authentication: JWT with RS256
  authorization: RBAC via decorator pattern
  input_validation: Pydantic models with strict mode
  secrets_management: Environment variables via python-dotenv

conventions:
  style_guide: PEP 8 + Black formatter
  linting: [ruff, mypy]
  type_checking: true
  error_handling_pattern: Custom exception hierarchy
  logging_framework: structlog with JSON output

architecture:
  structure: layered
  dependency_injection: true
  async_style: async/await throughout
  testing_strategy: pytest with fixtures and mocks
"""
```

**Practical Implications**: This structure allows automated tooling to enforce conventions, generate boilerplate, and inject context without manual intervention.

**Constraints**: DNA must remain concise (< 500 lines) or it becomes noise. Focus on decisions that affect code generation, not exhaustive documentation.

### 2. Context Injection Mechanisms

DNA must be automatically injected into LLM interactions, not manually copy-pasted.

```python
# llm_wrapper.py
from typing import List, Dict, Optional
from anthropic import Anthropic
from openai import OpenAI

class DNAAwareLLMClient:
    """
    Wrapper for LLM clients that automatically injects Project DNA.
    """
    
    def __init__(
        self,
        provider: str,
        api_key: str,
        dna: ProjectDNA
    ):
        self.dna = dna
        
        if provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-sonnet-4-20250514"
        elif provider == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4-turbo"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        self.provider = provider
    
    def generate_code(
        self,
        instruction: str,
        context: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """
        Generate code with automatic DNA injection.
        
        Args:
            instruction: What code to generate
            context: Additional context (file contents, etc.)
            temperature: LLM temperature
            
        Returns:
            Generated code as string
        """
        # Build prompt with DNA context prepended
        system_prompt = f"""{self.dna.to_prompt_context()}

You are generating code for this project. Follow all conventions above strictly.
Generate production-ready code with proper error handling, type hints, and logging."""

        user_prompt = f"""{instruction}

{'Additional context:\n' + context if context else ''}

Generate complete, runnable code. Include all imports and proper structure."""

        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        else:  # OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content

# Usage
dna = ProjectDNA(Path("dna.yaml"))
llm = DNAAwareLLMClient(
    provider="anthropic",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    dna=dna
)

code = llm.generate_code(
    instruction="Create an API endpoint to update user profile",
    context=open("models/user.py").read()
)
```

**Practical Implications**: Every code generation automatically conforms to project standards without developer intervention. Eliminates "fix-up" phase.

**Trade-offs**: Adds ~200-500 tokens to every LLM call, increasing latency by ~0.5s and cost by ~5%. This is worthwhile given the time saved on manual corrections.

### 3. DNA Evolution & Versioning

DNA must evolve with the project while maintaining consistency.

```python
# dna_versioning.py
from datetime import datetime
from typing import Optional
from pathlib import Path
import hashlib
import json

class DNAVersion:
    """Track DNA changes over time."""
    
    def __init__(self, dna_path: Path = Path("dna.yaml")):
        self.dna_path = dna_path
        self.version_log_path = dna_path.parent / "dna_versions.jsonl"
        
    def compute_hash(self) -> str:
        """Compute hash of current DNA content."""
        content = self.dna_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:12]
    
    def record_change(
        self,
        author: str,
        reason: str,
        breaking: bool = False
    ) -> str:
        """
        Record DNA change with metadata.
        
        Args:
            author: Who made the change
            reason: Why the change was made
            breaking: Whether this is a breaking change
            
        Returns:
            Version hash
        """
        version_hash = self.compute_hash()
        
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": version_hash,
            "author": author,
            "reason": reason,
            "breaking": breaking
        }
        
        with open(self.version_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            
        return version_hash
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve DNA change history."""
        if not self.version_log_path.exists():
            return []
            
        with open(self.version_log_path) as f:
            records = [json.loads(line) for line in f]
            
        return records[-limit:]
    
    def validate_compatibility(
        self,
        old_hash: str,
        new_hash: str
    ) -> bool:
        """
        Check if DNA change is backward compatible.
        
        Breaking changes require regeneration of dependent code.
        """
        history = self.get_history(limit=100)
        
        # Find if there's a breaking change between versions
        in_range = False
        for record in reversed(history):
            if record["version"] == new_hash:
                in_range = True
            if in_range and record["breaking"]:
                return False
            if record["version"] == old_hash:
                break
                
        return True

# Integration with CI/CD
def dna_validation_hook():
    """
    Pre-commit hook to validate DNA changes.
    """
    versioner = DNAVersion()
    
    # Check if DNA was modified
    dna_modified = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True
    ).stdout.strip().split("\n")
    
    if "dna.yaml" in dna_modified:
        print("DNA modification detected. Validation required.")
        
        # Prompt for change metadata
        author = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True
        ).stdout.strip()
        
        reason = input("Reason for DNA change: ")
        breaking = input("Is this a breaking change? (y/n): ").lower() == "y"
        
        version = versioner.record_change(author, reason, breaking)
        
        if breaking:
            print(f"\n⚠️  BREAKING CHANGE: Version {version}")
            print("Dependent code may need regeneration.")
            print("Review and update generated code accordingly.")
            
        # Stage the version log
        subprocess.run(["git", "add", str(versioner.version_log_path)])
```

**Practical Implications**: DNA changes are tracked like schema migrations. Breaking changes trigger review and potential code regeneration.

**Constraints**: Teams must establish governance for DNA modifications. Recommend: DNA changes require PR review from senior engineers.

### 4. Multi-Context DNA Hierarchies

Complex projects need DNA specialization by domain or service.

```python
# dna_hierarchy.py
from typing import