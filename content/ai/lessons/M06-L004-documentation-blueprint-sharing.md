# Documentation & Blueprint Sharing: Engineering Effective AI Context

## Core Concepts

When you interact with an LLM, you're not just writing a query—you're providing a temporary knowledge base that exists only for the duration of that conversation. Unlike traditional software where documentation lives in wikis, READMEs, or knowledge bases that persist across sessions, LLM interactions are stateless. Each conversation starts from zero.

**The Engineering Shift:**

```python
# Traditional software development
class DatabaseConnection:
    """
    Developer writes code once, documentation persists in:
    - Code comments
    - API documentation
    - Wiki pages
    - Shared drive specifications
    
    These are referenced indefinitely by the team.
    """
    def connect(self, host: str, port: int) -> None:
        # Implementation details available to all developers
        pass

# LLM-based development
def query_llm_for_database_logic(context: str, question: str) -> str:
    """
    Every interaction requires providing:
    - System architecture
    - Coding standards
    - Business logic
    - Technical constraints
    
    If you don't include it, the LLM doesn't know it exists.
    """
    full_prompt = f"{context}\n\nQuestion: {question}"
    return llm.generate(full_prompt)
```

The fundamental difference: **documentation in LLM workflows must be actively injected into every interaction**, not passively referenced. This changes how you structure, store, and share technical knowledge.

**Why This Matters Now:**

1. **Context is expensive**: With typical context windows of 8K-128K tokens, every token costs money and latency. Poor documentation structure wastes both.

2. **Reproducibility is hard**: Unlike traditional code where the same input + state = same output, LLMs have temperature and sampling. Good documentation patterns are critical for consistent results.

3. **Knowledge compounds rapidly**: In a six-month project, you'll generate hundreds of prompts, specifications, and examples. Without systematic documentation, you'll rebuild the same context repeatedly.

4. **Team scale multiplies waste**: If three engineers each spend 10 minutes reconstructing project context daily, that's 37.5 hours/month lost to redundant work.

The engineers who master documentation patterns for LLM workflows achieve 3-5x faster iteration cycles and dramatically lower API costs.

## Technical Components

### 1. Context Documents: The Source-of-Truth Blueprint

A context document is a structured specification that serves as the "persistent memory" for your LLM interactions. Think of it as a technical specification document, but optimized for LLM parsing.

**Structure Pattern:**

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class ContextDocument:
    """
    Structured context for LLM interactions.
    Each section serves a specific purpose in guiding LLM behavior.
    """
    
    # High-level system purpose - frames all subsequent responses
    project_overview: str
    
    # Technical constraints that must never be violated
    hard_requirements: List[str]
    
    # Architectural decisions with rationale
    architecture_decisions: Dict[str, str]
    
    # Code examples showing style and patterns
    code_examples: List[str]
    
    # Known issues or edge cases to avoid
    anti_patterns: List[str]
    
    # Specific terminology or domain language
    glossary: Dict[str, str]

    def to_prompt(self, max_tokens: int = 4000) -> str:
        """
        Convert to prompt string with token budget awareness.
        Prioritizes critical information if token limit is approached.
        """
        sections = []
        
        # Priority 1: Project overview and hard requirements
        sections.append(f"# Project Context\n{self.project_overview}\n")
        sections.append("## Hard Requirements\n" + 
                       "\n".join(f"- {req}" for req in self.hard_requirements))
        
        # Priority 2: Code examples (show, don't just tell)
        if self.code_examples:
            sections.append("## Code Style Examples\n" + 
                           "\n\n".join(self.code_examples))
        
        # Priority 3: Architecture and glossary
        if self.architecture_decisions:
            arch_text = "\n".join(
                f"**{k}**: {v}" 
                for k, v in self.architecture_decisions.items()
            )
            sections.append(f"## Architecture Decisions\n{arch_text}")
        
        # Priority 4: Anti-patterns (if space remains)
        if self.anti_patterns:
            sections.append("## Avoid These Patterns\n" + 
                           "\n".join(f"- {ap}" for ap in self.anti_patterns))
        
        full_context = "\n\n".join(sections)
        
        # Rough token estimation (1 token ≈ 4 characters)
        estimated_tokens = len(full_context) // 4
        
        if estimated_tokens > max_tokens:
            # Truncate lower-priority sections
            return "\n\n".join(sections[:3])
        
        return full_context


# Example usage for a microservices API project
api_context = ContextDocument(
    project_overview="""
    Building a rate-limited API gateway for internal microservices.
    Python 3.11, FastAPI, Redis for rate limiting, PostgreSQL for audit logs.
    Must handle 10K requests/second with <50ms p99 latency.
    """,
    
    hard_requirements=[
        "All endpoints must have rate limiting (per-user and per-IP)",
        "Authentication via JWT tokens only, no API keys",
        "All database queries must use connection pooling",
        "Error responses must never expose internal service names",
    ],
    
    architecture_decisions={
        "Rate limiting": "Redis sorted sets for sliding window (chosen over fixed window for smoother experience)",
        "Service discovery": "Direct service URLs in config (not service mesh - kept simple for this scale)",
        "Logging": "Structured JSON to stdout (ingested by external log aggregator)",
    },
    
    code_examples=[
        """
# Correct error handling pattern
from fastapi import HTTPException, status

async def get_user(user_id: int) -> User:
    try:
        return await db.fetch_user(user_id)
    except UserNotFound:
        # External message - no internal details
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found"
        )
    except DatabaseError as e:
        # Log internally but return generic message
        logger.error(f"DB error fetching user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal service error"
        )
        """,
    ],
    
    anti_patterns=[
        "DO NOT use global variables for database connections",
        "DO NOT return raw database errors to clients",
        "DO NOT implement rate limiting in application code (use Redis)",
    ],
    
    glossary={
        "upstream service": "Internal microservice behind the gateway",
        "tenant": "Customer organization (multi-tenant system)",
        "rate bucket": "Redis sorted set tracking request timestamps per user",
    }
)

# Generate prompt for a new task
task_prompt = f"""{api_context.to_prompt()}

## Current Task
Implement a new endpoint GET /api/v1/analytics/summary that aggregates
data from three upstream services. Include proper error handling and rate limiting.
"""
```

**Practical Implications:**

- **Token efficiency**: A well-structured context document prevents redundant explanations. You reference it once rather than re-explaining architecture in each prompt.
- **Consistency**: The LLM generates code matching your established patterns because examples are always present.
- **Onboarding speed**: New team members (human or AI) understand the system by reading one document.

**Trade-offs:**

- **Maintenance burden**: Context documents become outdated. You need version control and update discipline.
- **Context bloat**: As projects grow, context documents expand. You'll need strategies to summarize or partition them.
- **Over-specification risk**: Too much context can anchor the LLM to existing patterns, making it harder to get creative solutions.

### 2. Modular Context: Composable Knowledge Blocks

Rather than one monolithic context document, advanced workflows use modular context blocks that compose based on the task.

```python
from typing import Protocol
from enum import Enum

class ContextModule(Protocol):
    """Interface for composable context blocks."""
    
    def render(self) -> str:
        """Return the context text for this module."""
        ...
    
    def estimated_tokens(self) -> int:
        """Estimate token count for budget planning."""
        ...


class SecurityContext:
    """Security-related context - included for endpoints handling auth/data."""
    
    def render(self) -> str:
        return """
## Security Requirements
- Input validation: Use Pydantic models for all request bodies
- SQL injection prevention: Use parameterized queries only (SQLAlchemy ORM preferred)
- JWT validation: Verify signature, expiration, and required claims (user_id, tenant_id)
- Rate limiting: Apply per-endpoint limits defined in rate_limits.yaml

## Authentication Flow
```python
from fastapi import Depends, HTTPException
from jose import JWTError, jwt

async def verify_token(token: str = Depends(oauth2_scheme)) -> Dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        if payload.get("exp", 0) < time.time():
            raise HTTPException(status_code=401, detail="Token expired")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```
"""
    
    def estimated_tokens(self) -> int:
        return len(self.render()) // 4


class DatabaseContext:
    """Database patterns - included for data access tasks."""
    
    def render(self) -> str:
        return """
## Database Patterns
- Connection pooling: Use SQLAlchemy engine with pool_size=20, max_overflow=10
- Transaction management: Use async context managers
- Query optimization: Always use indexes, explain plan queries >100ms

## Example Data Access Pattern
```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

async def get_user_transactions(
    session: AsyncSession, 
    user_id: int, 
    limit: int = 100
) -> List[Transaction]:
    stmt = (
        select(Transaction)
        .where(Transaction.user_id == user_id)
        .order_by(Transaction.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return result.scalars().all()
```
"""
    
    def estimated_tokens(self) -> int:
        return len(self.render()) // 4


class TaskType(Enum):
    NEW_ENDPOINT = "new_endpoint"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    DATABASE_QUERY = "database_query"


class ContextBuilder:
    """Compose context based on task type."""
    
    def __init__(self):
        self.modules = {
            "security": SecurityContext(),
            "database": DatabaseContext(),
            # Add more modules as needed
        }
    
    def build_context(
        self, 
        task_type: TaskType, 
        token_budget: int = 8000
    ) -> str:
        """Build context appropriate for task type within token budget."""
        
        # Define which modules are relevant for each task type
        task_modules = {
            TaskType.NEW_ENDPOINT: ["security", "database"],
            TaskType.BUG_FIX: ["security"],  # Lighter context for focused tasks
            TaskType.REFACTOR: [],  # Minimal context, focus on code itself
            TaskType.DATABASE_QUERY: ["database"],
        }
        
        selected_modules = [
            self.modules[name] 
            for name in task_modules.get(task_type, [])
        ]
        
        # Reserve tokens for base context and task description
        reserved_tokens = 1000
        available_tokens = token_budget - reserved_tokens
        
        # Build context with most critical modules first
        context_parts = []
        token_count = 0
        
        for module in selected_modules:
            module_tokens = module.estimated_tokens()
            if token_count + module_tokens <= available_tokens:
                context_parts.append(module.render())
                token_count += module_tokens
            else:
                # Token budget exceeded, skip remaining modules
                break
        
        return "\n\n".join(context_parts)


# Usage example
builder = ContextBuilder()

# For a new authenticated endpoint, include both security and database context
new_endpoint_context = builder.build_context(
    TaskType.NEW_ENDPOINT, 
    token_budget=6000
)

# For a focused bug fix, include only security context
bugfix_context = builder.build_context(
    TaskType.BUG_FIX,
    token_budget=3000
)
```

**Practical Implications:**

- **Cost optimization**: Pay only for relevant context. A simple bug fix doesn't need full system architecture.
- **Specialization**: Different engineers/tasks can use different context combinations without maintaining separate documents.
- **Dynamic scaling**: Automatically adjust context depth based on model's context window (GPT-4 vs. Claude vs. smaller models).

**Trade-offs:**

- **Complexity**: More moving parts mean more maintenance and potential for inconsistency between modules.
- **Missing context**: If module selection logic is wrong, the LLM lacks critical information and produces incorrect results.
- **Testing burden**: Need to validate that context combinations produce expected results for each task type.

### 3. Version Control & Evolution Tracking

Documentation is code. Treat it like code.

```python
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

@dataclass
class ContextVersion:
    """Versioned context document with change tracking."""
    
    version: str
    content: ContextDocument
    created_at: datetime
    change_summary: str
    author: str
    
    def save(self, directory: Path) -> None:
        """Save as JSON for version control."""
        filepath = directory / f"context_v{self.version}.json"
        
        data = {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "change_summary": self.change_summary,
            "author": self.author,
            "content": {
                "project_overview": self.content.project_overview,
                "hard_requirements": self.content.hard_requirements,
                "architecture_decisions": self.content.architecture_decisions,
                "code_examples": self.content.code_examples,
                "anti_patterns": self.content.anti_patterns,
                "glossary": self.content.glossary,
            }
        }
        
        filepath.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, filepath: Path) -> "ContextVersion":
        """Load from JSON."""
        data = json.loads(filepath.read_text())
        
        content = ContextDocument(
            project_overview=data["content"]["project_overview"],
            hard_requirements=data["content"]["hard_requirements"],
            architecture_decisions=data["content"]["architecture_decisions"],
            code_examples=data["content"]["code_examples"],
            anti_patterns=data["content"]["anti_patterns"],
            glossary=data["content"]["glossary"],
        )
        
        return cls(
            version=data["version"],
            content=content,
            created_at=datetime.fromisoformat(data["created_at"]),
            change_summary=data["change_summary"],
            author=data["author"],
        )


class ContextRepository:
    """Manage context document versions."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.repo_path.mkdir(exist_ok=True)
    
    def save_version(
        self, 
        content: ContextDocument, 
        change_summary: str,
        author: str
    ) -> ContextVersion:
        """Save new version with automatic versioning."""
        
        # Find latest version
        existing_versions = self.list_versions()
        if existing_versions:
            latest = existing_versions[-1]
            major, minor = map(int, latest.version.split("."))
            new_version =