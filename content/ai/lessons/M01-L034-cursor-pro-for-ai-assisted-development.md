# AI-Assisted Development: Engineering with LLM Code Editors

## Core Concepts

AI-assisted development tools fundamentally change the granularity of software engineering work. Instead of typing individual characters and lines, engineers now work at the level of intent—describing what code should do, and having an LLM generate implementation details.

### Traditional vs. AI-Assisted Development

```python
# Traditional approach: Manual implementation
# Time: ~15 minutes, cognitive load: high (syntax, edge cases, error handling)

def process_user_data(users: list[dict]) -> dict:
    """Manually typed, every character, every bracket."""
    result = {}
    for user in users:
        if user.get('active'):
            age = user.get('age', 0)
            if age >= 18:
                category = 'adult'
            else:
                category = 'minor'
            
            if category not in result:
                result[category] = []
            result[category].append({
                'name': user.get('name', 'Unknown'),
                'email': user.get('email', ''),
                'age': age
            })
    return result
```

```python
# AI-assisted approach: Intent-driven development
# Time: ~2 minutes, cognitive load: low (focus on logic, not syntax)

# Engineer writes comment or partial code:
# "Create a function that categorizes active users by adult/minor status"

# LLM generates complete implementation with better error handling:
from typing import TypedDict, Literal

class User(TypedDict):
    active: bool
    age: int
    name: str
    email: str

UserCategory = Literal['adult', 'minor']

def process_user_data(users: list[User]) -> dict[UserCategory, list[dict]]:
    """Categorize active users by age group."""
    result: dict[UserCategory, list[dict]] = {'adult': [], 'minor': []}
    
    for user in users:
        if not user.get('active', False):
            continue
            
        category: UserCategory = 'adult' if user.get('age', 0) >= 18 else 'minor'
        result[category].append({
            'name': user.get('name', 'Unknown'),
            'email': user.get('email', ''),
            'age': user.get('age', 0)
        })
    
    return result
```

The AI-assisted version took less time to produce and includes improvements the engineer might have missed: proper type hints, pre-initialized dictionary keys, early continue pattern, and consistent default handling.

### Key Engineering Insights

**1. Abstraction Level Shift:** You're no longer optimizing keystroke efficiency—you're optimizing prompt clarity and code review speed. The bottleneck moves from writing to verification.

**2. Context Management is Critical:** LLMs operate on a context window (typically 4K-128K tokens). The quality of generated code depends entirely on what's in that window: relevant files, function signatures, existing patterns, error messages.

**3. Determinism Trade-off:** Traditional IDEs are deterministic—same input, same autocomplete. LLMs are probabilistic—same prompt might yield different implementations. This requires different mental models and workflows.

### Why This Matters Now

The median time to write a function has dropped from ~15 minutes to ~3 minutes for experienced engineers using AI assistance. But the real value isn't speed—it's cognitive bandwidth. By offloading syntax and boilerplate, engineers can focus on architecture, edge cases, and system design. However, this only works if you understand how to manage context, verify outputs, and guide the LLM effectively.

## Technical Components

### 1. Context Window Management

**Technical Explanation:**  
Every LLM interaction includes a context window—the total text the model can "see" when generating code. This typically includes: your current file, manually added files, relevant codebase chunks (via embeddings search), and conversation history. The window has a hard token limit (1 token ≈ 0.75 words).

**Practical Implications:**  
If your context window is 32K tokens (~24K words) and you're working in a 500-line file (≈4K tokens), you have ~28K tokens for additional context. Add 3 related files (12K tokens) and conversation history (8K tokens), and you're using ~24K/32K (75%). At 80%+, some tools start dropping context—usually oldest conversation turns first.

**Real Constraints:**  
- Large files (>1000 lines) consume disproportionate context
- Including unnecessary files reduces relevance of remaining context
- Token counting is approximate (varies by tokenizer)
- Context refreshes on every request—no persistent state

**Concrete Example:**

```python
# BAD: Including entire Django models.py (5000 lines)
# in context to generate a simple view function
# Result: LLM has less room for relevant view examples

# GOOD: Extract only the relevant model
# my_models.py (manually added to context)
class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    published_at = models.DateTimeField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)

# Now generate view with focused context
# views.py (current file)
def article_detail(request, article_id):
    # Prompt: "Create a detail view with published articles only"
    # LLM has room for Django view patterns, this model, and best practices
    article = get_object_or_404(
        Article.objects.select_related('author'),
        id=article_id,
        published_at__lte=timezone.now()
    )
    return render(request, 'article_detail.html', {'article': article})
```

### 2. Multi-File Code Generation

**Technical Explanation:**  
AI code editors can generate or modify multiple files in a single request. This works through a combination of: (1) parsing your intent for affected files, (2) generating diffs for each file, (3) applying changes atomically or presenting for review.

**Practical Implications:**  
Multi-file generation is powerful for refactoring, feature scaffolding, and pattern replication. However, it increases error surface area—one mistake can propagate across files. Always review diffs file-by-file, not in aggregate.

**Real Constraints:**  
- More files = lower quality per file (attention dilution)
- Cross-file references can be inconsistent
- Rollback requires version control (built-in undo is limited)
- Some tools have file count limits (typically 5-10 per request)

**Concrete Example:**

```python
# Prompt: "Create a FastAPI endpoint for user registration with validation and tests"

# Generated file 1: app/models/user.py
from pydantic import BaseModel, EmailStr, Field

class UserRegistration(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    username: str = Field(..., min_length=3, max_length=50)
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepass123",
                "username": "johndoe"
            }
        }

# Generated file 2: app/routers/auth.py
from fastapi import APIRouter, HTTPException, status
from app.models.user import UserRegistration
import re

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user: UserRegistration):
    # Check password strength
    if not re.search(r"[A-Z]", user.password) or not re.search(r"[0-9]", user.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain uppercase and numbers"
        )
    
    # Actual registration logic would go here
    return {"message": "User registered successfully", "username": user.username}

# Generated file 3: tests/test_auth.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_register_valid_user():
    response = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "SecurePass123",
        "username": "testuser"
    })
    assert response.status_code == 201
    assert response.json()["username"] == "testuser"

def test_register_weak_password():
    response = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "weakpass",
        "username": "testuser"
    })
    assert response.status_code == 400
    assert "uppercase" in response.json()["detail"].lower()
```

### 3. Codebase Embeddings and Semantic Search

**Technical Explanation:**  
AI code editors create vector embeddings (numerical representations) of your codebase. When you ask a question or generate code, they semantically search these embeddings to find relevant files/functions, then include them in context. This is why the LLM can reference a utility function you wrote months ago without you explicitly mentioning it.

**Practical Implications:**  
- The LLM "knows" about patterns in your codebase without seeing everything
- Quality depends on embedding freshness (usually updates on file save)
- Works across languages—can find similar logic even if syntax differs
- More effective in larger codebases (>10K lines) where manual context selection is impractical

**Real Constraints:**  
- Embeddings take 10-60 seconds to generate for large codebases
- Semantic search can miss exact matches if terminology differs
- No guarantee of what's included—it's probabilistic
- Privacy: embeddings are typically generated locally, but verify for your tool

**Concrete Example:**

```python
# Existing codebase has a custom retry decorator (not in current context)
# utils/decorators.py (written 3 months ago)
import time
from functools import wraps

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
        return wrapper
    return decorator

# New file: services/external_api.py
# Prompt: "Create a function to fetch user data with retry logic"

# Generated code automatically discovers and uses existing decorator:
from utils.decorators import retry_on_failure
import httpx

@retry_on_failure(max_attempts=3, delay=2.0)
async def fetch_user_data(user_id: int) -> dict:
    """Fetch user data from external API with automatic retries."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()

# The LLM found retry_on_failure through semantic search for "retry" patterns,
# even though you didn't mention it in your prompt or include the file in context.
```

### 4. Conversation-Based Refinement

**Technical Explanation:**  
Unlike traditional autocomplete, AI code editors maintain conversation history. Each follow-up message includes previous exchanges, allowing iterative refinement without re-explaining context. This enables a "generate → review → refine" workflow.

**Practical Implications:**  
- First generation is rarely perfect—plan for 1-3 refinement iterations
- Conversation history consumes context window space
- Specific refinement requests ("add error handling") work better than vague ones ("make it better")
- Long conversations (>10 exchanges) often need reset to clear accumulated cruft

**Real Constraints:**  
- History is lost on session restart (unless explicitly saved)
- No branching—can't explore alternative implementations simultaneously
- Token limits apply to entire conversation, not just current message
- Model can "forget" early context in long conversations

**Concrete Example:**

```python
# Initial prompt: "Create a function to parse CSV data"

# Generation 1 (basic implementation):
import csv
from typing import List, Dict

def parse_csv(filepath: str) -> List[Dict[str, str]]:
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Follow-up 1: "Add error handling for missing files and invalid CSV"

# Generation 2 (adds error handling):
import csv
from typing import List, Dict
from pathlib import Path

def parse_csv(filepath: str) -> List[Dict[str, str]]:
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except csv.Error as e:
        raise ValueError(f"Invalid CSV format: {e}")

# Follow-up 2: "Add type hints for return value with actual column types"

# Generation 3 (adds TypedDict for structure):
import csv
from typing import List
from pathlib import Path
from typing import TypedDict

class CSVRow(TypedDict, total=False):
    # Adjust fields based on expected CSV structure
    id: str
    name: str
    email: str
    age: str

def parse_csv(filepath: str) -> List[CSVRow]:
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [CSVRow(**row) for row in reader]
    except csv.Error as e:
        raise ValueError(f"Invalid CSV format: {e}")
    except TypeError as e:
        raise ValueError(f"CSV columns don't match expected structure: {e}")
```

### 5. Inline vs. Composer Modes

**Technical Explanation:**  
AI code editors offer two primary interaction modes:

1. **Inline mode:** LLM suggestions appear directly in your editor as you type, similar to autocomplete but multi-line. Accepts with Tab, ignores by continuing to type.

2. **Composer mode:** Dedicated chat interface where you describe changes, review diffs, then apply. Full conversation history, multi-file support, explicit approval.

**Practical Implications:**  
- Inline for small, predictable completions (function bodies, boilerplate)
- Composer for exploratory work, refactoring, or when you need to see changes before applying
- Inline has lower latency (100-500ms) vs. composer (1-3s)
- Inline is more intrusive—can disrupt flow if suggestions are low-quality

**Real Constraints:**  
- Inline mode has smaller context window (typically 1/4 of composer)
- Can't use both simultaneously for same code section
- Inline suggestions expire quickly (usually after typing continues)
- Composer mode requires explicit commit—easy to lose work if not applied

**Concrete Example:**

```python
# INLINE MODE - Good use case
# Type: "def calculate_" and pause
# LLM immediately suggests (appears ghosted in editor):

def calculate_compound_interest(
    principal: float, 
    rate: float, 
    time: float, 
    n: int = 12
) -> float:
    """Calculate compound interest.
    
    Args:
        principal: Initial investment amount
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time: Time period in years
        n: Number of times interest compounds per year
    
    Returns:
        Final amount after compound interest
    """
    return principal * (1 + rate / n) ** (n * time)

# Press Tab to accept, or keep typing to ignore

# COMPOSER MODE - Good use case
# Complex refactoring across multiple files
# Prompt: "Refactor the authentication middleware to use dependency injection 
#          and separate concerns into auth.py and dependencies.py"

# Shows diff preview for both files before applying:
# ────── auth.py ──────
# - def authenticate_user(request: Request):
# + async def verify_token(token: str) -> User:
#       """Verify JWT token and return user."""
# +     # Token verification logic
# +     pass

# ────── dependencies.