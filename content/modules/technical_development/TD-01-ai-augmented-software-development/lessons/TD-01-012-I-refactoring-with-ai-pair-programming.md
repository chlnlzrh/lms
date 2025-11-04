# Refactoring with AI Pair-Programming

## Core Concepts

### Technical Definition

AI pair-programming for refactoring involves using language models as interactive code analysis and transformation tools. Unlike traditional refactoring tools that apply predefined transformations (rename variable, extract method), LLM-based refactoring operates through natural language instructions to understand intent, analyze context across multiple files, and propose structural changes that require semantic understanding.

The key distinction: traditional IDEs perform syntactic transformations (AST manipulation), while LLM-assisted refactoring combines syntactic awareness with semantic understanding of code purpose, patterns, and trade-offs.

### Engineering Analogy: Traditional vs. LLM-Assisted Refactoring

**Traditional Automated Refactoring:**

```python
# IDE refactoring: "Extract Method" - purely syntactic
class DataProcessor:
    def process(self, data: list[dict]) -> list[dict]:
        results = []
        for item in data:
            # Selected these 5 lines -> Extract Method
            if item.get('status') == 'active':
                validated = self._validate(item)
                if validated:
                    transformed = self._transform(validated)
                    results.append(transformed)
        return results
```

The IDE extracts the selected lines into a new method, but you must:
- Choose the extraction boundaries
- Name the method
- Identify all parameters
- Ensure no variable scope issues

**LLM-Assisted Refactoring:**

```python
# Prompt: "Refactor this to use a pipeline pattern with clear stages,
# making it easier to add new processing steps and test each stage independently"

from typing import Callable, TypeVar
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class PipelineStage:
    name: str
    process: Callable[[T], T | None]
    
class DataProcessor:
    def __init__(self):
        self.stages: list[PipelineStage] = [
            PipelineStage("filter_active", self._filter_active),
            PipelineStage("validate", self._validate),
            PipelineStage("transform", self._transform)
        ]
    
    def process(self, data: list[dict]) -> list[dict]:
        results = []
        for item in data:
            current = item
            for stage in self.stages:
                current = stage.process(current)
                if current is None:
                    break
            if current is not None:
                results.append(current)
        return results
    
    def _filter_active(self, item: dict) -> dict | None:
        return item if item.get('status') == 'active' else None
    
    def _validate(self, item: dict) -> dict | None:
        # validation logic
        return item
    
    def _transform(self, item: dict) -> dict:
        # transformation logic
        return item
```

The LLM understands the architectural goal (pipeline pattern), recognizes appropriate design patterns, maintains type safety, and creates a structure that addresses the specific extensibility requirement.

### Key Insights

**1. Context Aggregation Over File Boundaries**

Traditional refactoring tools operate on single-file ASTs. LLMs can hold multiple related files in context simultaneously, identifying cross-file dependencies and suggesting refactorings that span modules. This changes refactoring from local optimization to system-wide restructuring.

**2. Intent-Driven vs. Syntax-Driven Transformation**

When you tell an LLM "make this more testable," it understands dependency injection, composition over inheritance, and test seam creation. Traditional tools need explicit commands like "extract interface" - you must already know the solution pattern.

**3. Incremental Learning Loop**

The most effective pattern isn't "refactor this completely" but rather:
1. Ask for analysis of code smells
2. Request specific transformation
3. Review and provide feedback
4. Iterate on problematic sections

This mirrors actual pair programming, not batch processing.

### Why This Matters Now

Legacy codebases accumulate faster than teams can refactor them. The backlog of "technical debt" grows because manual refactoring is slow and risky. LLM-assisted refactoring doesn't eliminate the need for tests or human review, but it compresses the time from "we should refactor this" to "here's a concrete proposal" from days to minutes.

More critically: LLMs can explain *why* certain code is problematic and *what* patterns would improve it, serving as an on-demand architecture consultant for every developer, not just senior engineers.

## Technical Components

### 1. Context Window Management

**Technical Explanation:**

LLMs have fixed context windows (typically 8K-200K tokens). For refactoring, you must fit:
- Original code to refactor
- Related code (dependencies, callers)
- Instructions and constraints
- Room for the generated response

A common mistake is dumping entire files into the context when only specific functions need refactoring.

**Practical Implications:**

```python
# Bad: Dumping entire file (2000 lines)
"""
Refactor this file to use dependency injection:
<entire 2000-line file>
"""

# Good: Targeted context
"""
Refactor the UserService class to use dependency injection.

Current implementation:
<UserService class only - 50 lines>

Dependencies it uses:
<DatabaseConnection interface - 10 lines>
<EmailService interface - 8 lines>

Callers (for reference):
<snippet showing how UserService is instantiated - 5 lines>

Constraints:
- Maintain existing public API
- Constructor injection preferred
- Add type hints
"""
```

**Real Constraints:**

- Token limits force prioritization: show the code being changed, interfaces of dependencies, and representative usage examples
- Response length counts against context: if you need 500 lines of refactored code, ensure you have token budget
- Multiple iteration cycles consume context exponentially: each round includes previous code + new instructions

**Concrete Example:**

```python
import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for context management."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def prepare_refactoring_context(
    target_code: str,
    dependencies: list[str],
    max_tokens: int = 6000
) -> str:
    """Build context staying within token budget."""
    context_parts = [
        "Refactor the following code:\n\n",
        target_code,
        "\n\nDependencies:\n\n"
    ]
    
    current_tokens = estimate_tokens("".join(context_parts))
    remaining_tokens = max_tokens - current_tokens - 1000  # Reserve for response
    
    for dep in dependencies:
        dep_tokens = estimate_tokens(dep)
        if current_tokens + dep_tokens < remaining_tokens:
            context_parts.append(dep + "\n\n")
            current_tokens += dep_tokens
        else:
            # Truncate dependency to just signature
            signature = extract_signatures(dep)
            context_parts.append(signature + "\n\n")
    
    return "".join(context_parts)

def extract_signatures(code: str) -> str:
    """Extract just function/class signatures from code."""
    import ast
    tree = ast.parse(code)
    signatures = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            signatures.append(f"def {node.name}({', '.join(args)}): ...")
        elif isinstance(node, ast.ClassDef):
            signatures.append(f"class {node.name}: ...")
    
    return "\n".join(signatures)
```

### 2. Diff-Based Validation

**Technical Explanation:**

LLMs sometimes produce code that looks correct but subtly changes behavior. Diff-based validation means using standard diff tools to examine exactly what changed, making unintended modifications immediately visible.

**Practical Implications:**

Always generate refactored code to a separate location and use `diff` or structured comparison before accepting changes:

```python
from difflib import unified_diff
from typing import Iterator

def validate_refactoring(
    original_file: str,
    refactored_file: str,
    allowed_changes: set[str]
) -> tuple[bool, list[str]]:
    """
    Validate that refactoring only makes expected changes.
    
    Returns: (is_valid, list_of_issues)
    """
    with open(original_file) as f:
        original_lines = f.readlines()
    with open(refactored_file) as f:
        refactored_lines = f.readlines()
    
    diff = unified_diff(
        original_lines,
        refactored_lines,
        fromfile=original_file,
        tofile=refactored_file,
        lineterm=''
    )
    
    issues = []
    dangerous_changes = [
        'return ',  # Changed return value
        'if ',      # Changed conditional logic
        '==',       # Changed comparison
        'await ',   # Changed async behavior
    ]
    
    for line in diff:
        if line.startswith('-') and not line.startswith('---'):
            # Check if removed line contains logic
            for pattern in dangerous_changes:
                if pattern in line and pattern not in allowed_changes:
                    issues.append(f"Removed logic line: {line.strip()}")
        elif line.startswith('+') and not line.startswith('+++'):
            # Check if added line changes behavior
            for pattern in dangerous_changes:
                if pattern in line and pattern not in allowed_changes:
                    issues.append(f"Added logic line: {line.strip()}")
    
    return len(issues) == 0, issues

# Usage
valid, issues = validate_refactoring(
    'original/service.py',
    'refactored/service.py',
    allowed_changes={'return '}  # We expect return statement changes
)

if not valid:
    print("⚠️  Unexpected changes detected:")
    for issue in issues:
        print(f"  - {issue}")
```

**Real Constraints:**

- LLMs may reformat code (whitespace, line breaks) creating noisy diffs
- Variable renaming creates large diffs even for structural refactoring
- Semantic equivalence (e.g., `if x:` vs `if x is True:`) shows as diff but is functionally identical

**Concrete Example:**

```python
import ast

def semantic_diff(original_code: str, refactored_code: str) -> dict:
    """
    Compare code at AST level to detect semantic changes,
    ignoring formatting and variable names.
    """
    orig_tree = ast.parse(original_code)
    refac_tree = ast.parse(refactored_code)
    
    orig_structure = extract_structure(orig_tree)
    refac_structure = extract_structure(refac_tree)
    
    return {
        'functions_added': refac_structure['functions'] - orig_structure['functions'],
        'functions_removed': orig_structure['functions'] - refac_structure['functions'],
        'classes_added': refac_structure['classes'] - orig_structure['classes'],
        'classes_removed': orig_structure['classes'] - refac_structure['classes'],
        'imports_added': refac_structure['imports'] - orig_structure['imports'],
        'imports_removed': orig_structure['imports'] - refac_structure['imports'],
    }

def extract_structure(tree: ast.AST) -> dict:
    """Extract structural elements from AST."""
    structure = {
        'functions': set(),
        'classes': set(),
        'imports': set()
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Use signature, not name, for renamed function detection
            arg_types = [type(arg).__name__ for arg in node.args.args]
            structure['functions'].add(f"{len(node.args.args)}args_{len(node.body)}stmts")
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            structure['classes'].add(f"{len(methods)}methods")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                structure['imports'].add(alias.name)
    
    return structure
```

### 3. Test Preservation and Enhancement

**Technical Explanation:**

Refactoring must preserve existing test coverage while ideally improving testability. LLM-assisted refactoring should include updating tests to work with the new structure without weakening assertions or coverage.

**Practical Implications:**

```python
# Original tightly-coupled code
class OrderProcessor:
    def process_order(self, order_id: str) -> bool:
        # Direct database access - hard to test
        conn = psycopg2.connect("dbname=prod user=app")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE id = %s", (order_id,))
        order = cursor.fetchone()
        
        # Direct email sending - hard to test
        smtp = smtplib.SMTP('smtp.example.com')
        smtp.send_message(f"Order {order_id} processed")
        
        return True

# Original test - requires real database and email server
def test_process_order():
    processor = OrderProcessor()
    result = processor.process_order("ORDER123")
    assert result == True  # Weak assertion
```

**After LLM-assisted refactoring with test improvement:**

```python
from typing import Protocol
from dataclasses import dataclass

class OrderRepository(Protocol):
    def get_order(self, order_id: str) -> dict | None: ...
    def update_order(self, order_id: str, status: str) -> None: ...

class NotificationService(Protocol):
    def notify(self, message: str) -> None: ...

@dataclass
class Order:
    id: str
    customer_email: str
    status: str
    total: float

class OrderProcessor:
    def __init__(
        self,
        repository: OrderRepository,
        notifications: NotificationService
    ):
        self.repository = repository
        self.notifications = notifications
    
    def process_order(self, order_id: str) -> Order | None:
        order_data = self.repository.get_order(order_id)
        if not order_data:
            return None
        
        order = Order(**order_data)
        self.repository.update_order(order_id, "processed")
        self.notifications.notify(f"Order {order_id} processed")
        
        return order

# Enhanced test with proper mocking and assertions
from unittest.mock import Mock
import pytest

def test_process_order_success():
    # Arrange
    mock_repo = Mock(spec=OrderRepository)
    mock_repo.get_order.return_value = {
        'id': 'ORDER123',
        'customer_email': 'customer@example.com',
        'status': 'pending',
        'total': 99.99
    }
    
    mock_notifications = Mock(spec=NotificationService)
    
    processor = OrderProcessor(mock_repo, mock_notifications)
    
    # Act
    result = processor.process_order('ORDER123')
    
    # Assert
    assert result is not None
    assert result.id == 'ORDER123'
    assert result.total == 99.99
    mock_repo.update_order.assert_called_once_with('ORDER123', 'processed')
    mock_notifications.notify.assert_called_once()

def test_process_order_not_found():
    mock_repo = Mock(spec=OrderRepository)
    mock_repo.get_order.return_value = None
    mock_notifications = Mock(spec=NotificationService)
    
    processor = OrderProcessor(mock_repo, mock_notifications)
    result = processor.process_order('INVALID')
    
    assert result is None
    mock_notifications.notify.assert_not_called()
```

**Real Constraints:**

- Test refactoring must happen in same commit as code refactoring to avoid broken builds
- Coverage percentage should stay same or increase
- Test execution time shouldn't significantly increase (avoid integration tests replacing unit tests)

### 4. Iterative Refinement Protocol

**Technical Explanation:**

Complex refactorings rarely succeed in one pass. The effective pattern is: analyze → propose → review → refine. Each iteration should address specific issues from the previous pass.

**Practical Implications:**

```python
from dataclasses import dataclass