# Feature Decomposition with Claude/Cursor

## Core Concepts

Feature decomposition is the practice of breaking down complex software requirements into smaller, well-defined implementation units that can be executed independently while maintaining coherent architecture. When applied to AI-assisted development with LLM tools, this practice becomes both more critical and more powerful—critical because LLMs work best with focused, well-scoped tasks, and powerful because they can help you decompose features in ways that reveal hidden complexity early.

### Traditional vs. AI-Assisted Feature Decomposition

**Traditional Approach:**
```python
# Developer receives requirement: "Add user authentication"
# Manual decomposition process:
# 1. Think through all components (30-60 min)
# 2. Create task list in project manager
# 3. Start coding, discover missing pieces mid-implementation
# 4. Context switch between architecture and implementation

class AuthenticationSystem:
    def __init__(self):
        # Developer realizes: need password hashing
        # Developer realizes: need session management
        # Developer realizes: need rate limiting
        # Developer realizes: need password reset flow
        # Each discovery interrupts implementation flow
        pass
```

**AI-Assisted Approach:**
```python
# Developer provides requirement to LLM with decomposition prompt
# LLM generates structured breakdown in seconds:

"""
Feature: User Authentication
├── 1. Data Layer
│   ├── User model with password hash field
│   ├── Session storage mechanism
│   └── Migration scripts
├── 2. Security Components
│   ├── Password hashing utility (bcrypt)
│   ├── JWT token generation/validation
│   └── Rate limiting middleware
├── 3. Core Authentication Logic
│   ├── Registration handler
│   ├── Login handler
│   └── Logout handler
├── 4. Password Recovery
│   ├── Reset token generation
│   ├── Email notification system
│   └── Reset confirmation handler
└── 5. Integration Points
    ├── Middleware for protected routes
    ├── Error handling for auth failures
    └── Audit logging
"""

# Developer now implements each component with focused prompts
# Discovers architectural issues before writing code
# Maintains clear implementation order
```

### Key Engineering Insights

**1. Decomposition Surfaces Hidden Dependencies Early**

Poor decomposition leads to discovered dependencies during implementation. When you ask an LLM to decompose a feature, it typically identifies cross-cutting concerns (logging, error handling, testing) that developers often add reactively. This shifts discovery from implementation-time to planning-time, reducing costly refactors.

**2. Granularity Directly Impacts LLM Output Quality**

LLMs have a quality curve relative to task scope. Tasks that are too broad produce generic, incomplete code. Tasks that are too narrow produce over-engineered solutions disconnected from broader context. The optimal granularity is typically a single function or class that has clear inputs/outputs and can be tested independently.

**3. Decomposition Creates Natural Checkpoints**

Each decomposed unit becomes a verification point. Instead of generating 500 lines of code and hoping it works, you generate 50 lines, verify, then proceed. This creates a feedback loop that catches conceptual errors early when they're cheap to fix.

### Why This Matters Now

Modern LLM coding assistants can generate substantial code volumes quickly—but volume without structure creates technical debt at unprecedented speed. A developer using an LLM without proper decomposition can create a working-but-unmaintainable system in hours. Proper decomposition harnesses the speed while maintaining architectural integrity.

Consider the cost difference: A poorly decomposed feature might generate 1000 lines of code in 30 minutes that takes 10 hours to debug and refactor. A well-decomposed feature might take 90 minutes to plan and implement but require only 30 minutes of debugging. The 3x time investment in decomposition yields a 10x reduction in debugging time.

## Technical Components

### 1. Decomposition Scope Definition

**Technical Explanation:**

Scope definition establishes the boundaries of what will be implemented, explicitly stating what's included and excluded. This prevents scope creep in LLM interactions where conversational context can drift toward related but out-of-scope functionality.

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ScopeType(Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    FUTURE = "future"

@dataclass
class FeatureScope:
    feature_name: str
    in_scope: List[str]
    out_of_scope: List[str]
    dependencies: List[str]
    success_criteria: List[str]

# Example: API rate limiting feature
rate_limiting_scope = FeatureScope(
    feature_name="API Rate Limiting",
    in_scope=[
        "Per-user request counting",
        "Configurable rate limits per endpoint",
        "HTTP 429 responses when limit exceeded",
        "Rate limit headers in responses"
    ],
    out_of_scope=[
        "Distributed rate limiting across servers",
        "Dynamic rate limit adjustment",
        "Rate limit analytics dashboard",
        "Per-client custom rate limits"
    ],
    dependencies=[
        "Redis or in-memory cache",
        "User authentication system",
        "Middleware framework"
    ],
    success_criteria=[
        "Blocks requests exceeding 100/minute per user",
        "Returns correct retry-after header",
        "Performance overhead < 5ms per request"
    ]
)

def format_scope_for_llm(scope: FeatureScope) -> str:
    """Format scope for inclusion in LLM prompts."""
    return f"""
Feature: {scope.feature_name}

MUST INCLUDE:
{chr(10).join(f"- {item}" for item in scope.in_scope)}

MUST EXCLUDE:
{chr(10).join(f"- {item}" for item in scope.out_of_scope)}

REQUIRED DEPENDENCIES:
{chr(10).join(f"- {item}" for item in scope.dependencies)}

SUCCESS CRITERIA:
{chr(10).join(f"- {item}" for item in scope.success_criteria)}
"""
```

**Practical Implications:**

When you prefix LLM prompts with explicit scope, you reduce hallucinated features by approximately 60-70%. The LLM stays constrained to the specified boundaries, and you can reference the scope document when generated code includes out-of-scope functionality.

**Trade-offs:**

Strict scope definition adds upfront time (10-15 minutes per feature) but prevents mid-implementation scope expansion. The cost is front-loaded planning time; the benefit is predictable implementation timeline.

### 2. Dependency Graph Construction

**Technical Explanation:**

A dependency graph maps which decomposed units must be implemented before others. This creates a valid implementation order and reveals circular dependencies that indicate architectural problems.

```python
from typing import Dict, Set, List
from collections import defaultdict, deque

class DependencyGraph:
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def add_dependency(self, task: str, depends_on: str) -> None:
        """Add dependency: task depends on depends_on."""
        self.graph[task].add(depends_on)
        self.reverse_graph[depends_on].add(task)
    
    def topological_sort(self) -> List[str]:
        """Return valid implementation order or raise if cycle detected."""
        in_degree = defaultdict(int)
        all_tasks = set(self.graph.keys()) | set(self.reverse_graph.keys())
        
        for task in all_tasks:
            in_degree[task] = len(self.graph[task])
        
        queue = deque([task for task in all_tasks if in_degree[task] == 0])
        result = []
        
        while queue:
            task = queue.popleft()
            result.append(task)
            
            for dependent in self.reverse_graph[task]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(all_tasks):
            raise ValueError("Circular dependency detected")
        
        return result
    
    def get_parallel_batches(self) -> List[Set[str]]:
        """Group tasks that can be implemented in parallel."""
        in_degree = defaultdict(int)
        all_tasks = set(self.graph.keys()) | set(self.reverse_graph.keys())
        
        for task in all_tasks:
            in_degree[task] = len(self.graph[task])
        
        batches = []
        remaining = set(all_tasks)
        
        while remaining:
            # Find all tasks with no remaining dependencies
            batch = {task for task in remaining if in_degree[task] == 0}
            if not batch:
                raise ValueError("Circular dependency detected")
            
            batches.append(batch)
            remaining -= batch
            
            # Update in-degrees
            for task in batch:
                for dependent in self.reverse_graph[task]:
                    in_degree[dependent] -= 1
        
        return batches

# Example: E-commerce checkout feature
checkout_deps = DependencyGraph()

# Define task dependencies
checkout_deps.add_dependency("validate_cart", "cart_model")
checkout_deps.add_dependency("calculate_shipping", "address_validator")
checkout_deps.add_dependency("process_payment", "payment_gateway_client")
checkout_deps.add_dependency("create_order", "validate_cart")
checkout_deps.add_dependency("create_order", "calculate_shipping")
checkout_deps.add_dependency("create_order", "process_payment")
checkout_deps.add_dependency("send_confirmation", "create_order")
checkout_deps.add_dependency("send_confirmation", "email_service")

# Get implementation order
implementation_order = checkout_deps.topological_sort()
print("Implementation order:", implementation_order)

# Get parallelizable batches
batches = checkout_deps.get_parallel_batches()
print("\nParallel implementation batches:")
for i, batch in enumerate(batches, 1):
    print(f"Batch {i}: {batch}")
```

**Practical Implications:**

Dependency graphs let you parallelize LLM interactions. Instead of waiting for sequential code generation, you can open multiple conversations for independent components. This can reduce total implementation time by 40-50% for features with high parallelism potential.

**Constraints:**

Building the dependency graph requires understanding the feature architecture upfront. For novel features in unfamiliar domains, you may need to create an initial draft implementation plan with an LLM, then refine the dependency graph based on that output.

### 3. Context-Preserving Task Specifications

**Technical Explanation:**

Each decomposed task needs sufficient context to be implemented correctly without requiring the full feature context. This means including relevant type definitions, interface contracts, error handling requirements, and architectural constraints in each task specification.

```python
from typing import TypedDict, Literal
from datetime import datetime

class TaskSpecification(TypedDict):
    task_id: str
    title: str
    description: str
    context: dict
    inputs: dict
    outputs: dict
    constraints: list[str]
    test_criteria: list[str]

def create_task_spec(
    task_id: str,
    title: str,
    description: str,
    **kwargs
) -> TaskSpecification:
    """Create a complete task specification for LLM consumption."""
    return TaskSpecification(
        task_id=task_id,
        title=title,
        description=description,
        context=kwargs.get('context', {}),
        inputs=kwargs.get('inputs', {}),
        outputs=kwargs.get('outputs', {}),
        constraints=kwargs.get('constraints', []),
        test_criteria=kwargs.get('test_criteria', [])
    )

# Example: Task specification for password hashing utility
password_hash_spec = create_task_spec(
    task_id="AUTH-003",
    title="Password Hashing Utility",
    description="Implement secure password hashing and verification using bcrypt",
    context={
        "architecture": "Stateless utility module, no database dependencies",
        "security_requirement": "Must use bcrypt with cost factor >= 12",
        "integration_point": "Called by registration and login handlers"
    },
    inputs={
        "hash_password": {
            "password": "str (plain text password)",
            "returns": "str (bcrypt hash)"
        },
        "verify_password": {
            "password": "str (plain text password)",
            "password_hash": "str (bcrypt hash)",
            "returns": "bool (True if valid)"
        }
    },
    outputs={
        "hash_format": "bcrypt hash string (60 characters)",
        "exceptions": "ValueError for invalid inputs"
    },
    constraints=[
        "Must not log passwords or hashes",
        "Must handle unicode passwords correctly",
        "Must be thread-safe",
        "Execution time must be 100-300ms (bcrypt intentional slowness)"
    ],
    test_criteria=[
        "Same password hashed twice produces different hashes",
        "Correct password verifies as True",
        "Incorrect password verifies as False",
        "Empty string password raises ValueError"
    ]
)

def format_task_for_llm(spec: TaskSpecification) -> str:
    """Format task specification as LLM prompt."""
    context_str = "\n".join(f"- {k}: {v}" for k, v in spec['context'].items())
    constraints_str = "\n".join(f"- {c}" for c in spec['constraints'])
    tests_str = "\n".join(f"- {t}" for t in spec['test_criteria'])
    
    return f"""
Implement: {spec['title']}

{spec['description']}

CONTEXT:
{context_str}

FUNCTION SIGNATURES:
```python
{format_function_signatures(spec['inputs'])}
```

CONSTRAINTS:
{constraints_str}

MUST PASS THESE TESTS:
{tests_str}

Generate complete implementation with type hints, docstrings, and error handling.
"""

def format_function_signatures(inputs: dict) -> str:
    """Generate function signature strings from input spec."""
    signatures = []
    for func_name, params in inputs.items():
        param_strs = [f"{k}: {v}" for k, v in params.items() if k != 'returns']
        return_type = params.get('returns', 'None')
        signatures.append(f"def {func_name}({', '.join(param_strs)}) -> {return_type}:")
    return "\n".join(signatures)
```

**Practical Implications:**

Rich task specifications reduce back-and-forth clarification with the LLM. Instead of generating code, realizing it doesn't meet constraints, and regenerating, you get usable code in the first generation 80%+ of the time.

**Trade-offs:**

Creating detailed specifications takes time—approximately 5-10 minutes per task. For trivial tasks (simple getters/setters), this overhead isn't justified. Use detailed specs for tasks with complexity, security requirements, or performance constraints.

### 4. Validation Checkpoints

**Technical Explanation:**

Validation checkpoints are automated checks that verify each decomposed unit meets its specification before proceeding to dependent tasks. This prevents cascading failures where a defect in component A causes failures in components B, C, and D.

```python
from typing import Callable, Any
import inspect
import time

class ValidationCheckpoint:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.checks: list[tuple[str, Callable]] = []
        self.results: dict[str, bool] = {}
    
    def add_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Add a validation check."""
        self.checks.append((name, check_fn))
    
    def run_all(self) -> tuple[bool, dict[str, Any]]:
        """Run all validation checks and return pass/fail with details."""
        results = {}
        all_passed = True
        
        for name, check_fn in self.checks:
            try:
                start = time.time()
                passed = check_fn()
                duration = time.time() - start
                
                results[name] = {
                    "passed": passed,
                    "duration_ms": round(duration * 1000, 2)
                }
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                results[name] = {
                    "passed": False,
                