# Automated Test Generation with LLMs

## Core Concepts

### Technical Definition

Automated test generation using LLMs involves leveraging language models to analyze code structure, behavior, and context to produce executable test cases. Unlike traditional rule-based test generation tools that rely on static analysis and predefined templates, LLM-based approaches understand semantic meaning, infer edge cases from natural language specifications, and generate human-readable test code that covers both happy paths and failure scenarios.

### Engineering Analogy: Traditional vs. LLM-Based Approaches

**Traditional Approach:**

```python
# Traditional property-based testing with manual specification
from hypothesis import given, strategies as st

class OrderProcessor:
    def calculate_total(self, items: list[dict]) -> float:
        return sum(item['price'] * item['quantity'] for item in items)

# Engineer manually writes test strategies
@given(st.lists(
    st.fixed_dictionaries({
        'price': st.floats(min_value=0.01, max_value=1000.0),
        'quantity': st.integers(min_value=1, max_value=100)
    }),
    min_size=1,
    max_size=10
))
def test_calculate_total_properties(items):
    processor = OrderProcessor()
    result = processor.calculate_total(items)
    assert result >= 0
    # Engineer must manually identify properties to test
```

**LLM-Assisted Approach:**

```python
# LLM analyzes code and generates comprehensive test suite
import anthropic
from typing import Optional

def generate_tests_for_function(source_code: str, context: str = "") -> str:
    """
    Generate comprehensive test suite using LLM analysis.
    """
    client = anthropic.Anthropic()
    
    prompt = f"""Analyze this Python function and generate a comprehensive pytest test suite.
Include:
- Edge cases (empty inputs, boundary values, null/None)
- Type validation
- Business logic validation
- Error conditions
- Integration scenarios if context suggests them

Function to test:
```python
{source_code}
```

Additional context: {context}

Generate complete, runnable pytest code with fixtures and parametrize decorators."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Example usage
source = """
class OrderProcessor:
    def calculate_total(self, items: list[dict]) -> float:
        if not items:
            raise ValueError("Items list cannot be empty")
        return sum(item['price'] * item['quantity'] for item in items)
"""

tests = generate_tests_for_function(
    source,
    context="E-commerce system processing customer orders"
)
# LLM generates tests for: empty list, missing keys, negative values,
# zero values, float precision, large numbers, type mismatches
```

The LLM approach identifies edge cases an engineer might miss (missing dictionary keys, mixed numeric types, float precision issues) and generates readable test code that documents intent.

### Key Insights That Change Engineering Perspective

**1. Tests as Documentation Synthesis:** LLMs excel at inferring expected behavior from variable names, type hints, and docstrings—treating code as a specification document. This shifts testing from "what did I implement?" to "what should this do according to all available context?"

**2. Semantic Understanding Over Syntactic Patterns:** Traditional tools match syntactic patterns; LLMs understand that a function named `validate_email` should test malformed emails, boundary cases in domain lengths, and internationalized addresses—without explicit templates.

**3. Context Accumulation:** LLMs can analyze an entire codebase's patterns. If your repository consistently handles `None` checks in a specific way, generated tests will reflect those conventions automatically.

### Why This Matters Now

The shift to microservices and API-first architectures has exploded the surface area requiring test coverage. Teams face:

- **API contract testing** across dozens of endpoints
- **Integration testing** with unreliable third-party services
- **Regression testing** as legacy systems are refactored
- **Security testing** for injection attacks and authorization failures

Manual test writing consumes 30-40% of development time. LLM-based generation doesn't replace engineers but accelerates coverage from 60% to 85%+ by automating the mechanical aspects while engineers focus on complex business logic scenarios.

## Technical Components

### Component 1: Code Context Extraction

**Technical Explanation:**

Effective test generation requires extracting comprehensive context from source code: function signatures, type annotations, docstrings, error handling patterns, dependencies, and surrounding code. This context informs what to test and how.

```python
import ast
import inspect
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class FunctionContext:
    name: str
    signature: str
    docstring: str
    parameters: List[Dict[str, Any]]
    return_type: str
    raises: List[str]
    calls_external: bool
    source_code: str

def extract_function_context(func_source: str) -> FunctionContext:
    """
    Parse function source to extract testable context.
    """
    tree = ast.parse(func_source)
    func_def = None
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    
    if not func_def:
        raise ValueError("No function definition found")
    
    # Extract parameters with type hints
    params = []
    for arg in func_def.args.args:
        param_info = {
            'name': arg.arg,
            'type': ast.unparse(arg.annotation) if arg.annotation else 'Any',
            'has_default': False
        }
        params.append(param_info)
    
    # Extract raises clauses from docstring and code
    raises = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Raise):
            if isinstance(node.exc, ast.Call):
                raises.append(node.exc.func.id)
    
    # Check for external calls
    calls_external = any(
        isinstance(node, ast.Call) 
        for node in ast.walk(func_def)
    )
    
    return FunctionContext(
        name=func_def.name,
        signature=ast.unparse(func_def),
        docstring=ast.get_docstring(func_def) or "",
        parameters=params,
        return_type=ast.unparse(func_def.returns) if func_def.returns else 'Any',
        raises=raises,
        calls_external=calls_external,
        source_code=func_source
    )

# Example usage
sample_code = """
def process_payment(amount: float, currency: str = "USD") -> dict:
    '''
    Process payment transaction.
    Raises ValueError if amount is negative or currency is unsupported.
    '''
    if amount < 0:
        raise ValueError("Amount cannot be negative")
    if currency not in ["USD", "EUR", "GBP"]:
        raise ValueError(f"Unsupported currency: {currency}")
    return {"status": "success", "amount": amount, "currency": currency}
"""

context = extract_function_context(sample_code)
print(f"Function: {context.name}")
print(f"Parameters: {context.parameters}")
print(f"Raises: {context.raises}")
```

**Practical Implications:**

This context becomes the prompt foundation. Without structured extraction, LLMs receive raw code and miss critical details buried in type hints or exception handling. Structured context enables targeted test generation.

**Real Constraints:**

- **Dynamic code:** Functions using `exec()`, `eval()`, or heavy metaprogramming resist static analysis
- **External dependencies:** Context extraction doesn't understand third-party library semantics
- **Incomplete type hints:** Legacy code without annotations reduces generation quality

### Component 2: Test Intent Specification

**Technical Explanation:**

Raw test generation produces generic tests. Specifying *test intent*—what properties, scenarios, or risks to validate—dramatically improves relevance. This involves engineering test categories and constraints into prompts.

```python
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

class TestCategory(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"
    SECURITY = "security"
    PERFORMANCE = "performance"

@dataclass
class TestSpec:
    categories: List[TestCategory]
    coverage_targets: List[str]  # Specific conditions to test
    mock_external: bool
    framework: str = "pytest"
    
def build_test_prompt(context: FunctionContext, spec: TestSpec) -> str:
    """
    Build targeted test generation prompt.
    """
    prompt_parts = [
        f"Generate {spec.framework} tests for this Python function:",
        f"```python\n{context.source_code}\n```",
        f"\nFunction: {context.name}",
        f"Parameters: {', '.join(p['name'] + ': ' + p['type'] for p in context.parameters)}",
        f"Returns: {context.return_type}",
    ]
    
    if context.raises:
        prompt_parts.append(f"Raises: {', '.join(context.raises)}")
    
    prompt_parts.append("\nGenerate tests for:")
    
    category_requirements = {
        TestCategory.UNIT: "- Basic functionality with typical inputs\n- Return value validation",
        TestCategory.EDGE_CASE: "- Boundary values (empty, zero, max)\n- None/null handling\n- Type mismatches",
        TestCategory.SECURITY: "- Injection attempts\n- Authorization bypasses\n- Input sanitization",
        TestCategory.INTEGRATION: "- External service failures\n- Network timeouts\n- Database transaction rollbacks"
    }
    
    for category in spec.categories:
        if category in category_requirements:
            prompt_parts.append(category_requirements[category])
    
    if spec.coverage_targets:
        prompt_parts.append("\nSpecifically test these scenarios:")
        for target in spec.coverage_targets:
            prompt_parts.append(f"- {target}")
    
    if spec.mock_external and context.calls_external:
        prompt_parts.append("\nMock all external calls using unittest.mock or pytest fixtures.")
    
    prompt_parts.append("\nGenerate complete, runnable test code with imports and fixtures.")
    
    return "\n".join(prompt_parts)

# Example usage
spec = TestSpec(
    categories=[TestCategory.UNIT, TestCategory.EDGE_CASE, TestCategory.SECURITY],
    coverage_targets=[
        "Negative amounts should raise ValueError",
        "Unsupported currencies should raise ValueError",
        "Currency code injection attempts (e.g., 'USD; DROP TABLE')"
    ],
    mock_external=False
)

prompt = build_test_prompt(context, spec)
print(prompt)
```

**Practical Implications:**

Structured specifications prevent the "50 generic tests" problem. Engineers define risk areas (security, concurrency, data corruption) and get focused tests rather than exhaustive but shallow coverage.

**Trade-offs:**

- **Over-specification:** Too many constraints can limit LLM creativity in finding unexpected edge cases
- **Maintenance overhead:** Test specs become code that requires updating
- **Category ambiguity:** A test might span multiple categories (security + integration)

### Component 3: Test Validation and Execution Pipeline

**Technical Explanation:**

Generated tests must be validated before deployment. This requires parsing the LLM output, executing tests in isolated environments, measuring coverage, and iterating on failures.

```python
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Tuple, Optional

class TestValidator:
    def __init__(self, venv_path: Optional[str] = None):
        self.venv_path = venv_path
    
    def extract_code_blocks(self, llm_response: str) -> List[str]:
        """Extract Python code blocks from LLM response."""
        pattern = r'```python\n(.*?)\n```'
        matches = re.findall(pattern, llm_response, re.DOTALL)
        return matches
    
    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if generated test code is syntactically valid."""
        try:
            compile(code, '<string>', 'exec')
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    def execute_tests(self, test_code: str, source_code: str) -> dict:
        """
        Execute generated tests in isolated environment.
        Returns execution results with coverage metrics.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Write source code
            source_file = tmp_path / "module.py"
            source_file.write_text(source_code)
            
            # Write test code
            test_file = tmp_path / "test_module.py"
            test_file.write_text(test_code)
            
            # Execute pytest with coverage
            cmd = [
                "pytest",
                str(test_file),
                "--cov=module",
                "--cov-report=json",
                "--json-report",
                "--json-report-file=report.json",
                "-v"
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse results
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "coverage": self._parse_coverage(tmp_path / "coverage.json"),
                    "test_count": self._count_tests(result.stdout)
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": "Test execution timeout (>30s)",
                    "coverage": 0,
                    "test_count": 0
                }
    
    def _parse_coverage(self, coverage_file: Path) -> float:
        """Extract coverage percentage from coverage.json."""
        if not coverage_file.exists():
            return 0.0
        
        import json
        with open(coverage_file) as f:
            data = json.load(f)
            return data.get('totals', {}).get('percent_covered', 0.0)
    
    def _count_tests(self, stdout: str) -> int:
        """Count number of tests executed from pytest output."""
        match = re.search(r'(\d+) passed', stdout)
        return int(match.group(1)) if match else 0

# Example usage
validator = TestValidator()

generated_test = """
import pytest
from module import process_payment

def test_valid_payment():
    result = process_payment(100.0, "USD")
    assert result["status"] == "success"
    assert result["amount"] == 100.0

def test_negative_amount():
    with pytest.raises(ValueError, match="Amount cannot be negative"):
        process_payment(-10.0, "USD")

def test_invalid_currency():
    with pytest.raises(ValueError, match="Unsupported currency"):
        process_payment(100.0, "JPY")
"""

# Validate syntax
valid, msg = validator.validate_syntax(generated_test)
print(f"Syntax validation: {valid} - {msg}")

# Execute tests
results = validator.execute_tests(generated_test, sample_code)
print(f"Tests passed: {results['success']}")
print(f"Coverage: {results['coverage']}%")
print(f"Test count: {results['test_count']}")
```

**Practical Implications:**

Automated validation catches non-executable tests immediately. In production workflows, failed tests trigger re-generation with error context, creating a self-improving loop.

**Real Constraints:**

- **Environment isolation:** Tests requiring databases, APIs, or specific system configurations need complex setup
- **Flaky tests:** LLMs occasionally generate tests with timing dependencies or random values
- **Security concerns:** Executing arbitrary LLM-generated code requires sandboxing

### Component 4: Iterative Refinement Loop

**Technical Explanation:**

Initial test generation rarely achieves target coverage. Iterative refinement analyzes coverage gaps