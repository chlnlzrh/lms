# AI-Guided Test Case Generation

## Core Concepts

Test case generation has historically been one of software engineering's most time-consuming tasks. Traditional approaches require developers to manually reason about edge cases, boundary conditions, and input variations—a process that's both exhaustive and error-prone.

AI-guided test case generation fundamentally shifts this paradigm. Instead of manually enumerating test scenarios, you leverage language models to explore the solution space systematically, generating diverse test cases that exercise code paths you might not have considered.

### Traditional vs. AI-Guided Approach

```python
# Traditional: Manual test case enumeration
def test_parse_date_traditional():
    # Developer manually thinks of cases
    assert parse_date("2024-01-15") == datetime(2024, 1, 15)
    assert parse_date("2024/01/15") == datetime(2024, 1, 15)
    assert parse_date("01-15-2024") == datetime(2024, 1, 15)
    # Missed: leap years, timezone handling, invalid dates, etc.

# AI-Guided: Systematic exploration
def generate_test_cases_with_ai(function_code: str, function_signature: str) -> list[dict]:
    """Generate comprehensive test cases using LLM reasoning."""
    prompt = f"""
    Analyze this function and generate comprehensive test cases:
    
    {function_code}
    
    Generate test cases covering:
    1. Happy path scenarios
    2. Boundary conditions
    3. Invalid inputs
    4. Edge cases specific to the domain
    
    Return as JSON array with: input, expected_output, category, rationale
    """
    
    response = call_llm(prompt)
    return parse_json_response(response)
    # Result: 50+ test cases including leap year handling, 
    # timezone edge cases, malformed inputs, etc.
```

### Key Insights

**Pattern Recognition at Scale:** LLMs have been trained on millions of test files. They've internalized common testing patterns, edge cases, and failure modes across countless codebases. This means they can suggest test scenarios that reflect collective engineering wisdom.

**Domain-Aware Generation:** When you provide context about what your code does, LLMs can reason about domain-specific edge cases. A date parser needs different test cases than a JSON validator, and AI models understand these nuances.

**Iterative Refinement:** Unlike static test generators, AI-guided approaches can respond to feedback. You can request "more boundary conditions" or "focus on security vulnerabilities" and get relevant results.

### Why This Matters Now

Test coverage directly correlates with production reliability, but comprehensive testing requires exponential effort as code complexity grows. With AI assistance:

- **Coverage increases 3-5x** with same developer time investment
- **Edge cases surface earlier** in development, not in production
- **Regression test suites expand automatically** when code changes
- **Domain expertise gets encoded** into test generation prompts

The critical shift: testing moves from exhaustive manual enumeration to strategic guidance of AI exploration.

## Technical Components

### 1. Context Extraction and Code Analysis

Before generating test cases, you need to provide the LLM with sufficient context about what the code does, its dependencies, and its constraints.

**Technical Explanation:**

Effective test generation requires the model to understand not just syntax, but semantics. This means extracting:
- Function signatures and type hints
- Docstrings and inline comments
- Dependencies and imported modules
- Surrounding code context (class structure, related functions)

```python
import ast
import inspect
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class FunctionContext:
    name: str
    signature: str
    source_code: str
    docstring: str
    dependencies: List[str]
    type_hints: Dict[str, str]

def extract_function_context(func: callable) -> FunctionContext:
    """Extract comprehensive context for test generation."""
    source = inspect.getsource(func)
    tree = ast.parse(source)
    func_def = tree.body[0]
    
    # Extract type hints
    type_hints = {}
    for arg in func_def.args.args:
        if arg.annotation:
            type_hints[arg.arg] = ast.unparse(arg.annotation)
    
    if func_def.returns:
        type_hints['return'] = ast.unparse(func_def.returns)
    
    # Extract dependencies (imports used in function)
    dependencies = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Name):
            dependencies.append(node.id)
    
    return FunctionContext(
        name=func.__name__,
        signature=str(inspect.signature(func)),
        source_code=source,
        docstring=inspect.getdoc(func) or "",
        dependencies=list(set(dependencies)),
        type_hints=type_hints
    )

# Example usage
def calculate_shipping_cost(weight_kg: float, distance_km: int, is_express: bool = False) -> float:
    """Calculate shipping cost based on weight, distance, and service level.
    
    Args:
        weight_kg: Package weight in kilograms (must be positive)
        distance_km: Shipping distance in kilometers (0-10000)
        is_express: Whether to use express shipping
        
    Returns:
        Total shipping cost in USD
        
    Raises:
        ValueError: If weight or distance are out of valid ranges
    """
    if weight_kg <= 0:
        raise ValueError("Weight must be positive")
    if distance_km < 0 or distance_km > 10000:
        raise ValueError("Distance must be between 0 and 10000 km")
    
    base_cost = weight_kg * 0.5 + distance_km * 0.1
    return base_cost * 1.5 if is_express else base_cost

context = extract_function_context(calculate_shipping_cost)
print(f"Function: {context.name}")
print(f"Signature: {context.signature}")
print(f"Type hints: {context.type_hints}")
```

**Practical Implications:**

More context = better test cases. However, there's a trade-off with token usage. For complex functions, you may need to summarize context rather than including entire class definitions.

**Real Constraints:**

- Context window limits (8k-200k tokens depending on model)
- Cost scales linearly with context size
- Too much context can confuse the model; focus on relevance

### 2. Structured Prompt Engineering for Test Generation

Raw "generate tests" prompts produce inconsistent results. Structured prompts with explicit categories and output formats yield reliable, comprehensive test suites.

**Technical Explanation:**

Effective prompts specify:
1. Test categories to cover
2. Output format (JSON for parsing)
3. Constraints and assumptions
4. Examples of desired test structure

```python
from typing import List, Literal
from pydantic import BaseModel
import json

class TestCase(BaseModel):
    category: Literal["happy_path", "boundary", "invalid_input", "edge_case", "error_handling"]
    input_args: dict
    expected_output: Any
    expected_exception: str | None = None
    rationale: str

def build_test_generation_prompt(context: FunctionContext) -> str:
    """Build structured prompt for test case generation."""
    return f"""Generate comprehensive test cases for the following function:

Function: {context.name}
Signature: {context.signature}
Documentation: {context.docstring}

Source Code:
```python
{context.source_code}
```

Generate test cases in these categories:

1. HAPPY PATH (3-5 cases): Normal, expected usage scenarios
2. BOUNDARY CONDITIONS (4-6 cases): Min/max values, limits, thresholds
3. INVALID INPUTS (3-5 cases): Wrong types, null values, malformed data
4. EDGE CASES (3-5 cases): Domain-specific unusual scenarios
5. ERROR HANDLING (2-4 cases): Expected exceptions and error states

For each test case, provide:
- category: One of the categories above
- input_args: Dictionary of function arguments
- expected_output: Expected return value (if not raising exception)
- expected_exception: Exception class name if one should be raised
- rationale: Brief explanation of what this tests

Return ONLY a JSON array of test cases. Example format:
[
  {{
    "category": "boundary",
    "input_args": {{"weight_kg": 0.001, "distance_km": 0, "is_express": false}},
    "expected_output": 0.0005,
    "expected_exception": null,
    "rationale": "Minimum valid weight and distance"
  }},
  {{
    "category": "invalid_input",
    "input_args": {{"weight_kg": -5, "distance_km": 100, "is_express": false}},
    "expected_output": null,
    "expected_exception": "ValueError",
    "rationale": "Negative weight should raise ValueError"
  }}
]

Generate 15-20 test cases covering all categories."""

def parse_test_cases(response: str) -> List[TestCase]:
    """Parse LLM response into structured test cases."""
    # Extract JSON from response (LLMs sometimes add explanation text)
    json_start = response.find('[')
    json_end = response.rfind(']') + 1
    json_str = response[json_start:json_end]
    
    cases_data = json.loads(json_str)
    return [TestCase(**case) for case in cases_data]
```

**Practical Implications:**

Structured prompts with output schemas reduce post-processing complexity. Using Pydantic models ensures type safety and validation of generated tests.

**Real Constraints:**

- JSON parsing can fail if LLM adds commentary
- Need robust error handling for malformed responses
- Some models struggle with strict JSON formatting; consider using function calling APIs

### 3. Test Case Validation and Filtering

Not all generated test cases are correct or useful. Automated validation filters out invalid tests before they enter your test suite.

**Technical Explanation:**

Validation checks:
1. **Syntactic validity:** Can the test be executed?
2. **Logical consistency:** Does expected output match actual behavior?
3. **Uniqueness:** Are we duplicating existing tests?
4. **Value:** Does this test add meaningful coverage?

```python
import sys
from io import StringIO
from typing import Tuple

class TestValidator:
    def __init__(self, target_function: callable):
        self.target_function = target_function
        self.seen_inputs = set()
    
    def validate_test_case(self, test: TestCase) -> Tuple[bool, str]:
        """Validate a generated test case.
        
        Returns:
            (is_valid, reason)
        """
        # Check for duplicate inputs
        input_signature = json.dumps(test.input_args, sort_keys=True)
        if input_signature in self.seen_inputs:
            return False, "Duplicate test case"
        
        # Attempt to execute
        try:
            result = self.target_function(**test.input_args)
            
            # If expecting exception but got result
            if test.expected_exception:
                return False, f"Expected {test.expected_exception} but got result: {result}"
            
            # Validate output matches (with tolerance for floats)
            if isinstance(result, float) and isinstance(test.expected_output, (int, float)):
                if abs(result - test.expected_output) > 0.001:
                    return False, f"Expected {test.expected_output}, got {result}"
            elif result != test.expected_output:
                return False, f"Expected {test.expected_output}, got {result}"
            
            self.seen_inputs.add(input_signature)
            return True, "Valid"
            
        except Exception as e:
            exception_name = type(e).__name__
            
            # If we expected this exception
            if test.expected_exception and exception_name == test.expected_exception:
                self.seen_inputs.add(input_signature)
                return True, "Valid (expected exception)"
            
            # Unexpected exception
            return False, f"Unexpected exception: {exception_name}: {str(e)}"
    
    def filter_valid_tests(self, tests: List[TestCase]) -> List[TestCase]:
        """Filter to only valid, unique test cases."""
        valid_tests = []
        for test in tests:
            is_valid, reason = self.validate_test_case(test)
            if is_valid:
                valid_tests.append(test)
            else:
                print(f"Filtered out test: {reason}", file=sys.stderr)
        
        return valid_tests

# Example usage
validator = TestValidator(calculate_shipping_cost)
# Assume 'generated_tests' comes from LLM
valid_tests = validator.filter_valid_tests(generated_tests)
print(f"Kept {len(valid_tests)} of {len(generated_tests)} generated tests")
```

**Practical Implications:**

Validation catches LLM hallucinations and mathematical errors. Expect 70-90% of generated tests to be valid; the filtering step is essential.

**Real Constraints:**

- Validation requires executing tests, which may have side effects
- For functions with external dependencies (DB, API calls), validation needs mocking
- Floating-point comparisons need tolerance thresholds

### 4. Test Code Synthesis

Converting structured test cases into actual executable test code completes the generation pipeline.

**Technical Explanation:**

Generate test functions following project conventions (pytest, unittest, etc.) with proper assertions, error handling, and documentation.

```python
from textwrap import dedent

class TestCodeGenerator:
    def __init__(self, style: Literal["pytest", "unittest"] = "pytest"):
        self.style = style
    
    def generate_pytest_function(self, test: TestCase, function_name: str, index: int) -> str:
        """Generate pytest-style test function."""
        test_name = f"test_{function_name}_{test.category}_{index}"
        
        # Format input arguments
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in test.input_args.items())
        
        if test.expected_exception:
            # Test expecting exception
            code = dedent(f'''
            def {test_name}():
                """
                {test.rationale}
                Category: {test.category}
                """
                import pytest
                with pytest.raises({test.expected_exception}):
                    {function_name}({args_str})
            ''')
        else:
            # Test expecting return value
            code = dedent(f'''
            def {test_name}():
                """
                {test.rationale}
                Category: {test.category}
                """
                result = {function_name}({args_str})
                assert result == {repr(test.expected_output)}
            ''')
        
        return code.strip()
    
    def generate_test_file(self, tests: List[TestCase], function_name: str, 
                          imports: List[str]) -> str:
        """Generate complete test file."""
        header = "# Auto-generated test file\n"
        header += "# Generated by AI-guided test generation\n\n"
        
        # Add imports
        import_section = "\n".join(imports) + "\n\n"
        
        # Generate all test functions
        test_functions = []
        for idx, test in enumerate(tests):
            test_func = self.generate_pytest_function(test, function_name, idx)
            test_functions.append(test_func)
        
        return header + import_section + "\n\n".join(test_functions)

# Example usage
generator = TestCodeGenerator(style="pytest")
test_file_content = generator.generate_test_file(
    valid_tests,
    "calculate_shipping_cost",
    ["import pytest", "from shipping import calculate_shipping_cost"]
)

# Write to file
with open("test_shipping_generated.py", "w") as f:
    f.write(test_file_content)
```

**Practical Implications:**

Generated test code should be human-readable and maintainable. Include documentation explaining test purpose and category.

**Real Constraints:**

- Generated code must match project style guides
- Complex assertions (comparing nested structures) need custom formatting
- Test naming conventions vary by team

### 5. Coverage Analysis and Gap Identification

After generating initial tests, analyze coverage gaps and iteratively generate additional tests for uncovered paths.

**Technical Explanation:**

Use coverage tools to identify untested branches, then generate targeted tests for those specific code paths.

```python
import coverage
import ast

class CoverageGuidedGenerator:
    def __init__(self, target_module: str):
        self