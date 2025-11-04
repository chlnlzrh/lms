# Intelligent Code Commenting with LLMs

## Core Concepts

### Technical Definition

Intelligent code commenting uses language models to generate contextual, maintainable documentation by analyzing code structure, execution patterns, and intent. Unlike static documentation generators that extract type signatures and function names, LLMs interpret algorithmic logic, identify edge cases, and explain *why* code exists—not just *what* it does.

### Engineering Analogy: From Extraction to Interpretation

**Traditional approach (static analysis):**

```python
def calculate_risk_score(transactions: list[dict], threshold: float) -> float:
    """
    Calculate risk score.
    
    Args:
        transactions: list of dict
        threshold: float
    
    Returns:
        float
    """
    total = sum(t['amount'] for t in transactions if t['amount'] > threshold)
    return min(total / 10000, 1.0)
```

The documentation extracts type information but provides zero insight into the business logic, the magic number `10000`, or why we cap at `1.0`.

**LLM-enhanced approach:**

```python
def calculate_risk_score(transactions: list[dict], threshold: float) -> float:
    """
    Calculates normalized risk score based on high-value transaction exposure.
    
    Uses threshold filtering to focus on potentially fraudulent large transactions.
    The normalization factor (10000) represents maximum expected daily transaction
    volume for typical accounts. Score is capped at 1.0 to maintain consistency
    with downstream risk modeling that expects [0, 1] range.
    
    Args:
        transactions: Transaction records with 'amount' field in currency units
        threshold: Minimum transaction amount to consider risky (typically 500)
    
    Returns:
        Risk score between 0.0 (no risk) and 1.0 (maximum risk threshold)
    """
    total = sum(t['amount'] for t in transactions if t['amount'] > threshold)
    return min(total / 10000, 1.0)
```

The LLM explains the *intent* behind filtering, clarifies the domain context (fraud detection), documents the reasoning behind magic numbers, and specifies expected ranges.

### Key Insights That Change Engineering Thinking

**1. Documentation debt is now addressable at scale.** Legacy codebases with minimal comments can be systematically documented without manual archeology. The bottleneck shifts from "writing comments" to "validating LLM interpretations."

**2. Comments can be context-aware across the codebase.** LLMs can reference related functions, imported modules, and calling patterns to generate documentation that reflects actual usage, not just isolated function behavior.

**3. The value hierarchy inverts.** Instead of spending effort on trivial comments (`# increment counter`), you invest in prompting strategies that capture complex algorithmic intent and business logic that static analysis cannot infer.

### Why This Matters Now

Modern codebases contain millions of lines with documentation coverage often below 30%. Manual commenting doesn't scale, and documentation drift (where comments become outdated) creates maintenance hazards. LLMs can:

- **Generate initial documentation** for undocumented legacy code in hours, not months
- **Maintain consistency** across large teams with varying documentation styles
- **Reduce onboarding time** by 40-60% when new engineers can read intent-rich comments
- **Support code review** by flagging functions where generated comments don't match implementation (indicating unclear code)

The technology matured in 2023-2024 to where accuracy on well-structured code exceeds 85%, making it production-viable for most engineering teams.

## Technical Components

### 1. Context Construction: Code + Metadata

**Technical Explanation:**

LLMs need more than raw function code to generate useful comments. Effective context includes: the target function, its dependencies, call sites, type definitions, and surrounding module documentation. The context window (typically 8K-128K tokens) determines how much you can include.

**Practical Implications:**

A function isolated from its context produces generic documentation. Including the class definition, imports, and 2-3 usage examples produces documentation that explains *purpose* and *integration patterns*.

**Real Constraints:**

- Context windows cost tokens (money). A 128K context costs ~30x more than 4K.
- Including too much irrelevant code adds noise, reducing output quality.
- Dependency graphs can explode; you need heuristics for relevance filtering.

**Concrete Example:**

```python
import ast
from pathlib import Path

def extract_function_context(
    file_path: Path,
    function_name: str,
    max_context_lines: int = 50
) -> dict[str, str]:
    """
    Extracts function code plus surrounding context for LLM processing.
    
    Returns function source, class context (if method), and imports to help
    LLM understand dependencies and usage patterns.
    """
    source = file_path.read_text()
    tree = ast.parse(source)
    
    context = {
        "imports": [],
        "class_context": None,
        "function_code": None,
        "docstring_exists": False
    }
    
    # Extract imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            context["imports"].append(ast.unparse(node))
    
    # Find function and its class context
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    context["class_context"] = f"class {node.name}"
                    context["function_code"] = ast.unparse(item)
                    context["docstring_exists"] = ast.get_docstring(item) is not None
                    break
        elif isinstance(node, ast.FunctionDef) and node.name == function_name:
            context["function_code"] = ast.unparse(node)
            context["docstring_exists"] = ast.get_docstring(node) is not None
    
    return context

# Example usage
context = extract_function_context(
    Path("risk_analyzer.py"),
    "calculate_risk_score"
)

# Build prompt with structured context
prompt = f"""Generate a comprehensive docstring for this function.

Imports:
{chr(10).join(context['imports'])}

{'Class: ' + context['class_context'] if context['class_context'] else ''}

Function:
{context['function_code']}

Provide a docstring that explains purpose, algorithm, parameters, return value, and any domain-specific considerations."""
```

**Trade-off:** More context improves quality but increases latency and cost. For batch processing 1000 functions, tight context control is essential.

### 2. Prompt Engineering for Documentation Style

**Technical Explanation:**

Generic prompts produce generic comments. Effective prompts specify documentation standards (Google style, NumPy style), desired detail level, and domain terminology. The prompt acts as a style guide that the LLM follows.

**Practical Implications:**

Teams with established documentation standards can encode them in prompts, ensuring LLM-generated comments match hand-written ones. This makes automated documentation indistinguishable from human-written docs.

**Real Constraints:**

- Over-specifying format can reduce semantic quality (LLM focuses on structure over meaning)
- Domain jargon must be defined in the prompt or context; LLMs hallucinate technical terms
- Different LLMs respond to formatting instructions differently (few-shot examples work universally)

**Concrete Example:**

```python
from typing import Literal

def build_documentation_prompt(
    code: str,
    style: Literal["google", "numpy", "sphinx"] = "google",
    detail_level: Literal["brief", "comprehensive"] = "comprehensive",
    domain_terms: dict[str, str] | None = None
) -> str:
    """
    Constructs a prompt that generates style-consistent documentation.
    """
    
    style_examples = {
        "google": '''
Example Google-style docstring:
"""
Brief one-line summary.

Detailed explanation of behavior, algorithm, and edge cases.
Include information about complexity, side effects, or domain context.

Args:
    param_name: Description including type constraints, expected ranges,
        and any validation requirements.

Returns:
    Description of return value including type and possible values.

Raises:
    ExceptionType: When and why this exception occurs.
"""''',
        "numpy": '''
Example NumPy-style docstring:
"""
Brief one-line summary.

Detailed explanation with algorithm description and use cases.

Parameters
----------
param_name : type
    Description with constraints and expected behavior.

Returns
-------
type
    Description of returned value and possible states.
"""'''
    }
    
    domain_context = ""
    if domain_terms:
        domain_context = "\nDomain Terminology:\n"
        domain_context += "\n".join(f"- {term}: {definition}" 
                                     for term, definition in domain_terms.items())
    
    detail_instructions = {
        "brief": "Provide concise documentation focusing on purpose and interface.",
        "comprehensive": "Provide detailed documentation including algorithm explanation, "
                        "edge cases, performance considerations, and usage examples."
    }
    
    return f"""Generate a {style}-style docstring for this Python function.

{style_examples[style]}

Requirements:
- {detail_instructions[detail_level]}
- Explain WHY the code exists, not just WHAT it does
- Document any magic numbers or non-obvious logic
- Include type information and expected value ranges
{domain_context}

Code to document:
```python
{code}
```

Return ONLY the docstring content (without the triple quotes)."""

# Example with domain terms
domain = {
    "risk score": "Normalized metric [0-1] representing fraud probability",
    "threshold": "Minimum transaction amount requiring additional scrutiny"
}

prompt = build_documentation_prompt(
    code="""def calculate_risk_score(transactions, threshold):
    total = sum(t['amount'] for t in transactions if t['amount'] > threshold)
    return min(total / 10000, 1.0)""",
    style="google",
    detail_level="comprehensive",
    domain_terms=domain
)
```

**Trade-off:** Longer prompts with examples increase cost per function but can improve consistency by 25-40% in blind evaluations.

### 3. Validation and Quality Control

**Technical Explanation:**

LLM outputs require validation because they can hallucinate parameters, misinterpret logic, or generate outdated references. Automated validation checks structural correctness, parameter alignment, and semantic consistency.

**Practical Implications:**

A validation pipeline catches errors before human review, reducing review time by 60-70%. Engineers only examine flagged outputs, not every generated comment.

**Real Constraints:**

- Semantic validation (does comment match code intent?) requires heuristics or second LLM call
- False positives waste reviewer time; tuning thresholds requires calibration datasets
- Validation adds latency; batch processing needs async validation

**Concrete Example:**

```python
import ast
import difflib
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    confidence: float

def validate_generated_docstring(
    function_code: str,
    generated_docstring: str
) -> ValidationResult:
    """
    Validates LLM-generated docstring against function signature.
    
    Checks parameter alignment, return type documentation, and
    basic semantic consistency. Returns validation result with
    specific error messages for human review.
    """
    errors = []
    warnings = []
    
    # Parse function to extract signature
    try:
        tree = ast.parse(function_code)
        func = next(node for node in ast.walk(tree) 
                   if isinstance(node, ast.FunctionDef))
    except Exception as e:
        return ValidationResult(False, [f"Failed to parse code: {e}"], [], 0.0)
    
    # Extract parameter names from code
    code_params = {arg.arg for arg in func.args.args}
    
    # Extract parameter names from docstring (simple regex approach)
    import re
    doc_params = set(re.findall(r'(?:Args?|Parameters?):\s*\n\s*(\w+)', 
                                 generated_docstring, re.MULTILINE))
    # More robust: parse Google/NumPy style specifically
    param_pattern = r'^\s*(\w+)\s*[:\(]'
    in_params_section = False
    for line in generated_docstring.split('\n'):
        if re.match(r'^\s*(Args?|Parameters?):', line):
            in_params_section = True
            continue
        if in_params_section:
            if re.match(r'^\s*(Returns?|Raises?|Examples?):', line):
                break
            match = re.match(param_pattern, line)
            if match:
                doc_params.add(match.group(1))
    
    # Check parameter alignment
    missing_in_doc = code_params - doc_params
    extra_in_doc = doc_params - code_params
    
    if missing_in_doc:
        errors.append(f"Parameters in code but not documented: {missing_in_doc}")
    if extra_in_doc:
        errors.append(f"Parameters documented but not in code: {extra_in_doc}")
    
    # Check return documentation
    has_return = func.returns is not None or any(
        isinstance(node, ast.Return) and node.value is not None 
        for node in ast.walk(func)
    )
    has_return_doc = bool(re.search(r'Returns?:', generated_docstring))
    
    if has_return and not has_return_doc:
        warnings.append("Function returns value but docstring has no Returns section")
    elif not has_return and has_return_doc:
        warnings.append("Docstring documents return but function returns None")
    
    # Check for hallucinated technical terms
    suspicious_terms = ["quantum", "blockchain", "AI-powered", "revolutionary"]
    found_suspicious = [term for term in suspicious_terms 
                       if term.lower() in generated_docstring.lower()]
    if found_suspicious:
        warnings.append(f"Potentially hallucinated terms: {found_suspicious}")
    
    # Calculate confidence score
    confidence = 1.0
    confidence -= 0.3 * len(errors)
    confidence -= 0.1 * len(warnings)
    confidence = max(0.0, confidence)
    
    is_valid = len(errors) == 0 and confidence > 0.6
    
    return ValidationResult(is_valid, errors, warnings, confidence)

# Example usage
code = """def calculate_risk_score(transactions: list[dict], threshold: float) -> float:
    total = sum(t['amount'] for t in transactions if t['amount'] > threshold)
    return min(total / 10000, 1.0)"""

docstring = """Calculates risk score based on transaction amounts.

Args:
    transactions: List of transaction dictionaries
    threshold: Minimum amount threshold
    account_id: Account identifier

Returns:
    Risk score between 0 and 1"""

result = validate_generated_docstring(code, docstring)
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")
print(f"Confidence: {result.confidence:.2f}")

# Output:
# Valid: False
# Errors: ['Parameters documented but not in code: {\'account_id\'}']
# Warnings: []
# Confidence: 0.70
```

**Trade-off:** Validation catches 80-90% of hallucinations but cannot detect subtle semantic errors (e.g., claiming an algorithm is O(n) when it's O(n²)).

### 4. Batch Processing and Caching

**Technical Explanation:**

Processing thousands of functions requires batching requests, caching results, and implementing retry logic. LLM APIs have rate limits (requests/minute) and token limits (tokens/request), requiring queue management.

**Practical Implications:**

Naive sequential processing of 5000 functions at 2s/request takes 2.8 hours. Batching with concurrency and caching reduces this to 15-20 minutes.

**Real Constraints:**

- Rate limits vary by API tier (60 req/min to 3500 req/min)
- Caching requires stable code hashing to detect changes
- Partial failures need idempotent retry logic

**Concrete Example:**

```python
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Callable, Any
import aiohttp

class DocumentationGenerator:
    """
    