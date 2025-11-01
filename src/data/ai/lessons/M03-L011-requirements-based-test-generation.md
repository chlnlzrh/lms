# Requirements-Based Test Generation

## Core Concepts

Requirements-based test generation uses LLMs to automatically create test cases, test data, and validation logic from requirements documents, user stories, or API specifications. Instead of manually translating requirements into test scenarios—a process that's both time-consuming and error-prone—you leverage language models to parse specifications and generate comprehensive test coverage.

### Traditional vs. Modern Approach

**Traditional test generation:**

```python
# Manual process: Read requirement, write tests by hand
# Requirement: "User API should validate email format and return 400 for invalid emails"

import unittest
from api import create_user

class TestUserAPI(unittest.TestCase):
    def test_valid_email(self):
        response = create_user({"email": "user@example.com"})
        self.assertEqual(response.status_code, 201)
    
    def test_invalid_email_no_at(self):
        response = create_user({"email": "userexample.com"})
        self.assertEqual(response.status_code, 400)
    
    # Engineer manually thinks of edge cases...
    # May miss: empty string, multiple @, no domain, etc.
```

**LLM-assisted requirements-based generation:**

```python
import anthropic
from typing import List, Dict
import json

def generate_tests_from_requirement(requirement: str) -> List[Dict]:
    """
    Generate comprehensive test cases from a text requirement.
    Returns structured test cases with inputs, expected outputs, and rationale.
    """
    client = anthropic.Anthropic()
    
    prompt = f"""Given this API requirement:

{requirement}

Generate comprehensive test cases covering:
1. Happy path scenarios
2. Edge cases
3. Invalid inputs
4. Boundary conditions

Return JSON array with format:
[
  {{
    "test_name": "descriptive_name",
    "input": {{"field": "value"}},
    "expected_status": 201,
    "expected_response": {{}},
    "rationale": "why this case matters"
  }}
]"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract JSON from response
    content = response.content[0].text
    # Find JSON array in response
    start = content.find('[')
    end = content.rfind(']') + 1
    test_cases = json.loads(content[start:end])
    
    return test_cases

# Usage
requirement = """
User creation endpoint POST /api/users
- Accepts JSON with 'email' and 'name' fields
- Email must be valid format (contains @, has domain)
- Name must be 2-50 characters
- Returns 201 on success with user object
- Returns 400 for validation failures with error details
"""

tests = generate_tests_from_requirement(requirement)
print(f"Generated {len(tests)} test cases")
# Output: Generated 12 test cases
# Includes cases you might not have thought of:
# - Email with multiple dots, Unicode characters in name,
# - SQL injection attempts, XSS in name field, etc.
```

The LLM approach generates 3-5x more test cases in seconds, including edge cases that manual testers often miss. The key shift: you move from test **creation** to test **curation**—reviewing and selecting from comprehensive generated tests rather than inventing each one.

### Key Engineering Insights

**1. Completeness vs. Relevance Trade-off**: LLMs generate exhaustive test scenarios, but not all are valuable for your context. A payment API needs different coverage than an internal dashboard. You'll spend less time creating tests but more time evaluating which tests matter.

**2. Requirement Precision Amplification**: LLMs expose ambiguities in requirements ruthlessly. When a requirement says "validate email," does that include internationalized domains? Disposable email providers? The generation process forces requirement clarification earlier in the development cycle.

**3. Test Data Generation is the Hidden Value**: Beyond test structure, LLMs excel at generating realistic, diverse test data—names from different cultures, edge-case timestamps, malformed JSON that still parses—scenarios that expose bugs manual testing misses.

### Why This Matters Now

Test coverage directly correlates with production reliability, but comprehensive testing is expensive. Requirements-based generation makes exhaustive testing economically viable:

- **Regression testing**: Generate tests for legacy code from documentation
- **API contract testing**: Auto-generate tests from OpenAPI specs
- **Refactoring safety**: Quickly create test harnesses before modifying code
- **Documentation validation**: Ensure docs match actual behavior

With test generation, you shift testing left without increasing developer burden. Requirements change? Regenerate tests in minutes, not days.

## Technical Components

### 1. Requirement Parsing and Structuring

LLMs parse unstructured requirements (user stories, PRDs, technical specs) into structured test parameters. This is fundamentally a semantic extraction problem.

**Technical Explanation**: The LLM identifies entities (inputs, outputs, conditions, constraints) and relationships (validation rules, dependencies, state transitions). It maps free-form text to test case schema.

**Practical Implementation**:

```python
from typing import TypedDict, List, Optional
from anthropic import Anthropic

class TestCase(TypedDict):
    scenario: str
    preconditions: List[str]
    input_data: dict
    expected_output: dict
    expected_status: int
    test_category: str  # "happy_path", "edge_case", "security", "error_handling"

def parse_requirement_to_structured_tests(
    requirement: str,
    context: Optional[str] = None
) -> List[TestCase]:
    """
    Parse requirement into structured test cases with explicit categories.
    """
    client = Anthropic()
    
    system_prompt = """You are a test engineer parsing requirements into test cases.
Extract:
- All input parameters and their constraints
- Expected outputs for each scenario
- Edge cases and boundary conditions
- Error conditions
- Security considerations (injection, overflow, etc.)

Return valid JSON only, no markdown."""

    user_prompt = f"""Requirement:
{requirement}

{f"Additional Context: {context}" if context else ""}

Generate test cases in this exact JSON structure:
[
  {{
    "scenario": "brief description",
    "preconditions": ["state or setup required"],
    "input_data": {{"param": "value"}},
    "expected_output": {{"field": "value"}},
    "expected_status": 200,
    "test_category": "happy_path|edge_case|security|error_handling"
  }}
]"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8000,
        temperature=0.3,  # Lower temperature for more consistent structure
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    content = response.content[0].text
    # Robust JSON extraction
    start = content.find('[')
    end = content.rfind(']') + 1
    if start == -1 or end == 0:
        raise ValueError("No valid JSON array in response")
    
    tests = json.loads(content[start:end])
    return tests
```

**Constraints and Trade-offs**:
- **Ambiguity handling**: LLMs make assumptions when requirements are vague. You must validate generated tests against intent.
- **Context window limits**: Complex requirements may exceed token limits. Break into smaller chunks or use summarization.
- **Consistency**: Same requirement may generate different tests across runs. Use temperature=0 and explicit schemas for reproducibility.

**Example**:

```python
requirement = """
Password reset endpoint POST /api/auth/reset-password
Input: email address
Behavior:
- If email exists: send reset link, return 200
- If email doesn't exist: return 200 (don't leak user existence)
- Rate limit: max 3 requests per email per hour
- Reset links expire after 1 hour
"""

tests = parse_requirement_to_structured_tests(requirement)
# Generates:
# - Happy path: valid email, reset sent
# - Security: email enumeration prevention verified
# - Rate limiting: 4th request blocked
# - Timing: reset link invalid after 61 minutes
# - Edge cases: malformed emails, SQL injection attempts
```

### 2. Test Data Synthesis

Generating realistic, diverse test data is where LLMs provide asymmetric value. They create contextually appropriate data that exposes real-world issues.

**Technical Explanation**: LLMs generate data that respects semantic constraints (valid credit card checksums, realistic addresses) while introducing variation that manual testers skip (Unicode edge cases, boundary values, cultural diversity).

**Practical Implementation**:

```python
from typing import Literal
import json

def generate_test_data(
    field_name: str,
    field_type: str,
    constraints: dict,
    count: int = 10,
    include_invalid: bool = True
) -> dict:
    """
    Generate diverse test data for a specific field.
    
    Args:
        field_name: Semantic name (e.g., "email", "phone_number")
        field_type: Data type (string, int, date, etc.)
        constraints: Validation rules (min_length, max_length, pattern, etc.)
        count: Number of valid examples
        include_invalid: Whether to generate invalid examples
    """
    client = Anthropic()
    
    prompt = f"""Generate test data for field: {field_name}
Type: {field_type}
Constraints: {json.dumps(constraints)}

Generate {count} VALID examples and {count if include_invalid else 0} INVALID examples.

Include diverse cases:
- Boundary values (min, max, just over/under limits)
- Cultural diversity (international names, addresses)
- Special characters (Unicode, emoji, escape sequences)
- Common real-world variations
- Edge cases that often cause bugs

Return JSON:
{{
  "valid": ["example1", "example2", ...],
  "invalid": ["bad1", "bad2", ...],
  "explanations": {{"example": "why this is interesting"}}
}}"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=0.7,  # Higher temp for more diversity
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.content[0].text
    start = content.find('{')
    end = content.rfind('}') + 1
    return json.loads(content[start:end])

# Example usage
email_data = generate_test_data(
    field_name="user_email",
    field_type="string",
    constraints={
        "format": "email",
        "max_length": 255,
        "required": True
    },
    count=15
)

print(f"Valid emails: {len(email_data['valid'])}")
# Output includes:
# - Standard: user@example.com
# - Subdomain: user@mail.company.co.uk
# - Plus addressing: user+tag@example.com
# - International: user@例え.jp
# - Long local part: very.long.email.address@example.com
# Invalid includes:
# - Missing @, double @@, spaces, too long, etc.
```

**Constraints and Trade-offs**:
- **Validation coverage**: LLMs understand common formats but may miss domain-specific validation rules. Always verify generated invalid data actually fails your validation.
- **Randomness vs. reproducibility**: Higher temperature creates diversity but reduces repeatability. Use seeds or explicit examples for CI/CD.
- **Cost**: Generating large datasets token-by-token is expensive. Cache commonly used data, generate in batches.

### 3. Test Code Generation

Converting structured test cases into executable code—the final mile that determines practical value.

**Technical Explanation**: LLMs translate test case specifications into framework-specific code (pytest, Jest, JUnit) with proper assertions, setup/teardown, mocking, and error handling.

**Practical Implementation**:

```python
def generate_executable_tests(
    test_cases: List[TestCase],
    framework: Literal["pytest", "unittest", "jest"] = "pytest",
    api_client_name: str = "api_client"
) -> str:
    """
    Generate executable test code from structured test cases.
    
    Returns complete, runnable test file with imports and fixtures.
    """
    client = Anthropic()
    
    test_cases_json = json.dumps(test_cases, indent=2)
    
    prompt = f"""Generate {framework} test code for these test cases:

{test_cases_json}

Requirements:
1. Complete, runnable code with all imports
2. Use {api_client_name} for API calls (assume it's provided as fixture)
3. Proper assertions for expected outputs and status codes
4. Descriptive test names and docstrings
5. Handle preconditions with fixtures or setup
6. Group related tests into classes
7. Include type hints

Return ONLY the Python code, no markdown or explanation."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8000,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    
    code = response.content[0].text
    # Clean up any markdown formatting
    code = code.replace("```python", "").replace("```", "").strip()
    return code

# Example usage
test_cases = [
    {
        "scenario": "Valid user creation",
        "preconditions": ["Database is empty"],
        "input_data": {"email": "test@example.com", "name": "Test User"},
        "expected_output": {"id": "any", "email": "test@example.com"},
        "expected_status": 201,
        "test_category": "happy_path"
    },
    {
        "scenario": "Duplicate email rejection",
        "preconditions": ["User with email exists"],
        "input_data": {"email": "existing@example.com", "name": "Another User"},
        "expected_output": {"error": "Email already exists"},
        "expected_status": 409,
        "test_category": "error_handling"
    }
]

test_code = generate_executable_tests(test_cases, framework="pytest")
print(test_code)
```

**Generated output** (example):

```python
import pytest
from typing import Dict, Any

@pytest.fixture
def clean_database(api_client):
    """Ensure database is clean before tests."""
    api_client.reset_database()
    yield
    api_client.reset_database()

class TestUserCreation:
    """Test cases for user creation endpoint."""
    
    def test_valid_user_creation(self, api_client, clean_database):
        """
        Verify successful user creation with valid inputs.
        Expected: 201 status with user object including generated ID.
        """
        response = api_client.post(
            "/api/users",
            json={"email": "test@example.com", "name": "Test User"}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["email"] == "test@example.com"
        assert data["name"] == "Test User"
    
    def test_duplicate_email_rejection(self, api_client, clean_database):
        """
        Verify duplicate email addresses are rejected.
        Expected: 409 status with error message.
        """
        # Create initial user
        api_client.post(
            "/api/users",
            json={"email": "existing@example.com", "name": "First User"}
        )
        
        # Attempt duplicate
        response = api_client.post(
            "/api/users",
            json={"email": "existing@example.com", "name": "Another User"}
        )
        
        assert response.status_code == 409
        data = response.json()
        assert "error" in data
        assert "already exists" in data["error"].lower()
```

**Constraints and Trade-offs**:
- **Framework specifics**: Generated code may not follow your team's conventions. Use few-shot examples in the prompt with your style.
- **Complex setup**: Multi-step preconditions (database seeds, auth tokens) may need manual refinement.
- **Assertion specificity**: LLMs may use loose assertions (e.g., `assert "error" in response`). Review and tighten for production.

### 4. Requirement Coverage Analysis

Analyzing whether generated tests adequately cover all requirement aspects—the