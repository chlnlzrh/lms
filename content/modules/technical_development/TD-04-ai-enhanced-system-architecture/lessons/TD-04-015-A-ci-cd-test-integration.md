# CI/CD Test Integration for LLM Applications

## Core Concepts

Testing LLM-powered applications in continuous integration pipelines requires fundamentally different approaches than traditional software testing. Unlike deterministic functions that return identical outputs for identical inputs, LLMs introduce controlled non-determinism that demands new testing methodologies, evaluation metrics, and quality gates.

### Traditional vs. LLM Testing Architecture

```python
# Traditional deterministic testing
def calculate_tax(income: float, rate: float) -> float:
    """Pure function with deterministic output."""
    return income * rate

def test_calculate_tax():
    result = calculate_tax(100000, 0.25)
    assert result == 25000.0  # Exact equality works
    assert result == calculate_tax(100000, 0.25)  # Idempotent


# LLM-powered testing requires statistical validation
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib

@dataclass
class EvaluationResult:
    input_hash: str
    output: str
    latency_ms: float
    token_count: int
    passed_criteria: Dict[str, bool]
    confidence_score: float

async def llm_extract_entities(text: str, model_config: Dict) -> str:
    """Non-deterministic function requiring statistical validation."""
    # Simulated LLM call
    await asyncio.sleep(0.1)
    return '{"person": "John", "location": "Paris"}'

async def test_llm_extract_entities():
    """Test using statistical validation over multiple runs."""
    test_input = "John traveled to Paris last summer."
    results: List[EvaluationResult] = []
    
    # Run multiple times to assess consistency
    for i in range(5):
        start = asyncio.get_event_loop().time()
        output = await llm_extract_entities(test_input, {"temperature": 0.0})
        latency = (asyncio.get_event_loop().time() - start) * 1000
        
        # Validate semantic correctness, not exact string match
        criteria = {
            "contains_person": "person" in output.lower(),
            "contains_location": "location" in output.lower(),
            "valid_json": is_valid_json(output),
            "within_latency_sla": latency < 500
        }
        
        results.append(EvaluationResult(
            input_hash=hashlib.sha256(test_input.encode()).hexdigest()[:8],
            output=output,
            latency_ms=latency,
            token_count=len(output.split()),
            passed_criteria=criteria,
            confidence_score=sum(criteria.values()) / len(criteria)
        ))
    
    # Assert statistical properties
    avg_confidence = sum(r.confidence_score for r in results) / len(results)
    assert avg_confidence >= 0.90, f"Average confidence {avg_confidence} below threshold"
    
    # Assert consistency (outputs should be similar)
    unique_outputs = len(set(r.output for r in results))
    assert unique_outputs <= 2, f"Too much variation: {unique_outputs} unique outputs"

def is_valid_json(s: str) -> bool:
    import json
    try:
        json.loads(s)
        return True
    except:
        return False
```

### Engineering Insights That Change Your Approach

1. **Shift from Exact Assertions to Quality Boundaries**: Traditional `assert result == expected` fails with LLMs. Instead, define acceptable quality ranges using semantic similarity, rubric-based scoring, and statistical thresholds.

2. **Regression Detection Requires Baseline Distributions**: Store not just expected outputs, but distributions of quality metrics from previous runs. Regressions manifest as statistical shifts, not exact mismatches.

3. **Flaky Tests Are Features, Not Bugs**: Some output variation is expected. The goal is to quantify and bound that variation, not eliminate it. A test that passes 95% of the time might be functioning correctly.

4. **Cost and Latency Are First-Class Quality Metrics**: Unlike traditional tests where execution cost is negligible, each LLM test run costs real money and time. Build cost-aware test suites that maximize signal per dollar spent.

### Why This Matters Now

Production LLM applications fail in ways that only emerge at scale: cumulative costs spiral, latencies hit p99 outliers, prompt injections bypass validation, and model updates silently degrade performance. Without robust CI/CD integration, these issues reach production. The industry is rapidly consolidating around patterns that work—implementing them now prevents costly rewrites later.

## Technical Components

### 1. Evaluation Metrics Framework

LLM testing requires multiple orthogonal metrics evaluated together to form a comprehensive quality signal.

```python
from typing import Protocol, List, Tuple
from dataclasses import dataclass
from enum import Enum
import re

class MetricType(Enum):
    SEMANTIC_SIMILARITY = "semantic"
    STRUCTURAL_VALIDITY = "structural"
    SAFETY_COMPLIANCE = "safety"
    PERFORMANCE = "performance"
    COST_EFFICIENCY = "cost"

@dataclass
class MetricResult:
    metric_type: MetricType
    score: float  # 0.0 to 1.0
    passed: bool
    threshold: float
    metadata: Dict[str, Any]

class EvaluationMetric(Protocol):
    """Protocol for evaluation metrics."""
    def evaluate(self, input_text: str, output_text: str, 
                 expected: str, metadata: Dict) -> MetricResult:
        ...

class SemanticSimilarityMetric:
    """Evaluate semantic similarity using embedding distance."""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
    
    def evaluate(self, input_text: str, output_text: str, 
                 expected: str, metadata: Dict) -> MetricResult:
        # In production, use actual embeddings
        # This simulates cosine similarity
        similarity = self._compute_similarity(output_text, expected)
        
        return MetricResult(
            metric_type=MetricType.SEMANTIC_SIMILARITY,
            score=similarity,
            passed=similarity >= self.threshold,
            threshold=self.threshold,
            metadata={"method": "cosine_similarity"}
        )
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Simplified similarity for demonstration."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

class StructuralValidityMetric:
    """Validate output structure matches expected format."""
    
    def __init__(self, format_type: str, threshold: float = 1.0):
        self.format_type = format_type
        self.threshold = threshold
    
    def evaluate(self, input_text: str, output_text: str, 
                 expected: str, metadata: Dict) -> MetricResult:
        validators = {
            "json": self._validate_json,
            "markdown": self._validate_markdown,
            "code": self._validate_code
        }
        
        validator = validators.get(self.format_type, lambda x: (False, {}))
        is_valid, details = validator(output_text)
        
        return MetricResult(
            metric_type=MetricType.STRUCTURAL_VALIDITY,
            score=1.0 if is_valid else 0.0,
            passed=is_valid,
            threshold=self.threshold,
            metadata=details
        )
    
    def _validate_json(self, text: str) -> Tuple[bool, Dict]:
        import json
        try:
            parsed = json.loads(text)
            return True, {"keys": list(parsed.keys()) if isinstance(parsed, dict) else []}
        except json.JSONDecodeError as e:
            return False, {"error": str(e)}
    
    def _validate_markdown(self, text: str) -> Tuple[bool, Dict]:
        has_headers = bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))
        return has_headers, {"has_headers": has_headers}
    
    def _validate_code(self, text: str) -> Tuple[bool, Dict]:
        # Check if code block is present and syntactically valid
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
        if not code_blocks:
            return False, {"error": "No code blocks found"}
        
        try:
            compile(code_blocks[0], '<string>', 'exec')
            return True, {"blocks_count": len(code_blocks)}
        except SyntaxError as e:
            return False, {"error": str(e)}

class SafetyComplianceMetric:
    """Check for safety issues, PII leakage, prompt injection."""
    
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email"),
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', "credit_card")
        ]
    
    def evaluate(self, input_text: str, output_text: str, 
                 expected: str, metadata: Dict) -> MetricResult:
        violations = []
        
        # Check for PII leakage
        for pattern, pii_type in self.pii_patterns:
            if re.search(pattern, output_text):
                violations.append(f"PII_{pii_type}")
        
        # Check for prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard all prior",
            "system:",
        ]
        for pattern in injection_patterns:
            if pattern.lower() in output_text.lower():
                violations.append("prompt_injection")
        
        passed = len(violations) == 0
        
        return MetricResult(
            metric_type=MetricType.SAFETY_COMPLIANCE,
            score=1.0 if passed else 0.0,
            passed=passed,
            threshold=self.threshold,
            metadata={"violations": violations}
        )

class PerformanceMetric:
    """Evaluate latency and throughput characteristics."""
    
    def __init__(self, max_latency_ms: float = 1000.0):
        self.max_latency_ms = max_latency_ms
    
    def evaluate(self, input_text: str, output_text: str, 
                 expected: str, metadata: Dict) -> MetricResult:
        latency = metadata.get("latency_ms", 0.0)
        passed = latency <= self.max_latency_ms
        
        return MetricResult(
            metric_type=MetricType.PERFORMANCE,
            score=max(0.0, 1.0 - (latency / self.max_latency_ms)),
            passed=passed,
            threshold=1.0,
            metadata={
                "latency_ms": latency,
                "max_latency_ms": self.max_latency_ms
            }
        )
```

**Practical Implications**: Combine multiple metrics to catch different failure modes. Semantic similarity catches content drift, structural validity catches format breaks, safety catches security issues, and performance catches latency regressions.

**Real Constraints**: Each metric adds test execution time and complexity. Start with 3-4 critical metrics and expand based on observed failure patterns. Semantic similarity metrics requiring embeddings add ~50-100ms per evaluation.

### 2. Test Case Management and Versioning

```python
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import hashlib

@dataclass
class TestCase:
    id: str
    input_text: str
    expected_output: str
    metadata: Dict[str, Any]
    tags: List[str]
    created_at: str
    version: int

@dataclass
class TestCaseVersion:
    test_case_id: str
    version: int
    changes: Dict[str, Any]
    changed_by: str
    changed_at: str
    reason: str

class TestSuiteManager:
    """Manage versioned test cases with git-like semantics."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.test_cases_path = base_path / "test_cases"
        self.history_path = base_path / "history"
        self.test_cases_path.mkdir(parents=True, exist_ok=True)
        self.history_path.mkdir(parents=True, exist_ok=True)
    
    def add_test_case(self, input_text: str, expected_output: str,
                      tags: List[str], metadata: Optional[Dict] = None) -> TestCase:
        """Add a new test case with automatic versioning."""
        test_id = self._generate_id(input_text)
        
        test_case = TestCase(
            id=test_id,
            input_text=input_text,
            expected_output=expected_output,
            metadata=metadata or {},
            tags=tags,
            created_at=datetime.utcnow().isoformat(),
            version=1
        )
        
        self._save_test_case(test_case)
        return test_case
    
    def update_test_case(self, test_id: str, changes: Dict[str, Any],
                        reason: str, changed_by: str) -> TestCase:
        """Update test case and record version history."""
        test_case = self.load_test_case(test_id)
        
        # Record version history
        version_record = TestCaseVersion(
            test_case_id=test_id,
            version=test_case.version,
            changes=changes,
            changed_by=changed_by,
            changed_at=datetime.utcnow().isoformat(),
            reason=reason
        )
        self._save_version_history(version_record)
        
        # Apply changes
        for key, value in changes.items():
            if hasattr(test_case, key):
                setattr(test_case, key, value)
        
        test_case.version += 1
        self._save_test_case(test_case)
        
        return test_case
    
    def load_test_cases(self, tags: Optional[List[str]] = None) -> List[TestCase]:
        """Load test cases, optionally filtered by tags."""
        test_cases = []
        
        for file_path in self.test_cases_path.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
                test_case = TestCase(**data)
                
                if tags is None or any(tag in test_case.tags for tag in tags):
                    test_cases.append(test_case)
        
        return test_cases
    
    def load_test_case(self, test_id: str) -> TestCase:
        """Load specific test case by ID."""
        file_path = self.test_cases_path / f"{test_id}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
            return TestCase(**data)
    
    def get_history(self, test_id: str) -> List[TestCaseVersion]:
        """Get version history for a test case."""
        history_file = self.history_path / f"{test_id}_history.json"
        
        if not history_file.exists():
            return []
        
        with open(history_file, 'r') as f:
            history_data = json.load(f)
            return [TestCaseVersion(**record) for record in history_data]
    
    def _generate_id(self, input_text: str) -> str:
        """Generate unique ID from input text."""
        hash_obj = hashlib.sha256(input