# Test Plan Documentation for AI Systems

## Core Concepts

Test plan documentation for AI systems represents a fundamental shift from traditional software testing paradigms. Where conventional software follows deterministic paths—the same input reliably produces the same output—AI systems introduce probabilistic behavior that demands new documentation strategies.

**Traditional vs. AI Testing Documentation:**

```python
# Traditional Software Testing
def calculate_tax(income: float, rate: float) -> float:
    """Tax calculation - deterministic output."""
    return income * rate

# Test Plan: Verify output equals income * rate
# Expected: calculate_tax(100000, 0.25) == 25000.0
# Result: Pass/Fail (binary)

# AI System Testing
import anthropic

def extract_invoice_data(invoice_text: str) -> dict:
    """Extract structured data from invoice - probabilistic output."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Extract vendor, date, amount, items from:\n{invoice_text}"
        }]
    )
    # Output varies: phrasing differs, format may shift, edge cases unpredictable
    return response.content

# Test Plan: Define acceptable variance boundaries
# Expected: Correct vendor in 98% of cases, amount exact in 99.5% cases
# Result: Statistical distribution with confidence intervals
```

The key insight: **AI test plans document acceptable ranges of behavior rather than exact expected outputs**. You're testing whether the system performs within operational boundaries, not whether it matches a predetermined string.

### Why This Matters NOW

Production AI systems fail differently than traditional software. A database query either returns results or throws an error. An LLM might return plausible-sounding but completely fabricated data, subtly misinterpret instructions, or degrade performance on edge cases you never anticipated. Without rigorous test documentation that captures these failure modes, you're deploying systems you can't reliably maintain or improve.

Three critical realities drive the need for robust AI test documentation:

1. **Non-determinism requires statistical validation**: You need frameworks to measure consistency across runs
2. **Emergent behavior demands comprehensive scenario coverage**: Models behave unexpectedly on unanticipated inputs
3. **Model updates break existing functionality**: Provider model changes can silently degrade your system's performance

Engineers who master AI test plan documentation can confidently deploy, monitor, and iterate on AI systems. Those who don't find themselves fighting production fires with no systematic way to diagnose or prevent failures.

## Technical Components

### 1. Test Case Structure for Probabilistic Systems

Traditional test cases specify exact inputs and expected outputs. AI test cases must define:

- **Input specification**: The prompt/context provided
- **Expected behavior range**: Acceptable output boundaries, not exact matches
- **Success criteria**: Measurable thresholds (accuracy, format compliance, latency)
- **Failure modes**: Specific unacceptable behaviors to detect

```python
from typing import TypedDict, Literal
from dataclasses import dataclass
from enum import Enum

class TestResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    DEGRADED = "degraded"  # Works but below optimal thresholds

@dataclass
class AITestCase:
    test_id: str
    category: str  # "extraction", "classification", "generation", etc.
    input_data: str
    expected_behavior: dict  # Defines acceptable ranges
    success_criteria: dict  # Measurable thresholds
    failure_patterns: list[str]  # Known bad behaviors
    
    def evaluate(self, actual_output: str) -> TestResult:
        """Evaluate if output meets success criteria."""
        pass  # Implementation depends on test type

# Example: Document classification test
doc_classification_test = AITestCase(
    test_id="classify_001",
    category="classification",
    input_data="Q3 revenue increased 23% YoY driven by enterprise segment...",
    expected_behavior={
        "document_type": ["earnings_report", "financial_statement"],
        "confidence_min": 0.85,
        "reasoning_required": True
    },
    success_criteria={
        "correct_category_rate": 0.95,  # 95% accuracy required
        "avg_confidence": 0.90,
        "max_latency_ms": 2000
    },
    failure_patterns=[
        "classifies as 'unknown' or 'other'",
        "provides category without confidence score",
        "hallucinates information not in input"
    ]
)
```

**Practical Implications**: This structure forces you to explicitly define what "good enough" means before testing. You'll catch ambiguous requirements early—if you can't specify acceptable behavior ranges, your system requirements aren't clear enough to build.

**Real Constraints**: Defining expected behavior ranges requires domain expertise. You need subject matter knowledge to determine if 95% accuracy is achievable/necessary, or if 85% is acceptable. This upfront work pays dividends when debugging production issues.

### 2. Golden Dataset Construction

Golden datasets serve as your ground truth for evaluating AI system behavior. Unlike traditional test data that validates logic paths, AI golden datasets must capture:

- **Representative distribution**: Real-world input variety
- **Edge cases**: Unusual but valid inputs that expose model limitations
- **Adversarial examples**: Inputs designed to trigger known failure modes
- **Temporal stability markers**: Samples to detect performance drift over time

```python
import json
from datetime import datetime
from pathlib import Path

class GoldenDatasetEntry:
    def __init__(
        self,
        input_text: str,
        expected_output: dict,
        metadata: dict
    ):
        self.input_text = input_text
        self.expected_output = expected_output
        self.metadata = metadata
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "input": self.input_text,
            "expected": self.expected_output,
            "metadata": self.metadata,
            "created_at": self.created_at
        }

class GoldenDataset:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.entries: list[GoldenDatasetEntry] = []
        self.load()
    
    def load(self):
        """Load existing golden dataset."""
        if self.dataset_path.exists():
            with open(self.dataset_path, 'r') as f:
                data = json.load(f)
                for entry in data:
                    self.entries.append(GoldenDatasetEntry(
                        entry['input'],
                        entry['expected'],
                        entry['metadata']
                    ))
    
    def add_entry(
        self,
        input_text: str,
        expected_output: dict,
        category: str,
        difficulty: str = "standard"
    ):
        """Add new golden dataset entry."""
        entry = GoldenDatasetEntry(
            input_text=input_text,
            expected_output=expected_output,
            metadata={
                "category": category,
                "difficulty": difficulty,
                "source": "manual_curation"
            }
        )
        self.entries.append(entry)
    
    def save(self):
        """Persist golden dataset."""
        with open(self.dataset_path, 'w') as f:
            json.dump(
                [e.to_dict() for e in self.entries],
                f,
                indent=2
            )
    
    def get_by_category(self, category: str) -> list[GoldenDatasetEntry]:
        """Retrieve entries by category."""
        return [e for e in self.entries if e.metadata['category'] == category]

# Usage: Building a golden dataset for invoice extraction
dataset = GoldenDataset(Path("golden_invoices.json"))

# Standard case
dataset.add_entry(
    input_text="Invoice #1234\nDate: 2024-01-15\nTotal: $450.00\nVendor: Acme Corp",
    expected_output={
        "invoice_number": "1234",
        "date": "2024-01-15",
        "total": 450.00,
        "vendor": "Acme Corp"
    },
    category="standard_format"
)

# Edge case: Missing invoice number
dataset.add_entry(
    input_text="Receipt\nDate: Jan 15, 2024\nAmount Due: $450\nFrom: Acme Corp",
    expected_output={
        "invoice_number": None,
        "date": "2024-01-15",
        "total": 450.00,
        "vendor": "Acme Corp"
    },
    category="missing_fields",
    difficulty="hard"
)

# Adversarial case: Ambiguous formatting
dataset.add_entry(
    input_text="Acme Corp - $450.00 - 1/15/24 - Ref: 1234",
    expected_output={
        "invoice_number": "1234",
        "date": "2024-01-15",
        "total": 450.00,
        "vendor": "Acme Corp"
    },
    category="non_standard_format",
    difficulty="hard"
)

dataset.save()
```

**Practical Implications**: Your golden dataset becomes your regression test suite. Every production failure should add an entry, ensuring you never regress on solved problems.

**Real Constraints**: Golden datasets require ongoing curation. Budget 20-30% of development time for dataset maintenance. Stale datasets fail to catch emerging issues.

### 3. Evaluation Metrics and Acceptance Thresholds

AI test plans must specify measurable metrics and define acceptable performance thresholds. Common metrics include:

- **Accuracy/Precision/Recall**: For classification tasks
- **Exact/Fuzzy Match Rates**: For extraction tasks
- **Semantic Similarity**: For generation tasks
- **Latency Percentiles**: For performance requirements
- **Cost per Request**: For operational viability

```python
from typing import Any
import json
from difflib import SequenceMatcher

class EvaluationMetrics:
    @staticmethod
    def exact_match(expected: Any, actual: Any) -> bool:
        """Binary exact match."""
        return expected == actual
    
    @staticmethod
    def fuzzy_string_match(expected: str, actual: str, threshold: float = 0.9) -> bool:
        """Fuzzy string matching using sequence similarity."""
        similarity = SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
        return similarity >= threshold
    
    @staticmethod
    def numeric_tolerance(expected: float, actual: float, tolerance: float = 0.01) -> bool:
        """Numeric match within tolerance."""
        return abs(expected - actual) <= tolerance
    
    @staticmethod
    def field_accuracy(expected: dict, actual: dict, required_fields: list[str]) -> float:
        """Calculate accuracy across required fields."""
        matches = sum(
            1 for field in required_fields
            if field in actual and expected.get(field) == actual.get(field)
        )
        return matches / len(required_fields)

class TestEvaluator:
    def __init__(self, acceptance_thresholds: dict):
        self.thresholds = acceptance_thresholds
        self.metrics = EvaluationMetrics()
    
    def evaluate_extraction(
        self,
        expected: dict,
        actual: dict,
        required_fields: list[str]
    ) -> dict:
        """Evaluate extraction task performance."""
        results = {
            "field_accuracy": self.metrics.field_accuracy(expected, actual, required_fields),
            "exact_matches": {},
            "fuzzy_matches": {}
        }
        
        for field in required_fields:
            if field not in actual:
                results["exact_matches"][field] = False
                results["fuzzy_matches"][field] = False
                continue
            
            expected_val = str(expected.get(field, ""))
            actual_val = str(actual.get(field, ""))
            
            results["exact_matches"][field] = self.metrics.exact_match(
                expected_val, actual_val
            )
            results["fuzzy_matches"][field] = self.metrics.fuzzy_string_match(
                expected_val, actual_val
            )
        
        # Check against acceptance thresholds
        results["passes_threshold"] = (
            results["field_accuracy"] >= self.thresholds.get("min_field_accuracy", 0.9)
        )
        
        return results

# Usage example
evaluator = TestEvaluator(acceptance_thresholds={
    "min_field_accuracy": 0.95,  # 95% of fields must be correct
    "min_exact_match_rate": 0.90  # 90% must be exact matches
})

expected_invoice = {
    "invoice_number": "INV-1234",
    "date": "2024-01-15",
    "total": "450.00",
    "vendor": "Acme Corporation"
}

actual_invoice = {
    "invoice_number": "INV-1234",
    "date": "2024-01-15",
    "total": "450.00",
    "vendor": "Acme Corp"  # Slight variation
}

results = evaluator.evaluate_extraction(
    expected_invoice,
    actual_invoice,
    required_fields=["invoice_number", "date", "total", "vendor"]
)

print(f"Field Accuracy: {results['field_accuracy']:.2%}")
print(f"Passes Threshold: {results['passes_threshold']}")
print(f"Exact Matches: {sum(results['exact_matches'].values())}/{len(results['exact_matches'])}")
```

**Practical Implications**: Explicit thresholds prevent subjective "looks good" evaluations. When production performance degrades, you have objective metrics to quantify the regression.

**Real Constraints**: Thresholds depend on business requirements. Financial data extraction might require 99.9% accuracy on amounts, while content categorization might accept 85% accuracy. Document the business justification for each threshold.

### 4. Regression Detection and Version Control

AI systems degrade silently. Model provider updates, prompt changes, or data drift can break previously working functionality. Test plans must include regression detection mechanisms.

```python
import hashlib
from datetime import datetime
from typing import Optional

class TestRun:
    def __init__(
        self,
        test_suite_version: str,
        model_identifier: str,
        prompt_hash: str
    ):
        self.test_suite_version = test_suite_version
        self.model_identifier = model_identifier
        self.prompt_hash = prompt_hash
        self.timestamp = datetime.now().isoformat()
        self.results: list[dict] = []
    
    def add_result(self, test_id: str, metrics: dict):
        """Record individual test result."""
        self.results.append({
            "test_id": test_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
    
    def summary(self) -> dict:
        """Aggregate test run summary."""
        if not self.results:
            return {}
        
        total_tests = len(self.results)
        passed_tests = sum(
            1 for r in self.results 
            if r["metrics"].get("passes_threshold", False)
        )
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "model": self.model_identifier,
            "prompt_hash": self.prompt_hash,
            "timestamp": self.timestamp
        }

class RegressionDetector:
    def __init__(self, baseline_run: TestRun):
        self.baseline = baseline_run
    
    def detect_regressions(
        self,
        current_run: TestRun,
        tolerance: float = 0.05  # 5% degradation threshold
    ) -> dict:
        """Compare current run against baseline to detect regressions."""
        baseline_summary = self.baseline.summary()
        current_summary = current_run.summary()
        
        baseline_pass_rate = baseline_summary.get("pass_rate", 0)
        current_pass_rate = current_summary.get("pass_rate", 0)
        
        degradation = baseline_pass_rate - current_pass_rate
        
        regressions = {
            "has_regression": degradation > tolerance,
            "baseline_