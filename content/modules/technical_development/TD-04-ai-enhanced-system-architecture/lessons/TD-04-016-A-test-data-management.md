# Test Data Management for AI Systems

## Core Concepts

Test data management in AI systems fundamentally differs from traditional software testing. In conventional applications, you write deterministic tests with fixed inputs and expected outputs. In AI systems, outputs are probabilistic, context-dependent, and often nuanced. This shift requires a completely different approach to how you collect, curate, version, and utilize test data.

### Traditional vs. Modern Approach

```python
# Traditional Software Testing
def test_calculate_discount():
    """Deterministic: same input always produces same output"""
    result = calculate_discount(price=100, discount_percent=10)
    assert result == 90  # Always passes or always fails

# AI System Testing
def test_extract_invoice_amount():
    """Probabilistic: output varies based on model, prompt, context"""
    invoice_text = "Total amount due: $1,234.56"
    result = llm.extract(invoice_text, schema=InvoiceSchema)
    
    # Simple assertion often insufficient
    assert result.amount == 1234.56  # May fail intermittently
    
    # Need richer validation
    assert 1200 <= result.amount <= 1300  # Range checking
    assert result.currency == "USD"
    assert result.confidence > 0.8  # Model's certainty
```

The fundamental difference: traditional tests verify **implementation correctness**, while AI tests verify **behavior quality**. Your test data becomes a behavioral specification, defining acceptable response patterns rather than exact outputs.

### Key Insights That Change Your Thinking

**1. Test data is your ground truth.** In AI systems, your test dataset defines what "correct" means. Unlike traditional software where correctness is provable through logic, AI correctness is empirically demonstrated through examples. If your test data is biased, incomplete, or incorrect, your system will perpetuate those flaws.

**2. Test data evolves as your system evolves.** You'll continuously discover edge cases, user behaviors, and failure modes. Your test suite grows organically, capturing institutional knowledge about what your AI should and shouldn't do.

**3. Volume and diversity matter more than coverage.** Traditional testing aims for code coverage; AI testing aims for behavioral coverage. A hundred variations of invoice formats teach your testing system more than perfectly covering every code path.

### Why This Matters Now

AI systems are moving from experimental to production-critical. Companies are discovering that:

- **40-60% of AI project time** goes to data-related work, including test data
- **Regression in AI behavior** is subtle and insidious—small prompt changes cascade unpredictably
- **Compliance and auditability** require documented test cases showing system behavior
- **Cost optimization** demands knowing which test cases actually provide signal vs. noise

Without systematic test data management, you're flying blind. You can't reliably deploy, debug, or improve your AI systems.

## Technical Components

### 1. Test Case Structure and Schema

Unlike simple input-output pairs, AI test cases need rich metadata to be useful.

```python
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class TestCategory(Enum):
    HAPPY_PATH = "happy_path"
    EDGE_CASE = "edge_case"
    ADVERSARIAL = "adversarial"
    REGRESSION = "regression"

@dataclass
class AITestCase:
    """Structured test case for AI system behavior"""
    
    # Core test data
    test_id: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    
    # Metadata for organization
    category: TestCategory
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    # Quality validation
    acceptance_criteria: Dict[str, Any] = field(default_factory=dict)
    min_confidence: float = 0.0
    max_latency_ms: Optional[int] = None
    
    # Provenance tracking
    source: str = ""  # "production", "synthetic", "manual"
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    failure_count: int = 0
    
    # Context for debugging
    notes: str = ""
    related_ticket: Optional[str] = None

# Example usage
test_case = AITestCase(
    test_id="invoice_extract_001",
    input_data={
        "text": "Invoice #12345\nTotal: $1,234.56\nDue: 2024-01-15",
        "extraction_fields": ["invoice_number", "total", "due_date"]
    },
    expected_output={
        "invoice_number": "12345",
        "total": 1234.56,
        "due_date": "2024-01-15"
    },
    category=TestCategory.HAPPY_PATH,
    tags=["invoice", "financial", "date_parsing"],
    acceptance_criteria={
        "exact_match": ["invoice_number"],
        "numeric_tolerance": {"total": 0.01},
        "date_format": "ISO8601"
    },
    min_confidence=0.85,
    max_latency_ms=2000,
    source="production",
    created_by="data_team"
)
```

**Practical Implications:**

- **Structured schemas enable automation**: You can batch-process tests, generate reports, and track regressions systematically
- **Rich metadata enables debugging**: When tests fail, you know why they exist and what they're protecting against
- **Acceptance criteria encode domain knowledge**: Different fields need different validation strategies

**Trade-offs:**

- More structure means more upfront work per test case
- Over-specification can make tests brittle to benign changes
- Balance between completeness and maintainability

### 2. Test Data Collection Strategies

You need multiple sources of test data, each serving different purposes.

```python
import json
import random
from typing import Iterator
from pathlib import Path

class TestDataCollector:
    """Collect and organize test data from multiple sources"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def collect_from_production(
        self, 
        production_logs: List[Dict[str, Any]],
        sample_rate: float = 0.1
    ) -> List[AITestCase]:
        """Sample real production traffic for test cases"""
        
        test_cases = []
        for log in production_logs:
            if random.random() > sample_rate:
                continue
                
            # Only promote logs with human verification
            if not log.get("human_verified"):
                continue
            
            test_case = AITestCase(
                test_id=f"prod_{log['request_id']}",
                input_data=log["input"],
                expected_output=log["verified_output"],
                category=TestCategory.HAPPY_PATH,
                tags=log.get("tags", []),
                source="production",
                description=f"Real production case from {log['timestamp']}"
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def generate_synthetic_edge_cases(
        self,
        base_template: Dict[str, Any],
        variations: List[Dict[str, Any]]
    ) -> List[AITestCase]:
        """Create synthetic edge cases by systematic variation"""
        
        test_cases = []
        for i, variation in enumerate(variations):
            modified_input = base_template.copy()
            modified_input.update(variation["input_changes"])
            
            test_case = AITestCase(
                test_id=f"synthetic_edge_{i:03d}",
                input_data=modified_input,
                expected_output=variation["expected_output"],
                category=TestCategory.EDGE_CASE,
                tags=["synthetic"] + variation.get("tags", []),
                source="synthetic",
                description=variation.get("description", "")
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def capture_failure_cases(
        self,
        error_logs: List[Dict[str, Any]]
    ) -> List[AITestCase]:
        """Convert production failures into regression tests"""
        
        test_cases = []
        for error in error_logs:
            test_case = AITestCase(
                test_id=f"regression_{error['incident_id']}",
                input_data=error["failing_input"],
                expected_output=error["corrected_output"],
                category=TestCategory.REGRESSION,
                tags=["regression", error["error_type"]],
                source="production",
                description=f"Regression test for incident {error['incident_id']}",
                related_ticket=error.get("ticket_id"),
                notes=error.get("root_cause", "")
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def save_test_suite(self, test_cases: List[AITestCase], suite_name: str):
        """Persist test suite to disk"""
        output_file = self.storage_path / f"{suite_name}.jsonl"
        
        with open(output_file, 'w') as f:
            for test_case in test_cases:
                # Convert to dict for serialization
                test_dict = {
                    "test_id": test_case.test_id,
                    "input_data": test_case.input_data,
                    "expected_output": test_case.expected_output,
                    "category": test_case.category.value,
                    "tags": test_case.tags,
                    "description": test_case.description,
                    "acceptance_criteria": test_case.acceptance_criteria,
                    "source": test_case.source,
                    "created_at": test_case.created_at.isoformat()
                }
                f.write(json.dumps(test_dict) + "\n")
```

**Practical Implications:**

- **Production sampling** gives you real-world distribution but requires human verification
- **Synthetic generation** enables comprehensive edge case coverage but may miss unknown unknowns
- **Failure capture** prevents regressions but can lead to over-fitting to past failures

**Real Constraints:**

- Production data may contain PII requiring anonymization
- Synthetic data may not reflect actual user behavior
- Too many regression tests slow down development

### 3. Versioning and Dataset Management

AI systems change frequently—models update, prompts evolve, requirements shift. Your test data must be versioned like code.

```python
from typing import Set
import hashlib
from dataclasses import asdict

class TestDataVersionManager:
    """Version control for test datasets"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def compute_dataset_hash(self, test_cases: List[AITestCase]) -> str:
        """Create reproducible hash of test suite"""
        
        # Sort by test_id for consistency
        sorted_cases = sorted(test_cases, key=lambda x: x.test_id)
        
        # Hash relevant fields only (exclude timestamps)
        hash_content = []
        for tc in sorted_cases:
            hash_fields = {
                "test_id": tc.test_id,
                "input_data": tc.input_data,
                "expected_output": tc.expected_output,
                "category": tc.category.value,
                "acceptance_criteria": tc.acceptance_criteria
            }
            hash_content.append(json.dumps(hash_fields, sort_keys=True))
        
        content_str = "\n".join(hash_content)
        return hashlib.sha256(content_str.encode()).hexdigest()[:12]
    
    def create_version(
        self, 
        test_cases: List[AITestCase],
        version_name: str,
        description: str = ""
    ) -> str:
        """Create a named version of test suite"""
        
        dataset_hash = self.compute_dataset_hash(test_cases)
        version_id = f"{version_name}_{dataset_hash}"
        
        version_dir = self.base_path / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Save test cases
        test_file = version_dir / "tests.jsonl"
        with open(test_file, 'w') as f:
            for tc in test_cases:
                f.write(json.dumps(asdict(tc), default=str) + "\n")
        
        # Save metadata
        metadata = {
            "version_id": version_id,
            "version_name": version_name,
            "description": description,
            "dataset_hash": dataset_hash,
            "test_count": len(test_cases),
            "created_at": datetime.now().isoformat(),
            "category_breakdown": self._count_by_category(test_cases),
            "tag_breakdown": self._count_by_tags(test_cases)
        }
        
        with open(version_dir / "metadata.json", 'w') as f:
            json.dumps(metadata, indent=2)
        
        return version_id
    
    def load_version(self, version_id: str) -> List[AITestCase]:
        """Load a specific version of test suite"""
        
        test_file = self.base_path / version_id / "tests.jsonl"
        test_cases = []
        
        with open(test_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Reconstruct AITestCase from dict
                test_case = AITestCase(
                    test_id=data["test_id"],
                    input_data=data["input_data"],
                    expected_output=data["expected_output"],
                    category=TestCategory(data["category"]),
                    tags=data.get("tags", []),
                    description=data.get("description", ""),
                    acceptance_criteria=data.get("acceptance_criteria", {}),
                    source=data.get("source", ""),
                    created_at=datetime.fromisoformat(data["created_at"])
                )
                test_cases.append(test_case)
        
        return test_cases
    
    def diff_versions(
        self, 
        version_id_1: str, 
        version_id_2: str
    ) -> Dict[str, Set[str]]:
        """Compare two versions to see what changed"""
        
        v1_cases = {tc.test_id for tc in self.load_version(version_id_1)}
        v2_cases = {tc.test_id for tc in self.load_version(version_id_2)}
        
        return {
            "added": v2_cases - v1_cases,
            "removed": v1_cases - v2_cases,
            "unchanged": v1_cases & v2_cases
        }
    
    def _count_by_category(self, test_cases: List[AITestCase]) -> Dict[str, int]:
        counts = {}
        for tc in test_cases:
            counts[tc.category.value] = counts.get(tc.category.value, 0) + 1
        return counts
    
    def _count_by_tags(self, test_cases: List[AITestCase]) -> Dict[str, int]:
        counts = {}
        for tc in test_cases:
            for tag in tc.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts
```

**Practical Implications:**

- **Hashing enables reproducibility**: You can reference exact test suites in CI/CD pipelines
- **Diffing enables change tracking**: You see exactly what behavioral coverage changed
- **Metadata enables analysis**: You can track test suite composition over time

**Trade-offs:**

- Storage grows linearly with versions (mitigate with deduplication)
- More versions create navigation complexity (use meaningful names)
- Perfect versioning adds overhead (version at meaningful milestones)

### 4. Evaluation Harness

Running tests against AI systems requires specialized evaluation logic.

```python
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class AITestEvaluator:
    """Execute and evaluate AI test cases"""
    
    def __init__(
        self, 
        model_function: Callable,
        max_workers: int = 5
    ):
        self.model_function = model_function
        self.