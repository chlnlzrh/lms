# Best Practice Documentation for LLM Systems

## Core Concepts

Traditional software documentation focuses on *what the code does*. When working with LLM-based systems, documentation must shift to explain *why decisions were made and what outcomes were expected*. This fundamental difference emerges because LLMs are non-deterministic components whose behavior changes based on subtle input variations, model updates, and context accumulation.

### Traditional vs. LLM System Documentation

```python
# Traditional API Documentation
def calculate_tax(amount: float, state: str) -> float:
    """
    Calculate sales tax for a given amount and state.
    
    Args:
        amount: Purchase amount in USD
        state: Two-letter state code
    
    Returns:
        Tax amount in USD
    """
    tax_rates = {"CA": 0.0725, "TX": 0.0625, "NY": 0.08}
    return amount * tax_rates.get(state, 0.0)

# LLM System Documentation - Insufficient
def extract_entities(text: str) -> dict:
    """Extract named entities from text using LLM."""
    response = llm.complete(f"Extract entities from: {text}")
    return parse_response(response)

# LLM System Documentation - Best Practice
def extract_entities(text: str, model_version: str = "v2.1") -> dict:
    """
    Extract named entities (person, organization, location) from text.
    
    Design Decisions:
    - Uses zero-shot extraction (no examples) because our evaluation showed
      92% accuracy vs 94% with 3-shot, but 3x faster (230ms vs 680ms avg)
    - Structured JSON output enforced via response_format to eliminate parsing errors
    - Temperature=0 for consistency across extractions
    
    Known Limitations:
    - Accuracy drops to ~78% for texts >2000 tokens (context dilution)
    - May miss abbreviated organization names (e.g., "IBM" vs "International Business Machines")
    
    Prompt Evolution:
    - v1.0: Simple "extract entities" - 67% accuracy
    - v2.0: Added output format specification - 85% accuracy
    - v2.1: Added "include confidence scores" - 92% accuracy (current)
    
    Evaluation Criteria:
    - Tested on 500-document benchmark (see tests/benchmark_entities.py)
    - Primary metric: F1 score on entity boundaries
    - Secondary: Extraction latency p95 < 500ms
    
    Args:
        text: Input text (optimal length: 100-2000 tokens)
        model_version: Prompt version to use (default: latest stable)
    
    Returns:
        {
            "persons": [{"text": str, "confidence": float}],
            "organizations": [...],
            "locations": [...]
        }
    
    Example:
        >>> extract_entities("Apple Inc. CEO Tim Cook visited London.")
        {
            "persons": [{"text": "Tim Cook", "confidence": 0.95}],
            "organizations": [{"text": "Apple Inc.", "confidence": 0.98}],
            "locations": [{"text": "London", "confidence": 0.92}]
        }
    """
    prompt = PROMPT_REGISTRY["entity_extraction"][model_version]
    response = llm.complete(
        prompt.format(text=text),
        temperature=0,
        response_format={"type": "json_object"}
    )
    return parse_and_validate_entities(response)
```

The enhanced documentation captures critical context that enables future engineers (or your future self) to understand not just *what* the system does, but *why it works this way* and *when it might fail*.

### Why This Matters Now

LLM systems exhibit three characteristics that make documentation critical:

1. **Non-deterministic behavior**: The same input can produce different outputs, making traditional "this function always returns X" documentation insufficient.

2. **Prompt archaeology problem**: Six months from now, you won't remember why you included that specific phrase in your prompt. Without documentation, teams waste weeks re-discovering what works.

3. **Invisible degradation**: Model updates, prompt drift, and changing data distributions can silently degrade system performance. Documentation enables you to detect and diagnose these issues.

Recent data from production LLM systems shows teams spend 40-60% of debugging time reconstructing context that should have been documented initially. This lesson provides the frameworks to avoid that waste.

## Technical Components

### Component 1: Prompt Versioning and Lineage

Every prompt must be versioned with a clear lineage showing how it evolved and why changes were made.

**Technical Implementation:**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import json

@dataclass
class PromptVersion:
    """Complete specification for a versioned prompt."""
    version: str
    template: str
    created_at: datetime
    created_by: str
    parent_version: Optional[str]
    change_rationale: str
    evaluation_results: Dict[str, float]
    known_issues: List[str]
    
    def render(self, **kwargs) -> str:
        """Render prompt with runtime parameters."""
        return self.template.format(**kwargs)

class PromptRegistry:
    """Central registry for all prompts with full version history."""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.prompts: Dict[str, Dict[str, PromptVersion]] = {}
        self._load_from_disk()
    
    def register(self, name: str, version: PromptVersion) -> None:
        """Register a new prompt version with validation."""
        if name not in self.prompts:
            self.prompts[name] = {}
        
        # Validate parent exists if specified
        if version.parent_version:
            if version.parent_version not in self.prompts[name]:
                raise ValueError(f"Parent version {version.parent_version} not found")
        
        self.prompts[name][version.version] = version
        self._save_to_disk(name, version)
    
    def get(self, name: str, version: str = "latest") -> PromptVersion:
        """Retrieve specific prompt version."""
        if version == "latest":
            version = max(self.prompts[name].keys())
        return self.prompts[name][version]
    
    def get_lineage(self, name: str, version: str) -> List[PromptVersion]:
        """Trace complete evolution history of a prompt."""
        lineage = []
        current = self.prompts[name][version]
        
        while current:
            lineage.append(current)
            if current.parent_version:
                current = self.prompts[name][current.parent_version]
            else:
                current = None
        
        return list(reversed(lineage))
    
    def _save_to_disk(self, name: str, version: PromptVersion) -> None:
        """Persist prompt version to disk."""
        path = f"{self.storage_path}/{name}/{version.version}.json"
        # Implementation details omitted for brevity
    
    def _load_from_disk(self) -> None:
        """Load all prompt versions from disk on initialization."""
        # Implementation details omitted for brevity

# Usage Example
registry = PromptRegistry("./prompts")

# Register initial version
v1 = PromptVersion(
    version="1.0",
    template="Summarize this text: {text}",
    created_at=datetime.now(),
    created_by="engineer@example.com",
    parent_version=None,
    change_rationale="Initial implementation for MVP",
    evaluation_results={"rouge_l": 0.42, "faithfulness": 0.78},
    known_issues=["Often includes details not in source text"]
)
registry.register("summarization", v1)

# Register improved version
v2 = PromptVersion(
    version="2.0",
    template="Summarize this text using only information present in the source. Do not add external knowledge.\n\nText: {text}",
    created_at=datetime.now(),
    created_by="engineer@example.com",
    parent_version="1.0",
    change_rationale="Added explicit constraint to prevent hallucination after observing 23% hallucination rate in v1.0",
    evaluation_results={"rouge_l": 0.39, "faithfulness": 0.91},
    known_issues=["Slightly lower ROUGE-L but much better faithfulness"]
)
registry.register("summarization", v2)

# Retrieve and use
prompt = registry.get("summarization", "2.0")
formatted = prompt.render(text="Recent advances in quantum computing...")
```

**Practical Implications:**

- Version control enables A/B testing between prompt versions in production
- When issues arise, you can quickly roll back to known-good versions
- New team members can understand the system's evolution without tribal knowledge

**Real Constraints:**

- Overhead: Each prompt version requires ~5-10 minutes to document properly
- Storage: Comprehensive versioning can generate hundreds of versions for mature systems
- Trade-off: Documentation time vs. future debugging time (typically 1:10 ratio)

### Component 2: Evaluation Harness Documentation

Every LLM component needs documented evaluation criteria and reproducible test harnesses.

```python
from typing import Callable, List, Tuple
import statistics

class EvaluationHarness:
    """
    Standardized evaluation framework for LLM components.
    
    Design Philosophy:
    - Every component must have quantitative evaluation metrics
    - Evaluations must be reproducible (fixed test sets, versioned)
    - Include both quality metrics and operational metrics (latency, cost)
    """
    
    def __init__(
        self,
        name: str,
        test_cases: List[Tuple[dict, dict]],  # (input, expected_output)
        metrics: List[Callable],
        version: str = "1.0"
    ):
        self.name = name
        self.test_cases = test_cases
        self.metrics = metrics
        self.version = version
    
    def evaluate(
        self,
        component_fn: Callable,
        track_latency: bool = True,
        track_tokens: bool = True
    ) -> dict:
        """
        Run complete evaluation suite against a component.
        
        Returns comprehensive results including:
        - Metric scores (accuracy, F1, etc.)
        - Operational metrics (latency p50/p95/p99, token usage)
        - Per-example results for failure analysis
        """
        results = {
            "test_cases_run": len(self.test_cases),
            "metric_scores": {},
            "latency_ms": [],
            "tokens_used": [],
            "failures": []
        }
        
        for idx, (input_data, expected) in enumerate(self.test_cases):
            import time
            start = time.time()
            
            try:
                actual = component_fn(**input_data)
                latency_ms = (time.time() - start) * 1000
                
                if track_latency:
                    results["latency_ms"].append(latency_ms)
                
                # Run all metrics
                for metric_fn in self.metrics:
                    metric_name = metric_fn.__name__
                    if metric_name not in results["metric_scores"]:
                        results["metric_scores"][metric_name] = []
                    
                    score = metric_fn(expected, actual)
                    results["metric_scores"][metric_name].append(score)
                
            except Exception as e:
                results["failures"].append({
                    "test_case_idx": idx,
                    "error": str(e),
                    "input": input_data
                })
        
        # Aggregate metrics
        for metric_name, scores in results["metric_scores"].items():
            results["metric_scores"][metric_name] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores)
            }
        
        if results["latency_ms"]:
            sorted_latencies = sorted(results["latency_ms"])
            n = len(sorted_latencies)
            results["latency_percentiles"] = {
                "p50": sorted_latencies[int(n * 0.5)],
                "p95": sorted_latencies[int(n * 0.95)],
                "p99": sorted_latencies[int(n * 0.99)]
            }
        
        return results

# Define custom metrics
def entity_f1_score(expected: dict, actual: dict) -> float:
    """
    Calculate F1 score for entity extraction.
    
    Considerations:
    - Exact string match required (could be relaxed for fuzzy matching)
    - Entity type must match
    - Position-independent comparison
    """
    expected_entities = set(
        (e["text"], e["type"]) 
        for e in expected.get("entities", [])
    )
    actual_entities = set(
        (e["text"], e["type"]) 
        for e in actual.get("entities", [])
    )
    
    if not expected_entities and not actual_entities:
        return 1.0
    
    true_positives = len(expected_entities & actual_entities)
    false_positives = len(actual_entities - expected_entities)
    false_negatives = len(expected_entities - actual_entities)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

# Create evaluation harness
test_cases = [
    (
        {"text": "Apple Inc. CEO Tim Cook visited Paris last week."},
        {
            "entities": [
                {"text": "Apple Inc.", "type": "organization"},
                {"text": "Tim Cook", "type": "person"},
                {"text": "Paris", "type": "location"}
            ]
        }
    ),
    # ... more test cases
]

harness = EvaluationHarness(
    name="entity_extraction_eval",
    test_cases=test_cases,
    metrics=[entity_f1_score],
    version="1.0"
)

# Run evaluation
results = harness.evaluate(extract_entities)
print(f"Mean F1 Score: {results['metric_scores']['entity_f1_score']['mean']:.3f}")
print(f"Latency P95: {results['latency_percentiles']['p95']:.1f}ms")
```

**Practical Implications:**

- Quantitative metrics enable objective comparison between prompt versions
- Reproducible evaluations catch regressions before production deployment
- Latency tracking identifies performance degradation early

**Real Constraints:**

- Creating quality test sets requires 2-4 hours of effort per component
- Test set size trade-off: 50 examples gives 95% confidence interval of ±14%, 200 examples reduces to ±7%
- Evaluation execution time: 200 test cases at 300ms each = 60 seconds per evaluation run

### Component 3: Decision Context Records

Document *why* architectural decisions were made, including rejected alternatives.

```python
from enum import Enum
from typing import List, Optional

class DecisionStatus(Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"

@dataclass
class ArchitecturalDecisionRecord:
    """
    Architecture Decision Record (ADR) adapted for LLM systems.
    
    Format based on Michael Nygard's ADR pattern, extended for ML systems.
    """
    
    id: str  # e.g., "ADR-001"
    title: str
    date: datetime
    status: DecisionStatus
    context: str  # What forces are at play?
    decision: str  # What we decided to do
    consequences: str  # What becomes easier/harder as a result
    
    # LLM-specific additions
    alternatives_considered: List[str]
    evaluation_data: Optional[dict]  # Quantitative comparison