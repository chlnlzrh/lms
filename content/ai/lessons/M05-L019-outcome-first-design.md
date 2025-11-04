# Outcome-First Design: Engineering AI Systems That Deliver Value

## Core Concepts

**Technical Definition:** Outcome-First Design is an engineering methodology where you define measurable success criteria and evaluation frameworks before implementing AI/LLM features. Rather than starting with "let's use an LLM to process this data," you begin with "we need 95% accuracy classifying support tickets into 8 categories within 200ms at $0.001 per request."

### Traditional vs. Outcome-First Engineering

```python
# Traditional approach: Technology-first
def build_support_classifier():
    """Let's use GPT-4 because it's powerful!"""
    model = "gpt-4"
    prompt = "Classify this support ticket..."
    # Build first, measure later (if at all)
    return call_llm(model, prompt)

# Outcome-first approach: Requirements-driven
from typing import TypedDict, Literal
from dataclasses import dataclass

class ClassificationRequirements:
    """Define success before implementation"""
    target_accuracy: float = 0.95
    max_latency_ms: int = 200
    max_cost_per_request: float = 0.001
    required_categories: list[str] = [
        "billing", "technical", "account", "feature_request",
        "bug_report", "sales", "cancellation", "other"
    ]

@dataclass
class ClassificationResult:
    category: str
    confidence: float
    latency_ms: int
    cost: float
    
    def meets_requirements(self, req: ClassificationRequirements) -> tuple[bool, str]:
        """Verify against defined outcomes"""
        if self.latency_ms > req.max_latency_ms:
            return False, f"Latency {self.latency_ms}ms exceeds {req.max_latency_ms}ms"
        if self.cost > req.max_cost_per_request:
            return False, f"Cost ${self.cost} exceeds ${req.max_cost_per_request}"
        if self.confidence < 0.9:  # Proxy for accuracy
            return False, f"Confidence {self.confidence} too low"
        return True, "All requirements met"

def build_classifier_to_spec(requirements: ClassificationRequirements):
    """Implementation driven by measurable outcomes"""
    # Now choose technology based on requirements
    # Maybe GPT-4 is too expensive/slow, use GPT-3.5-turbo or fine-tuned model
    pass
```

### Key Engineering Insights

**1. Metrics Drive Architecture Decisions:** When you define that customer sentiment analysis must process 10,000 messages/hour at $50/month total cost, you immediately know GPT-4 won't work. You'll explore prompt optimization, model selection, caching strategies, or fine-tuning smaller models before writing a line of code.

**2. Failure Modes Become Explicit:** Instead of discovering in production that your LLM occasionally returns JSON with incorrect field names, outcome-first design forces you to define: "System must return valid JSON matching schema 99.9% of time, with fallback handling for parse failures."

**3. Evaluation IS the Product:** With traditional software, tests verify implementation. With LLMs, the evaluation framework often represents more engineering complexity than the LLM call itself. Your evaluation code is not a testing afterthought—it's core product infrastructure.

### Why This Matters Now

LLMs create a dangerous illusion: they work immediately with minimal code. A 5-line OpenAI API call produces seemingly magical results. This ease breeds production disasters:

- **Cost Surprise:** "Our chatbot costs $12,000/month" (discovered after launch)
- **Quality Drift:** "Accuracy dropped from 87% to 62%" (no monitoring existed)
- **Latency Cascade:** "P95 latency is 8 seconds" (never measured during development)

Outcome-first design applies traditional engineering discipline to a technology that feels like it doesn't need it. It prevents the pattern where engineers spend 1 hour building an LLM feature and 100 hours firefighting production issues.

## Technical Components

### 1. Quantitative Success Criteria Definition

**Technical Explanation:** Before implementation, define numerical thresholds across quality, performance, and cost dimensions. These aren't aspirational—they're hard requirements that determine technology choices.

**Practical Implementation:**

```python
from enum import Enum
from typing import Optional
import time

class MetricType(Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    LATENCY = "latency_ms"
    COST = "cost_usd"
    THROUGHPUT = "requests_per_second"

class SuccessCriterion:
    """Single measurable requirement"""
    def __init__(
        self,
        metric: MetricType,
        operator: str,  # ">", "<", ">=", "<=", "=="
        threshold: float,
        critical: bool = True  # Does failure block deployment?
    ):
        self.metric = metric
        self.operator = operator
        self.threshold = threshold
        self.critical = critical
    
    def evaluate(self, value: float) -> bool:
        ops = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: abs(v - t) < 1e-9
        }
        return ops[self.operator](value, self.threshold)
    
    def __repr__(self) -> str:
        criticality = "CRITICAL" if self.critical else "DESIRED"
        return f"{criticality}: {self.metric.value} {self.operator} {self.threshold}"

class OutcomeSpecification:
    """Complete requirements definition"""
    def __init__(self, name: str):
        self.name = name
        self.criteria: list[SuccessCriterion] = []
    
    def add_criterion(self, *args, **kwargs) -> 'OutcomeSpecification':
        self.criteria.append(SuccessCriterion(*args, **kwargs))
        return self
    
    def evaluate(self, metrics: dict[MetricType, float]) -> tuple[bool, list[str]]:
        """Check if implementation meets specification"""
        failures = []
        for criterion in self.criteria:
            value = metrics.get(criterion.metric)
            if value is None:
                failures.append(f"Missing metric: {criterion.metric.value}")
                continue
            
            if not criterion.evaluate(value):
                status = "FAIL" if criterion.critical else "SUBOPTIMAL"
                failures.append(
                    f"{status}: {criterion.metric.value}={value} "
                    f"(required {criterion.operator} {criterion.threshold})"
                )
        
        critical_failures = [f for f in failures if f.startswith("FAIL")]
        return len(critical_failures) == 0, failures

# Example: Email categorization system
email_spec = (
    OutcomeSpecification("email_categorization")
    .add_criterion(MetricType.ACCURACY, ">=", 0.92, critical=True)
    .add_criterion(MetricType.LATENCY, "<=", 500, critical=True)
    .add_criterion(MetricType.COST, "<=", 0.002, critical=True)
    .add_criterion(MetricType.PRECISION, ">=", 0.88, critical=False)
)

print("Requirements:")
for criterion in email_spec.criteria:
    print(f"  {criterion}")
```

**Real Constraints:** Early requirements feel arbitrary. How do you know you need 92% accuracy vs 87%? This requires business context: "If accuracy drops below 92%, users start manually checking every categorization, negating the automation value." Engineering rigor means challenging stakeholders to quantify these thresholds, not accepting "as good as possible."

### 2. Structured Evaluation Harness

**Technical Explanation:** An evaluation harness is infrastructure that measures your LLM system against success criteria using representative test data. Unlike unit tests that verify code correctness, evaluation harnesses verify emergent behavior against examples.

**Practical Implementation:**

```python
from typing import Callable, Any
import json
from datetime import datetime
from pathlib import Path

class EvaluationExample:
    """Single test case with expected outcome"""
    def __init__(self, input_data: Any, expected: Any, metadata: dict = None):
        self.input = input_data
        self.expected = expected
        self.metadata = metadata or {}

class EvaluationResult:
    """Result from evaluating one example"""
    def __init__(self, example: EvaluationExample, actual: Any, 
                 latency_ms: float, cost: float):
        self.example = example
        self.actual = actual
        self.latency_ms = latency_ms
        self.cost = cost
        self.correct: Optional[bool] = None
        self.score: Optional[float] = None

class EvaluationHarness:
    """Framework for systematic LLM evaluation"""
    
    def __init__(
        self,
        name: str,
        examples: list[EvaluationExample],
        scoring_fn: Callable[[Any, Any], float],
        specification: OutcomeSpecification
    ):
        self.name = name
        self.examples = examples
        self.scoring_fn = scoring_fn
        self.specification = specification
        self.results: list[EvaluationResult] = []
    
    def run(self, implementation: Callable[[Any], Any]) -> dict:
        """Execute evaluation against implementation"""
        self.results = []
        
        for example in self.examples:
            start = time.perf_counter()
            
            try:
                actual = implementation(example.input)
                latency_ms = (time.perf_counter() - start) * 1000
                
                # Estimate cost (would be tracked in real implementation)
                cost = self._estimate_cost(example.input, actual)
                
                result = EvaluationResult(example, actual, latency_ms, cost)
                result.score = self.scoring_fn(actual, example.expected)
                result.correct = result.score >= 0.9  # Threshold for "correct"
                
            except Exception as e:
                # Failures are results too
                result = EvaluationResult(example, None, 0, 0)
                result.score = 0.0
                result.correct = False
                result.error = str(e)
            
            self.results.append(result)
        
        return self._generate_report()
    
    def _estimate_cost(self, input_data: Any, output: Any) -> float:
        """Estimate API cost (simplified)"""
        # Real implementation would track actual API costs
        input_tokens = len(str(input_data)) / 4
        output_tokens = len(str(output)) / 4
        return (input_tokens * 0.03 + output_tokens * 0.06) / 1_000_000
    
    def _generate_report(self) -> dict:
        """Calculate aggregate metrics"""
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        
        metrics = {
            MetricType.ACCURACY: correct / total if total > 0 else 0,
            MetricType.LATENCY: sum(r.latency_ms for r in self.results) / total,
            MetricType.COST: sum(r.cost for r in self.results) / total,
        }
        
        meets_spec, failures = self.specification.evaluate(metrics)
        
        return {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
            "total_examples": total,
            "metrics": {k.value: v for k, v in metrics.items()},
            "meets_specification": meets_spec,
            "failures": failures,
            "results": self.results
        }
    
    def save_report(self, report: dict, path: Path):
        """Persist evaluation results"""
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to serializable format
        serializable = report.copy()
        serializable["results"] = [
            {
                "input": r.example.input,
                "expected": r.example.expected,
                "actual": r.actual,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "cost": r.cost
            }
            for r in report["results"]
        ]
        
        with open(path / filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        return path / filename

# Example usage
def exact_match_scoring(actual: str, expected: str) -> float:
    """Simple scoring: 1.0 for exact match, 0.0 otherwise"""
    return 1.0 if actual.strip().lower() == expected.strip().lower() else 0.0

examples = [
    EvaluationExample(
        "I need to update my billing info",
        "billing",
        metadata={"difficulty": "easy"}
    ),
    EvaluationExample(
        "The app crashes when I try to export data",
        "bug_report",
        metadata={"difficulty": "medium"}
    ),
    # In practice: 50-500+ examples
]

harness = EvaluationHarness(
    name="email_categorization_v1",
    examples=examples,
    scoring_fn=exact_match_scoring,
    specification=email_spec
)
```

**Trade-offs:** Building evaluation harnesses takes time—often longer than implementing the LLM call itself. The payoff comes from rapid iteration: once built, you can test prompt changes, model swaps, or architecture modifications in minutes with confidence. Skip this for throwaway prototypes, but treat it as mandatory infrastructure for production systems.

### 3. Baseline Establishment

**Technical Explanation:** A baseline is a minimal-effort implementation that sets the performance floor. It answers: "What's the simplest approach that could possibly work?" and provides a quantitative reference point for evaluating sophisticated solutions.

**Practical Implementation:**

```python
from typing import Protocol
import random

class Classifier(Protocol):
    """Interface for any classification approach"""
    def classify(self, text: str) -> str:
        ...

class RandomBaseline:
    """Simplest possible baseline: random guessing"""
    def __init__(self, categories: list[str]):
        self.categories = categories
    
    def classify(self, text: str) -> str:
        return random.choice(self.categories)

class KeywordBaseline:
    """Simple rule-based baseline"""
    def __init__(self, keyword_map: dict[str, list[str]]):
        self.keyword_map = keyword_map
    
    def classify(self, text: str) -> str:
        text_lower = text.lower()
        
        # Score each category by keyword matches
        scores = {}
        for category, keywords in self.keyword_map.items():
            scores[category] = sum(
                1 for keyword in keywords 
                if keyword in text_lower
            )
        
        # Return category with most matches, or "other"
        if max(scores.values()) == 0:
            return "other"
        return max(scores, key=scores.get)

class SimpleLLMBaseline:
    """Basic LLM with minimal prompt engineering"""
    def __init__(self, categories: list[str]):
        self.categories = categories
        self.prompt_template = (
            "Classify the following text into one of these categories: {categories}\n\n"
            "Text: {text}\n\n"
            "Category:"
        )
    
    def classify(self, text: str) -> str:
        prompt = self.prompt_template.format(
            categories=", ".join(self.categories),
            text=text
        )
        # response = call_llm(prompt)  # Actual LLM call
        # return response.strip()
        return "billing"  # Placeholder

# Baseline comparison framework
class BaselineComparison:
    """Compare multiple approaches systematically"""
    
    def __init__(self, harness: EvaluationHarness):
        self.harness = harness
        self.baseline_results: dict[str, dict] = {}
    
    def add_baseline(