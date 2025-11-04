# New Feature Evaluation: Engineering Rigor for LLM Capabilities

## Core Concepts

### Technical Definition

New feature evaluation in LLM systems is the systematic process of assessing whether newly released capabilities, model versions, or API features deliver measurable improvements for your specific use case—without introducing regressions, cost overruns, or reliability issues. Unlike traditional software where features are deterministic and well-documented, LLM capabilities are probabilistic, context-dependent, and often under-specified in their actual performance characteristics.

### Engineering Analogy: Database Index Evaluation

Consider how you'd evaluate adding a new index to a production database:

**Traditional Database Index (Deterministic):**
```python
# Before: Full table scan
SELECT * FROM users WHERE email = 'user@example.com'
# Query time: 450ms (predictable)

# After: B-tree index on email column
CREATE INDEX idx_users_email ON users(email);
SELECT * FROM users WHERE email = 'user@example.com'
# Query time: 12ms (predictable)

# Evaluation is straightforward:
# - Run EXPLAIN ANALYZE
# - Measure query time improvement
# - Calculate index storage cost
# - Decision is binary and reproducible
```

**LLM Feature Evaluation (Probabilistic):**
```python
# Before: Using GPT-3.5 for classification
def classify_support_ticket_v1(ticket_text: str) -> str:
    response = llm_client.complete(
        model="gpt-3.5-turbo",
        prompt=f"Classify this support ticket: {ticket_text}",
        temperature=0
    )
    return response.content

# After: New model version or feature (e.g., structured outputs)
def classify_support_ticket_v2(ticket_text: str) -> str:
    response = llm_client.complete(
        model="gpt-4-turbo",
        prompt=f"Classify this support ticket: {ticket_text}",
        response_format={"type": "json_schema", "schema": ticket_schema},
        temperature=0
    )
    return response.content

# Evaluation is complex:
# - Results vary across inputs (probabilistic)
# - Need statistical significance testing
# - Cost/latency trade-offs aren't linear
# - Quality improvements may not generalize
# - Regression testing is non-trivial
```

### Key Insights

**1. The Regression Paradox**: A new LLM feature may improve average performance while catastrophically failing on edge cases that matter most to your business. Traditional software versioning assumptions (newer = better) don't apply.

**2. Cost Asymmetry**: LLM feature improvements often come with non-linear cost increases. A 10% quality improvement might cost 3x more per request, making the ROI calculation critical rather than obvious.

**3. Evaluation Debt**: Unlike traditional features where unit tests suffice, LLM features require maintaining evaluation datasets, running expensive benchmark suites, and continuously monitoring production behavior. This evaluation infrastructure is as important as the feature itself.

### Why This Matters NOW

The LLM landscape is evolving rapidly—new models ship monthly, API features change frequently, and providers deprecate capabilities with short notice. Engineers who treat LLM features like traditional software updates will experience:

- **Production incidents** from untested regressions (e.g., new model refuses previously acceptable prompts)
- **Budget overruns** from naive adoption of more expensive features without ROI analysis
- **Technical debt** from missing the optimal time to upgrade (staying on deprecated models vs. premature optimization)

Organizations that systematically evaluate LLM features ship faster, spend less, and maintain higher reliability than those that chase every new release.

## Technical Components

### 1. Evaluation Dataset Construction

**Technical Explanation:**

An evaluation dataset is a curated collection of input-output pairs that represents the distribution and edge cases of your production workload. Unlike traditional ML datasets focused on training, LLM evaluation datasets prioritize:

- **Coverage**: Capturing rare but critical failure modes
- **Ground truth**: Clear, unambiguous expected outputs
- **Versioning**: Tracking how expectations evolve with business requirements
- **Cost efficiency**: Small enough to run frequently, large enough for statistical significance

**Practical Implementation:**

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class EvaluationCase:
    """Single test case for LLM evaluation"""
    id: str
    input_text: str
    expected_output: str
    category: str  # e.g., "edge_case", "common_case", "adversarial"
    metadata: Dict[str, any]
    created_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "input": self.input_text,
            "expected": self.expected_output,
            "category": self.category,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

class EvaluationDataset:
    """Manages versioned evaluation datasets"""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.cases: List[EvaluationCase] = []
    
    def add_case(self, case: EvaluationCase) -> None:
        """Add a single evaluation case"""
        self.cases.append(case)
    
    def sample_production_traffic(
        self, 
        production_logs: List[Dict],
        sample_rate: float = 0.01,
        min_confidence: float = 0.95
    ) -> None:
        """Sample high-confidence production cases for evaluation"""
        for log in production_logs:
            if log.get("confidence", 0) >= min_confidence:
                if hash(log["request_id"]) % 100 < (sample_rate * 100):
                    case = EvaluationCase(
                        id=log["request_id"],
                        input_text=log["input"],
                        expected_output=log["output"],
                        category="production_sample",
                        metadata={"confidence": log["confidence"]},
                        created_at=datetime.now()
                    )
                    self.add_case(case)
    
    def save(self, filepath: str) -> None:
        """Save dataset to disk with versioning"""
        data = {
            "name": self.name,
            "version": self.version,
            "cases": [case.to_dict() for case in self.cases]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'EvaluationDataset':
        """Load versioned dataset from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset = cls(name=data["name"], version=data["version"])
        for case_dict in data["cases"]:
            case = EvaluationCase(
                id=case_dict["id"],
                input_text=case_dict["input"],
                expected_output=case_dict["expected"],
                category=case_dict["category"],
                metadata=case_dict["metadata"],
                created_at=datetime.fromisoformat(case_dict["created_at"])
            )
            dataset.add_case(case)
        return dataset
```

**Real Constraints:**

- Dataset size vs. cost: Running 1,000 cases against a $0.01/request API costs $10 per evaluation
- Ground truth quality: Manual labeling is expensive; high-confidence production samples provide cheaper alternatives
- Staleness: Evaluation datasets become outdated as business requirements shift

**Concrete Example:**

For a support ticket classifier, your evaluation dataset might include:
- 50 common cases (normal tickets)
- 30 edge cases (ambiguous tickets, multiple issues)
- 20 adversarial cases (spam, abuse, edge-of-policy)
- 100 production samples (actual tickets with high-confidence labels)

Total: 200 cases × $0.015 per GPT-4 request = $3 per evaluation run

### 2. Comparative Evaluation Framework

**Technical Explanation:**

A comparative evaluation framework runs your current implementation alongside a candidate feature, measuring differences across multiple dimensions: accuracy, latency, cost, and failure modes. This requires isolated execution environments to prevent cross-contamination and statistical testing to determine significance.

**Practical Implementation:**

```python
from typing import Callable, List, Tuple
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class EvaluationResult:
    """Results from a single evaluation run"""
    case_id: str
    output: str
    latency_ms: float
    cost_usd: float
    success: bool
    error: Optional[str] = None

class ComparativeEvaluator:
    """Compare two LLM implementations systematically"""
    
    def __init__(
        self,
        baseline_fn: Callable[[str], str],
        candidate_fn: Callable[[str], str],
        dataset: EvaluationDataset
    ):
        self.baseline_fn = baseline_fn
        self.candidate_fn = candidate_fn
        self.dataset = dataset
    
    def _run_single_case(
        self, 
        case: EvaluationCase, 
        fn: Callable
    ) -> EvaluationResult:
        """Execute a single evaluation case with timing and error handling"""
        start_time = time.perf_counter()
        try:
            output = fn(case.input_text)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Estimate cost based on tokens (simplified)
            estimated_cost = self._estimate_cost(case.input_text, output)
            
            return EvaluationResult(
                case_id=case.id,
                output=output,
                latency_ms=latency_ms,
                cost_usd=estimated_cost,
                success=True
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return EvaluationResult(
                case_id=case.id,
                output="",
                latency_ms=latency_ms,
                cost_usd=0,
                success=False,
                error=str(e)
            )
    
    def _estimate_cost(self, input_text: str, output_text: str) -> float:
        """Rough token-based cost estimation"""
        # Approximate: 1 token ≈ 4 characters
        input_tokens = len(input_text) / 4
        output_tokens = len(output_text) / 4
        
        # Example pricing for GPT-4 Turbo
        input_cost = input_tokens * (0.01 / 1000)
        output_cost = output_tokens * (0.03 / 1000)
        return input_cost + output_cost
    
    def run_comparison(
        self, 
        parallel: bool = True,
        max_workers: int = 5
    ) -> Tuple[List[EvaluationResult], List[EvaluationResult]]:
        """Run both implementations across the entire dataset"""
        baseline_results = []
        candidate_results = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Run baseline
                baseline_futures = {
                    executor.submit(self._run_single_case, case, self.baseline_fn): case
                    for case in self.dataset.cases
                }
                for future in as_completed(baseline_futures):
                    baseline_results.append(future.result())
                
                # Run candidate
                candidate_futures = {
                    executor.submit(self._run_single_case, case, self.candidate_fn): case
                    for case in self.dataset.cases
                }
                for future in as_completed(candidate_futures):
                    candidate_results.append(future.result())
        else:
            for case in self.dataset.cases:
                baseline_results.append(
                    self._run_single_case(case, self.baseline_fn)
                )
                candidate_results.append(
                    self._run_single_case(case, self.candidate_fn)
                )
        
        return baseline_results, candidate_results
    
    def generate_report(
        self,
        baseline_results: List[EvaluationResult],
        candidate_results: List[EvaluationResult]
    ) -> Dict:
        """Generate comparative metrics report"""
        def calculate_metrics(results: List[EvaluationResult]) -> Dict:
            successful = [r for r in results if r.success]
            return {
                "success_rate": len(successful) / len(results),
                "avg_latency_ms": statistics.mean([r.latency_ms for r in successful]),
                "p95_latency_ms": statistics.quantiles(
                    [r.latency_ms for r in successful], n=20
                )[18],  # 95th percentile
                "total_cost_usd": sum(r.cost_usd for r in successful),
                "errors": [r.error for r in results if not r.success]
            }
        
        baseline_metrics = calculate_metrics(baseline_results)
        candidate_metrics = calculate_metrics(candidate_results)
        
        return {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "improvements": {
                "latency_reduction_pct": (
                    (baseline_metrics["avg_latency_ms"] - 
                     candidate_metrics["avg_latency_ms"]) /
                    baseline_metrics["avg_latency_ms"] * 100
                ),
                "cost_change_pct": (
                    (candidate_metrics["total_cost_usd"] - 
                     baseline_metrics["total_cost_usd"]) /
                    baseline_metrics["total_cost_usd"] * 100
                ),
                "reliability_change": (
                    candidate_metrics["success_rate"] - 
                    baseline_metrics["success_rate"]
                )
            }
        }
```

**Real Constraints:**

- Parallel execution can hit rate limits; implement exponential backoff
- Statistical significance requires sufficient sample size (typically 100+ cases)
- Cost of evaluation compounds: testing 5 candidate features = 6x dataset cost

### 3. Quality Scoring with LLM-as-Judge

**Technical Explanation:**

For open-ended outputs (summarization, generation), traditional exact-match metrics fail. LLM-as-judge uses a more capable model to score outputs on rubric-based criteria. While this introduces additional cost and latency, it enables automated evaluation of subjective quality.

**Practical Implementation:**

```python
from typing import Dict, List
from enum import Enum

class ScoringCriteria(Enum):
    ACCURACY = "factual accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance to query"
    CONCISENESS = "conciseness"

@dataclass
class QualityScore:
    """Structured quality assessment"""
    criteria: ScoringCriteria
    score: int  # 1-5 scale
    reasoning: str

class LLMJudge:
    """Use a strong model to evaluate outputs"""
    
    def __init__(self, judge_model: str = "gpt-4"):
        self.judge_model = judge_model
    
    def score_output(
        self,
        input_text: str,
        output_text: str,
        expected_output: str,
        criteria: List[ScoringCriteria]
    ) -> List[QualityScore]:
        """Score an output across multiple criteria"""
        scores = []
        
        for criterion in criteria:
            prompt = f"""You are evaluating an AI system's output quality.

INPUT: {input_text}

EXPECTED OUTPUT: {expected_output}

ACTUAL OUTPUT: {output_text}

CRITERION: {criterion.value}

Score the ACTUAL OUTPUT on a 1-5 scale for {criterion.value}:
1 = Poor, 2 = Below Average, 3 = Average, 4 = Good, 5 = Excellent

Provide your score and brief reasoning in this format:
SCORE: <number>
REASONING: <explanation>"""

            response = self._call_judge_model(prompt)
            score_value, reasoning = self._parse_judge_response(response)
            
            scores.append(QualityScore(
                criteria=