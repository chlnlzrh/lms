# Prompt Regression Testing

## Core Concepts

Prompt regression testing applies traditional software testing principles to LLM interactions, establishing automated verification that prompt modifications don't degrade model outputs. Unlike deterministic code where identical inputs guarantee identical outputs, LLMs introduce probabilistic behavior requiring statistical approaches to test stability.

### Engineering Analogy: API Testing vs. Prompt Testing

```python
# Traditional API regression testing
def test_user_registration():
    response = api.register_user({
        "email": "test@example.com",
        "password": "secure123"
    })
    assert response.status_code == 201
    assert response.json()["id"] is not None
    assert response.json()["email"] == "test@example.com"
    # Deterministic: same input → same output

# Prompt regression testing
def test_sentiment_classification():
    prompt = "Classify sentiment: 'This product exceeded expectations'"
    responses = [llm.generate(prompt, temperature=0.0) for _ in range(10)]
    
    # Must handle non-deterministic behavior
    positive_count = sum(1 for r in responses if "positive" in r.lower())
    assert positive_count >= 8  # 80% threshold for pass
    
    # Semantic equivalence, not exact match
    assert all(
        semantic_similarity(r, "positive sentiment") > 0.85 
        for r in responses
    )
```

The fundamental difference: traditional testing verifies exact outputs, while prompt testing verifies output characteristics across a distribution of responses.

### Why This Matters Now

Production LLM applications face unique challenges:

1. **Silent degradation**: Prompt changes can subtly degrade quality without errors
2. **Context dependencies**: Model updates, API changes, or temperature variations affect outputs
3. **Scale complexity**: Hundreds of prompts across multiple use cases require systematic validation
4. **Cost implications**: Each test invocation costs money and time, requiring strategic test design

Organizations deploying LLMs at scale report that **60-80% of production issues stem from prompt changes** made without adequate regression coverage. A financial services company discovered their summarization prompt degraded accuracy from 94% to 71% after a "minor" wording change—caught only by customer complaints, not tests.

### Key Engineering Insights

**Insight 1: Temperature=0 doesn't guarantee determinism.** Even with temperature 0, models exhibit variation due to floating-point precision, batching effects, and API-level changes. Design tests assuming 5-10% natural variation.

**Insight 2: Eval-driven development beats post-hoc testing.** Writing evaluations before prompt optimization provides a feedback loop 10-20x faster than manual review cycles.

**Insight 3: Statistical significance requires sample sizes.** Testing prompts once gives false confidence. Minimum viable regression testing requires 5-10 samples per prompt at critical checkpoints.

## Technical Components

### 1. Evaluation Metrics and Scoring Functions

Regression tests require quantifiable metrics that map LLM outputs to pass/fail criteria.

```python
from typing import Callable, List, Dict, Any
from dataclasses import dataclass
import re

@dataclass
class EvalResult:
    score: float  # 0.0 to 1.0
    passed: bool
    details: Dict[str, Any]

class PromptEvaluator:
    """Base evaluator with common scoring patterns"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    def exact_match(self, output: str, expected: str) -> EvalResult:
        """Binary match - use sparingly for LLMs"""
        match = output.strip().lower() == expected.strip().lower()
        return EvalResult(
            score=1.0 if match else 0.0,
            passed=match,
            details={"expected": expected, "got": output}
        )
    
    def contains_keywords(
        self, 
        output: str, 
        required: List[str],
        forbidden: List[str] = None
    ) -> EvalResult:
        """Check presence/absence of specific terms"""
        output_lower = output.lower()
        
        required_found = [kw for kw in required if kw.lower() in output_lower]
        forbidden_found = [kw for kw in (forbidden or []) if kw.lower() in output_lower]
        
        score = len(required_found) / len(required) if required else 1.0
        if forbidden_found:
            score *= 0.5  # Penalty for forbidden terms
        
        return EvalResult(
            score=score,
            passed=score >= self.threshold and not forbidden_found,
            details={
                "required_found": required_found,
                "required_missing": list(set(required) - set(required_found)),
                "forbidden_found": forbidden_found
            }
        )
    
    def structural_format(
        self, 
        output: str, 
        pattern: str
    ) -> EvalResult:
        """Validate output structure with regex"""
        matches = re.match(pattern, output.strip(), re.DOTALL)
        
        return EvalResult(
            score=1.0 if matches else 0.0,
            passed=bool(matches),
            details={"pattern": pattern, "output_length": len(output)}
        )
    
    def semantic_similarity(
        self, 
        output: str, 
        reference: str,
        model: Any = None  # Embedding model
    ) -> EvalResult:
        """Compare semantic meaning using embeddings"""
        # Simplified - in production, use actual embedding model
        from difflib import SequenceMatcher
        
        # Placeholder: use sentence-transformers or similar in production
        similarity = SequenceMatcher(None, output.lower(), reference.lower()).ratio()
        
        return EvalResult(
            score=similarity,
            passed=similarity >= self.threshold,
            details={
                "reference": reference,
                "similarity": similarity
            }
        )

# Practical usage
evaluator = PromptEvaluator(threshold=0.8)

# Test classification output
output = "Sentiment: POSITIVE (confidence: 0.92)"
result = evaluator.contains_keywords(
    output,
    required=["sentiment", "positive"],
    forbidden=["negative", "neutral"]
)
print(f"Passed: {result.passed}, Score: {result.score}")

# Test structured output
json_output = '{"status": "success", "value": 42}'
result = evaluator.structural_format(
    json_output,
    r'\{.*"status":\s*"[^"]+",.*"value":\s*\d+.*\}'
)
print(f"Valid JSON structure: {result.passed}")
```

**Practical Implications:**

- **Exact matching** works only for highly constrained outputs (classification labels, specific formats)
- **Keyword checking** balances flexibility with validation, ideal for checking technical requirements
- **Structural validation** ensures parseable outputs before downstream processing
- **Semantic similarity** handles paraphrasing but requires embedding models and careful threshold tuning

**Trade-offs:**

- Stricter metrics (exact match) catch more regressions but create brittle tests
- Looser metrics (semantic similarity) reduce false positives but may miss subtle degradations
- Embedding-based comparisons add latency and cost to test suites

### 2. Test Case Generation and Management

Effective regression testing requires representative test cases covering edge cases, variations, and production distribution.

```python
from typing import List, Optional
from dataclasses import dataclass
import json

@dataclass
class TestCase:
    """Single test case with input, expected behavior, metadata"""
    id: str
    prompt_template: str
    input_vars: Dict[str, str]
    evaluator: Callable
    eval_params: Dict[str, Any]
    tags: List[str]
    priority: int  # 1=critical, 2=important, 3=nice-to-have

class TestSuite:
    """Manage and execute test cases for prompt regression"""
    
    def __init__(self, name: str):
        self.name = name
        self.test_cases: List[TestCase] = []
    
    def add_test(
        self,
        test_id: str,
        prompt_template: str,
        input_vars: Dict[str, str],
        evaluator: Callable,
        eval_params: Dict[str, Any],
        tags: List[str] = None,
        priority: int = 2
    ):
        """Add test case to suite"""
        self.test_cases.append(TestCase(
            id=test_id,
            prompt_template=prompt_template,
            input_vars=input_vars,
            evaluator=evaluator,
            eval_params=eval_params,
            tags=tags or [],
            priority=priority
        ))
    
    def generate_variants(
        self,
        base_case: TestCase,
        variations: Dict[str, List[str]]
    ) -> List[TestCase]:
        """Create test variations from base case"""
        variants = []
        
        for var_name, var_values in variations.items():
            for idx, value in enumerate(var_values):
                new_vars = base_case.input_vars.copy()
                new_vars[var_name] = value
                
                variants.append(TestCase(
                    id=f"{base_case.id}_var_{var_name}_{idx}",
                    prompt_template=base_case.prompt_template,
                    input_vars=new_vars,
                    evaluator=base_case.evaluator,
                    eval_params=base_case.eval_params,
                    tags=base_case.tags + [f"variant:{var_name}"],
                    priority=base_case.priority + 1
                ))
        
        return variants
    
    def filter_by_priority(self, max_priority: int) -> List[TestCase]:
        """Get tests up to specified priority level"""
        return [tc for tc in self.test_cases if tc.priority <= max_priority]
    
    def filter_by_tags(self, tags: List[str]) -> List[TestCase]:
        """Get tests matching any specified tag"""
        return [
            tc for tc in self.test_cases 
            if any(tag in tc.tags for tag in tags)
        ]
    
    def export_to_file(self, filepath: str):
        """Serialize test suite for version control"""
        data = {
            "name": self.name,
            "test_cases": [
                {
                    "id": tc.id,
                    "prompt_template": tc.prompt_template,
                    "input_vars": tc.input_vars,
                    "eval_params": tc.eval_params,
                    "tags": tc.tags,
                    "priority": tc.priority
                }
                for tc in self.test_cases
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

# Example: Building a test suite for email classification
suite = TestSuite("email_classifier")

# Critical test case
suite.add_test(
    test_id="urgent_detection_basic",
    prompt_template="Classify if urgent: {email_text}",
    input_vars={"email_text": "URGENT: Server down, production affected"},
    evaluator=lambda output: "urgent" in output.lower(),
    eval_params={"expected": "urgent"},
    tags=["classification", "urgent"],
    priority=1
)

# Generate edge case variants
base_case = suite.test_cases[0]
variants = suite.generate_variants(
    base_case,
    variations={
        "email_text": [
            "urgent: please review when possible",  # Ambiguous urgency
            "URGENT!!!!! CLICK HERE NOW",            # Spam pattern
            "Urgent care appointment reminder"       # Healthcare context
        ]
    }
)

for variant in variants:
    suite.test_cases.append(variant)

# Filter for CI/CD - only run priority 1 tests
critical_tests = suite.filter_by_priority(max_priority=1)
print(f"Critical tests for CI: {len(critical_tests)}")

# Full suite for nightly runs
print(f"Full suite size: {len(suite.test_cases)}")
```

**Practical Implications:**

- **Priority tiering** enables fast CI checks (critical tests only) and comprehensive nightly runs
- **Variant generation** systematically explores edge cases without manual duplication
- **Tagging** allows selective test execution (e.g., "run all classification tests")
- **Version control** for test suites enables tracking which tests caught regressions

**Constraints:**

- Large test suites increase execution time and API costs
- Maintaining test cases as prompts evolve requires discipline
- Over-generation of variants leads to redundant coverage

### 3. Statistical Validation for Non-Deterministic Outputs

LLM outputs require statistical approaches to distinguish true regressions from natural variation.

```python
import statistics
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class StatisticalResult:
    mean_score: float
    std_dev: float
    confidence_interval: Tuple[float, float]
    passed: bool
    sample_size: int

class StatisticalValidator:
    """Validate LLM outputs across multiple samples"""
    
    def __init__(
        self, 
        n_samples: int = 10,
        confidence_level: float = 0.95,
        min_pass_rate: float = 0.8
    ):
        self.n_samples = n_samples
        self.confidence_level = confidence_level
        self.min_pass_rate = min_pass_rate
    
    def run_statistical_test(
        self,
        llm_generate: Callable,
        prompt: str,
        evaluator: Callable,
        temperature: float = 0.0
    ) -> StatisticalResult:
        """Execute prompt multiple times and validate statistically"""
        scores = []
        
        for i in range(self.n_samples):
            output = llm_generate(prompt, temperature=temperature)
            eval_result = evaluator(output)
            scores.append(eval_result.score)
        
        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        
        # Simplified confidence interval (use scipy.stats in production)
        margin = 1.96 * (std_dev / (len(scores) ** 0.5))  # 95% CI
        ci = (mean_score - margin, mean_score + margin)
        
        # Pass if lower bound of CI exceeds threshold
        passed = ci[0] >= self.min_pass_rate
        
        return StatisticalResult(
            mean_score=mean_score,
            std_dev=std_dev,
            confidence_interval=ci,
            passed=passed,
            sample_size=len(scores)
        )
    
    def compare_prompts(
        self,
        llm_generate: Callable,
        prompt_a: str,
        prompt_b: str,
        evaluator: Callable
    ) -> Dict[str, Any]:
        """A/B test two prompts statistically"""
        results_a = self.run_statistical_test(llm_generate, prompt_a, evaluator)
        results_b = self.run_statistical_test(llm_generate, prompt_b, evaluator)
        
        # Simple significance test (use proper t-test in production)
        score_diff = results_b.mean_score - results_a.mean_score
        pooled_std = ((results_a.std_dev ** 2 + results_b.std_dev ** 2) / 2) ** 0.5
        
        significant = abs(score_diff) > (1.96 * pooled_std)
        
        return {
            "prompt_a_score": results_a.mean_score,
            "prompt_b_score": results_b.mean_score,
            "difference": score_diff,
            "significant": significant,
            "recommendation": "B" if score_diff > 0 and significant else "A" if score_diff < 0 and significant else "No clear winner"
        }

# Mock LLM for demonstration
class MockLLM:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        import random
        random.seed(hash(prompt) + int(temperature * 100))
        
        # Simulate some variation
        if "classify