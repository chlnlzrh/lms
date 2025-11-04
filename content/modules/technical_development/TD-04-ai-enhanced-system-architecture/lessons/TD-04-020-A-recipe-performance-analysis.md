# Recipe Performance Analysis: Optimizing LLM Outputs Through Systematic Measurement

## Core Concepts

Recipe performance analysis is the systematic measurement and optimization of LLM prompt configurations (recipes) based on quantitative metrics. Unlike traditional software where performance means execution speed and memory usage, LLM recipe performance encompasses accuracy, consistency, cost, latency, and output quality—all of which must be balanced against each other.

### Traditional vs. Modern Approach

```python
# Traditional software optimization: Profile, identify bottleneck, fix
import time
from typing import List

def traditional_search(data: List[str], query: str) -> List[str]:
    """Linear search - easy to measure, obvious optimization path"""
    start = time.time()
    results = [item for item in data if query.lower() in item.lower()]
    latency = time.time() - start
    print(f"Latency: {latency:.3f}s, Results: {len(results)}")
    return results

# LLM recipe optimization: Multiple dimensions, unclear trade-offs
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class RecipeMetrics:
    """Multi-dimensional performance measurement"""
    accuracy: float  # 0-1, requires ground truth
    consistency: float  # 0-1, output variance across runs
    latency_p50: float  # median response time
    latency_p95: float  # 95th percentile response time
    cost_per_call: float  # dollars per invocation
    tokens_in: int  # average input tokens
    tokens_out: int  # average output tokens
    
    def score(self, weights: Optional[dict] = None) -> float:
        """Composite score with configurable priorities"""
        if weights is None:
            weights = {
                'accuracy': 0.4,
                'consistency': 0.2,
                'latency': 0.2,
                'cost': 0.2
            }
        
        # Normalize latency (lower is better, cap at 10s)
        latency_score = max(0, 1 - (self.latency_p95 / 10.0))
        # Normalize cost (lower is better, cap at $0.10)
        cost_score = max(0, 1 - (self.cost_per_call / 0.10))
        
        return (
            weights['accuracy'] * self.accuracy +
            weights['consistency'] * self.consistency +
            weights['latency'] * latency_score +
            weights['cost'] * cost_score
        )
```

The fundamental shift: **optimization becomes multi-objective and probabilistic**. You can't simply "make it faster"—you must decide which dimensions matter for your use case and measure how changes affect all of them.

### Why This Matters Now

1. **Non-determinism requires statistical measurement**: A single test run tells you almost nothing. You need aggregate metrics across dozens or hundreds of runs.

2. **Cost scales with usage**: A 2x improvement in prompt efficiency can mean the difference between $500/month and $1000/month at scale.

3. **Quality isn't binary**: Unlike traditional software where outputs are correct or incorrect, LLM outputs exist on a spectrum. You need frameworks to quantify "good enough."

4. **Recipe changes have non-obvious effects**: Adding three words to a prompt might improve accuracy by 15% but double latency. You won't know without measurement.

## Technical Components

### 1. Metric Definition and Collection

Effective recipe analysis requires defining metrics that align with your actual requirements, not just what's easy to measure.

```python
from typing import Any, Callable, List, Dict
import statistics
from datetime import datetime
import hashlib

class MetricCollector:
    """Systematic collection of recipe performance data"""
    
    def __init__(self):
        self.runs: List[Dict[str, Any]] = []
    
    def record_run(
        self,
        recipe_id: str,
        input_data: str,
        output: str,
        latency: float,
        tokens_in: int,
        tokens_out: int,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Record a single recipe execution"""
        run_data = {
            'recipe_id': recipe_id,
            'timestamp': datetime.utcnow().isoformat(),
            'input_hash': hashlib.md5(input_data.encode()).hexdigest(),
            'output': output,
            'latency': latency,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'ground_truth': ground_truth,
            'metadata': metadata or {}
        }
        self.runs.append(run_data)
    
    def analyze_recipe(
        self,
        recipe_id: str,
        accuracy_fn: Optional[Callable[[str, str], float]] = None,
        cost_per_1k_in: float = 0.01,  # $0.01 per 1K input tokens
        cost_per_1k_out: float = 0.03  # $0.03 per 1K output tokens
    ) -> RecipeMetrics:
        """Compute aggregate metrics for a recipe"""
        recipe_runs = [r for r in self.runs if r['recipe_id'] == recipe_id]
        
        if not recipe_runs:
            raise ValueError(f"No runs found for recipe {recipe_id}")
        
        # Latency statistics
        latencies = [r['latency'] for r in recipe_runs]
        latency_p50 = statistics.median(latencies)
        latency_p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        
        # Token statistics
        avg_tokens_in = statistics.mean([r['tokens_in'] for r in recipe_runs])
        avg_tokens_out = statistics.mean([r['tokens_out'] for r in recipe_runs])
        
        # Cost calculation
        cost_per_call = (
            (avg_tokens_in / 1000) * cost_per_1k_in +
            (avg_tokens_out / 1000) * cost_per_1k_out
        )
        
        # Accuracy (if ground truth available and accuracy function provided)
        accuracy = 0.0
        if accuracy_fn:
            accuracy_scores = []
            for run in recipe_runs:
                if run['ground_truth']:
                    score = accuracy_fn(run['output'], run['ground_truth'])
                    accuracy_scores.append(score)
            accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # Consistency: measure output variance for same inputs
        consistency = self._calculate_consistency(recipe_runs)
        
        return RecipeMetrics(
            accuracy=accuracy,
            consistency=consistency,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            cost_per_call=cost_per_call,
            tokens_in=int(avg_tokens_in),
            tokens_out=int(avg_tokens_out)
        )
    
    def _calculate_consistency(self, runs: List[Dict]) -> float:
        """Measure output consistency for identical inputs"""
        from collections import defaultdict
        
        # Group runs by input hash
        input_groups = defaultdict(list)
        for run in runs:
            input_groups[run['input_hash']].append(run['output'])
        
        # Calculate similarity within each group
        consistency_scores = []
        for outputs in input_groups.values():
            if len(outputs) < 2:
                continue
            
            # Simple string similarity (can be replaced with semantic similarity)
            similarities = []
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    sim = self._string_similarity(outputs[i], outputs[j])
                    similarities.append(sim)
            
            if similarities:
                consistency_scores.append(statistics.mean(similarities))
        
        return statistics.mean(consistency_scores) if consistency_scores else 1.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple character-level similarity"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # Jaccard similarity on character bigrams
        bigrams1 = set(s1[i:i+2] for i in range(len(s1) - 1))
        bigrams2 = set(s2[i:i+2] for i in range(len(s2) - 1))
        
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return intersection / union if union > 0 else 0.0
```

**Practical Implications:**
- You need a minimum of 20-30 runs per recipe for stable metrics
- Consistency measurement requires multiple runs with identical inputs
- Cost must factor in both input and output tokens with correct pricing
- Accuracy requires ground truth data, which is expensive to create

**Trade-offs:**
- More runs = better statistical confidence but higher measurement cost
- Simple similarity metrics are fast but miss semantic meaning
- Automated metrics may not capture subjective quality aspects

### 2. Test Dataset Construction

Performance analysis is only as good as your test data. You need representative samples that cover edge cases, typical inputs, and stress scenarios.

```python
from enum import Enum
from typing import Set
import random

class TestCaseType(Enum):
    TYPICAL = "typical"
    EDGE_CASE = "edge_case"
    STRESS = "stress"
    ADVERSARIAL = "adversarial"

@dataclass
class TestCase:
    input_text: str
    expected_output: Optional[str]
    case_type: TestCaseType
    tags: Set[str]
    
class TestDataset:
    """Curated test dataset for recipe evaluation"""
    
    def __init__(self):
        self.cases: List[TestCase] = []
    
    def add_case(
        self,
        input_text: str,
        expected_output: Optional[str] = None,
        case_type: TestCaseType = TestCaseType.TYPICAL,
        tags: Optional[Set[str]] = None
    ) -> None:
        """Add a test case to the dataset"""
        self.cases.append(TestCase(
            input_text=input_text,
            expected_output=expected_output,
            case_type=case_type,
            tags=tags or set()
        ))
    
    def get_sample(
        self,
        n: int = 30,
        distribution: Optional[Dict[TestCaseType, float]] = None
    ) -> List[TestCase]:
        """Get stratified sample of test cases"""
        if distribution is None:
            # Default: 70% typical, 15% edge, 10% stress, 5% adversarial
            distribution = {
                TestCaseType.TYPICAL: 0.70,
                TestCaseType.EDGE_CASE: 0.15,
                TestCaseType.STRESS: 0.10,
                TestCaseType.ADVERSARIAL: 0.05
            }
        
        sample = []
        for case_type, proportion in distribution.items():
            type_cases = [c for c in self.cases if c.case_type == case_type]
            n_sample = int(n * proportion)
            if type_cases:
                sample.extend(random.sample(
                    type_cases,
                    min(n_sample, len(type_cases))
                ))
        
        return sample
    
    def get_by_tags(self, tags: Set[str]) -> List[TestCase]:
        """Filter test cases by tags"""
        return [c for c in self.cases if c.tags & tags]

# Example: Building a test dataset for sentiment analysis
def build_sentiment_test_dataset() -> TestDataset:
    dataset = TestDataset()
    
    # Typical cases
    dataset.add_case(
        "The product works great and arrived quickly!",
        "positive",
        TestCaseType.TYPICAL,
        {"sentiment", "product_review"}
    )
    
    dataset.add_case(
        "Terrible quality, broke after one use.",
        "negative",
        TestCaseType.TYPICAL,
        {"sentiment", "product_review"}
    )
    
    # Edge cases: Mixed sentiment
    dataset.add_case(
        "Great quality but horrible customer service.",
        "mixed",
        TestCaseType.EDGE_CASE,
        {"sentiment", "mixed", "product_review"}
    )
    
    # Stress test: Very long input
    dataset.add_case(
        "The product is " + "absolutely amazing " * 50 + "!",
        "positive",
        TestCaseType.STRESS,
        {"sentiment", "long_input"}
    )
    
    # Adversarial: Sarcasm
    dataset.add_case(
        "Oh great, another broken item. Just what I needed.",
        "negative",
        TestCaseType.ADVERSARIAL,
        {"sentiment", "sarcasm"}
    )
    
    return dataset
```

**Practical Implications:**
- Test datasets should mirror production data distribution
- Edge cases often reveal the biggest performance differences between recipes
- You need labeled data (expected outputs) for accuracy measurement
- Small datasets (30-50 cases) are sufficient for initial comparison

**Real Constraints:**
- Creating ground truth is expensive—prioritize cases that matter most
- Test data becomes stale—production patterns shift over time
- Adversarial cases are hard to anticipate before deployment

### 3. Comparative Analysis

The goal isn't absolute performance—it's choosing the best recipe for your requirements. Comparative analysis makes trade-offs explicit.

```python
from typing import List, Tuple
import json

class RecipeComparison:
    """Compare multiple recipes on same test dataset"""
    
    def __init__(self, collector: MetricCollector):
        self.collector = collector
    
    def compare_recipes(
        self,
        recipe_ids: List[str],
        priority_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Generate comparison report for multiple recipes"""
        
        metrics_by_recipe = {}
        for recipe_id in recipe_ids:
            try:
                metrics = self.collector.analyze_recipe(recipe_id)
                metrics_by_recipe[recipe_id] = metrics
            except ValueError:
                continue
        
        if not metrics_by_recipe:
            return {"error": "No valid recipe data"}
        
        # Calculate composite scores
        scores = {
            recipe_id: metrics.score(priority_weights)
            for recipe_id, metrics in metrics_by_recipe.items()
        }
        
        # Rank recipes
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Find Pareto optimal recipes (not dominated on any metric)
        pareto_optimal = self._find_pareto_optimal(metrics_by_recipe)
        
        return {
            'ranked_recipes': ranked,
            'pareto_optimal': pareto_optimal,
            'detailed_metrics': {
                recipe_id: {
                    'accuracy': m.accuracy,
                    'consistency': m.consistency,
                    'latency_p50': m.latency_p50,
                    'latency_p95': m.latency_p95,
                    'cost_per_call': m.cost_per_call,
                    'composite_score': scores[recipe_id]
                }
                for recipe_id, m in metrics_by_recipe.items()
            },
            'recommendations': self._generate_recommendations(
                metrics_by_recipe,
                ranked
            )
        }
    
    def _find_pareto_optimal(
        self,
        metrics_by_recipe: Dict[str, RecipeMetrics]
    ) -> List[str]:
        """Find recipes not dominated on all metrics"""
        pareto_set = []
        
        for recipe_id, metrics in metrics_by_recipe.items():
            is_dominated = False
            
            for other_id, other_metrics in metrics_by_recipe.items():
                if recipe_id == other_id:
                    continue
                
                # Check if other recipe dominates this one
                if (other_metrics.accuracy >= metrics.accuracy and
                    other_metrics.consistency >= metrics.consistency