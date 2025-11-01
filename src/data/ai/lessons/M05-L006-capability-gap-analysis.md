# Capability Gap Analysis: Engineering Assessment Framework for AI Systems

## Core Concepts

**Technical Definition:** Capability gap analysis is a systematic engineering methodology for identifying the delta between what an AI system can reliably deliver and what your application requirements demand. It combines empirical testing, performance profiling, and constraint mapping to produce quantifiable assessments that inform architecture decisions, model selection, and implementation strategies.

### Traditional vs. Modern Approach

```python
# Traditional Software: Capabilities are deterministic and documented
class TraditionalAPI:
    """Capabilities are explicit contracts"""
    
    def parse_date(self, date_str: str) -> datetime:
        """
        Supports: ISO8601, RFC3339
        Does NOT support: Natural language ("next Tuesday")
        Failure mode: Raises ValueError with specific error
        Performance: O(1), <1ms
        """
        return datetime.fromisoformat(date_str)
    
    # You know EXACTLY what works and what doesn't
    # Binary: works or throws exception
    # No surprises in production

# AI System: Capabilities are probabilistic and emergent
class LLMDateParser:
    """Capabilities are fuzzy boundaries"""
    
    def parse_date(self, date_str: str) -> datetime:
        """
        Supports: ??? (ISO8601? Natural language? Ambiguous formats?)
        Performance: ??? (varies by input complexity, 50-500ms)
        Accuracy: ??? (depends on phrasing, context, edge cases)
        Failure mode: ??? (wrong date? hallucination? refusal?)
        Cost: ??? (tokens vary by prompt + response)
        """
        response = llm.complete(f"Parse this date: {date_str}")
        return self._extract_date(response)
    
    # You must DISCOVER capabilities through testing
    # Continuous spectrum: excellent -> good -> poor -> fails
    # Production surprises are the default without gap analysis
```

The fundamental difference: traditional software has **defined capabilities**, AI systems have **discovered capabilities**. Gap analysis is the engineering discipline that maps that discovery process into actionable decisions.

### Why This Matters Now

Three converging factors make capability gap analysis critical:

1. **Model proliferation**: 50+ viable models with overlapping but distinct capabilities. Choosing wrong costs 10-100x in latency, cost, or accuracy.
2. **Hidden performance cliffs**: LLMs perform excellently on 95% of inputs, then catastrophically fail on edge cases you didn't test. Gap analysis finds those cliffs before users do.
3. **Economics**: The difference between GPT-4 and a fine-tuned small model is $0.01 vs $0.0001 per request. At 10M requests/month, that's $100K vs $1K. Gap analysis tells you which you actually need.

**Key Insight:** The most expensive AI systems are those built without understanding capability gaps—they either over-provision (10x cost for unneeded capability) or under-deliver (launch to production, then discover the model can't handle 20% of real traffic).

## Technical Components

### 1. Capability Taxonomy Construction

Build a structured map of capabilities your application requires, organized by criticality and measurability.

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Callable

class Criticality(Enum):
    CRITICAL = "must_work"      # System unusable if fails
    IMPORTANT = "should_work"   # Degraded experience if fails
    NICE_TO_HAVE = "could_work" # Enhancement if works

class Measurability(Enum):
    OBJECTIVE = "objective"     # Right/wrong answer exists
    SUBJECTIVE = "subjective"   # Quality judgment required
    BEHAVIORAL = "behavioral"   # Complex multi-turn evaluation

@dataclass
class Capability:
    """Single testable capability requirement"""
    id: str
    description: str
    criticality: Criticality
    measurability: Measurability
    success_threshold: float  # 0.0 - 1.0
    test_cases: List[dict]
    evaluation_fn: Callable

# Example: Customer support chatbot taxonomy
capabilities = [
    Capability(
        id="intent_classification",
        description="Classify customer intent into 12 categories",
        criticality=Criticality.CRITICAL,
        measurability=Measurability.OBJECTIVE,
        success_threshold=0.95,
        test_cases=[
            {"input": "I want to return my order", "expected": "return_request"},
            {"input": "My package never arrived", "expected": "delivery_issue"},
            # ... 200 more test cases covering edge cases
        ],
        evaluation_fn=lambda pred, expected: pred == expected
    ),
    Capability(
        id="empathy_tone",
        description="Respond with appropriate empathy to frustrated customers",
        criticality=Criticality.IMPORTANT,
        measurability=Measurability.SUBJECTIVE,
        success_threshold=0.80,
        test_cases=[
            {
                "input": "This is the third time I've contacted you! Nothing works!",
                "quality_dimensions": ["acknowledges_frustration", "apologetic", "action_oriented"]
            }
        ],
        evaluation_fn=lambda response: score_empathy(response)
    ),
    Capability(
        id="policy_lookup",
        description="Accurately retrieve and apply return policy details",
        criticality=Criticality.CRITICAL,
        measurability=Measurability.OBJECTIVE,
        success_threshold=0.98,
        test_cases=[
            {
                "input": "Can I return this after 45 days?",
                "expected_facts": ["30_day_policy", "no_after_30_days"],
                "expected_answer": "no"
            }
        ],
        evaluation_fn=lambda response, facts: all(f in response for f in facts)
    )
]
```

**Practical Implications:**
- **Criticality drives model selection**: CRITICAL capabilities may require multiple models, human-in-loop, or guardrails
- **Measurability determines test methodology**: Objective = automated CI/CD, Subjective = human evaluation batch, Behavioral = live A/B testing
- **Success thresholds create clear go/no-go**: "Works well enough" becomes "exceeds 95% accuracy on test suite"

**Trade-offs:**
- Comprehensive taxonomy = more work upfront, but prevents costly late discoveries
- Too many capabilities = analysis paralysis. Start with 5-10 critical ones.
- Success thresholds too high = may force expensive models unnecessarily

### 2. Empirical Performance Profiling

Systematically measure actual capability performance across models, prompts, and conditions.

```python
import time
from typing import Dict, Any
import numpy as np

@dataclass
class PerformanceProfile:
    """Measured capability performance for a specific model+prompt"""
    capability_id: str
    model_name: str
    prompt_template: str
    
    # Accuracy metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Performance metrics
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Cost metrics
    avg_prompt_tokens: int
    avg_completion_tokens: int
    cost_per_request: float
    
    # Reliability metrics
    error_rate: float
    timeout_rate: float
    
    # Edge case analysis
    worst_performing_categories: List[tuple[str, float]]

class CapabilityProfiler:
    """Profile model performance on specific capability"""
    
    def __init__(self, capability: Capability):
        self.capability = capability
        
    def profile_model(
        self,
        model_name: str,
        prompt_template: str,
        model_fn: Callable,
        cost_per_1k_tokens: Dict[str, float]
    ) -> PerformanceProfile:
        """Run full performance profile on model"""
        
        results = []
        latencies = []
        costs = []
        errors = 0
        
        for test_case in self.capability.test_cases:
            try:
                # Measure latency
                start = time.perf_counter()
                response = model_fn(
                    prompt_template.format(**test_case),
                    model=model_name
                )
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                
                # Measure accuracy
                is_correct = self.capability.evaluation_fn(
                    response.content,
                    test_case.get("expected")
                )
                results.append(is_correct)
                
                # Measure cost
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                cost = (
                    prompt_tokens * cost_per_1k_tokens["prompt"] / 1000 +
                    completion_tokens * cost_per_1k_tokens["completion"] / 1000
                )
                costs.append(cost)
                
            except Exception as e:
                errors += 1
                results.append(False)
        
        # Calculate aggregate metrics
        accuracy = np.mean(results)
        
        # Find worst-performing subcategories
        category_performance = self._analyze_categories(results)
        
        return PerformanceProfile(
            capability_id=self.capability.id,
            model_name=model_name,
            prompt_template=prompt_template,
            accuracy=accuracy,
            precision=self._calculate_precision(results),
            recall=self._calculate_recall(results),
            f1_score=self._calculate_f1(results),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            avg_prompt_tokens=int(np.mean([c[0] for c in costs])),
            avg_completion_tokens=int(np.mean([c[1] for c in costs])),
            cost_per_request=np.mean(costs),
            error_rate=errors / len(self.capability.test_cases),
            timeout_rate=0.0,  # Track separately
            worst_performing_categories=category_performance[:5]
        )
    
    def _analyze_categories(self, results: List[bool]) -> List[tuple[str, float]]:
        """Identify which types of inputs perform worst"""
        # Group results by test case category/tags
        # Return sorted by performance (worst first)
        pass

# Usage: Compare 3 models on same capability
profiler = CapabilityProfiler(capabilities[0])  # intent_classification

profiles = [
    profiler.profile_model("gpt-4o", prompt_v1, gpt4_fn, {"prompt": 0.005, "completion": 0.015}),
    profiler.profile_model("gpt-4o-mini", prompt_v1, gpt4mini_fn, {"prompt": 0.00015, "completion": 0.0006}),
    profiler.profile_model("claude-3-haiku", prompt_v2, haiku_fn, {"prompt": 0.00025, "completion": 0.00125})
]

# Now you have quantitative comparison
for p in profiles:
    print(f"{p.model_name}: {p.accuracy:.1%} accuracy, "
          f"${p.cost_per_request*1000:.2f}/1K requests, "
          f"{p.p95_latency_ms:.0f}ms p95")
```

**Practical Implications:**
- **P95/P99 latency matters more than average**: User experience is determined by slow requests
- **Worst-performing categories reveal gaps**: "95% accuracy" might hide "30% accuracy on edge cases that represent 20% of real traffic"
- **Cost scales non-linearly**: 2x longer prompts = 2x cost, but better prompts might enable 10x cheaper model

**Real Constraint:** Running full profiles is expensive. Start with 100-test baseline, expand to 1000+ for critical capabilities only after you've narrowed to 2-3 candidate models.

### 3. Gap Quantification & Decision Matrices

Transform performance profiles into actionable decisions using structured comparison.

```python
@dataclass
class CapabilityGap:
    """Quantified gap between requirement and capability"""
    capability_id: str
    required_threshold: float
    measured_performance: float
    gap: float  # Negative = exceeds requirement
    gap_severity: str  # "critical", "significant", "minor", "met"
    
    # What it would take to close gap
    estimated_improvement_methods: List[Dict[str, Any]]
    
    # Impact if gap not closed
    business_impact: str
    mitigation_options: List[str]

class GapAnalyzer:
    """Analyze capability gaps and generate recommendations"""
    
    def analyze_gaps(
        self,
        capabilities: List[Capability],
        profiles: List[PerformanceProfile]
    ) -> Dict[str, List[CapabilityGap]]:
        """Generate gap analysis for all capability-model combinations"""
        
        gaps_by_model = {}
        
        for profile in profiles:
            gaps = []
            
            # Find matching capability
            cap = next(c for c in capabilities if c.id == profile.capability_id)
            
            # Calculate gap
            gap = cap.success_threshold - profile.accuracy
            
            if gap > 0:
                # Determine severity
                if gap > 0.15:
                    severity = "critical"
                elif gap > 0.05:
                    severity = "significant"
                else:
                    severity = "minor"
                
                # Estimate what it would take to close gap
                improvement_methods = self._estimate_improvements(
                    gap, profile, cap
                )
                
                gaps.append(CapabilityGap(
                    capability_id=cap.id,
                    required_threshold=cap.success_threshold,
                    measured_performance=profile.accuracy,
                    gap=gap,
                    gap_severity=severity,
                    estimated_improvement_methods=improvement_methods,
                    business_impact=self._assess_business_impact(cap, gap),
                    mitigation_options=self._generate_mitigations(cap, gap)
                ))
            
            gaps_by_model[profile.model_name] = gaps
        
        return gaps_by_model
    
    def _estimate_improvements(
        self,
        gap: float,
        profile: PerformanceProfile,
        capability: Capability
    ) -> List[Dict[str, Any]]:
        """Estimate methods to close gap and their likelihood"""
        
        methods = []
        
        # Prompt engineering
        if gap < 0.10:
            methods.append({
                "method": "prompt_optimization",
                "estimated_improvement": "3-8%",
                "effort": "low",
                "timeline": "1-3 days",
                "cost_impact": "none"
            })
        
        # Few-shot examples
        if gap < 0.15:
            methods.append({
                "method": "few_shot_examples",
                "estimated_improvement": "5-12%",
                "effort": "low",
                "timeline": "2-5 days",
                "cost_impact": "10-20% token increase"
            })
        
        # Fine-tuning
        if gap < 0.20:
            methods.append({
                "method": "fine_tuning",
                "estimated_improvement": "10-25%",
                "effort": "medium",
                "timeline": "1-2 weeks",
                "cost_impact": "training cost + hosting"
            })
        
        # Larger model
        methods.append({
            "method": "upgrade_to_larger_model",
            "estimated_improvement": "varies widely",
            "effort": "low",
            "timeline": "immediate",
            "cost_impact": "2-10x per request"
        })
        
        # Ensemble/cascade
        if capability.criticality == Criticality.CRITICAL:
            methods.append({
                "method": "model_cascade",
                "estimated_improvement": "15-30%",
                "effort": "medium",
                "timeline": "1 week",
                "cost_impact": "20-50% increase (smart routing)"
            })
        
        return methods
    
    def generate_recommendation(
        self,