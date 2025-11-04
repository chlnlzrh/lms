# Tool Selection Decision Trees: Engineering Framework for AI/LLM Technology Choices

## Core Concepts

### What Is a Tool Selection Decision Tree?

A tool selection decision tree is a structured, deterministic framework for choosing between different AI/LLM tools, services, and architectures based on measurable technical constraints and requirements. Unlike traditional software where you might choose between databases or message queues based on well-established performance metrics, AI tool selection involves novel variables: token costs, latency-quality trade-offs, context window limits, and rapidly evolving capability landscapes.

**Traditional approach:**
```python
def select_database(requirements):
    # Simple decision based on stable, well-known metrics
    if requirements.transactions_per_second > 10000:
        return "high_performance_sql"
    elif requirements.schema_flexibility == "high":
        return "document_store"
    else:
        return "standard_sql"
```

**Modern AI tool selection:**
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ModelCapability(Enum):
    TEXT_GENERATION = "text"
    CODE_GENERATION = "code"
    MULTIMODAL = "multimodal"
    REASONING = "reasoning"

@dataclass
class AIRequirements:
    max_latency_ms: int
    max_cost_per_1k_requests: float
    min_quality_threshold: float
    context_window_tokens: int
    capability: ModelCapability
    privacy_level: str  # "public", "private", "confidential"
    expected_daily_requests: int

def select_ai_tool(requirements: AIRequirements) -> dict:
    """
    Multi-dimensional decision tree for AI tool selection.
    Returns recommended configuration with trade-off explanations.
    """
    decision = {
        "tool_type": None,
        "deployment": None,
        "model_class": None,
        "reasoning": []
    }
    
    # Privacy constraints eliminate entire categories
    if requirements.privacy_level == "confidential":
        decision["deployment"] = "self_hosted"
        decision["reasoning"].append(
            "Privacy requirement eliminates cloud APIs"
        )
    
    # Cost and scale interact non-linearly
    monthly_cost = (requirements.expected_daily_requests * 30 
                    * requirements.max_cost_per_1k_requests / 1000)
    
    if monthly_cost > 5000 and requirements.expected_daily_requests > 100000:
        decision["deployment"] = "self_hosted"
        decision["reasoning"].append(
            f"Scale ({requirements.expected_daily_requests}/day) "
            f"makes self-hosting cost-effective (${monthly_cost}/mo API cost)"
        )
    
    # Context window constraints limit model choices
    if requirements.context_window_tokens > 32000:
        decision["model_class"] = "long_context_specialist"
        decision["reasoning"].append(
            "Context requirement >32k eliminates standard models"
        )
    
    # Latency requirements interact with deployment
    if requirements.max_latency_ms < 500:
        if decision["deployment"] != "self_hosted":
            decision["reasoning"].append(
                "Sub-500ms latency difficult with API round-trips"
            )
        decision["model_class"] = "small_fast_model"
    
    return decision
```

### The Engineering Shift

Traditional software tool selection operates on stable metrics collected over years. AI tool selection requires thinking about:

1. **Cost volatility**: Pricing changes monthly; a tool that's cost-effective today may not be tomorrow
2. **Capability emergence**: New models can make yesterday's "impossible" problems trivial
3. **Multi-objective optimization**: You're balancing 5+ conflicting constraints simultaneously
4. **Deployment flexibility**: The same model can be API, self-hosted, or embedded with 100x cost differences

### Why This Matters Now

The AI tool landscape doubled in complexity in 2024 alone. Without a systematic decision framework, engineering teams:

- **Overspend by 10-50x** using high-capability models for simple tasks
- **Miss SLA requirements** by not accounting for P99 latency in API calls
- **Hit scaling walls** at 10k daily requests when architecture assumed unlimited API access
- **Rebuild entire systems** when requirements change slightly because initial choice was inflexible

A decision tree framework makes these trade-offs explicit and revisable as constraints change.

## Technical Components

### 1. Constraint Hierarchy and Elimination Criteria

Not all requirements are equal. Some constraints eliminate entire categories of solutions immediately. Understanding which constraints are *eliminators* vs. *optimizers* prevents wasted analysis.

**Technical explanation:**

Eliminators are hard constraints that prune the decision tree. Optimizers are soft constraints you balance in remaining choices.

```python
from typing import List, Dict, Any

class ConstraintType(Enum):
    ELIMINATOR = "eliminator"
    OPTIMIZER = "optimizer"

@dataclass
class Constraint:
    name: str
    type: ConstraintType
    threshold: Any
    current_value: Any
    
    def is_satisfied(self) -> bool:
        if isinstance(self.threshold, (int, float)):
            return self.current_value <= self.threshold
        return self.current_value == self.threshold

def evaluate_tool_option(
    option: Dict[str, Any],
    constraints: List[Constraint]
) -> tuple[bool, List[str]]:
    """
    Evaluate if tool option meets constraints.
    Returns (is_viable, reasons_for_elimination).
    """
    eliminators = [c for c in constraints if c.type == ConstraintType.ELIMINATOR]
    
    reasons = []
    for constraint in eliminators:
        option_value = option.get(constraint.name)
        
        if constraint.name == "data_privacy" and constraint.threshold == "on_premise":
            if option.get("deployment") == "cloud_api":
                reasons.append(
                    f"Privacy requires on-premise but option is cloud API"
                )
                
        elif constraint.name == "max_latency_ms":
            if option_value > constraint.threshold:
                reasons.append(
                    f"Latency {option_value}ms exceeds limit {constraint.threshold}ms"
                )
                
        elif constraint.name == "required_capability":
            if constraint.threshold not in option.get("capabilities", []):
                reasons.append(
                    f"Missing required capability: {constraint.threshold}"
                )
    
    is_viable = len(reasons) == 0
    return is_viable, reasons

# Example usage
tool_options = [
    {
        "name": "cloud_api_large",
        "deployment": "cloud_api",
        "latency_p50_ms": 800,
        "latency_p99_ms": 2000,
        "capabilities": ["text", "code", "reasoning"],
        "cost_per_1k": 0.03
    },
    {
        "name": "self_hosted_small",
        "deployment": "on_premise",
        "latency_p50_ms": 200,
        "latency_p99_ms": 400,
        "capabilities": ["text", "code"],
        "cost_per_1k": 0.002
    }
]

constraints = [
    Constraint("data_privacy", ConstraintType.ELIMINATOR, "on_premise", None),
    Constraint("max_latency_ms", ConstraintType.ELIMINATOR, 500, None),
    Constraint("cost_per_1k", ConstraintType.OPTIMIZER, 0.01, None)
]

for option in tool_options:
    # Update constraint current values from option
    for constraint in constraints:
        if constraint.name in option:
            constraint.current_value = option[constraint.name]
    
    viable, reasons = evaluate_tool_option(option, constraints)
    print(f"\n{option['name']}: {'✓ VIABLE' if viable else '✗ ELIMINATED'}")
    for reason in reasons:
        print(f"  - {reason}")
```

**Practical implications:**

- Check eliminators first—saves hours of detailed analysis
- Privacy/compliance eliminators are binary: cloud vs. self-hosted
- Latency eliminators depend on P99, not P50 (APIs have long tails)

**Real constraints:**

Privacy requirements can force 10x cost increases (self-hosting vs. API). This is non-negotiable—optimize everything else around it.

### 2. Cost Modeling Across Scale

AI tool costs are highly non-linear with scale. The optimal choice at 1k requests/day is often wrong at 100k requests/day.

**Technical explanation:**

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class CostModel:
    name: str
    fixed_monthly_cost: float
    variable_cost_per_1k: float
    setup_cost: float
    min_viable_scale: int
    
    def monthly_cost(self, requests_per_day: int) -> float:
        monthly_requests = requests_per_day * 30
        variable = (monthly_requests / 1000) * self.variable_cost_per_1k
        return self.fixed_monthly_cost + variable
    
    def total_cost_6_months(self, requests_per_day: int) -> float:
        """Include setup cost amortized over 6 months."""
        return self.setup_cost + (6 * self.monthly_cost(requests_per_day))

# Define cost models for different approaches
api_model = CostModel(
    name="Cloud API",
    fixed_monthly_cost=0,
    variable_cost_per_1k=0.02,
    setup_cost=0,
    min_viable_scale=1
)

self_hosted_gpu = CostModel(
    name="Self-hosted GPU",
    fixed_monthly_cost=800,  # GPU instance rental
    variable_cost_per_1k=0.001,
    setup_cost=5000,  # Engineering time
    min_viable_scale=10000
)

self_hosted_optimized = CostModel(
    name="Self-hosted Optimized",
    fixed_monthly_cost=2000,  # Multiple GPUs
    variable_cost_per_1k=0.0005,
    setup_cost=15000,  # Optimization engineering
    min_viable_scale=50000
)

def find_cost_optimal_solution(
    daily_requests: int,
    options: List[CostModel],
    time_horizon_months: int = 6
) -> tuple[CostModel, Dict[str, float]]:
    """
    Find most cost-effective solution at given scale.
    Returns chosen model and cost breakdown.
    """
    valid_options = [
        opt for opt in options 
        if daily_requests >= opt.min_viable_scale
    ]
    
    if not valid_options:
        # Fall back to API for below minimum scale
        valid_options = [api_model]
    
    costs = {}
    for option in valid_options:
        if time_horizon_months == 6:
            costs[option.name] = option.total_cost_6_months(daily_requests)
        else:
            monthly = option.monthly_cost(daily_requests)
            setup_amortized = option.setup_cost / time_horizon_months
            costs[option.name] = (monthly * time_horizon_months 
                                  + option.setup_cost)
    
    optimal = min(valid_options, key=lambda x: costs[x.name])
    return optimal, costs

# Analyze at different scales
scales = [1000, 10000, 50000, 200000]
print("Cost Analysis Across Scale\n" + "="*50)

for scale in scales:
    optimal, costs = find_cost_optimal_solution(
        scale,
        [api_model, self_hosted_gpu, self_hosted_optimized]
    )
    print(f"\n{scale:,} requests/day:")
    print(f"  Optimal: {optimal.name} (${costs[optimal.name]:,.0f} / 6mo)")
    print(f"  All options:")
    for name, cost in sorted(costs.items(), key=lambda x: x[1]):
        print(f"    {name}: ${cost:,.0f}")
```

**Practical implications:**

- Below 10k requests/day: APIs almost always win
- 10k-50k requests/day: Break-even zone; depends on growth trajectory
- Above 100k requests/day: Self-hosting typically 3-5x cheaper

**Real constraints:**

Setup costs ($5k-$20k engineering time) mean you need 6+ month commitment to justify self-hosting. If project might be cancelled, stay with APIs.

### 3. Latency Budgets and Architecture Implications

Latency requirements dictate not just model choice, but entire architecture patterns. A 100ms requirement is fundamentally different from 1000ms.

**Technical explanation:**

```python
import time
from typing import Optional
from enum import Enum

class LatencyBudget(Enum):
    INTERACTIVE = 100      # User waiting, typing paused
    RESPONSIVE = 500       # User waiting, expects quick response
    ASYNC = 2000           # Background processing
    BATCH = 10000          # Offline processing

@dataclass
class LatencyProfile:
    model_inference_p50_ms: int
    model_inference_p99_ms: int
    network_roundtrip_p50_ms: int
    network_roundtrip_p99_ms: int
    preprocessing_ms: int
    postprocessing_ms: int
    
    def total_p50_latency(self) -> int:
        return (self.model_inference_p50_ms + 
                self.network_roundtrip_p50_ms +
                self.preprocessing_ms + 
                self.postprocessing_ms)
    
    def total_p99_latency(self) -> int:
        """P99 network and inference compound, not add linearly."""
        return (self.model_inference_p99_ms + 
                self.network_roundtrip_p99_ms +
                self.preprocessing_ms + 
                self.postprocessing_ms)

# Compare deployment options for latency
cloud_api_profile = LatencyProfile(
    model_inference_p50_ms=400,
    model_inference_p99_ms=1200,
    network_roundtrip_p50_ms=50,
    network_roundtrip_p99_ms=300,
    preprocessing_ms=10,
    postprocessing_ms=10
)

self_hosted_large = LatencyProfile(
    model_inference_p50_ms=600,
    model_inference_p99_ms=900,
    network_roundtrip_p50_ms=5,   # Local network
    network_roundtrip_p99_ms=20,
    preprocessing_ms=10,
    postprocessing_ms=10
)

self_hosted_small = LatencyProfile(
    model_inference_p50_ms=150,
    model_inference_p99_ms=250,
    network_roundtrip_p50_ms=5,
    network_roundtrip_p99_ms=20,
    preprocessing_ms=10,
    postprocessing_ms=10
)

def meets_latency_budget(
    profile: LatencyProfile,
    budget: LatencyBudget,
    sla_percentile: str = "p99"
) -> tuple[bool, int, str]:
    """
    Check if latency profile meets budget.
    Returns (meets_budget, actual_latency, recommendation).
    """
    if sla_percentile == "p99":
        actual = profile.total_p99_latency()
    else:
        actual = profile.total_p50_latency()
    
    meets = actual <= budget.value
    
    recommendation = ""
    if not meets:
        overage = actual - budget.value
        if overage < 200:
            recommendation = "Consider caching or speculative execution"
        elif profile.network_roundtrip_p99_ms > 100:
            recommendation = "Network latency dominates—move to self-hosted"
        else:
            recommendation = "Model too slow—switch to smaller/faster model"
    
    return meets, actual, recommendation

# Evaluate options against budget
budget = LatencyBudget.RESPONSIVE  # 500ms requirement

profiles = [
    ("Cloud API", cloud_api_profile),
    ("Self-hosted Large", self_hosted_large),
    ("Self-hosted Small", self_hosted_small)
]

print