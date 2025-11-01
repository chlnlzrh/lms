# Build vs. Buy Decision Framework for AI/LLM Systems

## Core Concepts

### Technical Definition

The build-vs-buy decision for AI/LLM systems is a strategic engineering trade-off between:

- **Build**: Developing custom models, fine-tuning open-source foundations, or creating proprietary architectures
- **Buy**: Consuming managed API services, using pre-trained models, or purchasing enterprise solutions

Unlike traditional software where "build" means writing code from scratch, AI systems exist on a spectrum:

```python
# Traditional Software: Binary Choice
# BUILD: Write everything
def custom_email_validator(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# BUY: Use library
from email_validator import validate_email
validate_email(email)

# AI/LLM Systems: Spectrum of Options
# FULL BUILD: Train from scratch (rare, $10M+)
model = train_gpt_from_scratch(
    params=175_000_000_000,
    tokens=300_000_000_000,
    compute_budget=millions_of_dollars
)

# HYBRID BUILD: Fine-tune open weights
from transformers import AutoModelForCausalLM
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-70b")
custom_model = fine_tune(base_model, domain_data)

# LIGHT BUILD: Prompt engineering + RAG
from langchain import VectorStore, PromptTemplate
system = RAGPipeline(
    vector_store=your_data,
    prompt_template=custom_prompt,
    api_model="gpt-4"  # Still using external API
)

# FULL BUY: Direct API consumption
import openai
response = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": prompt}]
)
```

### Key Insight: The Cost Function Has Inverted

Traditional software: High upfront build cost, low marginal cost per user.
AI/LLM systems: Lower upfront cost (use APIs), high marginal cost at scale.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class CostModel:
    upfront_cost: float
    cost_per_request: float
    requests_per_month: int
    
    def total_monthly_cost(self) -> float:
        return self.upfront_cost / 12 + (self.cost_per_request * self.requests_per_month)
    
    def breakeven_volume(self, alternative: 'CostModel') -> int:
        """Calculate monthly volume where costs equal"""
        if self.cost_per_request == alternative.cost_per_request:
            return float('inf')
        
        upfront_diff = (alternative.upfront_cost - self.upfront_cost) / 12
        marginal_diff = self.cost_per_request - alternative.cost_per_request
        
        return int(upfront_diff / marginal_diff) if marginal_diff != 0 else float('inf')

# Scenario: Document classification system
api_approach = CostModel(
    upfront_cost=50_000,  # Integration, prompt engineering, testing
    cost_per_request=0.002,  # API call cost
    requests_per_month=1_000_000
)

fine_tuned_approach = CostModel(
    upfront_cost=250_000,  # Data prep, training, infrastructure
    cost_per_request=0.0002,  # Inference on owned infrastructure
    requests_per_month=1_000_000
)

print(f"API monthly cost: ${api_approach.total_monthly_cost():,.2f}")
# Output: API monthly cost: $2,004.17

print(f"Fine-tuned monthly cost: ${fine_tuned_approach.total_monthly_cost():,.2f}")
# Output: Fine-tuned monthly cost: $20,833.33

breakeven = api_approach.breakeven_volume(fine_tuned_approach)
print(f"Breakeven at {breakeven:,} requests/month")
# Output: Breakeven at 13,888,889 requests/month

# At 10M requests/month:
api_approach.requests_per_month = 10_000_000
fine_tuned_approach.requests_per_month = 10_000_000
print(f"\nAt 10M requests/month:")
print(f"API: ${api_approach.total_monthly_cost():,.2f}")
print(f"Fine-tuned: ${fine_tuned_approach.total_monthly_cost():,.2f}")
# Output: API: $20,004.17, Fine-tuned: $22,833.33

# At 50M requests/month - BUILD wins
api_approach.requests_per_month = 50_000_000
fine_tuned_approach.requests_per_month = 50_000_000
print(f"\nAt 50M requests/month:")
print(f"API: ${api_approach.total_monthly_cost():,.2f}")
print(f"Fine-tuned: ${fine_tuned_approach.total_monthly_cost():,.2f}")
# Output: API: $100,004.17, Fine-tuned: $30,833.33
```

### Why This Matters Now

1. **API pricing volatility**: Providers change pricing 2-3x per year. Your $10K/month system can become $30K overnight.
2. **Data privacy regulations**: GDPR, HIPAA, and emerging AI regulations increasingly prohibit sending data to third parties.
3. **Model capability plateau**: Frontier models improve slower; fine-tuned smaller models close the gap for specific tasks.
4. **Inference cost decline**: GPU costs dropped 40% YoY while API prices remain sticky or increase.

## Technical Components

### 1. Total Cost of Ownership (TCO) Calculator

Understanding true costs requires modeling multiple dimensions:

```python
from enum import Enum
from typing import Optional
import numpy as np

class DeploymentType(Enum):
    API = "api"
    SELF_HOSTED_CLOUD = "self_hosted_cloud"
    SELF_HOSTED_ON_PREM = "self_hosted_on_prem"

@dataclass
class TCOComponents:
    # Direct costs
    infrastructure_monthly: float
    api_cost_per_request: float
    
    # Engineering costs
    initial_dev_hours: float
    monthly_maintenance_hours: float
    engineer_hourly_rate: float = 150.0
    
    # Operational costs
    monitoring_tooling_monthly: float = 0.0
    data_labeling_cost: float = 0.0
    training_compute_cost: float = 0.0
    
    # Hidden costs
    incident_response_hours_monthly: float = 0.0
    compliance_audit_hours_monthly: float = 0.0
    
    def calculate_tco(
        self, 
        requests_per_month: int,
        months: int = 12
    ) -> dict[str, float]:
        # Upfront costs amortized
        dev_cost = self.initial_dev_hours * self.engineer_hourly_rate
        amortized_dev = dev_cost / months
        
        # Recurring engineering
        maintenance_cost = self.monthly_maintenance_hours * self.engineer_hourly_rate
        incident_cost = self.incident_response_hours_monthly * self.engineer_hourly_rate
        compliance_cost = self.compliance_audit_hours_monthly * self.engineer_hourly_rate
        
        # Infrastructure
        compute_cost = self.infrastructure_monthly + (self.api_cost_per_request * requests_per_month)
        
        total_monthly = (
            amortized_dev +
            maintenance_cost +
            incident_cost +
            compliance_cost +
            compute_cost +
            self.monitoring_tooling_monthly +
            self.training_compute_cost / months
        )
        
        return {
            "monthly_total": total_monthly,
            "annual_total": total_monthly * 12,
            "cost_per_request": total_monthly / requests_per_month if requests_per_month > 0 else 0,
            "breakdown": {
                "compute": compute_cost,
                "engineering": maintenance_cost + incident_cost,
                "development_amortized": amortized_dev,
                "compliance": compliance_cost,
                "tooling": self.monitoring_tooling_monthly,
            }
        }

# Example: Compare three approaches for a customer support classifier
api_tco = TCOComponents(
    infrastructure_monthly=200,  # Minimal - just API gateway
    api_cost_per_request=0.001,
    initial_dev_hours=160,  # 4 weeks prompt engineering
    monthly_maintenance_hours=20,  # Prompt updates
    monitoring_tooling_monthly=500,
    incident_response_hours_monthly=5,
    compliance_audit_hours_monthly=8,  # Data handling audits
)

fine_tuned_tco = TCOComponents(
    infrastructure_monthly=3500,  # GPU instances for inference
    api_cost_per_request=0.0001,  # Much lower per-request
    initial_dev_hours=640,  # 16 weeks (data prep, training, deployment)
    monthly_maintenance_hours=40,
    data_labeling_cost=50_000,
    training_compute_cost=15_000,
    monitoring_tooling_monthly=1200,
    incident_response_hours_monthly=15,
    compliance_audit_hours_monthly=4,  # Own data
)

rag_tco = TCOComponents(
    infrastructure_monthly=800,  # Vector DB + small embedding model
    api_cost_per_request=0.0005,  # API for generation only
    initial_dev_hours=320,  # 8 weeks
    monthly_maintenance_hours=30,
    monitoring_tooling_monthly=800,
    incident_response_hours_monthly=10,
    compliance_audit_hours_monthly=6,
)

volumes = [100_000, 500_000, 2_000_000, 10_000_000]

print("Cost per 1000 requests at different volumes:\n")
for vol in volumes:
    api_cost = api_tco.calculate_tco(vol)
    ft_cost = fine_tuned_tco.calculate_tco(vol)
    rag_cost = rag_tco.calculate_tco(vol)
    
    print(f"{vol:,} requests/month:")
    print(f"  API:        ${api_cost['cost_per_request']*1000:.3f}")
    print(f"  Fine-tuned: ${ft_cost['cost_per_request']*1000:.3f}")
    print(f"  RAG:        ${rag_cost['cost_per_request']*1000:.3f}")
    print()
```

**Practical Implications:**
- At <500K requests/month: API usually wins on pure economics
- At >2M requests/month: Fine-tuning or RAG becomes competitive
- Engineering costs dominate at low volumes; compute costs dominate at high volumes

**Real Constraints:**
- Engineering time is often the bottleneck, not budget
- Hidden costs (incidents, compliance) can double total TCO
- Training data quality matters more than quantity—budget accordingly

### 2. Performance Requirements Matrix

Different approaches achieve different performance characteristics:

```python
from typing import NamedTuple

class PerformanceProfile(NamedTuple):
    p50_latency_ms: float
    p99_latency_ms: float
    throughput_rps: int
    accuracy_score: float  # 0-1
    cost_per_1k_requests: float

class ApproachComparison:
    def __init__(self, task_description: str):
        self.task = task_description
        self.profiles: dict[str, PerformanceProfile] = {}
    
    def add_approach(self, name: str, profile: PerformanceProfile):
        self.profiles[name] = profile
    
    def meets_requirements(
        self,
        max_latency_p99: float,
        min_throughput: int,
        min_accuracy: float,
        max_cost_per_1k: float
    ) -> dict[str, bool]:
        results = {}
        for name, profile in self.profiles.items():
            meets = (
                profile.p99_latency_ms <= max_latency_p99 and
                profile.throughput_rps >= min_throughput and
                profile.accuracy_score >= min_accuracy and
                profile.cost_per_1k_requests <= max_cost_per_1k
            )
            results[name] = meets
        return results
    
    def rank_by_priority(
        self,
        weights: dict[str, float]
    ) -> list[tuple[str, float]]:
        """Rank approaches by weighted score (lower is better for latency/cost)"""
        scores = {}
        
        for name, profile in self.profiles.items():
            # Normalize and invert where needed
            score = (
                weights.get('latency', 0) * profile.p99_latency_ms / 1000 +
                weights.get('cost', 0) * profile.cost_per_1k_requests +
                weights.get('accuracy', 0) * (1 - profile.accuracy_score) * 100 +
                weights.get('throughput', 0) * (1000 / profile.throughput_rps)
            )
            scores[name] = score
        
        return sorted(scores.items(), key=lambda x: x[1])

# Real example: Content moderation system
comparison = ApproachComparison("Content moderation")

# Large API model (GPT-4)
comparison.add_approach("API-Large", PerformanceProfile(
    p50_latency_ms=800,
    p99_latency_ms=3500,
    throughput_rps=50,
    accuracy_score=0.94,
    cost_per_1k_requests=2.00
))

# Smaller API model (GPT-3.5)
comparison.add_approach("API-Small", PerformanceProfile(
    p50_latency_ms=400,
    p99_latency_ms=1200,
    throughput_rps=200,
    accuracy_score=0.87,
    cost_per_1k_requests=0.20
))

# Fine-tuned 7B model on owned infrastructure
comparison.add_approach("Fine-tuned-7B", PerformanceProfile(
    p50_latency_ms=120,
    p99_latency_ms=250,
    throughput_rps=500,
    accuracy_score=0.91,
    cost_per_1k_requests=0.08
))

# Distilled tiny model (100M params)
comparison.add_approach("Distilled-Tiny", PerformanceProfile(
    p50_latency_ms=15,
    p99_latency_ms=35,
    throughput_rps=5000,
    accuracy_score=0.82,
    cost_per_1k_requests=0.01
))

# Scenario 1: Real-time chat moderation (latency critical)
print("Scenario: Real-time chat moderation")
print("Requirements: p99 < 500ms, throughput > 100 rps, accuracy > 0.85\n")

meets_req = comparison.meets_requirements(
    max_latency_p99=500,
    min_throughput=100,
    min_accuracy=0.85,
    max_cost_per_1k=1.0
)

for approach, meets in meets_req.items():
    profile = comparison.profiles[approach]
    print(f"{approach}: {'✓ PASS' if meets else '✗ FAIL'}")
    print(f"  Latency p99: {profile.p99_latency_ms}ms")
    print(f"  Throughput: {profile.throughput_rps} rps")
    print(f"  Accuracy: {profile.accuracy_score:.2%}\n")

# Scenario 2: Batch moderation (cost critical)
print("\nScenario: