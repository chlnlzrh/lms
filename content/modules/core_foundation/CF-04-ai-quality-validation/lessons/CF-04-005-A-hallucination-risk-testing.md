# Hallucination Risk Testing

## Core Concepts

Large language models generate plausible-sounding text that may be factually incorrect, inconsistent, or entirely fabricated—a phenomenon called hallucination. Unlike traditional software bugs that fail predictably, hallucinations are probabilistic failures that emerge from the statistical nature of neural networks. For production systems, this represents a fundamental reliability challenge: your system can confidently produce incorrect outputs with no internal error signal.

### Engineering Analogy: Deterministic vs. Probabilistic Failure Modes

Traditional software testing:

```python
def calculate_discount(price: float, discount_percent: float) -> float:
    """Traditional deterministic function - fails predictably."""
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)

# Test cases catch all failure modes
assert calculate_discount(100, 10) == 90.0
assert calculate_discount(100, 0) == 100.0
try:
    calculate_discount(100, 150)  # Predictably fails
except ValueError:
    pass  # Expected behavior
```

LLM-based systems:

```python
from typing import Optional
import anthropic

def extract_discount_from_email(email_text: str, client: anthropic.Client) -> Optional[float]:
    """LLM-based extraction - fails probabilistically."""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"Extract the discount percentage from this email. Return only the number.\n\n{email_text}"
        }]
    )
    
    try:
        return float(response.content[0].text.strip().rstrip('%'))
    except (ValueError, AttributeError):
        return None

# Same input can produce different outputs
email = "Thanks for your loyalty! Enjoy savings on your next purchase."
# Run 1: Returns 10.0 (hallucinated)
# Run 2: Returns None (correct - no discount mentioned)
# Run 3: Returns 15.0 (hallucinated different value)
```

The critical difference: traditional code fails at boundaries you define. LLMs fail in ways you cannot fully enumerate. A passing test today doesn't guarantee the same behavior tomorrow, even with identical inputs and model versions.

### Key Insights That Change Engineering Approach

**Hallucinations are not bugs to fix—they're inherent system properties to measure and mitigate.** You cannot eliminate them through prompt engineering alone. Your testing strategy must shift from "verify correctness" to "quantify risk and implement guardrails."

**Confidence scores are unreliable indicators.** LLMs often express high confidence in hallucinated content. The model's internal probability distribution doesn't directly correspond to factual accuracy.

**Hallucination rates vary dramatically by task type.** Closed-domain tasks with verifiable facts (e.g., extracting dates from structured text) have lower risk than open-domain generation (e.g., summarizing complex technical concepts). Your testing strategy must be task-specific.

### Why This Matters Now

Production LLM deployments are moving from chatbots to decision-critical systems: medical triage, legal document analysis, financial reporting, code generation. A hallucinated medical diagnosis or incorrect legal citation has severe consequences. The industry's current failure mode: teams discover catastrophic hallucinations in production because they tested like traditional software (happy path + edge cases) rather than measuring probabilistic failure rates across representative distributions.

## Technical Components

### 1. Hallucination Taxonomy and Detection

Hallucinations manifest in distinct categories requiring different detection approaches:

**Factual Hallucinations:** Claims contradicting known facts or verifiable data.

```python
import anthropic
from typing import Dict, List
import json

def detect_factual_hallucination(
    claim: str,
    ground_truth: str,
    client: anthropic.Client
) -> Dict[str, any]:
    """Compare generated claim against ground truth for factual accuracy."""
    
    prompt = f"""Compare this claim against the ground truth facts.

Claim: {claim}

Ground Truth: {ground_truth}

Respond in JSON format:
{{
    "accurate": true/false,
    "contradictions": ["list of specific contradictions"],
    "unsupported_claims": ["claims not in ground truth"]
}}"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)

# Example: Test product description generation
ground_truth = "Product X123: 16GB RAM, 512GB SSD, Intel i5 processor, released March 2024"
generated_claim = "Product X123 features 32GB RAM and AMD Ryzen processor, launched in 2023"

result = detect_factual_hallucination(generated_claim, ground_truth, client)
# {
#     "accurate": false,
#     "contradictions": ["Claims 32GB RAM vs actual 16GB", "Claims AMD vs actual Intel"],
#     "unsupported_claims": ["Launch year 2023 vs actual March 2024"]
# }
```

**Consistency Hallucinations:** Internal contradictions within generated text.

```python
def detect_consistency_hallucination(text: str, client: anthropic.Client) -> List[str]:
    """Find internal contradictions in generated text."""
    
    prompt = f"""Analyze this text for internal contradictions or inconsistent statements.

Text: {text}

List each contradiction found. If none, return empty list.
Format: JSON array of strings."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)

# Example: Inconsistent recommendation
text = """
For optimal performance, we recommend disabling caching to ensure fresh data.
Our caching layer provides 10x performance improvements and should always be enabled.
"""

contradictions = detect_consistency_hallucination(text, client)
# ["Contradictory caching advice: first recommends disabling, then says always enable"]
```

**Source Attribution Hallucinations:** Fabricated citations or misattributed quotes.

```python
import re
from typing import Set, Tuple

def verify_citations(
    generated_text: str,
    source_documents: List[str]
) -> Tuple[Set[str], Set[str]]:
    """Verify that citations in generated text exist in source documents."""
    
    # Extract citations (assuming format: [Source: document_name])
    citation_pattern = r'\[Source: ([^\]]+)\]'
    claimed_citations = set(re.findall(citation_pattern, generated_text))
    
    # Extract actual source names
    actual_sources = set(source_documents)
    
    # Find hallucinated citations
    hallucinated = claimed_citations - actual_sources
    verified = claimed_citations & actual_sources
    
    return verified, hallucinated

# Example
generated = """
The system architecture [Source: tech_spec_v2.pdf] uses microservices.
Performance benchmarks [Source: results_2024.xlsx] show 50ms latency.
Security audit [Source: pentest_final.doc] found no critical issues.
"""

sources = ["tech_spec_v2.pdf", "results_2024.xlsx"]
verified, hallucinated = verify_citations(generated, sources)

print(f"Verified citations: {verified}")
# {"tech_spec_v2.pdf", "results_2024.xlsx"}

print(f"Hallucinated citations: {hallucinated}")
# {"pentest_final.doc"}
```

**Practical Implications:** Build detection pipelines that run multiple validators in parallel. Factual hallucinations require external knowledge bases or ground truth datasets. Consistency checking can be self-contained but is computationally expensive (requires re-reading generated text). Source attribution is binary and fast to verify.

**Trade-offs:** Using LLMs to detect hallucinations (as shown above) introduces meta-hallucination risk—the detector itself can be wrong. For critical systems, prefer deterministic verification (exact string matching for citations, database lookups for facts) over LLM-based detection.

### 2. Benchmark Dataset Construction

Effective hallucination testing requires diverse, representative test sets that cover your production distribution.

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import random

class RiskLevel(Enum):
    LOW = "low"           # Factual, verifiable, closed-domain
    MEDIUM = "medium"     # Some ambiguity, requires reasoning
    HIGH = "high"         # Open-ended, opinion-based, rare facts

@dataclass
class TestCase:
    input_prompt: str
    ground_truth: Optional[str]
    risk_level: RiskLevel
    verification_method: str
    category: str

class HallucinationBenchmark:
    """Construct stratified benchmark for hallucination testing."""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
    
    def add_factual_cases(self, knowledge_base: Dict[str, str]) -> None:
        """Generate test cases from verified knowledge base."""
        for entity, facts in knowledge_base.items():
            # Positive case: should retrieve correctly
            self.test_cases.append(TestCase(
                input_prompt=f"What are the specifications of {entity}?",
                ground_truth=facts,
                risk_level=RiskLevel.LOW,
                verification_method="exact_match",
                category="factual_retrieval"
            ))
            
            # Negative case: should decline for unknown entities
            fake_entity = f"NONEXISTENT_{entity}_XYZ"
            self.test_cases.append(TestCase(
                input_prompt=f"What are the specifications of {fake_entity}?",
                ground_truth=None,  # Should refuse or indicate unknown
                risk_level=RiskLevel.HIGH,
                verification_method="refusal_detection",
                category="unknown_entity"
            ))
    
    def add_reasoning_cases(self, complex_scenarios: List[Dict]) -> None:
        """Add cases requiring multi-step reasoning."""
        for scenario in complex_scenarios:
            self.test_cases.append(TestCase(
                input_prompt=scenario["prompt"],
                ground_truth=scenario["expected_reasoning"],
                risk_level=RiskLevel.MEDIUM,
                verification_method="reasoning_validation",
                category="multi_step_reasoning"
            ))
    
    def add_edge_cases(self, ambiguous_inputs: List[str]) -> None:
        """Add deliberately ambiguous or trick questions."""
        for input_text in ambiguous_inputs:
            self.test_cases.append(TestCase(
                input_prompt=input_text,
                ground_truth=None,  # No single correct answer
                risk_level=RiskLevel.HIGH,
                verification_method="ambiguity_handling",
                category="edge_case"
            ))
    
    def get_stratified_sample(self, n: int) -> List[TestCase]:
        """Sample test cases proportionally by risk level."""
        low_risk = [tc for tc in self.test_cases if tc.risk_level == RiskLevel.LOW]
        med_risk = [tc for tc in self.test_cases if tc.risk_level == RiskLevel.MEDIUM]
        high_risk = [tc for tc in self.test_cases if tc.risk_level == RiskLevel.HIGH]
        
        # Target distribution: 50% low, 30% medium, 20% high
        sample = (
            random.sample(low_risk, min(len(low_risk), int(n * 0.5))) +
            random.sample(med_risk, min(len(med_risk), int(n * 0.3))) +
            random.sample(high_risk, min(len(high_risk), int(n * 0.2)))
        )
        
        return random.sample(sample, min(len(sample), n))

# Usage example
benchmark = HallucinationBenchmark()

# Add factual cases from your domain
product_kb = {
    "Model-A": "CPU: 8-core, RAM: 16GB, Storage: 512GB SSD",
    "Model-B": "CPU: 12-core, RAM: 32GB, Storage: 1TB SSD"
}
benchmark.add_factual_cases(product_kb)

# Add reasoning cases
reasoning_scenarios = [{
    "prompt": "If Model-A costs $1000 and Model-B costs 50% more, what's the price difference?",
    "expected_reasoning": "Model-B costs $1500 (50% more than $1000). Difference is $500."
}]
benchmark.add_reasoning_cases(reasoning_scenarios)

# Add edge cases
benchmark.add_edge_cases([
    "What's the best product?",  # Subjective, should hedge
    "Tell me about Model-C",     # Doesn't exist in KB
    "How much does everything cost?"  # Ambiguous reference
])

# Get balanced test set
test_set = benchmark.get_stratified_sample(100)
```

**Practical Implications:** Your benchmark must over-represent high-risk cases relative to production distribution. In production, 90% of queries might be low-risk, but your test set should be 50% low-risk to catch failure modes. Track hallucination rates separately by risk level—a 5% hallucination rate on high-risk cases might be acceptable, but 5% on low-risk cases indicates systemic issues.

**Real Constraints:** Building comprehensive benchmarks is labor-intensive. Budget 2-4 hours per 10 high-quality test cases. Prioritize coverage over size—100 diverse, well-crafted cases outperform 1000 similar cases. Version control your benchmark and treat it as production code.

### 3. Automated Testing Pipelines

Production hallucination testing requires continuous monitoring, not one-time validation.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Callable
import anthropic
import statistics

@dataclass
class TestResult:
    test_case_id: str
    input_prompt: str
    generated_output: str
    ground_truth: Optional[str]
    is_hallucination: bool
    confidence_score: float
    latency_ms: float
    timestamp: datetime
    model_version: str

@dataclass
class HallucinationMetrics:
    total_tests: int
    hallucination_count: int
    hallucination_rate: float
    avg_confidence_when_hallucinating: float
    avg_confidence_when_correct: float
    by_risk_level: Dict[RiskLevel, float] = field(default_factory=dict)
    by_category: Dict[str, float] = field(default_factory=dict)

class HallucinationTestPipeline:
    """Automated pipeline for continuous hallucination testing."""
    
    def __init__(self, client: anthropic.Client):
        self.client = client
        self.results: List[TestResult] = []
        
    def run_test_suite(
        self,
        test_cases: List[TestCase],
        model: str,
        verifiers: Dict[str, Callable]
    ) -> HallucinationMetrics:
        """Execute full test suite and compute metrics."""
        
        for idx, test_case in enumerate(test_cases):
            start_time = datetime.now()
            
            # Generate response
            response = self.client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": test_case.input_prompt}]
            )
            
            generated = response.content[0].text
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            # Verify response using appropriate method
            verifier = verifiers.get(test_case.verification_method)
            if verifier:
                is_hallucination, confidence = verifier(
                    generated, 
                    test_case.ground_truth,
                    self.client
                )
            else:
                is_hallucination, confidence = False, 0.0
            
            # Store result
            self.results.append(TestResult(
                test_case_id=f"test_{idx}",
                input_prompt=test_case.input_prompt,