# Vendor Comparison & Cost Optimization

## Core Concepts

When engineers evaluate databases, they compare query performance, storage costs, and latency under specific workloads. LLM vendor selection requires similar rigor, but the variables are different: token pricing, context window limits, rate limits, and quality metrics that can't be reduced to milliseconds.

**Traditional API Selection:**
```python
# Traditional API evaluation: straightforward metrics
import time
import requests

def evaluate_api_endpoint(url: str, payload: dict) -> dict:
    start = time.time()
    response = requests.post(url, json=payload)
    latency = time.time() - start
    
    return {
        'latency_ms': latency * 1000,
        'cost_per_request': 0.001,  # Fixed pricing
        'success': response.status_code == 200
    }

# Clear winner: lowest latency + cost
```

**LLM Vendor Evaluation:**
```python
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    model: str

@dataclass
class VendorPricing:
    prompt_cost_per_1m: float  # Cost per 1M tokens
    completion_cost_per_1m: float
    context_window: int
    rate_limit_rpm: int

def calculate_request_cost(
    response: LLMResponse,
    pricing: VendorPricing
) -> float:
    prompt_cost = (response.prompt_tokens / 1_000_000) * pricing.prompt_cost_per_1m
    completion_cost = (response.completion_tokens / 1_000_000) * pricing.completion_cost_per_1m
    return prompt_cost + completion_cost

# Cost varies by tokens used, quality isn't guaranteed,
# latency depends on completion length
```

The shift from evaluating APIs to evaluating LLM vendors introduces three critical complexities:

1. **Variable costs per request**: Two identical prompts can cost different amounts based on output length
2. **Quality as primary constraint**: The cheapest, fastest model is worthless if output quality is insufficient
3. **Multi-dimensional optimization**: You're balancing cost, latency, quality, context limits, and rate limits simultaneously

**Why this matters now**: As of 2024, token costs vary by 100x across vendors for comparable quality, and new models launch monthly. An engineering team that hard-codes a single vendor will either overpay by 5-10x or miss performance improvements. Building vendor-agnostic infrastructure and systematic evaluation frameworks is the difference between $500/month and $5,000/month at modest scale.

## Technical Components

### 1. Token-Based Pricing Models

LLM vendors charge per token, not per request. A token is roughly 4 characters in English, but varies by language and tokenizer. Most vendors split pricing into prompt tokens (input) and completion tokens (output), with completion tokens typically costing 2-3x more.

**Technical implication**: Your cost is directly tied to prompt design and output length control. A verbose prompt style can double your costs with no quality improvement.

```python
from typing import Protocol
import tiktoken

class Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...

def estimate_cost(
    prompt: str,
    expected_completion_tokens: int,
    pricing: VendorPricing,
    encoding_name: str = "cl100k_base"
) -> dict:
    """Estimate cost before making API call."""
    encoder = tiktoken.get_encoding(encoding_name)
    prompt_tokens = len(encoder.encode(prompt))
    
    prompt_cost = (prompt_tokens / 1_000_000) * pricing.prompt_cost_per_1m
    completion_cost = (expected_completion_tokens / 1_000_000) * pricing.completion_cost_per_1m
    total_cost = prompt_cost + completion_cost
    
    return {
        'prompt_tokens': prompt_tokens,
        'estimated_completion_tokens': expected_completion_tokens,
        'prompt_cost': prompt_cost,
        'completion_cost': completion_cost,
        'total_cost': total_cost
    }

# Example: Compare two prompt styles
verbose_prompt = """
Please analyze the following customer feedback and provide a detailed 
summary of the key themes, sentiments, and actionable recommendations. 
Be thorough and comprehensive in your analysis.

Customer feedback: "Product works but shipping was slow"
"""

concise_prompt = """
Analyze this feedback for themes, sentiment, and recommendations:
"Product works but shipping was slow"
"""

pricing = VendorPricing(
    prompt_cost_per_1m=1.00,
    completion_cost_per_1m=3.00,
    context_window=128_000,
    rate_limit_rpm=500
)

verbose_cost = estimate_cost(verbose_prompt, 150, pricing)
concise_cost = estimate_cost(concise_prompt, 150, pricing)

print(f"Verbose prompt cost: ${verbose_cost['total_cost']:.6f}")
print(f"Concise prompt cost: ${concise_cost['total_cost']:.6f}")
print(f"Savings per request: {(1 - concise_cost['total_cost'] / verbose_cost['total_cost']) * 100:.1f}%")
```

**Real constraint**: Different vendors use different tokenizers. A prompt that's 1,000 tokens in one vendor's tokenizer might be 1,100 in another's, affecting cost comparisons. Always test with vendor-specific tokenizers.

### 2. Context Window Economics

Context window size determines how much text (prompt + completion) fits in a single request. Larger windows enable more sophisticated prompts but often cost more per token.

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ModelSpec:
    name: str
    context_window: int
    prompt_cost_per_1m: float
    completion_cost_per_1m: float

def analyze_context_strategy(
    documents: List[str],
    query: str,
    small_context_model: ModelSpec,
    large_context_model: ModelSpec,
    encoding_name: str = "cl100k_base"
) -> dict:
    """Compare cost of multiple small-context calls vs. one large-context call."""
    encoder = tiktoken.get_encoding(encoding_name)
    
    # Calculate tokens
    query_tokens = len(encoder.encode(query))
    doc_tokens = [len(encoder.encode(doc)) for doc in documents]
    total_doc_tokens = sum(doc_tokens)
    expected_completion = 200
    
    # Strategy 1: Process each document separately with small context model
    small_context_requests = len(documents)
    small_context_cost = 0
    for tokens in doc_tokens:
        prompt_tokens = query_tokens + tokens
        prompt_cost = (prompt_tokens / 1_000_000) * small_context_model.prompt_cost_per_1m
        completion_cost = (expected_completion / 1_000_000) * small_context_model.completion_cost_per_1m
        small_context_cost += prompt_cost + completion_cost
    
    # Strategy 2: Process all documents in one request with large context model
    large_context_prompt_tokens = query_tokens + total_doc_tokens
    
    if large_context_prompt_tokens > large_context_model.context_window:
        large_context_feasible = False
        large_context_cost = float('inf')
    else:
        large_context_feasible = True
        prompt_cost = (large_context_prompt_tokens / 1_000_000) * large_context_model.prompt_cost_per_1m
        completion_cost = (expected_completion / 1_000_000) * large_context_model.completion_cost_per_1m
        large_context_cost = prompt_cost + completion_cost
    
    return {
        'small_context_strategy': {
            'model': small_context_model.name,
            'requests': small_context_requests,
            'total_cost': small_context_cost,
            'cost_per_document': small_context_cost / len(documents)
        },
        'large_context_strategy': {
            'model': large_context_model.name,
            'requests': 1 if large_context_feasible else 0,
            'total_cost': large_context_cost if large_context_feasible else None,
            'feasible': large_context_feasible
        },
        'recommendation': 'large_context' if (large_context_feasible and large_context_cost < small_context_cost) else 'small_context'
    }

# Example comparison
small_model = ModelSpec("small-fast", 8_000, 0.50, 1.50)
large_model = ModelSpec("large-context", 128_000, 1.00, 3.00)

docs = [
    "Customer reported issue with login on mobile app..." * 50,
    "Billing question about subscription renewal..." * 50,
    "Feature request for dark mode..." * 50,
]

analysis = analyze_context_strategy(
    docs, 
    "Summarize common themes across these support tickets:",
    small_model,
    large_model
)

print(f"Small context: {analysis['small_context_strategy']['requests']} requests, ${analysis['small_context_strategy']['total_cost']:.4f}")
print(f"Large context: {analysis['large_context_strategy']['requests']} request, ${analysis['large_context_strategy']['total_cost']:.4f}")
print(f"Recommendation: {analysis['recommendation']}")
```

**Trade-off**: Large context windows enable sophisticated multi-document analysis in one call, but if you only need to process documents independently, multiple smaller requests may cost less despite higher per-token pricing.

### 3. Rate Limits and Throughput Optimization

Rate limits are specified in requests per minute (RPM) and tokens per minute (TPM). Exceeding them causes failed requests, not just slower responses. For production systems, rate limits often matter more than per-token cost.

```python
import asyncio
import time
from typing import List, Callable, Any
from dataclasses import dataclass
from collections import deque

@dataclass
class RateLimiter:
    requests_per_minute: int
    tokens_per_minute: int
    
    def __post_init__(self):
        self.request_times: deque = deque()
        self.token_times: deque = deque()
    
    async def acquire(self, estimated_tokens: int) -> None:
        """Wait until rate limits allow this request."""
        now = time.time()
        minute_ago = now - 60
        
        # Remove timestamps older than 1 minute
        while self.request_times and self.request_times[0] < minute_ago:
            self.request_times.popleft()
        while self.token_times and self.token_times[0][0] < minute_ago:
            self.token_times.popleft()
        
        # Check if we need to wait for request limit
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.acquire(estimated_tokens)
        
        # Check if we need to wait for token limit
        current_tokens = sum(tokens for _, tokens in self.token_times)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            wait_time = 60 - (now - self.token_times[0][0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.acquire(estimated_tokens)
        
        # Record this request
        self.request_times.append(now)
        self.token_times.append((now, estimated_tokens))

async def process_batch_with_rate_limit(
    prompts: List[str],
    llm_call: Callable,
    rate_limiter: RateLimiter,
    encoder: Any
) -> List[str]:
    """Process prompts respecting rate limits."""
    
    async def process_one(prompt: str) -> str:
        estimated_tokens = len(encoder.encode(prompt)) + 500  # prompt + expected completion
        await rate_limiter.acquire(estimated_tokens)
        return await llm_call(prompt)
    
    tasks = [process_one(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

# Example: Compare throughput with different rate limits
async def simulate_llm_call(prompt: str) -> str:
    """Simulate LLM API call with realistic latency."""
    await asyncio.sleep(0.5)  # Simulate 500ms API latency
    return f"Response to: {prompt[:50]}..."

async def benchmark_rate_limits():
    encoder = tiktoken.get_encoding("cl100k_base")
    prompts = [f"Analyze this data point {i}: ..." for i in range(100)]
    
    # Vendor A: Higher rate limits, higher cost
    vendor_a_limiter = RateLimiter(requests_per_minute=500, tokens_per_minute=90_000)
    
    # Vendor B: Lower rate limits, lower cost
    vendor_b_limiter = RateLimiter(requests_per_minute=200, tokens_per_minute=40_000)
    
    start = time.time()
    await process_batch_with_rate_limit(prompts, simulate_llm_call, vendor_a_limiter, encoder)
    vendor_a_time = time.time() - start
    
    start = time.time()
    await process_batch_with_rate_limit(prompts, simulate_llm_call, vendor_b_limiter, encoder)
    vendor_b_time = time.time() - start
    
    print(f"Vendor A (high limits): {vendor_a_time:.1f}s for 100 requests")
    print(f"Vendor B (low limits): {vendor_b_time:.1f}s for 100 requests")
    print(f"Throughput difference: {(vendor_b_time / vendor_a_time):.2f}x slower")

# asyncio.run(benchmark_rate_limits())
```

**Real constraint**: Rate limits are per API key. If you need higher throughput, you may need to pay for higher tier access or distribute load across multiple keys, adding infrastructure complexity.

### 4. Quality Metrics and Evaluation

Unlike traditional APIs where "success" is binary, LLM output quality exists on a spectrum. You need quantifiable metrics to compare vendors objectively.

```python
from typing import List, Dict
import re
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    vendor: str
    model: str
    accuracy: float
    avg_cost: float
    avg_latency: float
    pass_rate: float

def evaluate_structured_output(
    response: str,
    expected_fields: List[str]
) -> Dict[str, bool]:
    """Check if response contains expected structured data."""
    results = {}
    for field in expected_fields:
        # Simple check: field name appears with a value
        pattern = rf"{field}\s*[:=]\s*\S+"
        results[field] = bool(re.search(pattern, response, re.IGNORECASE))
    return results

def evaluate_factual_accuracy(
    response: str,
    ground_truth: Dict[str, str]
) -> float:
    """Check if response contains expected facts."""
    matches = 0
    for key, value in ground_truth.items():
        if value.lower() in response.lower():
            matches += 1
    return matches / len(ground_truth)

async def run_comparative_evaluation(
    test_cases: List[Dict],
    vendors: Dict[str, Callable]
) -> List[EvaluationResult]:
    """Run systematic comparison across vendors."""
    results = []
    
    for vendor_name, vendor_call in vendors.items():
        total_cost = 0
        total_latency = 0
        accuracy_scores = []
        passed = 0
        
        for test_case in test_cases:
            start = time.time()
            response