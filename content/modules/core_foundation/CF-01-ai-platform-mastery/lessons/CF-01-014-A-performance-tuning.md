# Performance Tuning for LLM Systems

## Core Concepts

Performance tuning in LLM systems is fundamentally different from traditional application optimization. In conventional systems, you optimize CPU cycles, memory allocation, and I/O operations. With LLMs, you're optimizing token processing throughput, managing multi-GB model weights, and balancing the tension between latency, cost, and output quality.

### Traditional vs. Modern Performance Optimization

```python
# Traditional API optimization: reduce database calls
class TraditionalUserService:
    def get_user_recommendations(self, user_id: str) -> list[str]:
        # Problem: N+1 queries
        user = db.query("SELECT * FROM users WHERE id = ?", user_id)
        preferences = db.query("SELECT * FROM preferences WHERE user_id = ?", user_id)
        items = db.query("SELECT * FROM items WHERE category = ?", preferences['category'])
        return [item['name'] for item in items]
    
    # Solution: single query with joins (10x faster)
    def get_user_recommendations_optimized(self, user_id: str) -> list[str]:
        results = db.query("""
            SELECT i.name FROM users u
            JOIN preferences p ON u.id = p.user_id
            JOIN items i ON i.category = p.category
            WHERE u.id = ?
        """, user_id)
        return [r['name'] for r in results]

# LLM optimization: reduce token processing and API calls
from typing import Optional
import asyncio
from dataclasses import dataclass
import time

@dataclass
class LLMResponse:
    content: str
    tokens_used: int
    latency_ms: float

class LLMUserService:
    def __init__(self, llm_client):
        self.client = llm_client
    
    # Problem: sequential calls, redundant context
    def analyze_user_behavior(self, user_data: dict) -> dict:
        start = time.time()
        
        # Call 1: Analyze preferences (500 tokens)
        pref_prompt = f"Analyze user preferences: {user_data}"
        preferences = self.client.generate(pref_prompt, max_tokens=200)
        
        # Call 2: Generate recommendations (600 tokens, re-sends user_data)
        rec_prompt = f"Based on user: {user_data} and preferences: {preferences}, recommend items"
        recommendations = self.client.generate(rec_prompt, max_tokens=200)
        
        # Call 3: Explain reasoning (700 tokens, re-sends everything)
        exp_prompt = f"User: {user_data}, Prefs: {preferences}, Recs: {recommendations}. Explain why."
        explanation = self.client.generate(exp_prompt, max_tokens=150)
        
        # Total: 3 API calls, ~1800 input tokens, ~550 output tokens
        # Latency: ~4500ms (3 sequential calls × ~1500ms each)
        return {
            'preferences': preferences,
            'recommendations': recommendations,
            'explanation': explanation,
            'latency_ms': (time.time() - start) * 1000
        }
    
    # Solution: single call with structured output, streaming
    async def analyze_user_behavior_optimized(self, user_data: dict) -> dict:
        start = time.time()
        
        # Single call with structured output format
        prompt = f"""Analyze this user and provide structured response:
User: {user_data}

Respond in this exact format:
PREFERENCES: [one line summary]
RECOMMENDATIONS: [comma-separated list]
REASONING: [brief explanation]
"""
        
        response = await self.client.generate_async(
            prompt,
            max_tokens=300,
            temperature=0.3  # Lower temp for structured output
        )
        
        # Parse structured response
        lines = response.split('\n')
        result = {
            'preferences': lines[0].replace('PREFERENCES: ', ''),
            'recommendations': lines[1].replace('RECOMMENDATIONS: ', '').split(', '),
            'explanation': lines[2].replace('REASONING: ', ''),
            'latency_ms': (time.time() - start) * 1000
        }
        
        # Total: 1 API call, ~600 input tokens, ~300 output tokens
        # Latency: ~1500ms (single call)
        # Result: 67% fewer tokens, 67% lower latency, 60% cost reduction
        return result
```

### Key Insights That Change Your Mental Model

**1. Latency is dominated by token generation, not model loading**: Each generated token requires a full forward pass through billions of parameters. A 100-token response takes 100× longer than a 1-token response, not 100 bytes more memory.

**2. Batch processing doesn't work like traditional systems**: You can't simply "batch 1000 requests" and expect 1000× throughput. LLM batching is constrained by memory (each request's KV cache must fit in VRAM simultaneously) and by the longest sequence in the batch.

**3. Caching strategies are inverted**: In traditional systems, you cache hot data. With LLMs, you cache prompt prefixes (system messages, few-shot examples) because re-processing the same tokens is pure waste.

### Why This Matters NOW

Production LLM systems routinely process millions of tokens daily. The difference between naive and optimized implementations:

- **Cost**: $100/day vs. $1,000/day for the same workload
- **Latency**: 5-second responses vs. 1-second responses (80% reduction)
- **Throughput**: 10 requests/minute vs. 100 requests/minute per instance

These aren't theoretical improvements—they're the difference between a system that ships and one that gets killed in code review.

## Technical Components

### 1. Token-Level Optimization

Tokens are the atomic unit of LLM cost and latency. Most engineers initially think in characters or words; you must think in tokens.

**Technical Explanation**: LLMs process text in tokens (subword units). GPT-style models use ~750 tokens per 1000 characters of English text. Each input token costs ~$0.01/1M tokens, each output token costs ~$0.03/1M tokens (3× more expensive because generation requires inference).

**Practical Implications**: A 2000-token prompt with 500-token response costs ~($0.02 input + $0.015 output) = $0.035 per request. At 1000 requests/day, that's $35/day or $1,050/month.

```python
from typing import List, Dict
import tiktoken

class TokenOptimizer:
    def __init__(self, model: str = "gpt-4"):
        # Use appropriate tokenizer for your model
        self.encoder = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Accurate token counting (not len(text.split()))"""
        return len(self.encoder.encode(text))
    
    def optimize_prompt(self, 
                       system_msg: str,
                       user_msg: str,
                       examples: List[Dict[str, str]]) -> tuple[str, Dict[str, int]]:
        """
        Optimize prompt by removing redundancy and measuring impact.
        """
        # Baseline: unoptimized prompt
        full_prompt = f"{system_msg}\n\n"
        for ex in examples:
            full_prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
        full_prompt += f"Input: {user_msg}\nOutput:"
        
        baseline_tokens = self.count_tokens(full_prompt)
        
        # Optimization 1: Compress system message
        # Remove filler words, use abbreviations
        compressed_system = system_msg.replace("Please ", "").replace(
            "You are an assistant that", "You"
        ).replace("as possible", "")
        
        # Optimization 2: Minimize example formatting
        # Remove labels, use separators
        optimized_prompt = f"{compressed_system}\n\n"
        for ex in examples:
            # Use minimal separators instead of labels
            optimized_prompt += f"{ex['input']}→{ex['output']}\n"
        optimized_prompt += f"{user_msg}→"
        
        optimized_tokens = self.count_tokens(optimized_prompt)
        
        return optimized_prompt, {
            'baseline_tokens': baseline_tokens,
            'optimized_tokens': optimized_tokens,
            'reduction_pct': ((baseline_tokens - optimized_tokens) / baseline_tokens) * 100,
            'cost_savings_per_1k_calls': (baseline_tokens - optimized_tokens) * 0.00001 * 1000
        }

# Example usage
optimizer = TokenOptimizer()
system_msg = "Please act as an assistant that extracts email addresses from text as accurately as possible."
user_msg = "Contact John at john@example.com or Mary at mary@test.org"
examples = [
    {"input": "Email me at bob@demo.com", "output": "bob@demo.com"},
    {"input": "Reach out to alice@foo.bar", "output": "alice@foo.bar"}
]

optimized, metrics = optimizer.optimize_prompt(system_msg, user_msg, examples)
print(f"Token reduction: {metrics['reduction_pct']:.1f}%")
print(f"Cost savings per 1K calls: ${metrics['cost_savings_per_1k_calls']:.2f}")
# Output: Token reduction: 23.4%
# Output: Cost savings per 1K calls: $0.28
```

**Real Constraints**: Over-compression reduces output quality. Test empirically to find the sweet spot. A 20% token reduction that increases error rate from 2% to 5% isn't worth it.

### 2. Prompt Caching and Prefix Reuse

LLMs process tokens sequentially. If the first 500 tokens of every request are identical (system prompt + examples), you're wasting compute.

**Technical Explanation**: Modern LLM APIs support prompt caching, where identical prefix tokens are processed once and their Key-Value (KV) cache is reused across requests. This reduces both latency (no re-processing) and cost (cached tokens often cost 90% less).

```python
from typing import Optional, Tuple
from dataclasses import dataclass
import hashlib

@dataclass
class CachedPrompt:
    prefix: str
    prefix_hash: str
    tokens: int

class PromptCacheManager:
    def __init__(self):
        self.cache: Dict[str, CachedPrompt] = {}
    
    def create_cached_prompt(self,
                            static_prefix: str,
                            dynamic_suffix: str) -> Tuple[str, bool]:
        """
        Separate static (cacheable) from dynamic parts.
        Returns: (full_prompt, cache_hit)
        """
        prefix_hash = hashlib.sha256(static_prefix.encode()).hexdigest()[:16]
        
        cache_hit = prefix_hash in self.cache
        
        if not cache_hit:
            self.cache[prefix_hash] = CachedPrompt(
                prefix=static_prefix,
                prefix_hash=prefix_hash,
                tokens=len(static_prefix.split())  # Simplified
            )
        
        # Full prompt: cached prefix + dynamic suffix
        full_prompt = f"{static_prefix}\n\n{dynamic_suffix}"
        
        return full_prompt, cache_hit
    
    def estimate_savings(self, 
                        requests_per_day: int,
                        prefix_tokens: int,
                        cache_hit_rate: float = 0.95) -> Dict[str, float]:
        """
        Calculate cost savings from prompt caching.
        """
        # Standard pricing (example)
        standard_cost_per_1m = 10.0  # $10 per 1M input tokens
        cached_cost_per_1m = 1.0     # $1 per 1M cached tokens (90% discount)
        
        daily_prefix_tokens = requests_per_day * prefix_tokens
        
        # Without caching: all tokens at standard rate
        cost_without = (daily_prefix_tokens / 1_000_000) * standard_cost_per_1m
        
        # With caching: most tokens at cached rate
        cached_tokens = daily_prefix_tokens * cache_hit_rate
        uncached_tokens = daily_prefix_tokens * (1 - cache_hit_rate)
        
        cost_with = (
            (cached_tokens / 1_000_000) * cached_cost_per_1m +
            (uncached_tokens / 1_000_000) * standard_cost_per_1m
        )
        
        return {
            'daily_cost_without_cache': cost_without,
            'daily_cost_with_cache': cost_with,
            'daily_savings': cost_without - cost_with,
            'monthly_savings': (cost_without - cost_with) * 30
        }

# Example: 10K requests/day with 1000-token static prefix
cache_mgr = PromptCacheManager()
savings = cache_mgr.estimate_savings(
    requests_per_day=10_000,
    prefix_tokens=1000,
    cache_hit_rate=0.95
)
print(f"Monthly savings: ${savings['monthly_savings']:.2f}")
# Output: Monthly savings: $2565.00
```

**Real Constraints**: Cache TTL varies by provider (typically 5-60 minutes). Design your prompt structure so static content stays truly static. Embedding timestamps or request IDs in the prefix breaks caching.

### 3. Parallel Request Management

Sequential LLM calls are a performance killer. Modern async patterns allow concurrent requests, but LLM APIs have rate limits you must respect.

```python
import asyncio
from typing import List, Dict, Any
import time
from collections import deque

class RateLimitedLLMClient:
    def __init__(self, 
                 requests_per_minute: int = 500,
                 tokens_per_minute: int = 150_000):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        
        # Sliding window rate limiters
        self.request_times = deque(maxlen=requests_per_minute)
        self.token_usage = deque(maxlen=1000)
        
    async def _wait_for_capacity(self, estimated_tokens: int):
        """
        Wait until rate limits allow this request.
        """
        while True:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.request_times and self.request_times[0] < now - 60:
                self.request_times.popleft()
            
            while self.token_usage and self.token_usage[0]['time'] < now - 60:
                self.token_usage.popleft()
            
            # Check if we have capacity
            current_rpm = len(self.request_times)
            current_tpm = sum(u['tokens'] for u in self.token_usage)
            
            if current_rpm < self.rpm_limit and current_tpm + estimated_tokens < self.tpm_limit:
                # Record this request
                self.request_times.append(now)
                self.token_usage.append({'time': now, 'tokens': estimated_tokens})
                return
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Rate-limited LLM generation.
        """
        estimated_tokens = len(prompt.split()) + max_tokens
        await self._wait_for_capacity(estimated_tokens)
        
        # Simulate API call
        await asyncio.sleep(0.5)  # Replace with actual LLM call
        return f"Response to: {prompt[:50]}..."
    
    async def batch_generate(self, prompts: List[str], max_tokens: int = 100) -> List[str]:
        """
        Process multiple prompts concurrently with rate limiting.
        """
        tasks = [self.generate(prompt, max_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks)

# Example: Process 100 requests
async def demo_parallel_processing():
    client = RateLimitedL