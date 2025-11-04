# Inference & Runtime Dynamics

## Core Concepts

Inference is the production phase of machine learning—when a trained model processes new inputs to generate predictions or outputs. Unlike training, which happens once (or periodically) to create the model, inference happens continuously in production, processing every user request, API call, or batch job.

For LLMs specifically, inference means generating text token-by-token based on input prompts. Each token generation requires a complete forward pass through billions of parameters, making LLM inference fundamentally different from traditional ML inference in scale, cost, and architectural constraints.

### Engineering Analogy: Database Query vs. Sequential Computation

**Traditional ML Inference (like a database lookup):**

```python
import numpy as np
from typing import Dict

class TraditionalMLModel:
    """Simple classification model - single forward pass"""
    
    def __init__(self, weights: np.ndarray):
        self.weights = weights
    
    def predict(self, features: np.ndarray) -> Dict[str, any]:
        """Single matrix multiplication, instant result"""
        logits = np.dot(features, self.weights)
        prediction = np.argmax(logits)
        
        return {
            'prediction': prediction,
            'confidence': float(np.max(logits)),
            'latency_ms': 2  # Predictable, ~constant time
        }

# Usage
model = TraditionalMLModel(weights=np.random.randn(100, 10))
result = model.predict(np.random.randn(100))
# Returns immediately with fixed computation cost
```

**LLM Inference (like iterative compilation):**

```python
from typing import List, Dict
import time

class LLMInference:
    """Simplified LLM - autoregressive token generation"""
    
    def __init__(self, model_params: int = 7_000_000_000):
        self.model_params = model_params
        self.vocab_size = 32_000
        
    def generate(self, prompt: str, max_tokens: int = 100) -> Dict[str, any]:
        """Each token requires full model forward pass"""
        tokens_generated = []
        total_compute = 0
        start_time = time.time()
        
        # Tokenize prompt (simplified)
        prompt_tokens = len(prompt.split())
        
        for i in range(max_tokens):
            # CRITICAL: Each iteration processes ALL previous tokens
            context_length = prompt_tokens + i
            
            # Compute cost: 2 * params * context_length (FLOPs)
            flops_this_token = 2 * self.model_params * context_length
            total_compute += flops_this_token
            
            # Simulate token generation
            next_token = f"token_{i}"
            tokens_generated.append(next_token)
            
            # Stop condition (simplified)
            if next_token == "token_50":  # Simulated EOS
                break
        
        return {
            'output': ' '.join(tokens_generated),
            'tokens_generated': len(tokens_generated),
            'total_flops': total_compute,
            'latency_seconds': time.time() - start_time,
            'avg_flops_per_token': total_compute / len(tokens_generated)
        }

# Usage
llm = LLMInference(model_params=7_000_000_000)
result = llm.generate("Explain quantum computing", max_tokens=100)

print(f"Generated {result['tokens_generated']} tokens")
print(f"Total compute: {result['total_flops']:.2e} FLOPs")
print(f"Latency: {result['latency_seconds']:.2f}s")
# Output varies dramatically by length - unpredictable cost
```

### Key Insights That Change Engineering Thinking

1. **Compute Cost Grows Quadratically**: Each new token must attend to all previous tokens. A 100-token response isn't 100x more expensive than 1 token—it's closer to 5,000x because you're processing (1+2+3+...+100) token contexts.

2. **Latency Is Sequential, Not Parallel**: You can't generate token 50 without first generating tokens 1-49. This fundamentally limits throughput optimization compared to embarrassingly parallel workloads.

3. **Memory Bandwidth Dominates**: For large models, moving parameters from memory to compute units takes longer than the actual math. A 70B parameter model at fp16 requires 140GB of memory transfers per token.

4. **The Prompt Is Processed Once, Output Is Iterative**: Processing a 1000-token prompt is a single parallel operation. Generating a 1000-token response is 1000 sequential operations. This asymmetry drives architectural decisions.

### Why This Matters NOW

Infrastructure costs for LLM inference can exceed $1M/month for moderate-scale applications. Understanding inference dynamics is the difference between a prototype that costs $50/day and a production system that costs $5,000/day with identical functionality. Engineers who optimize inference can deliver 10x cost savings or 5x latency improvements without changing model quality.

## Technical Components

### 1. Autoregressive Generation & KV Caching

**Technical Explanation:**

LLMs generate text autoregressively—each token depends on all previous tokens. Without optimization, this means recalculating attention for all previous tokens at each step, creating O(n²) complexity in sequence length.

KV (Key-Value) caching stores the attention keys and values from previous tokens, eliminating redundant computation. Instead of recomputing attention for all previous tokens, you only compute it for the new token and reuse cached values.

```python
import numpy as np
from typing import Tuple, Optional

class AttentionWithKVCache:
    """Simplified multi-head attention with KV caching"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projection matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        
        # KV cache: stored across generation steps
        self.k_cache: Optional[np.ndarray] = None
        self.v_cache: Optional[np.ndarray] = None
    
    def forward_without_cache(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Standard attention - recomputes everything"""
        seq_len = x.shape[0]
        
        # Project all tokens
        Q = x @ self.W_q  # (seq_len, d_model)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Attention
        scores = Q @ K.T / np.sqrt(self.d_k)
        attn_weights = self._softmax(scores)
        output = attn_weights @ V
        
        # FLOPs: 3 projections + attention computation
        flops = 3 * seq_len * self.d_model * self.d_model + \
                2 * seq_len * seq_len * self.d_model
        
        return output, flops
    
    def forward_with_cache(
        self, 
        x: np.ndarray,  # Just the NEW token
        use_cache: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Cached attention - only computes new token"""
        
        # Project only the new token
        q_new = x @ self.W_q  # (1, d_model)
        k_new = x @ self.W_k
        v_new = x @ self.W_v
        
        # Update cache
        if self.k_cache is None:
            # First token - initialize cache
            self.k_cache = k_new
            self.v_cache = v_new
        else:
            # Append to cache
            self.k_cache = np.vstack([self.k_cache, k_new])
            self.v_cache = np.vstack([self.v_cache, v_new])
        
        seq_len = self.k_cache.shape[0]
        
        # Attention with full cache
        scores = q_new @ self.k_cache.T / np.sqrt(self.d_k)
        attn_weights = self._softmax(scores)
        output = attn_weights @ self.v_cache
        
        # FLOPs: 3 projections for 1 token + attention over seq_len
        flops = 3 * self.d_model * self.d_model + \
                2 * seq_len * self.d_model
        
        return output, flops
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def clear_cache(self):
        """Reset cache between requests"""
        self.k_cache = None
        self.v_cache = None


# Comparison
attention = AttentionWithKVCache(d_model=512, n_heads=8)

# Scenario: Generate 100 tokens
seq_length = 100
flops_without_cache = 0
flops_with_cache = 0

print("=== Without KV Cache ===")
for i in range(1, seq_length + 1):
    # Must process all i tokens each time
    tokens = np.random.randn(i, 512)
    _, flops = attention.forward_without_cache(tokens)
    flops_without_cache += flops

print(f"Total FLOPs: {flops_without_cache:.2e}")

print("\n=== With KV Cache ===")
attention.clear_cache()
for i in range(seq_length):
    # Only process 1 new token
    new_token = np.random.randn(1, 512)
    _, flops = attention.forward_with_cache(new_token)
    flops_with_cache += flops

print(f"Total FLOPs: {flops_with_cache:.2e}")
print(f"\nSpeedup: {flops_without_cache / flops_with_cache:.1f}x")
```

**Practical Implications:**

- **Memory Trade-off**: KV cache for a 7B model with 2048 context requires ~16GB additional memory (2 layers × 32 layers × 2048 tokens × 4096 hidden × 2 bytes)
- **Batch Size Constraint**: Cached inference limits dynamic batching since each sequence has different cache states
- **Stateful Operations**: Cache must persist across generation steps but be cleared between requests

**Real Constraints:**

KV cache assumes the model processes tokens sequentially. Speculative decoding or parallel sampling invalidate cache assumptions. Cache memory grows linearly with context length—long conversations eventually exhaust GPU memory.

### 2. Batching Strategies & Throughput Optimization

**Technical Explanation:**

GPUs achieve peak efficiency with large matrix operations. Processing one request at a time wastes 90%+ of GPU capacity. Batching groups multiple requests into single operations, dramatically improving hardware utilization.

However, LLM inference has a critical challenge: requests finish at different times (different output lengths). Static batching waits for all requests in a batch to complete, wasting capacity. Continuous batching adds new requests as soon as slots free up.

```python
import time
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class Request:
    id: int
    prompt_tokens: int
    max_new_tokens: int
    tokens_generated: int = 0
    finished: bool = False

class InferenceBatcher:
    """Compare static vs continuous batching"""
    
    def __init__(
        self, 
        model_flops_per_token: int = 1_400_000_000_000,  # 7B model
        gpu_tflops: int = 312  # A100 theoretical
    ):
        self.model_flops_per_token = model_flops_per_token
        self.gpu_tflops = gpu_tflops * 1e12
        self.time_per_token = model_flops_per_token / self.gpu_tflops
    
    def static_batching(
        self, 
        requests: List[Request], 
        batch_size: int
    ) -> Dict[str, float]:
        """Process fixed batches - wait for all to complete"""
        total_time = 0
        total_tokens = 0
        
        # Process in fixed batches
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i+batch_size]
            
            # Find max output length in batch
            max_length = max(r.max_new_tokens for r in batch)
            
            # All requests process for max_length iterations
            # (short ones waste compute waiting)
            for step in range(max_length):
                batch_time = self.time_per_token * len(batch)
                total_time += batch_time
                total_tokens += len(batch)
            
            # Wasted tokens: sum of (max_length - actual_length)
            wasted = sum(max_length - r.max_new_tokens for r in batch)
            total_tokens -= wasted  # Actual useful tokens
        
        return {
            'total_time': total_time,
            'total_tokens': total_tokens,
            'throughput_tokens_per_sec': total_tokens / total_time,
            'avg_latency_per_request': total_time / len(requests)
        }
    
    def continuous_batching(
        self, 
        requests: List[Request], 
        max_batch_size: int
    ) -> Dict[str, float]:
        """Dynamic batching - replace finished requests immediately"""
        total_time = 0
        total_tokens = 0
        
        active_batch: List[Request] = []
        queue = requests.copy()
        request_times = {}
        
        step = 0
        while active_batch or queue:
            # Fill batch up to max size
            while len(active_batch) < max_batch_size and queue:
                req = queue.pop(0)
                active_batch.append(req)
                request_times[req.id] = step * self.time_per_token
            
            if not active_batch:
                break
            
            # Process one step for all active requests
            batch_time = self.time_per_token * len(active_batch)
            total_time += batch_time
            total_tokens += len(active_batch)
            
            # Update and remove finished requests
            for req in active_batch[:]:
                req.tokens_generated += 1
                if req.tokens_generated >= req.max_new_tokens:
                    active_batch.remove(req)
                    request_times[req.id] = (step + 1) * self.time_per_token - \
                                             request_times[req.id]
            
            step += 1
        
        return {
            'total_time': total_time,
            'total_tokens': total_tokens,
            'throughput_tokens_per_sec': total_tokens / total_time,
            'avg_latency_per_request': np.mean(list(request_times.values()))
        }


# Comparison: Mixed workload
np.random.seed(42)
requests = [
    Request(
        id=i, 
        prompt_tokens=100,
        max_new_tokens=np.random.randint(10, 200)  # Varied lengths
    )
    for i in range(32)
]

batcher = InferenceBatcher()

print("=== Static Batching (batch_size=8) ===")
static_results = batcher.static_batching(requests.copy(), batch_size=8)
print(f"Throughput: {static_results['throughput_tokens_per_sec']:.0f} tok/s")
print(f"Avg Latency: {static_results['avg_latency_per_request']:.3f}s")

print("\n=== Continuous