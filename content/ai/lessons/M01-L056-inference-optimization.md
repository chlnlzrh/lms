# Inference Optimization: Engineering Performance at Scale

## Core Concepts

Inference optimization is the engineering discipline of reducing latency, cost, and resource consumption when executing trained models in production. Unlike training optimization—which focuses on convergence speed and GPU utilization during the learning phase—inference optimization addresses the unique constraints of serving predictions: unpredictable traffic patterns, strict latency budgets, and cost-per-request economics.

### Traditional vs. Modern Approach

**Traditional approach (pre-transformer era):**

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Training phase
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Deployment: load full model into memory
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Inference: single prediction
def predict(features: np.ndarray) -> int:
    return model.predict(features.reshape(1, -1))[0]

# Cost model: fixed memory, predictable CPU per request
# 100MB model → $10/month server, ~50ms p95 latency
```

**Modern approach (LLM era):**

```python
from typing import List, Optional
import asyncio
from dataclasses import dataclass

@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int
    temperature: float
    request_id: str

class OptimizedInferenceEngine:
    def __init__(
        self,
        model_path: str,
        quantization: str = "int8",
        batch_size: int = 8,
        kv_cache_size: int = 2048
    ):
        # Model loading with optimization
        self.model = self._load_quantized_model(model_path, quantization)
        self.batch_size = batch_size
        self.kv_cache = self._initialize_kv_cache(kv_cache_size)
        self.request_queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        
    async def predict(self, request: InferenceRequest) -> str:
        # Dynamic batching: wait up to 10ms for more requests
        batch = await self._collect_batch(request, timeout_ms=10)
        
        # Batch inference with KV cache reuse
        results = await self._batch_inference(batch)
        
        return results[request.request_id]
    
    async def _collect_batch(
        self, 
        initial_request: InferenceRequest,
        timeout_ms: int
    ) -> List[InferenceRequest]:
        batch = [initial_request]
        deadline = asyncio.get_event_loop().time() + timeout_ms / 1000
        
        while len(batch) < self.batch_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=remaining
                )
                batch.append(req)
            except asyncio.TimeoutError:
                break
        
        return batch
    
    def _load_quantized_model(self, path: str, quantization: str):
        # Placeholder for actual model loading
        # Real implementation would use transformers, vLLM, etc.
        pass
    
    def _initialize_kv_cache(self, size: int):
        # Placeholder for KV cache initialization
        pass
    
    async def _batch_inference(self, batch: List[InferenceRequest]):
        # Placeholder for actual batch inference
        pass

# Cost model: variable memory, complex latency profile
# 7B parameter model:
# - Unoptimized: 28GB VRAM, $1000/month GPU, ~500ms p95
# - Optimized (int8 + batching): 7GB VRAM, $200/month GPU, ~80ms p95
# 5x cost reduction, 6x latency improvement
```

The fundamental shift: inference is no longer a simple function call but a complex system with queuing, batching, memory management, and precision trade-offs.

### Key Engineering Insights

1. **Latency is non-linear with model size**: Doubling parameters doesn't double latency—memory bandwidth becomes the bottleneck. A 13B model can be 4-5x slower than 7B, not 2x.

2. **Most computation is memory movement, not math**: Modern GPUs can perform trillions of operations per second, but moving weights from HBM to compute units takes time. Quantization and KV caching reduce bandwidth pressure.

3. **Batch size has diminishing returns**: Batching from 1→4 requests might reduce cost by 60%, but 16→32 only saves 10% while adding latency. The optimal batch size depends on your latency budget.

4. **First token vs. subsequent tokens**: Prefill (processing the prompt) is compute-bound; generation (producing tokens) is memory-bound. They require different optimization strategies.

### Why This Matters Now

LLM inference costs dominate AI budgets. A single 7B model serving 100 requests/second at 50ms latency costs $3,000-$10,000/month without optimization. With proper techniques, the same workload runs on $500/month hardware. At scale, inference optimization is the difference between profitable and unprofitable AI products.

## Technical Components

### 1. Quantization: Precision vs. Memory Trade-offs

Quantization reduces numerical precision to save memory and bandwidth. Neural networks store weights as 32-bit floats (FP32) by default—quantization converts them to 8-bit integers (INT8) or even 4-bit formats (INT4).

**Technical mechanism:**

```python
import torch
import numpy as np
from typing import Tuple

class QuantizedLinear:
    """8-bit quantized linear layer"""
    
    def __init__(self, weight: torch.Tensor):
        # Original: FP32, 4 bytes per parameter
        # Quantized: INT8, 1 byte per parameter + scale/zero-point
        self.quantized_weight, self.scale, self.zero_point = self._quantize(weight)
    
    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        # Symmetric quantization: map [-max, max] to [-127, 127]
        max_val = torch.max(torch.abs(tensor))
        scale = max_val / 127.0
        
        # Quantize: divide by scale, round, clamp to int8 range
        quantized = torch.clamp(
            torch.round(tensor / scale),
            -128,
            127
        ).to(torch.int8)
        
        return quantized, scale.item(), 0.0
    
    def _dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        # Convert back to FP32 for computation
        return quantized.to(torch.float32) * self.scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights on-the-fly (happens in fast cache)
        weight_fp32 = self._dequantize(self.quantized_weight)
        return torch.matmul(x, weight_fp32.T)

# Benchmark quantization impact
def benchmark_quantization():
    # 7B parameter model ≈ 7000M parameters
    # FP32: 7B * 4 bytes = 28GB
    # INT8: 7B * 1 byte = 7GB (4x reduction)
    # INT4: 7B * 0.5 bytes = 3.5GB (8x reduction)
    
    layer_size = (4096, 4096)  # Typical transformer layer
    weight = torch.randn(layer_size)
    
    # Measure memory
    fp32_size = weight.element_size() * weight.nelement()
    
    quant_layer = QuantizedLinear(weight)
    int8_size = (quant_layer.quantized_weight.element_size() * 
                 quant_layer.quantized_weight.nelement())
    
    print(f"FP32: {fp32_size / 1e6:.1f}MB")
    print(f"INT8: {int8_size / 1e6:.1f}MB ({fp32_size / int8_size:.1f}x smaller)")
    
    # Measure accuracy degradation
    x = torch.randn(1, 4096)
    
    fp32_output = torch.matmul(x, weight.T)
    int8_output = quant_layer.forward(x)
    
    error = torch.mean(torch.abs(fp32_output - int8_output))
    relative_error = error / torch.mean(torch.abs(fp32_output))
    
    print(f"Mean relative error: {relative_error.item():.4f}")
    # Typical output: 0.5-2% error, acceptable for most applications

benchmark_quantization()
```

**Practical implications:**

- INT8 quantization: 4x memory reduction, ~2-3x speedup, <1% accuracy loss
- INT4 quantization: 8x memory reduction, ~3-5x speedup, 2-5% accuracy loss
- Dynamic quantization: quantize at runtime (slower but no retraining needed)
- Static quantization: quantize once (faster, requires calibration data)

**Real constraints:**

Not all operations benefit equally. Attention computation (QKV matrices) sees bigger gains than activation functions. Some models are sensitive to quantization—always benchmark task-specific metrics.

### 2. KV Cache: Trading Memory for Speed

In autoregressive generation, each new token requires computing attention over all previous tokens. Without caching, this means recomputing the same key/value matrices repeatedly.

**Technical mechanism:**

```python
import torch
from typing import Optional, Tuple

class AttentionWithKVCache:
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Preallocate KV cache: [batch, n_heads, max_seq_len, d_head]
        self.max_seq_len = max_seq_len
        self.kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
    def forward(
        self,
        query: torch.Tensor,  # [batch, seq_len, d_model]
        key: torch.Tensor,
        value: torch.Tensor,
        use_cache: bool = True,
        cache_position: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        batch_size, seq_len, _ = query.shape
        
        # Reshape for multi-head attention
        q = query.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = key.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = value.view(batch_size, seq_len, self.n_heads, self.d_head)
        
        if use_cache:
            if self.kv_cache is None:
                # Initialize cache on first call
                self.kv_cache = (
                    torch.zeros(batch_size, self.n_heads, self.max_seq_len, self.d_head),
                    torch.zeros(batch_size, self.n_heads, self.max_seq_len, self.d_head)
                )
            
            k_cache, v_cache = self.kv_cache
            
            if cache_position is None:
                # Prefill: store entire sequence
                k_cache[:, :, :seq_len, :] = k.transpose(1, 2)
                v_cache[:, :, :seq_len, :] = v.transpose(1, 2)
                cache_position = seq_len
            else:
                # Generation: append single token
                k_cache[:, :, cache_position, :] = k.squeeze(1).transpose(0, 1)
                v_cache[:, :, cache_position, :] = v.squeeze(1).transpose(0, 1)
                cache_position += 1
            
            # Use cached keys/values for attention
            k = k_cache[:, :, :cache_position, :].transpose(1, 2)
            v = v_cache[:, :, cache_position, :].transpose(1, 2)
            
            self.kv_cache = (k_cache, v_cache)
        
        # Compute attention (simplified, no masking/scaling)
        scores = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        
        return output.view(batch_size, seq_len, self.d_model), self.kv_cache

# Benchmark KV cache impact
def benchmark_kv_cache():
    d_model, n_heads, max_seq_len = 4096, 32, 2048
    attention = AttentionWithKVCache(d_model, n_heads, max_seq_len)
    
    # Simulate generating 100 tokens
    batch_size = 1
    prompt_len = 512
    
    # Prefill phase
    prompt_qkv = torch.randn(batch_size, prompt_len, d_model)
    output, cache = attention.forward(prompt_qkv, prompt_qkv, prompt_qkv, use_cache=True)
    
    # Generation phase: with cache
    import time
    cache_position = prompt_len
    
    start = time.perf_counter()
    for _ in range(100):
        token_qkv = torch.randn(batch_size, 1, d_model)
        output, cache = attention.forward(
            token_qkv, token_qkv, token_qkv,
            use_cache=True,
            cache_position=cache_position
        )
        cache_position += 1
    cached_time = time.perf_counter() - start
    
    # Generation phase: without cache (recompute everything)
    attention.kv_cache = None
    start = time.perf_counter()
    full_seq = torch.randn(batch_size, prompt_len, d_model)
    for i in range(100):
        current_len = prompt_len + i + 1
        seq = torch.randn(batch_size, current_len, d_model)
        output, _ = attention.forward(seq, seq, seq, use_cache=False)
    uncached_time = time.perf_counter() - start
    
    print(f"With KV cache: {cached_time:.3f}s")
    print(f"Without KV cache: {uncached_time:.3f}s")
    print(f"Speedup: {uncached_time / cached_time:.1f}x")
    
    # Memory overhead
    cache_size = 2 * batch_size * n_heads * max_seq_len * (d_model // n_heads) * 4  # bytes
    print(f"Cache memory: {cache_size / 1e6:.1f}MB")

benchmark_kv_cache()
# Typical output: 10-50x speedup during generation
# Memory cost: ~800MB per sequence for 7B model
```

**Practical implications:**

- Essential for any autoregressive generation (text, code, etc.)
- Memory scales with sequence length: 2048 tokens ≈ 800MB for 7B model
- PagedAttention (used by vLLM) enables efficient cache management across requests

**Real constraints:**

KV cache dominates memory in production. A single GPU might fit 4 copies of a quantized 7B model but only serve 8-12 concurrent users due to cache overhead. Managing cache allocation is critical for throughput.

### 3. Batching Strategies: Latency vs. Throughput

Batching amortizes model loading and allows parallel processing, but introduces queuing delay. The optimal strategy depends on your service-level objectives (SL