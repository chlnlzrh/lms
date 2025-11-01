# Hardware Requirements & Scaling for Production LLM Deployments

## Core Concepts

When deploying large language models, hardware isn't just a cost center—it's the primary constraint that determines what's architecturally possible. Unlike traditional web services where you scale horizontally by adding commodity servers, LLM inference requires specialized hardware configurations where memory bandwidth, not compute throughput, typically becomes your bottleneck.

### Engineering Analogy: Traditional vs. LLM Serving

```python
# Traditional API Service (CPU-bound, horizontally scalable)
from flask import Flask
import multiprocessing

app = Flask(__name__)

@app.route('/analyze')
def analyze_text(text: str) -> dict:
    # Business logic fits in CPU cache
    # Each request: ~10MB memory, ~50ms CPU
    # Scale: add more $100/month instances
    return {"sentiment": "positive", "score": 0.87}

# Horizontally scalable: 10 instances = 10x throughput
```

```python
# LLM Inference (Memory bandwidth-bound, vertically scalable)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMService:
    def __init__(self, model_name: str):
        # Model weights: 13-175GB must stay in GPU VRAM
        # Each token generation: read ALL parameters
        # Bottleneck: memory bandwidth (not FLOPS)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        # Each token: ~100GB memory reads for 70B model
        # Memory bandwidth: 2TB/s (A100) limits tokens/sec
        # Scale: need bigger/more expensive GPUs, not more instances
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0])

# Not horizontally scalable: 10 GPUs ≠ 10x throughput per GPU
# Need sophisticated parallelism strategies
```

### Why This Matters NOW

The difference between a prototype that works on sample data and a production system serving real traffic comes down to understanding three hardware realities:

1. **Memory is the new compute**: A 70B parameter model requires reading ~140GB per token generated (FP16). At 2TB/s memory bandwidth (A100), you're physically limited to ~14 tokens/second per GPU, regardless of FLOPS.

2. **Batching dynamics are inverted**: Traditional services improve efficiency with larger batches. LLM inference with Key-Value (KV) cache means each additional sequence in a batch linearly increases memory consumption, creating a throughput/latency trade-off.

3. **Cost scales non-linearly**: Moving from a 7B to 70B model isn't 10x more expensive—it's often 30-50x more due to requiring multi-GPU setups, NVLink interconnects, and architectural complexity.

Understanding these constraints lets you make architectural decisions early—like choosing model sizes, quantization strategies, and deployment patterns—that determine whether your project is economically viable.

## Technical Components

### 1. GPU Memory Architecture and Parameter Storage

Modern LLMs require storing billions of parameters in GPU memory for fast access during inference. The memory requirements follow a predictable pattern based on model size and precision.

**Technical Explanation:**

```python
from dataclasses import dataclass
from typing import Literal
from enum import Enum

class Precision(Enum):
    FP32 = 4  # bytes per parameter
    FP16 = 2
    BF16 = 2
    INT8 = 1
    INT4 = 0.5

@dataclass
class ModelMemoryProfile:
    """Calculate memory requirements for LLM inference."""
    
    parameter_count: int  # Total parameters (e.g., 70B)
    precision: Precision
    sequence_length: int = 2048
    batch_size: int = 1
    
    def parameters_memory_gb(self) -> float:
        """Base model weights memory."""
        bytes_total = self.parameter_count * self.precision.value
        return bytes_total / (1024**3)
    
    def kv_cache_memory_gb(self) -> float:
        """Key-Value cache for attention mechanism.
        
        For decoder models: 2 (key+value) * num_layers * hidden_size * seq_len
        Approximation: ~2x num_parameters * seq_len / model_dim ratio
        """
        # Simplified: assumes typical architecture ratios
        # Real calculation needs num_layers, hidden_dim, num_heads
        bytes_per_token = (self.parameter_count / 1000) * 2 * self.precision.value
        total_tokens = self.sequence_length * self.batch_size
        bytes_total = bytes_per_token * total_tokens
        return bytes_total / (1024**3)
    
    def activation_memory_gb(self) -> float:
        """Intermediate activations during forward pass."""
        # Rough estimate: ~10-20% of parameter memory for single sequence
        return self.parameters_memory_gb() * 0.15 * self.batch_size
    
    def total_memory_gb(self) -> float:
        """Total GPU memory required with 20% overhead."""
        base = (self.parameters_memory_gb() + 
                self.kv_cache_memory_gb() + 
                self.activation_memory_gb())
        return base * 1.2  # Framework overhead

# Example calculations
llama70b_fp16 = ModelMemoryProfile(
    parameter_count=70_000_000_000,
    precision=Precision.FP16,
    sequence_length=2048,
    batch_size=1
)

print(f"Llama 70B FP16 Memory Breakdown:")
print(f"  Parameters: {llama70b_fp16.parameters_memory_gb():.1f} GB")
print(f"  KV Cache:   {llama70b_fp16.kv_cache_memory_gb():.1f} GB")
print(f"  Activations: {llama70b_fp16.activation_memory_gb():.1f} GB")
print(f"  Total:      {llama70b_fp16.total_memory_gb():.1f} GB")

# Compare quantization impact
llama70b_int4 = ModelMemoryProfile(
    parameter_count=70_000_000_000,
    precision=Precision.INT4,
    sequence_length=2048,
    batch_size=1
)

print(f"\nLlama 70B INT4 (quantized):")
print(f"  Total:      {llama70b_int4.total_memory_gb():.1f} GB")
print(f"  Memory reduction: {(1 - llama70b_int4.total_memory_gb() / llama70b_fp16.total_memory_gb()) * 100:.0f}%")
```

**Practical Implications:**

- **Single GPU limits**: A100 (80GB) can handle ~70B parameters at INT4, ~30B at FP16
- **Multi-GPU necessity**: 70B FP16 requires 2x A100 80GB minimum
- **KV cache growth**: Doubling sequence length or batch size doubles KV cache memory

**Real Constraints:**

The KV cache creates a critical trade-off. For a 70B model at FP16 with 2048 token context:
- Single sequence: ~10GB KV cache
- Batch of 8: ~80GB just for KV cache (more than model weights!)
- Long context (8K tokens): ~40GB per sequence

This is why production systems carefully balance batch size against latency requirements.

### 2. Memory Bandwidth and Token Generation Speed

Token generation is memory-bandwidth bound, not compute-bound. Each token requires loading all model parameters from GPU memory.

**Technical Explanation:**

```python
from typing import NamedTuple

class GPUSpec(NamedTuple):
    name: str
    memory_bandwidth_tbs: float  # TB/s
    vram_gb: int
    tflops_fp16: int

# Common GPU specifications
GPUS = {
    "A100_80GB": GPUSpec("A100 80GB", 2.0, 80, 312),
    "H100_80GB": GPUSpec("H100 80GB", 3.35, 80, 989),
    "V100_32GB": GPUSpec("V100 32GB", 0.9, 32, 125),
    "A10G_24GB": GPUSpec("A10G 24GB", 0.6, 24, 125),
}

def theoretical_tokens_per_second(
    gpu: GPUSpec,
    model_size_b: int,
    precision: Precision,
    batch_size: int = 1
) -> float:
    """Calculate theoretical max tokens/sec based on memory bandwidth.
    
    Each token generation requires reading all parameters once.
    Bandwidth limit = bytes_per_second / bytes_per_token
    """
    model_bytes = model_size_b * 1e9 * precision.value
    bandwidth_bytes_per_sec = gpu.memory_bandwidth_tbs * 1e12
    
    # Tokens per second (single sequence)
    tokens_per_sec_single = bandwidth_bytes_per_sec / model_bytes
    
    # With batching, can generate N tokens in similar time
    # (until KV cache fills memory)
    tokens_per_sec_batch = tokens_per_sec_single * batch_size
    
    return tokens_per_sec_batch

# Compare different hardware
for gpu_name, gpu in GPUS.items():
    tokens_sec = theoretical_tokens_per_second(
        gpu, 
        model_size_b=70,
        precision=Precision.FP16,
        batch_size=1
    )
    print(f"{gpu.name:12} | 70B FP16 | {tokens_sec:.1f} tok/s (theoretical)")

print("\nBatching impact on A100:")
for batch in [1, 4, 8, 16]:
    tokens_sec = theoretical_tokens_per_second(
        GPUS["A100_80GB"],
        model_size_b=70,
        precision=Precision.FP16,
        batch_size=batch
    )
    throughput = tokens_sec * batch
    print(f"  Batch {batch:2d}: {tokens_sec:.1f} tok/s/seq | {throughput:.0f} tok/s total")
```

**Practical Implications:**

- **H100 vs A100**: 1.7x bandwidth increase = 1.7x token generation speed
- **Compute is not the bottleneck**: H100 has 3x more TFLOPS than A100, but only 1.7x faster for inference
- **Batch size sweet spot**: Increase until KV cache fills memory

**Real Constraints:**

Production systems typically see 60-70% of theoretical bandwidth due to:
- Framework overhead (PyTorch/CUDA kernel launch)
- Non-contiguous memory access patterns
- CPU-GPU synchronization
- KV cache management overhead

### 3. Multi-GPU Parallelism Strategies

When models exceed single GPU memory, parallelism becomes mandatory. Three primary strategies exist, each with different trade-offs.

**Technical Explanation:**

```python
from enum import Enum
from typing import List
import math

class ParallelismStrategy(Enum):
    PIPELINE = "pipeline"
    TENSOR = "tensor"
    SEQUENCE = "sequence"

class MultiGPUConfig:
    """Calculate multi-GPU configuration requirements."""
    
    def __init__(
        self,
        model_size_b: int,
        precision: Precision,
        num_gpus: int,
        strategy: ParallelismStrategy,
        sequence_length: int = 2048
    ):
        self.model_size_b = model_size_b
        self.precision = precision
        self.num_gpus = num_gpus
        self.strategy = strategy
        self.sequence_length = sequence_length
    
    def memory_per_gpu_gb(self) -> float:
        """Memory required per GPU."""
        base_memory = self.model_size_b * self.precision.value
        
        if self.strategy == ParallelismStrategy.PIPELINE:
            # Each GPU holds consecutive layers
            # Memory split across GPUs, but each needs full activations
            model_memory = base_memory / self.num_gpus
            activation_memory = base_memory * 0.15  # Full activation memory
            
        elif self.strategy == ParallelismStrategy.TENSOR:
            # Each GPU holds fraction of each layer (weight matrices split)
            # Lower memory, but requires high-bandwidth interconnect
            model_memory = base_memory / self.num_gpus
            activation_memory = (base_memory * 0.15) / self.num_gpus
            
        elif self.strategy == ParallelismStrategy.SEQUENCE:
            # Each GPU processes different sequences
            # Full model replicated on each GPU
            model_memory = base_memory
            activation_memory = base_memory * 0.15
        
        # KV cache depends on strategy
        kv_cache = self._kv_cache_per_gpu()
        
        total_gb = (model_memory + activation_memory + kv_cache) / (1024**3)
        return total_gb * 1.2  # Overhead
    
    def _kv_cache_per_gpu(self) -> float:
        """KV cache size per GPU."""
        base_kv = (self.model_size_b / 1000) * 2 * self.precision.value * self.sequence_length
        
        if self.strategy == ParallelismStrategy.TENSOR:
            return base_kv / self.num_gpus
        else:
            return base_kv
    
    def communication_overhead_percent(self) -> float:
        """Estimate communication overhead."""
        if self.strategy == ParallelismStrategy.PIPELINE:
            # Only activation passing between stages
            return 5.0 + (self.num_gpus * 0.5)  # Increases with pipeline depth
        
        elif self.strategy == ParallelismStrategy.TENSOR:
            # All-reduce after each layer (bandwidth intensive)
            return 15.0 + (self.num_gpus * 2.0)  # Requires NVLink
        
        elif self.strategy == ParallelismStrategy.SEQUENCE:
            # No communication during forward pass
            return 0.0
    
    def effective_speedup(self) -> float:
        """Realistic speedup vs single GPU."""
        if self.strategy == ParallelismStrategy.SEQUENCE:
            # Perfect scaling for throughput (not latency)
            return self.num_gpus
        
        # Account for communication overhead
        overhead = self.communication_overhead_percent() / 100
        ideal_speedup = self.num_gpus if self.strategy == ParallelismStrategy.TENSOR else 1.0
        return ideal_speedup * (1 - overhead)

# Example: 70B model on different configurations
configs = [
    MultiGPUConfig(70, Precision.FP16, 2, ParallelismStrategy.PIPELINE),
    MultiGPUConfig(70, Precision.FP16, 2, ParallelismStrategy.TENSOR),
    MultiGPUConfig(70, Precision.FP16, 4, ParallelismStrategy.TENSOR),
    MultiGPUConfig(70, Precision.INT4, 1, ParallelismStrategy.SEQUENCE),
]

print("70B Model Multi-GPU Configurations:\n")
for config in configs:
    print(f"{config.strategy.value.title()} Parallelism | {config.num_gpus} GPUs | {config.precision.name}")
    print(f"  Memory/GPU: {config.memory_per_gpu_gb():.1f} GB")
    print(f"  Comm Overhead: {config.communication_overhead_percent():.1f}%")
    print(f"  Effective Speedup: {config.effective_speedup():.2f}x")
    print()
```

**