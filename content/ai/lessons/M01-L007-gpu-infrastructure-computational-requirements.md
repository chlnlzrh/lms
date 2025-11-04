# GPU Infrastructure & Computational Requirements

Modern AI development fundamentally changes how we think about computational resources. Unlike traditional software where a single powerful CPU suffices, AI workloads demand specialized hardware architectures that parallelize thousands of operations simultaneously. This lesson explores the technical realities of GPU infrastructure, helping you make informed decisions about training, fine-tuning, and deploying language models.

## Core Concepts

### Technical Definition

GPUs (Graphics Processing Units) are specialized processors designed for parallel computation. While CPUs excel at sequential processing with complex logic, GPUs contain thousands of simpler cores optimized for performing the same operation across massive datasets simultaneously. In AI contexts, this architecture accelerates matrix multiplication—the fundamental operation underlying neural network training and inference.

Modern AI workloads require understanding three computational paradigms:

- **Training**: Computing gradients across billions of parameters, requiring high-bandwidth memory and extreme parallel throughput
- **Fine-tuning**: Updating subset of parameters or adapters, requiring moderate compute but careful memory management
- **Inference**: Forward passes only, optimizable through quantization and batching

### Engineering Analogy

```python
import time
import numpy as np

def cpu_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Traditional CPU approach: sequential operations"""
    rows_a, cols_a = a.shape
    cols_b = b.shape[1]
    result = np.zeros((rows_a, cols_b))
    
    # Sequential iteration - one multiplication at a time
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i, j] += a[i, k] * b[k, j]
    return result

def gpu_matrix_multiply_simulation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GPU approach: vectorized parallel operations"""
    # All multiplications happen simultaneously across cores
    return np.dot(a, b)

# Benchmark comparison
size = 1024
matrix_a = np.random.rand(size, size)
matrix_b = np.random.rand(size, size)

# CPU-style sequential
start = time.time()
result_cpu = cpu_matrix_multiply(matrix_a[:32, :32], matrix_b[:32, :32])
cpu_time = time.time() - start

# GPU-style parallel (using numpy's optimized routines)
start = time.time()
result_gpu = gpu_matrix_multiply_simulation(matrix_a, matrix_b)
gpu_time = time.time() - start

print(f"CPU-style (32x32): {cpu_time*1000:.2f}ms")
print(f"GPU-style (1024x1024): {gpu_time*1000:.2f}ms")
print(f"GPU handles {(1024/32)**2:.0f}x more data in similar time")
```

The difference becomes critical when training a 7B parameter model requires approximately 10^21 floating-point operations per training token. Sequential processing would take years; parallel GPU computation reduces this to hours or minutes.

### Key Insights

**Memory bandwidth trumps compute speed**: Modern GPU architectures are often memory-bound, not compute-bound. A model with 70B parameters requires 140GB just to load weights in FP16 format. Memory bandwidth determines how quickly these parameters flow to compute cores.

**Precision trade-offs unlock scale**: Training typically uses FP32 (32-bit floating-point), but inference can use INT8 (8-bit integers) with minimal accuracy loss, reducing memory requirements by 4x and increasing throughput proportionally.

**Batch size is your leverage point**: GPUs excel at processing multiple inputs simultaneously. A single inference might utilize 10% of GPU capacity; batching 32 requests can approach 80%+ utilization with only marginal latency increase.

### Why This Matters Now

The economics of AI development have shifted dramatically. Cloud GPU instances cost $2-40/hour depending on capability. A poorly configured training run can waste thousands of dollars. Understanding computational requirements enables:

- **Accurate cost forecasting**: Estimate whether a project needs $100 or $100,000 in compute
- **Architecture decisions**: Choose between fine-tuning existing models vs. training from scratch
- **Deployment optimization**: Balance latency, throughput, and infrastructure costs
- **Bottleneck identification**: Diagnose whether slow performance stems from compute, memory, or I/O constraints

## Technical Components

### 1. GPU Memory Hierarchy and Bandwidth

Modern GPUs contain multiple memory tiers with vastly different performance characteristics:

```python
from typing import Dict, Tuple
import math

def calculate_memory_requirements(
    num_parameters: int,
    precision_bytes: int,
    batch_size: int,
    sequence_length: int,
    training: bool = False
) -> Dict[str, float]:
    """
    Calculate GPU memory requirements for LLM operations.
    
    Args:
        num_parameters: Model size (e.g., 7_000_000_000 for 7B model)
        precision_bytes: Bytes per parameter (4 for FP32, 2 for FP16, 1 for INT8)
        batch_size: Number of sequences processed simultaneously
        sequence_length: Tokens per sequence
        training: Whether calculating for training (needs gradients/optimizer states)
    
    Returns:
        Dictionary with memory breakdown in GB
    """
    # Model weights
    model_memory = (num_parameters * precision_bytes) / (1024**3)
    
    # Activations (intermediate layer outputs)
    # Approximation: roughly 4x model size for activations during forward pass
    activation_memory = (model_memory * 4 * batch_size * sequence_length) / 2048
    
    # Training-specific memory
    if training:
        # Gradients: same size as model
        gradient_memory = model_memory
        # Optimizer states (Adam uses 2x model size for momentum and variance)
        optimizer_memory = model_memory * 2
    else:
        gradient_memory = 0
        optimizer_memory = 0
    
    # KV cache for generation (stores keys and values for each layer)
    # Approximation: 2 * num_layers * hidden_dim * sequence_length * batch_size
    num_layers = math.log(num_parameters, 10) * 8  # Rough approximation
    hidden_dim = (num_parameters / (num_layers * 12)) ** 0.5
    kv_cache = (2 * num_layers * hidden_dim * sequence_length * batch_size * precision_bytes) / (1024**3)
    
    total = model_memory + activation_memory + gradient_memory + optimizer_memory + kv_cache
    
    return {
        "model_weights_gb": round(model_memory, 2),
        "activations_gb": round(activation_memory, 2),
        "gradients_gb": round(gradient_memory, 2),
        "optimizer_states_gb": round(optimizer_memory, 2),
        "kv_cache_gb": round(kv_cache, 2),
        "total_gb": round(total, 2),
        "recommended_gpu_memory_gb": round(total * 1.2, 2)  # 20% overhead
    }

# Example: 7B model inference
inference_7b = calculate_memory_requirements(
    num_parameters=7_000_000_000,
    precision_bytes=2,  # FP16
    batch_size=4,
    sequence_length=2048,
    training=False
)

print("7B Model Inference (FP16, batch=4, seq_len=2048):")
for key, value in inference_7b.items():
    print(f"  {key}: {value}")

# Example: 7B model training
training_7b = calculate_memory_requirements(
    num_parameters=7_000_000_000,
    precision_bytes=4,  # FP32 for training
    batch_size=2,
    sequence_length=2048,
    training=True
)

print("\n7B Model Training (FP32, batch=2, seq_len=2048):")
for key, value in training_7b.items():
    print(f"  {key}: {value}")
```

**Practical Implications**: A 7B model requires minimum 14GB VRAM for inference (FP16) but 80GB+ for training. This explains why consumer GPUs (16-24GB) can run inference but not train large models.

**Real Constraints**: Memory bandwidth limits throughput. A GPU with 900 GB/s bandwidth requires ~15ms just to load a 13B FP16 model into compute cores—this becomes the latency floor regardless of compute speed.

**Trade-offs**: Higher precision (FP32) improves numerical stability but doubles memory usage. Mixed-precision training uses FP16 for forward/backward passes but FP32 for weight updates, balancing speed and stability.

### 2. Compute Architecture: Cores, Tensor Cores, and FLOPs

GPU specifications advertise peak FLOPs (floating-point operations per second), but achievable performance depends on workload characteristics:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPUSpec:
    """Simplified GPU specifications"""
    name: str
    fp32_tflops: float  # TeraFLOPs for FP32
    fp16_tflops: float  # TeraFLOPs for FP16 (with Tensor Cores)
    memory_gb: int
    memory_bandwidth_gbps: int
    tdp_watts: int
    cost_per_hour: float

@dataclass
class WorkloadProfile:
    """Computational workload characteristics"""
    total_flops: float  # Total operations required
    memory_accessed_gb: float  # Total memory reads/writes
    precision: str  # 'fp32' or 'fp16'

def estimate_runtime(
    gpu: GPUSpec,
    workload: WorkloadProfile,
    efficiency: float = 0.6
) -> dict:
    """
    Estimate workload runtime considering compute and memory constraints.
    
    Args:
        gpu: GPU specifications
        workload: Workload characteristics
        efficiency: Achievable efficiency (0-1), typically 0.4-0.7 for real workloads
    
    Returns:
        Dictionary with timing breakdown and bottleneck identification
    """
    # Compute-bound time
    tflops = gpu.fp16_tflops if workload.precision == 'fp16' else gpu.fp32_tflops
    achievable_tflops = tflops * efficiency
    compute_time_seconds = workload.total_flops / (achievable_tflops * 1e12)
    
    # Memory-bound time
    memory_time_seconds = workload.memory_accessed_gb / gpu.memory_bandwidth_gbps
    
    # Actual time is the maximum (bottleneck)
    actual_time = max(compute_time_seconds, memory_time_seconds)
    bottleneck = "compute" if compute_time_seconds > memory_time_seconds else "memory"
    
    # Cost calculation
    cost = (actual_time / 3600) * gpu.cost_per_hour
    
    # Energy consumption
    energy_kwh = (actual_time / 3600) * (gpu.tdp_watts / 1000)
    
    return {
        "compute_time_sec": round(compute_time_seconds, 2),
        "memory_time_sec": round(memory_time_seconds, 2),
        "actual_time_sec": round(actual_time, 2),
        "bottleneck": bottleneck,
        "compute_utilization": round(compute_time_seconds / actual_time, 2),
        "cost_usd": round(cost, 2),
        "energy_kwh": round(energy_kwh, 3)
    }

# Define sample GPUs
gpu_consumer = GPUSpec(
    name="Consumer 24GB",
    fp32_tflops=35,
    fp16_tflops=70,
    memory_gb=24,
    memory_bandwidth_gbps=900,
    tdp_watts=350,
    cost_per_hour=1.50
)

gpu_datacenter = GPUSpec(
    name="Datacenter 80GB",
    fp32_tflops=20,
    fp16_tflops=300,
    memory_gb=80,
    memory_bandwidth_gbps=2000,
    tdp_watts=400,
    cost_per_hour=8.00
)

# Workload: 7B model inference on 1000 tokens
workload_inference = WorkloadProfile(
    total_flops=7e9 * 2 * 1000,  # params * 2 ops per param * tokens
    memory_accessed_gb=14,  # Load model once (FP16)
    precision='fp16'
)

# Workload: 7B model training on 10,000 tokens
workload_training = WorkloadProfile(
    total_flops=7e9 * 6 * 10000,  # 3x operations (forward, backward, optimizer)
    memory_accessed_gb=280,  # Multiple passes through weights
    precision='fp16'
)

print("Inference Comparison (1000 tokens):")
print(f"Consumer GPU: {estimate_runtime(gpu_consumer, workload_inference)}")
print(f"Datacenter GPU: {estimate_runtime(gpu_datacenter, workload_inference)}")

print("\nTraining Comparison (10,000 tokens):")
print(f"Consumer GPU: {estimate_runtime(gpu_consumer, workload_training)}")
print(f"Datacenter GPU: {estimate_runtime(gpu_datacenter, workload_training)}")
```

**Practical Implications**: Datacenter GPUs cost 5x more per hour but provide 4x+ throughput for large-batch workloads. For sustained training, higher upfront cost pays off quickly.

**Real Constraints**: Tensor Cores accelerate matrix multiplication but require specific dimensions (multiples of 8 for FP16). Misaligned operations fall back to slower CUDA cores.

**Trade-offs**: Consumer GPUs offer better FP32 performance (gaming optimization), while datacenter GPUs prioritize FP16/INT8 (AI optimization). Choose based on workload precision requirements.

### 3. Multi-GPU Scaling: Parallelism Strategies

Single GPUs hit memory limits around 13B-30B parameters. Larger models require distributing computation across multiple devices:

```python
from enum import Enum
from typing import List

class ParallelismStrategy(Enum):
    DATA = "data_parallelism"
    MODEL = "model_parallelism"
    PIPELINE = "pipeline_parallelism"
    TENSOR = "tensor_parallelism"

def calculate_multi_gpu_efficiency(
    num_gpus: int,
    strategy: ParallelismStrategy,
    model_size_gb: float,
    batch_size: int,
    communication_overhead: float = 0.1
) -> dict:
    """
    Estimate multi-GPU training efficiency and requirements.
    
    Args:
        num_gpus: Number of GPUs
        strategy: Parallelism approach
        model_size_gb: Model memory footprint
        batch_size: Total batch size across all GPUs
        communication_overhead: Fraction of time spent on GPU-to-GPU communication
    
    Returns:
        Efficiency metrics and recommendations
    """
    if strategy == ParallelismStrategy.DATA:
        # Each GPU has full model copy, processes subset of batch
        memory_per_gpu = model_size_gb
        effective_batch = batch_size
        # Communication for gradient synchronization
        comm_overhead = communication_overhead * (1 - 1/num_gpus)
        scaling_efficiency = (1 - comm_overhead)
        
    elif strategy == ParallelismStrategy.MODEL:
        # Model split across GPUs, each processes full batch
        memory_per_gpu = model_size_gb / num_gpus
        effective_batch = batch_size
        # High communication for activation passing
        comm_overhead = communication_overhead * 2
        scaling_efficiency = (1 - comm_overhead)
        
    elif strategy == ParallelismStrategy.PIPELINE:
        # Model split in stages, micro-batches flow through pipeline
        memory_per_gpu = model_size_gb / num_gpus
        effective_batch = batch_size
        # Pipeline bubbles reduce efficiency
        pipeline_bubble = 1 - (batch_size / (batch_size + num_gpus - 1))
        scaling_efficiency = 1 - pipeline_bubble - communication_overhead
        