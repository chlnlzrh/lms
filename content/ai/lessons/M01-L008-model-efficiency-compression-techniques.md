# Model Efficiency & Compression Techniques

## Core Concepts

Model efficiency and compression transform resource-intensive language models into deployable systems. Where uncompressed models demand 80GB of VRAM and 500ms inference latency, compressed variants achieve equivalent outputs using 8GB and 50ms—a 10x improvement in both dimensions without proportional quality degradation.

### Traditional vs. Modern Model Deployment

```python
# Traditional approach: Full precision model loading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Loads ~26GB for a 7B parameter model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float32  # 4 bytes per parameter
)
# Memory: 7B parameters × 4 bytes = 28GB
# Inference: ~800ms per 100 tokens on A100

# Modern approach: Quantized + optimized loading
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,      # 2 bytes per parameter
    device_map="auto",               # Automatic layer distribution
    load_in_4bit=True,               # 0.5 bytes per parameter
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # Nested quantization
)
# Memory: 7B parameters × 0.5 bytes = 3.5GB
# Inference: ~200ms per 100 tokens on RTX 4090
# Quality: <2% perplexity increase
```

The shift from naive deployment to compression-aware architecture isn't optional optimization—it's the difference between requiring $30,000 enterprise GPUs versus $1,500 consumer hardware for equivalent throughput.

### Engineering Insights

**Precision is negotiable, not sacred.** Neural networks store weights as floating-point numbers, but most weights contribute marginally to final outputs. A 7B parameter model contains ~6.8B redundant precision bits. Compression identifies and eliminates this redundancy.

**Memory bandwidth, not compute, bottlenecks inference.** Modern GPUs process teraflops but starve waiting for memory transfers. Reducing model size from 28GB to 4GB means transferring 85% less data per forward pass—directly proportional to latency reduction.

**Compression techniques stack multiplicatively.** Quantization + pruning + distillation don't add linearly—they compound. A 50% reduction from each technique yields 87.5% total reduction (0.5 × 0.5 × 0.5 = 0.125 remaining), not 150%.

### Why This Matters Now

The economics of LLM deployment hinge on compression. Serving one million requests daily:
- **Uncompressed (FP32):** 8× A100 GPUs, $50,000/month cloud costs
- **Quantized (INT4):** 1× RTX 4090, $150/month hardware amortization
- **Quality difference:** 1-3% task performance for 300× cost reduction

As models scale from 7B to 70B+ parameters, compression transitions from optimization to feasibility. A 70B FP32 model requires 280GB VRAM (multiple H100s); INT4 quantization fits it on a single consumer GPU.

## Technical Components

### 1. Quantization: Precision Reduction

Quantization maps high-precision floating-point weights to low-precision integers. FP32 (32 bits) → INT8 (8 bits) achieves 4× compression. INT4 (4 bits) reaches 8× compression.

**Technical Mechanism:**

```python
import torch
import torch.nn as nn

# Simulated quantization process
def quantize_tensor(tensor: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float, int]:
    """
    Symmetric quantization: maps float range to integer range.
    
    Returns:
        quantized_tensor: Integer representation
        scale: Float scaling factor
        zero_point: Integer offset
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    
    # Calculate quantization parameters
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - torch.round(min_val / scale)
    
    # Quantize
    quantized = torch.clamp(
        torch.round(tensor / scale + zero_point),
        qmin, qmax
    ).to(torch.int8)
    
    return quantized, scale.item(), int(zero_point.item())

def dequantize_tensor(
    quantized: torch.Tensor,
    scale: float,
    zero_point: int
) -> torch.Tensor:
    """Reconstruct approximate float values."""
    return (quantized.float() - zero_point) * scale

# Demonstration with model weights
layer = nn.Linear(1024, 1024)
original_weights = layer.weight.data
print(f"Original size: {original_weights.nbytes / 1024:.2f} KB")
print(f"Original dtype: {original_weights.dtype}")

# Quantize to INT8
quant_weights, scale, zero_point = quantize_tensor(original_weights, bits=8)
print(f"\nQuantized size: {quant_weights.nbytes / 1024:.2f} KB")
print(f"Compression ratio: {original_weights.nbytes / quant_weights.nbytes:.1f}x")

# Measure reconstruction error
reconstructed = dequantize_tensor(quant_weights, scale, zero_point)
mse = torch.mean((original_weights - reconstructed) ** 2)
print(f"Reconstruction MSE: {mse.item():.6f}")

# Practical inference comparison
input_tensor = torch.randn(32, 1024)  # Batch of 32

# Original inference
with torch.no_grad():
    output_original = nn.functional.linear(input_tensor, original_weights)

# Quantized inference (dequantized for compute)
with torch.no_grad():
    output_quantized = nn.functional.linear(
        input_tensor,
        reconstructed
    )

output_diff = torch.mean(torch.abs(output_original - output_quantized))
print(f"\nOutput difference: {output_diff.item():.6f}")
print(f"Relative error: {(output_diff / output_original.abs().mean()).item():.4%}")
```

**Practical Implications:**

- **INT8 quantization:** 4× memory reduction, <1% accuracy loss for most models
- **INT4 quantization:** 8× memory reduction, 1-3% accuracy loss, requires calibration
- **Dynamic vs. static quantization:** Dynamic quantizes per-batch (slower, more accurate), static uses fixed parameters (faster, less accurate)

**Real Constraints:**

- Matrix multiplication hardware optimizes for FP16/INT8, not arbitrary bit widths
- Activation quantization (not just weights) requires runtime calibration datasets
- Quantization-aware training (QAT) adds 2-3× training time but recovers 1-2% accuracy

**Concrete Example:**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configuration for 4-bit NormalFloat quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4: optimized for normally distributed weights
    bnb_4bit_use_double_quant=True,   # Quantize the quantization constants
    bnb_4bit_compute_dtype=torch.float16  # Computation happens in FP16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

# Result: 7B model in ~4GB VRAM instead of 14GB (FP16) or 28GB (FP32)
print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
```

### 2. Pruning: Structural Weight Elimination

Pruning removes unnecessary network connections or entire structures (neurons, attention heads, layers). Unstructured pruning removes individual weights; structured pruning removes organized components.

**Technical Mechanism:**

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class PrunableTransformerBlock(nn.Module):
    """Simplified transformer block for pruning demonstration."""
    
    def __init__(self, dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

def magnitude_pruning(module: nn.Module, amount: float = 0.3):
    """
    Prune weights with smallest absolute values.
    
    Args:
        module: Layer to prune
        amount: Fraction of weights to remove (0.3 = 30%)
    """
    prune.l1_unstructured(module, name='weight', amount=amount)
    # Makes pruning permanent (removes mask)
    prune.remove(module, 'weight')

def structured_pruning(module: nn.Linear, amount: float = 0.3, dim: int = 0):
    """
    Prune entire neurons/channels.
    
    Args:
        dim: 0 for output neurons, 1 for input connections
    """
    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=dim)
    prune.remove(module, 'weight')

# Demonstration
block = PrunableTransformerBlock()
mlp_layer = block.mlp[0]  # First MLP linear layer

# Count parameters before pruning
def count_nonzero_params(module: nn.Module) -> tuple[int, int]:
    total = 0
    nonzero = 0
    for param in module.parameters():
        total += param.numel()
        nonzero += torch.count_nonzero(param).item()
    return nonzero, total

before_nonzero, before_total = count_nonzero_params(mlp_layer)
print(f"Before pruning: {before_nonzero}/{before_total} parameters ({before_nonzero/before_total:.2%} dense)")

# Apply 40% magnitude pruning
magnitude_pruning(mlp_layer, amount=0.4)

after_nonzero, after_total = count_nonzero_params(mlp_layer)
print(f"After pruning: {after_nonzero}/{after_total} parameters ({after_nonzero/after_total:.2%} dense)")
print(f"Actual sparsity: {1 - after_nonzero/after_total:.2%}")

# Measure inference speed impact (structured pruning required for speedup)
input_tensor = torch.randn(32, 128, 768)  # [batch, seq_len, dim]

import time

# Benchmark unpruned
unpruned_block = PrunableTransformerBlock()
start = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        _ = unpruned_block(input_tensor)
unpruned_time = time.perf_counter() - start

# Benchmark pruned
pruned_block = PrunableTransformerBlock()
for name, module in pruned_block.named_modules():
    if isinstance(module, nn.Linear):
        structured_pruning(module, amount=0.3, dim=0)

start = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        _ = pruned_block(input_tensor)
pruned_time = time.perf_counter() - start

print(f"\nInference time - Unpruned: {unpruned_time*10:.2f}ms")
print(f"Inference time - Pruned: {pruned_time*10:.2f}ms")
print(f"Speedup: {unpruned_time/pruned_time:.2f}x")
```

**Practical Implications:**

- **Unstructured pruning:** Achieves 50-90% sparsity but requires sparse matrix kernels for speedup
- **Structured pruning:** Lower sparsity (30-50%) but immediate speedup on standard hardware
- **Iterative pruning:** Prune → finetune → prune cycles maintain quality better than one-shot pruning

**Real Constraints:**

- Generic PyTorch doesn't accelerate unstructured sparsity—needs specialized libraries (TorchSparse, DeepSparse)
- Pruning attention heads arbitrarily breaks multi-head attention mechanics
- Accuracy recovery via finetuning requires 5-10% of original training compute

### 3. Knowledge Distillation: Student-Teacher Training

Distillation trains a smaller "student" model to mimic a larger "teacher" model's outputs, not just match training labels. The student learns compressed representations of the teacher's knowledge.

**Technical Mechanism:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TeacherModel(nn.Module):
    """Large, accurate model (e.g., 12-layer transformer)."""
    
    def __init__(self, vocab_size: int = 10000, hidden_dim: int = 768, num_layers: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, nhead=12, dim_feedforward=hidden_dim*4)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

class StudentModel(nn.Module):
    """Smaller, efficient model (e.g., 3-layer transformer)."""
    
    def __init__(self, vocab_size: int = 10000, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=hidden_dim*4)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.7
) -> torch.Tensor:
    """
    Combines soft targets (teacher) with hard targets (labels).
    
    Args:
        temperature: Softens probability distributions (higher