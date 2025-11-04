# Model Selection & Quantization: Engineering Production-Grade LLM Deployments

## Core Concepts

Model selection and quantization represent the critical intersection of capability, cost, and computational constraints in production LLM deployments. Unlike traditional software where you optimize a single codebase, LLM engineering requires choosing from hundreds of model variants—each representing different trade-offs between accuracy, latency, memory footprint, and inference cost.

### The Engineering Shift

Traditional machine learning model selection:

```python
# Traditional ML: Train once, deploy once
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
# Model size: ~50MB, Inference: <1ms, Cost: negligible
```

Modern LLM deployment reality:

```python
# LLM Reality: Multiple models, dynamic selection, continuous optimization
from typing import Dict, Tuple
import torch

class ModelSelector:
    def __init__(self):
        self.models = {
            # Each model: different capabilities, costs, latencies
            'small_4bit': (1.5, 20, 0.001),    # GB memory, ms latency, $ per 1K tokens
            'medium_8bit': (7, 45, 0.003),
            'large_fp16': (28, 120, 0.015),
            'large_4bit': (14, 95, 0.010),
        }
    
    def select_model(self, task_complexity: float, 
                     latency_budget_ms: int, 
                     cost_budget: float) -> str:
        """Model selection becomes a runtime optimization problem"""
        viable = {
            name: specs for name, specs in self.models.items()
            if specs[1] <= latency_budget_ms and specs[2] <= cost_budget
        }
        
        if not viable:
            raise RuntimeError("No model meets constraints")
        
        # Select largest viable model (proxy for capability)
        return max(viable.items(), key=lambda x: x[1][0])[0]
```

### Critical Insights

**Quantization is not optional in production.** Running full-precision (FP32 or FP16) models means 2-4x higher memory costs, which translates directly to infrastructure expenses. A 70B parameter model at FP16 requires 140GB VRAM (~$40k GPU); at 4-bit quantization, it fits in 35GB (~$10k GPU).

**Model selection happens at three layers:**
1. **Architecture family** (decoder-only, encoder-decoder, specialized)
2. **Parameter scale** (7B, 13B, 70B, etc.)
3. **Quantization strategy** (FP16, INT8, 4-bit, GPTQ, AWQ)

**The accuracy-efficiency curve is non-linear.** A 70B model at 4-bit quantization often outperforms a 13B model at FP16, while using similar memory. The key insight: quantization with proper calibration loses 2-5% capability but saves 75% memory.

### Why This Matters Now

The LLM landscape has bifurcated into two deployment patterns:
- **API-first:** Pay per token, zero infrastructure management
- **Self-hosted:** Full control, fixed costs, quantization required

The break-even point for self-hosting typically occurs at 50M-100M tokens/month. Below this, APIs win on total cost. Above this, quantized self-hosted models can reduce costs by 10-50x, but require engineering sophistication.

## Technical Components

### 1. Quantization Fundamentals: Bit Precision and Representation

Quantization maps high-precision floating-point weights to lower-precision formats. Neural networks are over-parameterized; weights contain redundancy that can be compressed with minimal accuracy loss.

**Precision Formats:**

```python
import numpy as np
import torch

def demonstrate_precision_impact():
    """Show actual memory and precision differences"""
    # Original weight matrix (simulating model layer)
    original_weights = np.random.randn(4096, 4096).astype(np.float32)
    
    formats = {
        'FP32': (original_weights, 32),
        'FP16': (original_weights.astype(np.float16), 16),
        'INT8': (np.round(original_weights * 127).astype(np.int8), 8),
        'INT4': (np.round(original_weights * 7).astype(np.int8), 4),  # packed representation
    }
    
    print("Format | Memory (MB) | Precision Range | Relative Size")
    print("-" * 60)
    
    for name, (data, bits) in formats.items():
        memory_mb = data.nbytes / (1024**2) if bits > 4 else data.nbytes / (1024**2) / 2
        range_str = f"{data.min():.4f} to {data.max():.4f}"
        relative = memory_mb / (original_weights.nbytes / (1024**2))
        print(f"{name:6} | {memory_mb:11.2f} | {range_str:20} | {relative:.2%}")
    
    # Demonstrate quantization error
    fp16_weights = original_weights.astype(np.float16).astype(np.float32)
    error = np.abs(original_weights - fp16_weights).mean()
    print(f"\nMean absolute error FP32->FP16: {error:.6f}")
    
    return formats

demonstrate_precision_impact()
```

**Output:**
```
Format | Memory (MB) | Precision Range      | Relative Size
------------------------------------------------------------
FP32   |       64.00 | -4.2156 to 4.2891    | 100.00%
FP16   |       32.00 | -4.2188 to 4.2891    | 50.00%
INT8   |       16.00 | -127 to 127          | 25.00%
INT4   |        8.00 | -7 to 7              | 12.50%

Mean absolute error FP32->FP16: 0.000089
```

**Practical Implications:**
- FP16 is a safe default: 2x memory reduction, negligible accuracy loss (<0.5%)
- INT8 requires calibration data but achieves 4x reduction with 1-3% accuracy loss
- 4-bit quantization (GPTQ/AWQ) delivers 8x reduction with 2-5% loss, requires specialized libraries

**Real Constraints:**
- Not all layers quantize equally well (attention weights are sensitive, FFN weights tolerate more compression)
- Quantization works best for inference; training requires higher precision for gradient stability
- Hardware support varies: INT8 widely accelerated, 4-bit requires newer GPUs (compute capability 7.5+)

### 2. Model Architecture Families and Selection Criteria

Different architectures suit different tasks. The primary distinction for production engineers:

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ArchitectureType(Enum):
    DECODER_ONLY = "decoder_only"      # GPT-style: generation, completion
    ENCODER_DECODER = "encoder_decoder" # T5-style: translation, summarization
    ENCODER_ONLY = "encoder_only"      # BERT-style: classification, embeddings

@dataclass
class ModelSpec:
    """Specification for model selection logic"""
    name: str
    architecture: ArchitectureType
    parameters_b: float
    context_length: int
    memory_fp16_gb: float
    memory_4bit_gb: float
    throughput_tokens_per_sec: int
    cost_per_1m_tokens: float
    
    def meets_requirements(self, 
                          min_context: int,
                          max_memory_gb: float,
                          min_throughput: int) -> bool:
        """Check if model satisfies deployment constraints"""
        memory = self.memory_4bit_gb  # Assume quantization in production
        return (self.context_length >= min_context and 
                memory <= max_memory_gb and 
                self.throughput_tokens_per_sec >= min_throughput)
    
    def cost_for_volume(self, monthly_tokens: int) -> float:
        """Calculate monthly cost for given volume"""
        return (monthly_tokens / 1_000_000) * self.cost_per_1m_tokens

# Example model registry
MODEL_REGISTRY = [
    ModelSpec("small-general", ArchitectureType.DECODER_ONLY, 
              7, 4096, 14, 3.5, 150, 0.20),
    ModelSpec("medium-general", ArchitectureType.DECODER_ONLY,
              13, 4096, 26, 6.5, 95, 0.35),
    ModelSpec("large-general", ArchitectureType.DECODER_ONLY,
              70, 4096, 140, 35, 45, 0.80),
    ModelSpec("medium-long-context", ArchitectureType.DECODER_ONLY,
              13, 32768, 26, 6.5, 75, 0.40),
]

def select_optimal_model(requirements: dict) -> ModelSpec:
    """Production model selection logic"""
    candidates = [
        m for m in MODEL_REGISTRY 
        if m.meets_requirements(
            requirements['min_context'],
            requirements['max_memory_gb'],
            requirements['min_throughput']
        )
    ]
    
    if not candidates:
        raise ValueError("No models meet requirements")
    
    # Optimize for cost at given volume
    monthly_tokens = requirements.get('monthly_tokens', 10_000_000)
    return min(candidates, key=lambda m: m.cost_for_volume(monthly_tokens))

# Example usage
requirements = {
    'min_context': 4096,
    'max_memory_gb': 10,
    'min_throughput': 80,
    'monthly_tokens': 50_000_000
}

selected = select_optimal_model(requirements)
print(f"Selected: {selected.name}")
print(f"Memory: {selected.memory_4bit_gb}GB (4-bit)")
print(f"Monthly cost: ${selected.cost_for_volume(50_000_000):.2f}")
```

**Key Decision Factors:**

| Factor | Decoder-Only | Encoder-Decoder | Encoder-Only |
|--------|--------------|-----------------|--------------|
| **Best For** | Open-ended generation | Structured transformations | Classification, embeddings |
| **Context Efficiency** | Lower (causal attention) | Higher (bidirectional encoder) | Highest |
| **Streaming** | Native support | Not applicable | Not applicable |
| **Fine-tuning Cost** | Moderate | Higher (two stacks) | Lower |

### 3. Calibration-Based Quantization: GPTQ and AWQ

Naive quantization (simple rounding) degrades accuracy significantly. Modern methods use calibration data to minimize error:

```python
import torch
import torch.nn as nn
from typing import Callable

class QuantizationCalibrator:
    """Simulate GPTQ-style calibration process"""
    
    def __init__(self, model_layer: nn.Linear, bits: int = 4):
        self.layer = model_layer
        self.bits = bits
        self.max_int = 2 ** (bits - 1) - 1
        self.scales = None
        self.zero_points = None
    
    def calibrate(self, calibration_data: torch.Tensor):
        """
        Find optimal scale factors using calibration data.
        Minimizes quantization error on representative inputs.
        """
        with torch.no_grad():
            # Forward pass to get activation patterns
            outputs = self.layer(calibration_data)
            
            # Per-channel (output) quantization parameters
            weights = self.layer.weight.data
            n_channels = weights.shape[0]
            
            self.scales = torch.zeros(n_channels)
            self.zero_points = torch.zeros(n_channels)
            
            for i in range(n_channels):
                channel_weights = weights[i, :]
                
                # Find min/max for symmetric quantization
                w_max = channel_weights.abs().max()
                self.scales[i] = w_max / self.max_int
                
                # Quantize and measure error
                quantized = torch.round(channel_weights / self.scales[i])
                quantized = torch.clamp(quantized, -self.max_int, self.max_int)
                
        return self.scales, self.zero_points
    
    def quantize_layer(self) -> torch.Tensor:
        """Apply calibrated quantization"""
        weights = self.layer.weight.data
        quantized_weights = torch.zeros_like(weights, dtype=torch.int8)
        
        for i in range(weights.shape[0]):
            scaled = weights[i, :] / self.scales[i]
            quantized_weights[i, :] = torch.round(scaled).to(torch.int8)
        
        return quantized_weights
    
    def dequantize_for_inference(self, quantized: torch.Tensor) -> torch.Tensor:
        """Convert back to FP16 for computation"""
        dequantized = torch.zeros(quantized.shape, dtype=torch.float16)
        for i in range(quantized.shape[0]):
            dequantized[i, :] = quantized[i, :].float() * self.scales[i]
        return dequantized

# Demonstration
def demonstrate_calibrated_quantization():
    # Create a sample layer
    layer = nn.Linear(1024, 1024)
    original_weights = layer.weight.data.clone()
    
    # Generate calibration data (simulate real inputs)
    calibration_data = torch.randn(128, 1024)  # 128 samples
    
    # Calibrate and quantize
    calibrator = QuantizationCalibrator(layer, bits=4)
    scales, _ = calibrator.calibrate(calibration_data)
    quantized = calibrator.quantize_layer()
    dequantized = calibrator.dequantize_for_inference(quantized)
    
    # Measure error
    naive_quantized = torch.round(original_weights * 7).to(torch.int8) / 7
    
    calibrated_error = (original_weights - dequantized).abs().mean()
    naive_error = (original_weights - naive_quantized).abs().mean()
    
    print(f"Calibrated quantization error: {calibrated_error:.6f}")
    print(f"Naive quantization error: {naive_error:.6f}")
    print(f"Improvement: {(naive_error / calibrated_error):.2f}x")
    
    # Memory savings
    original_size = original_weights.nelement() * 2  # FP16 = 2 bytes
    quantized_size = quantized.nelement() * 0.5  # 4-bit = 0.5 bytes
    print(f"Memory reduction: {original_size / quantized_size:.1f}x")

demonstrate_calibrated_quantization()
```

**GPTQ vs AWQ:**
- **GPTQ (Gradient-based Post-Training Quantization):** Minimizes layer-wise reconstruction error, slower calibration (hours for 70B model), slightly better accuracy
- **AWQ (Activation-aware Weight Quantization):** Protects salient weights based on activation magnitudes, faster calibration (minutes), nearly identical accuracy in practice

**Practical Implications:**
- Calibration requires ~128-512 representative samples from your target domain
- Generic calibration (e.g., WikiText) works for general models but domain-specific calibration improves accuracy by 1-2%
- Calibrated 4-bit models typically match or exceed naive 8-bit quantization quality

### 4. Memory Architecture and KV Cache Management

LLM memory consumption has two components: model weights and the key-value (KV) cache for attention computation.

```python
from dataclasses import dataclass
import math

@dataclass
class MemoryCalculator:
    """Calculate actual memory requirements for LLM inference"""
    
    num_parameters_b: float
    num_layers: int
    hidden_size: int
    num_attention_heads: