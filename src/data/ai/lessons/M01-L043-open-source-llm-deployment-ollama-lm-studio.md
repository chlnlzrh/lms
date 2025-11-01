# Open-Source LLM Deployment: Local Inference Infrastructure

## Core Concepts

Open-source LLM deployment refers to running large language models on your own infrastructure rather than through third-party APIs. This involves downloading model weights, managing inference engines, and handling the complete execution stack from tokenization through text generation.

### Engineering Analogy

**Traditional API-Based Approach:**
```python
import requests
from typing import Dict, Any

def generate_with_api(prompt: str, api_key: str) -> str:
    """API-based inference: simple but opaque"""
    response = requests.post(
        "https://api.provider.com/v1/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"]

# Black box: no control over model, latency, data routing
result = generate_with_api("Analyze this data...", api_key="sk-...")
```

**Local Deployment Approach:**
```python
import subprocess
import json
from typing import Dict, Any, Optional

class LocalLLMEngine:
    """Local inference: full control and transparency"""
    
    def __init__(self, model_name: str, host: str = "localhost", port: int = 11434):
        self.model_name = model_name
        self.base_url = f"http://{host}:{port}"
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self) -> None:
        """Verify model is pulled and ready"""
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        if self.model_name not in result.stdout:
            subprocess.run(["ollama", "pull", self.model_name], check=True)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        context_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate with full observability"""
        import requests
        import time
        
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if context_window:
            payload["options"]["num_ctx"] = context_window
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        inference_time = time.time() - start_time
        
        # Full metrics visibility
        return {
            "text": result["response"],
            "inference_time": inference_time,
            "tokens_evaluated": result.get("eval_count", 0),
            "tokens_per_second": result.get("eval_count", 0) / inference_time,
            "model_load_duration": result.get("load_duration", 0) / 1e9,
            "prompt_eval_duration": result.get("prompt_eval_duration", 0) / 1e9,
        }

# Full control: model selection, resource allocation, data locality
engine = LocalLLMEngine("mistral:7b-instruct")
result = engine.generate("Analyze this data...", temperature=0.3)
print(f"Generated in {result['inference_time']:.2f}s at {result['tokens_per_second']:.1f} tok/s")
```

The fundamental shift is from **consumption** to **operation**. You're no longer calling a service—you're managing an inference system with its own performance characteristics, resource requirements, and operational complexity.

### Key Insights

**Data Sovereignty**: Your prompts and completions never leave your infrastructure. This isn't just privacy theater—it's architectural isolation. Sensitive data processing doesn't require vendor NDAs, compliance attestations, or trust in third-party security controls.

**Performance Predictability**: Local deployment means deterministic latency bounds. No network hops, no rate limits, no shared tenancy interference. Your p99 latency is determined by your hardware and model choice, not external factors.

**Cost Structure Inversion**: API pricing is per-token with variable costs. Local deployment has fixed hardware costs with near-zero marginal cost per inference. The breakeven point typically occurs around 5-10 million tokens/month for CPU inference, lower for GPU.

**Model Control**: You select exact model versions, quantization levels, and can swap implementations without code changes. Want to A/B test quantized vs. full-precision? Deploy both and route traffic programmatically.

### Why This Matters Now

The quantization revolution (GGUF format, 4-bit/8-bit inference) has made 7B-13B parameter models viable on consumer hardware. A model that would have required $10K+ in GPUs 18 months ago now runs acceptably on a $2K workstation or $200/month cloud instance. The price-performance inflection point has arrived.

Second, open model quality has reached production viability. Models like Mistral 7B, Llama 3 8B, and Qwen 2.5 7B match or exceed GPT-3.5 performance on many tasks while fitting in 6GB RAM quantized. The quality gap that justified API costs has narrowed dramatically.

## Technical Components

### 1. Inference Runtime Architecture

Local LLM deployment requires an inference server that loads model weights, manages GPU/CPU execution, and exposes an API. The runtime handles tokenization, attention computation, and decoding.

**Technical Implementation:**

```python
import psutil
import GPUtil
from dataclasses import dataclass
from typing import Optional, List
import subprocess
import json

@dataclass
class SystemResources:
    """System capability detection"""
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    gpu_name: Optional[str]

def detect_system_resources() -> SystemResources:
    """Detect available compute resources"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            primary_gpu = gpus[0]
            return SystemResources(
                cpu_cores=psutil.cpu_count(logical=False),
                ram_gb=psutil.virtual_memory().total / (1024**3),
                gpu_available=True,
                gpu_memory_gb=primary_gpu.memoryTotal / 1024,
                gpu_name=primary_gpu.name
            )
    except:
        pass
    
    return SystemResources(
        cpu_cores=psutil.cpu_count(logical=False),
        ram_gb=psutil.virtual_memory().total / (1024**3),
        gpu_available=False,
        gpu_memory_gb=0.0,
        gpu_name=None
    )

def recommend_model_configuration(resources: SystemResources) -> Dict[str, Any]:
    """Recommend model size and quantization based on hardware"""
    
    if resources.gpu_available and resources.gpu_memory_gb >= 12:
        return {
            "model_size": "13b",
            "quantization": "q4_K_M",  # 4-bit medium quality
            "context_window": 4096,
            "batch_size": 512,
            "num_gpu_layers": -1,  # Full GPU offload
            "expected_vram": "8-10GB",
            "expected_speed": "30-50 tok/s"
        }
    elif resources.gpu_available and resources.gpu_memory_gb >= 6:
        return {
            "model_size": "7b",
            "quantization": "q4_K_M",
            "context_window": 4096,
            "batch_size": 256,
            "num_gpu_layers": -1,
            "expected_vram": "5-6GB",
            "expected_speed": "40-70 tok/s"
        }
    elif resources.ram_gb >= 16:
        return {
            "model_size": "7b",
            "quantization": "q4_K_S",  # 4-bit small for CPU
            "context_window": 2048,
            "batch_size": 128,
            "num_gpu_layers": 0,  # CPU only
            "expected_ram": "6-8GB",
            "expected_speed": "5-15 tok/s"
        }
    else:
        return {
            "model_size": "3b",
            "quantization": "q4_K_S",
            "context_window": 2048,
            "batch_size": 64,
            "num_gpu_layers": 0,
            "expected_ram": "3-4GB",
            "expected_speed": "8-20 tok/s"
        }

# Practical usage
resources = detect_system_resources()
config = recommend_model_configuration(resources)

print(f"System: {resources.cpu_cores} CPU cores, {resources.ram_gb:.1f}GB RAM")
if resources.gpu_available:
    print(f"GPU: {resources.gpu_name} with {resources.gpu_memory_gb:.1f}GB VRAM")
print(f"\nRecommended configuration:")
print(f"  Model: {config['model_size']} parameter with {config['quantization']} quantization")
print(f"  Expected performance: {config['expected_speed']}")
```

**Practical Implications:**

Inference runtimes like Ollama and llama.cpp automatically detect hardware and configure execution accordingly, but understanding the mapping between resources and performance is critical for capacity planning. A 7B model at Q4_K_M quantization requires approximately 4.5GB of weights plus 1-2GB for KV cache and activation memory.

**Real Constraints:**

CPU inference is 5-10x slower than GPU inference for the same model. However, it's often sufficient for batch processing, development work, or low-throughput production use. The decision isn't binary—hybrid deployments can use CPU for development and GPU for production without code changes.

### 2. Model Formats and Quantization

Quantization reduces model precision from 16-bit floats to 8-bit, 4-bit, or even lower representations. This dramatically reduces memory footprint and increases throughput, with controlled quality degradation.

**Technical Implementation:**

```python
from enum import Enum
from typing import Dict, List
import re

class QuantizationType(Enum):
    """Common quantization schemes"""
    Q2_K = "q2_K"      # 2.5-3.0 bits per weight
    Q3_K_S = "q3_K_S"  # 3.4 bits per weight, small
    Q4_K_S = "q4_K_S"  # 4.25 bits per weight, small
    Q4_K_M = "q4_K_M"  # 4.83 bits per weight, medium (recommended)
    Q5_K_S = "q5_K_S"  # 5.25 bits per weight, small
    Q5_K_M = "q5_K_M"  # 5.66 bits per weight, medium
    Q6_K = "q6_K"      # 6.5 bits per weight
    Q8_0 = "q8_0"      # 8.5 bits per weight
    F16 = "f16"        # 16-bit float (full precision)

@dataclass
class ModelVariant:
    """Model configuration with size estimates"""
    base_model: str
    parameters: str
    quantization: QuantizationType
    size_gb: float
    expected_quality: str
    use_case: str

def calculate_model_size(param_count_b: float, quant: QuantizationType) -> float:
    """Estimate model size in GB"""
    bits_per_weight = {
        QuantizationType.Q2_K: 2.75,
        QuantizationType.Q3_K_S: 3.4,
        QuantizationType.Q4_K_S: 4.25,
        QuantizationType.Q4_K_M: 4.83,
        QuantizationType.Q5_K_S: 5.25,
        QuantizationType.Q5_K_M: 5.66,
        QuantizationType.Q6_K: 6.5,
        QuantizationType.Q8_0: 8.5,
        QuantizationType.F16: 16.0,
    }
    
    # Parameter count in billions * bits per weight / 8 bits per byte / 1024^3 for GB
    return (param_count_b * 1e9 * bits_per_weight[quant]) / (8 * 1024**3)

def generate_model_matrix() -> List[ModelVariant]:
    """Generate decision matrix for model selection"""
    variants = []
    
    configs = [
        (7, "7b", [
            (QuantizationType.Q4_K_S, "Good", "CPU inference, memory constrained"),
            (QuantizationType.Q4_K_M, "Very Good", "Balanced CPU/GPU, recommended"),
            (QuantizationType.Q5_K_M, "Excellent", "GPU inference, quality priority"),
            (QuantizationType.Q8_0, "Near-perfect", "Quality testing, benchmarking"),
        ]),
        (13, "13b", [
            (QuantizationType.Q4_K_M, "Very Good", "Standard GPU deployment"),
            (QuantizationType.Q5_K_M, "Excellent", "High-end GPU, quality priority"),
        ]),
        (34, "34b", [
            (QuantizationType.Q4_K_M, "Very Good", "Multi-GPU or high VRAM"),
        ])
    ]
    
    for param_count, param_str, quant_configs in configs:
        for quant, quality, use_case in quant_configs:
            variants.append(ModelVariant(
                base_model="mistral",
                parameters=param_str,
                quantization=quant,
                size_gb=calculate_model_size(param_count, quant),
                expected_quality=quality,
                use_case=use_case
            ))
    
    return variants

# Practical decision support
def select_optimal_model(
    available_memory_gb: float,
    gpu_available: bool,
    quality_priority: bool = False
) -> List[ModelVariant]:
    """Filter models based on constraints"""
    
    variants = generate_model_matrix()
    suitable = []
    
    for variant in variants:
        # Add 2GB overhead for KV cache and activations
        total_required = variant.size_gb + 2.0
        
        if total_required <= available_memory_gb:
            if quality_priority:
                if variant.expected_quality in ["Excellent", "Near-perfect"]:
                    suitable.append(variant)
            else:
                suitable.append(variant)
    
    # Sort by size descending (prefer larger models if they fit)
    suitable.sort(key=lambda x: x.size_gb, reverse=True)
    return suitable[:3]  # Top 3 recommendations

# Example usage
recommendations = select_optimal_model(
    available_memory_gb=16.0,
    gpu_available=True,
    quality_priority=False
)

print("Top model recommendations:")
for i, model in enumerate(recommendations, 1):
    print(f"\n{i}. {model.base_model}:{model.parameters}-{model.quantization.value}")
    print(f"   Size: {model.size_gb:.1f}GB")
    print(f"   Quality: {model.expected_quality}")
    print(f"   Use case: {model.use_case}")
```

**Practical Implications:**

Q4_K_M quantization is the sweet spot for most deployments—it provides 97-99% of full-precision quality at 25-30% of the memory footprint. The K-quant variants use importance-aware quantization, preserving precision in critical weight matrices while aggressively quantizing less important weights.

**Real Constraints:**

Quality degradation becomes noticeable below Q