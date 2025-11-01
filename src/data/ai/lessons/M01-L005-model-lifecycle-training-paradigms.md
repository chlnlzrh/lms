# Model Lifecycle & Training Paradigms

## Core Concepts

### Technical Definition

The model lifecycle encompasses the complete journey from data collection to production deployment and maintenance. Training paradigms define how models acquire capabilities—whether through full training from scratch, transfer learning, fine-tuning, or prompting. Understanding these paradigms determines your engineering approach, resource allocation, and what's actually feasible for your use case.

Most engineers transitioning to AI/LLM work assume they'll be training models. In reality, you'll spend 95% of your time working with pre-trained models, and understanding the training paradigm that created them fundamentally changes how you use them.

### Engineering Analogy: Traditional vs. Modern Approaches

**Traditional Software Development:**
```python
# You write the logic explicitly
def classify_email(email_text: str) -> str:
    spam_indicators = ['click here', 'limited time', 'congratulations']
    score = sum(1 for indicator in spam_indicators if indicator in email_text.lower())
    return 'spam' if score >= 2 else 'not_spam'

# You control: logic, edge cases, performance
# You deploy: your code
# You maintain: fix bugs, add rules
```

**ML Model Development:**
```python
# You provide examples, the model learns patterns
training_data = [
    ("Click here for free money!", "spam"),
    ("Meeting notes from today", "not_spam"),
    # ... thousands more examples
]

# Training creates the logic
model = train_classifier(training_data)

# You control: data, architecture, hyperparameters
# You deploy: learned weights + inference code
# You maintain: retrain with new data, monitor drift
```

**LLM Paradigm (Pre-trained Models):**
```python
# You describe the task; the model already has capabilities
prompt = """
Classify this email as spam or not_spam:
Email: Click here for free money!
Classification:
"""

response = llm.generate(prompt)

# You control: prompt, model selection, examples
# You deploy: API calls or model weights
# You maintain: prompt engineering, example curation
```

The shift: from writing logic → curating training data → crafting instructions for pre-trained models.

### Key Insights

1. **Pre-training is capital-intensive, fine-tuning is feasible, prompting is accessible**: Training a foundation model costs millions. Fine-tuning costs hundreds to thousands. Prompting costs pennies per use.

2. **The paradigm determines your control surface**: With traditional code, you control logic. With fine-tuning, you control examples. With prompting, you control instructions. Different problems require different control points.

3. **Model capabilities are frozen at training time**: A model trained in 2023 doesn't know about 2024 events. Understanding training cutoff dates and knowledge limitations prevents confusion and bugs.

4. **Training paradigms compose**: Foundation model → instruction fine-tuning → task-specific fine-tuning → few-shot prompting. Each layer adds capabilities.

### Why This Matters Now

The LLM landscape is fragmenting into specialized paradigms:
- **Edge deployment** requires quantized models (4-bit, 8-bit)
- **Real-time systems** need understanding of latency/throughput trade-offs
- **Cost optimization** demands knowing when to prompt vs. fine-tune vs. build custom
- **Compliance** increasingly requires training provenance and data lineage

Engineers who understand these trade-offs make better architectural decisions and avoid expensive mistakes.

## Technical Components

### 1. Pre-training: Foundation Model Creation

**Technical Explanation:**

Pre-training creates a foundation model by training on massive text corpora (hundreds of billions to trillions of tokens). The model learns language patterns, factual knowledge, reasoning capabilities, and implicit biases from the training data.

```python
# Conceptual pre-training loop (simplified)
# In reality, this runs on thousands of GPUs for months
from typing import List, Iterator
import random

def pretrain_step(
    model: 'LanguageModel',
    text_batch: List[str],
    learning_rate: float = 1e-4
) -> float:
    """Single pre-training step with next-token prediction."""
    total_loss = 0.0
    
    for text in text_batch:
        tokens = tokenize(text)
        
        # Predict each token given previous context
        for i in range(1, len(tokens)):
            context = tokens[:i]
            target = tokens[i]
            
            prediction = model.forward(context)
            loss = cross_entropy_loss(prediction, target)
            total_loss += loss
            
            # Update model weights
            model.backward(loss, learning_rate)
    
    return total_loss / len(text_batch)

# Training data: books, websites, code, papers
training_corpus = load_corpus()  # TB-scale dataset
for epoch in range(epochs):
    for batch in training_corpus.iter_batches(batch_size=1024):
        loss = pretrain_step(model, batch)
        if step % 1000 == 0:
            save_checkpoint(model, f"checkpoint_{step}.pt")
```

**Practical Implications:**

- **You won't do this yourself**: Pre-training requires $2M-$100M in compute
- **Training data determines capabilities**: A model trained primarily on English will be weak in other languages
- **Training cutoff is permanent**: Models don't know events after training completion
- **Emergent capabilities appear at scale**: Reasoning abilities emerge around 10B+ parameters

**Real Constraints:**

```python
# Cost calculation for pre-training
def estimate_pretraining_cost(
    num_tokens: int = 1_000_000_000_000,  # 1 trillion tokens
    model_params: int = 7_000_000_000,    # 7B parameters
    gpu_hours_per_token: float = 1e-12,   # Approximate
    gpu_cost_per_hour: float = 2.50       # A100 80GB
) -> dict:
    """Estimate pre-training costs."""
    
    # Compute FLOPs: ~6 * params * tokens for forward + backward
    total_flops = 6 * model_params * num_tokens
    
    # GPU throughput: ~312 TFLOPS for A100
    gpu_flops_per_second = 312e12
    total_seconds = total_flops / gpu_flops_per_second
    total_hours = total_seconds / 3600
    
    cost = total_hours * gpu_cost_per_hour
    
    return {
        "total_gpu_hours": int(total_hours),
        "estimated_cost_usd": int(cost),
        "training_days": int(total_hours / 24),
        "num_gpus_for_30_days": int(total_hours / (24 * 30))
    }

# Example: 7B parameter model on 1T tokens
print(estimate_pretraining_cost())
# Output: ~$2-5M, 80-200 days on hundreds of GPUs
```

**Concrete Example:**

```python
# What pre-training gives you: general language understanding
def demonstrate_pretrained_capabilities(llm):
    """Pre-trained models can do tasks they weren't explicitly trained for."""
    
    # Translation (learned from multilingual web data)
    print(llm.generate("Translate to French: Hello, world!"))
    # Output: "Bonjour, monde!"
    
    # Code completion (learned from GitHub/Stack Overflow)
    print(llm.generate("def fibonacci(n):\n    "))
    # Output: "if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    
    # Question answering (learned from Wikipedia, books)
    print(llm.generate("What is the capital of France?"))
    # Output: "Paris"
    
    # These capabilities emerge from next-token prediction on diverse data
```

### 2. Fine-tuning: Task Specialization

**Technical Explanation:**

Fine-tuning adapts a pre-trained model to specific tasks or domains by continuing training on a smaller, curated dataset. This is orders of magnitude cheaper than pre-training and allows customization without starting from scratch.

```python
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

class FineTuningDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(self, examples: List[Tuple[str, str]]):
        """
        Args:
            examples: List of (input, expected_output) pairs
        """
        self.examples = examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict:
        input_text, output_text = self.examples[idx]
        return {
            "input": input_text,
            "output": output_text
        }

def finetune_model(
    base_model: 'PreTrainedModel',
    training_data: List[Tuple[str, str]],
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-5  # Lower than pre-training
) -> 'FineTunedModel':
    """Fine-tune a pre-trained model on task-specific data."""
    
    dataset = FineTuningDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate)
    
    base_model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass: compute loss on expected outputs
            outputs = base_model(
                input_ids=batch["input"],
                labels=batch["output"]
            )
            loss = outputs.loss
            
            # Backward pass: update weights
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return base_model

# Example: Fine-tune for customer support
support_examples = [
    ("Customer: How do I reset my password?", 
     "Agent: Click 'Forgot Password' on the login page..."),
    ("Customer: My order hasn't arrived",
     "Agent: Let me check your order status..."),
    # ... hundreds to thousands more examples
]

finetuned_model = finetune_model(base_model, support_examples)
```

**Practical Implications:**

- **Data quality > quantity**: 1,000 high-quality examples beat 10,000 noisy ones
- **Catastrophic forgetting**: Fine-tuning can degrade general capabilities if data is too narrow
- **Overfitting risk**: Small datasets require careful regularization
- **Format sensitivity**: The model learns the exact format you train it on

**Real Constraints:**

```python
def estimate_finetuning_requirements(
    num_examples: int,
    model_size: str = "7B",
    num_epochs: int = 3
) -> dict:
    """Estimate fine-tuning resource needs."""
    
    # Approximate GPU memory and time requirements
    memory_gb = {
        "7B": 24,    # Fits on 1x A100 40GB with gradient checkpointing
        "13B": 48,   # Needs 1x A100 80GB or 2x A100 40GB
        "70B": 160   # Needs 2x A100 80GB or 4x A100 40GB
    }
    
    # Time per 1000 examples (approximate)
    hours_per_1k = {
        "7B": 0.5,
        "13B": 1.0,
        "70B": 3.0
    }
    
    total_hours = (num_examples / 1000) * hours_per_1k[model_size] * num_epochs
    cost_estimate = total_hours * 2.50  # A100 hourly rate
    
    return {
        "min_gpu_memory_gb": memory_gb[model_size],
        "estimated_hours": round(total_hours, 1),
        "estimated_cost_usd": round(cost_estimate, 2),
        "recommended_examples": max(500, num_examples)
    }

# Example: Fine-tune 7B model on 2000 examples
print(estimate_finetuning_requirements(2000, "7B"))
# Output: ~24GB GPU, ~3 hours, ~$7.50
```

**Concrete Example:**

```python
# Before fine-tuning: generic responses
base_response = base_model.generate(
    "Customer: My order #12345 hasn't arrived"
)
print(base_response)
# Output: "I understand you're waiting for an order. Order tracking typically..."

# After fine-tuning: company-specific responses
finetuned_response = finetuned_model.generate(
    "Customer: My order #12345 hasn't arrived"
)
print(finetuned_response)
# Output: "Let me check order #12345 in our system. Our standard shipping..."
# Uses company terminology, follows support script, checks order format
```

### 3. Parameter-Efficient Fine-Tuning (PEFT)

**Technical Explanation:**

PEFT methods like LoRA (Low-Rank Adaptation) fine-tune models by training only a small set of additional parameters rather than all model weights. This reduces memory usage by 3-10x and speeds up training while maintaining quality.

```python
from typing import Optional
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of the adaptation matrices (lower = fewer params)
            alpha: Scaling factor for adaptation
        """
        super().__init__()
        
        self.rank = rank
        self.scaling = alpha / rank
        
        # Original weight matrix (frozen during fine-tuning)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )
        
        # Low-rank adaptation matrices (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combines frozen and adapted weights."""
        # Original transformation
        result = torch.matmul(x, self.weight.T)
        
        # Add low-rank adaptation: BA(x)
        adaptation = torch.matmul(
            torch.matmul(x, self.lora_A.T),
            self.lora_B.T
        ) * self.scaling
        
        return result + adaptation
    
    def trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return self.lora_A.numel() + self.lora_B.numel()

# Compare parameter counts
def compare_finetuning_methods(
    model_params: int = 7_000_000_000,  # 7B model
    rank: int = 8
):
    """Compare full fine-tuning vs LoRA."""
    
    # Full fine-tuning: train all parameters
    full_params = model_params
    
    # LoRA: train only low-rank adaptations
    # Assuming ~40% of params are in linear layers
    linear_params = model_params * 0.4
    lora_params = linear_params * (2 * rank / 4096)  # Typical hidden dim
    
    print(f"Full fine-tuning: {full_params:,} parameters")
    print(f"LoRA fine-tuning: {int(lora_params):,} parameters")
    print(f"Reduction: {full_params / lora_params:.1f}x fewer parameters")
    print(f"Memory savings: ~{int((1 - lora