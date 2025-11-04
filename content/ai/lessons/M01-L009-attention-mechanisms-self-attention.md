# Attention Mechanisms & Self-Attention

## Core Concepts

### Technical Definition

Attention mechanisms are learnable, dynamic weighting systems that allow neural networks to focus on relevant parts of input sequences when processing each element. Self-attention, specifically, computes these weights by comparing each position in a sequence against all other positions within the same sequence, creating a context-aware representation where each token "sees" and weighs the importance of every other token.

Unlike fixed-window context or recurrent processing, attention creates direct connections between all sequence positions simultaneously, with connection strengths learned through gradient descent based on semantic relationships rather than positional proximity.

### Engineering Analogy: Database Query vs. Attention

```python
from typing import List, Dict, Tuple
import numpy as np

# Traditional Approach: Fixed-window context (like sliding window queries)
def fixed_window_context(tokens: List[str], position: int, window: int = 2) -> List[str]:
    """Each token only sees nearby tokens (like SQL with LIMIT/OFFSET)"""
    start = max(0, position - window)
    end = min(len(tokens), position + window + 1)
    context = tokens[start:end]
    # Equal weight to all tokens in window, zero weight to others
    return context

# Modern Approach: Attention (like dynamic JOINs with learned weights)
def attention_context(
    tokens: List[str], 
    position: int, 
    embeddings: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """
    Each token queries ALL tokens, getting weighted contributions
    Like: SELECT * FROM sequence WHERE semantic_similarity(query, key) > threshold
    But weights are learned and differentiable, not rule-based
    """
    query = embeddings[position]  # What we're looking for
    keys = embeddings              # What's available to match against
    
    # Compute relevance scores (dot product measures alignment)
    scores = np.dot(keys, query)   # Shape: (seq_len,)
    
    # Convert to probabilities (softmax = differentiable argmax)
    weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # Return context with learned importance weights
    return tokens, weights

# Example: Processing "The cat sat on the mat"
tokens = ["The", "cat", "sat", "on", "the", "mat"]
embeddings = np.random.randn(6, 128)  # 128-dim embeddings

# Fixed window: "sat" only sees ["cat", "sat", "on"]
print("Fixed window context for 'sat':")
print(fixed_window_context(tokens, 2, window=1))

# Attention: "sat" sees ALL tokens with learned weights
print("\nAttention context for 'sat':")
context, weights = attention_context(tokens, 2, embeddings)
for token, weight in zip(context, weights):
    print(f"  {token}: {weight:.3f}")
# Typically learns: "cat" (subject) and "mat" (object) get high weights
```

The key difference: fixed approaches use hard-coded rules (distance, syntax trees, regex patterns). Attention learns which connections matter through backpropagation, discovering relationships that might not follow linguistic rules.

### Key Insights That Change Engineering Thinking

**1. Sequence processing isn't inherently sequential:** Before attention, we thought processing "the cat sat" required three sequential steps (RNNs) or fixed rules (CNNs). Attention shows that relationships can be computed in parallel by comparing all pairs simultaneously. This is why transformers train 10-100x faster than RNNs.

**2. Context is learned, not engineered:** You don't specify "subject relates to verb" or "pronouns refer to nearest noun." The network learns these patterns from data by adjusting query/key/value matrices. This means attention can discover non-obvious relationships (like "it" referring to "cat" five sentences back).

**3. Computational cost grows quadratically:** Comparing N tokens pairwise requires O(N²) operations. This isn't an implementation detail—it's fundamental to all-to-all comparison. Understanding this trade-off drives architecture decisions (sparse attention, chunking, compression).

### Why This Matters NOW

Attention is the computational primitive underlying every major LLM breakthrough since 2017. GPT, BERT, Claude, Llama—all are variations of the same attention-based transformer architecture. Understanding attention mechanisms means:

- **Debugging model behavior:** Why does the model forget context at token 4000? Attention patterns degrade.
- **Optimizing costs:** Sequence length directly impacts compute cost via O(N²). Doubling length quadruples attention cost.
- **Architectural decisions:** Should you use retrieval-augmented generation (RAG) or long-context models? Depends on attention's scaling properties.
- **Fine-tuning strategies:** Where should you add LoRA adapters? Typically in attention layers, where context integration happens.

## Technical Components

### 1. Query, Key, Value Projections

**Technical Explanation:**

Self-attention transforms each input token into three vectors: Query (Q), Key (K), and Value (V). These are created by multiplying input embeddings by learned weight matrices W_Q, W_K, W_V:

```python
import torch
import torch.nn as nn
from typing import Tuple

class AttentionProjections(nn.Module):
    def __init__(self, d_model: int, d_k: int):
        """
        Args:
            d_model: Input embedding dimension (e.g., 768)
            d_k: Dimension of query/key/value vectors (e.g., 64)
        """
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
        Returns:
            Q, K, V: [batch_size, seq_len, d_k] each
        """
        Q = self.W_q(x)  # "What am I looking for?"
        K = self.W_k(x)  # "What do I contain?"
        V = self.W_v(x)  # "What information do I provide?"
        return Q, K, V

# Example: Process a sentence
d_model, d_k, seq_len = 768, 64, 10
x = torch.randn(1, seq_len, d_model)  # Batch=1, 10 tokens, 768-dim embeddings

projections = AttentionProjections(d_model, d_k)
Q, K, V = projections(x)

print(f"Input shape: {x.shape}")
print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
print(f"\nWeight matrices:")
print(f"W_q: {projections.W_q.weight.shape}")  # [64, 768]
print(f"Trainable parameters: {d_model * d_k * 3:,}")  # 147,456 for this head
```

**Practical Implications:**

- **Why three projections?** Separating "what to look for" (Q) from "what's available" (K) from "what to retrieve" (V) gives the model flexibility. A token might match many keys but only retrieve specific values.
- **Dimension reduction:** Typically d_k < d_model (e.g., 64 vs. 768). This reduces computation cost and acts as regularization.
- **No sharing:** Each attention head has separate W_Q, W_K, W_V matrices, letting different heads learn different relationship types.

**Real Constraints:**

- **Memory:** Storing Q, K, V triples requires 3x the input memory before computing attention scores.
- **Initialization matters:** Poor initialization can cause attention to collapse (all weights uniform) or explode (gradients blow up). Standard practice: Xavier/Kaiming initialization.

**Concrete Example:**

```python
# Visualize what projections do to word relationships
words = ["cat", "sat", "mat"]
# After projection, semantically similar words should have similar keys
# And tokens looking for similar info should have similar queries

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a @ b) / (torch.norm(a) * torch.norm(b))

# Simulate: "sat" queries for subjects and objects
Q_sat = Q[0, 1, :]  # Query from "sat"
K_cat = K[0, 0, :]  # Key from "cat"
K_mat = K[0, 2, :]  # Key from "mat"

print(f"Attention score (sat→cat): {cosine_similarity(Q_sat, K_cat):.3f}")
print(f"Attention score (sat→mat): {cosine_similarity(Q_sat, K_mat):.3f}")
# After training, Q_sat should align with K_cat and K_mat (subject/object of "sat")
```

### 2. Scaled Dot-Product Attention

**Technical Explanation:**

The attention mechanism computes relevance scores between queries and keys, normalizes them to probabilities, then uses these to weight values:

```python
def scaled_dot_product_attention(
    Q: torch.Tensor,  # [batch, seq_len, d_k]
    K: torch.Tensor,  # [batch, seq_len, d_k]
    V: torch.Tensor,  # [batch, seq_len, d_k]
    mask: torch.Tensor = None,  # [batch, seq_len, seq_len]
    dropout: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard attention formula:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    d_k = Q.size(-1)
    
    # Step 1: Compute attention scores (how much each query matches each key)
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq_len, seq_len]
    
    # Step 2: Scale by sqrt(d_k) to prevent vanishing gradients
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Step 3: Apply mask (e.g., prevent attending to future tokens)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Convert to probabilities (each query's weights sum to 1)
    attention_weights = torch.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]
    
    if dropout > 0:
        attention_weights = torch.nn.functional.dropout(attention_weights, p=dropout)
    
    # Step 5: Weighted sum of values
    output = torch.matmul(attention_weights, V)  # [batch, seq_len, d_k]
    
    return output, attention_weights

# Example usage
output, attn_weights = scaled_dot_product_attention(Q, K, V)

print(f"Output shape: {output.shape}")  # [1, 10, 64]
print(f"Attention weights shape: {attn_weights.shape}")  # [1, 10, 10]
print(f"\nAttention weight matrix (first 5x5):")
print(attn_weights[0, :5, :5])
print(f"\nEach row sums to 1: {attn_weights[0].sum(dim=1)[:5]}")
```

**Practical Implications:**

- **Scaling factor (1/√d_k):** Without this, dot products grow large as d_k increases, pushing softmax into regions with tiny gradients. Scaling keeps scores in a reasonable range.
- **Softmax creates competition:** Increasing attention to one token necessarily decreases attention to others (zero-sum over each row).
- **Output is context-aware:** Each output position is a weighted blend of all value vectors, with weights determined by query-key similarity.

**Real Constraints:**

- **Memory bottleneck:** Storing the [seq_len × seq_len] attention matrix is the primary memory cost. For seq_len=2048, this is 4M floats ≈ 16MB per layer per sample.
- **Numerical stability:** Softmax can overflow/underflow. Implementations subtract max before exponentiating.

**Concrete Example:**

```python
# Demonstrate attention pattern for "The cat sat on the mat"
seq_len = 6
tokens = ["The", "cat", "sat", "on", "the", "mat"]

# Create simple embeddings where related words are similar
embeddings = torch.tensor([
    [1.0, 0.0],  # The (determiner)
    [0.0, 1.0],  # cat (noun)
    [0.5, 0.5],  # sat (verb, relates to both subject and object)
    [0.3, 0.3],  # on (preposition)
    [1.0, 0.0],  # the (determiner)
    [0.0, 1.0],  # mat (noun)
]).unsqueeze(0)  # Add batch dimension

# Simple projection (identity for illustration)
Q = K = V = embeddings

output, attn = scaled_dot_product_attention(Q, K, V)

print("Attention from 'sat' (row 2) to all tokens:")
for i, token in enumerate(tokens):
    print(f"  {token}: {attn[0, 2, i]:.3f}")
# Should show high attention to "cat" and "mat" (subject and object)
```

### 3. Multi-Head Attention

**Technical Explanation:**

Instead of computing attention once, multi-head attention runs multiple attention operations in parallel with different learned projections, then concatenates results:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (e.g., 768)
            num_heads: Number of parallel attention heads (e.g., 12)
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Single projection for all heads (more efficient than separate matrices)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,  # [batch, seq_len, d_model]
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Step 1: Project and split into multiple heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, d_k]
        
        # Step 2: Apply attention for each head in parallel
        output, attn_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout.p)
        
        # Step 3: Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch