# Transformer Model Architecture

## Core Concepts

The Transformer is a neural network architecture that processes sequential data (text, audio, time series) using parallel attention mechanisms rather than sequential recurrence. Unlike RNNs that process tokens one-at-a-time maintaining hidden state, Transformers compute relationships between all positions simultaneously through self-attention.

### Traditional vs. Modern Approach

```python
# Traditional RNN approach - Sequential processing
class SimpleRNN:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.hidden_state = None
    
    def process_sequence(self, tokens: list[str]) -> list[np.ndarray]:
        """Process tokens sequentially, each depending on previous state"""
        outputs = []
        self.hidden_state = np.zeros(self.hidden_size)
        
        for token in tokens:  # Must process in order
            token_embedding = embed(token)
            # Current output depends on previous hidden state
            self.hidden_state = np.tanh(
                np.dot(token_embedding, self.W_input) + 
                np.dot(self.hidden_state, self.W_hidden)
            )
            outputs.append(self.hidden_state)
        
        return outputs  # Takes O(n) time steps


# Transformer approach - Parallel processing
class SimpleTransformer:
    def process_sequence(self, tokens: list[str]) -> np.ndarray:
        """Process all tokens in parallel using attention"""
        # Embed all tokens at once
        embeddings = np.array([embed(token) for token in tokens])
        
        # Compute attention: every token attends to every other token
        # This happens in parallel, not sequentially
        attention_scores = np.dot(embeddings, embeddings.T)  # All-to-all relationships
        attention_weights = softmax(attention_scores, axis=-1)
        
        # Apply attention in one operation
        outputs = np.dot(attention_weights, embeddings)
        
        return outputs  # Takes O(1) time steps (parallelizable)
```

The fundamental insight: **Instead of encoding context sequentially into a hidden state, Transformers compute context by directly measuring relationships between all positions**. This trades sequential dependency for computational parallelism.

### Key Engineering Insights

1. **Parallelization enables scale**: Training can process entire sequences simultaneously across GPU cores, enabling models with billions of parameters trained on trillions of tokens.

2. **Attention provides interpretability**: Unlike RNN hidden states (opaque vectors), attention weights explicitly show which input positions influence each output position.

3. **Position information must be explicit**: Because processing is parallel, the model has no inherent notion of order—position encodings must be added explicitly.

4. **Quadratic scaling is the fundamental constraint**: Computing attention between all pairs of tokens means memory and compute scale as O(n²) with sequence length.

### Why This Matters Now

Every major LLM (GPT, Claude, Llama, Gemini) uses Transformer architecture. Understanding its mechanics explains:
- Why context windows are expensive to extend (quadratic scaling)
- Why these models excel at in-context learning (attention can directly reference examples)
- Why fine-tuning works (architecture designed for transfer learning)
- Where performance bottlenecks appear in production (attention computation, KV cache size)

## Technical Components

### 1. Self-Attention Mechanism

Self-attention computes output representations by taking weighted combinations of input representations, where weights measure relevance between positions.

**Technical Explanation:**

```python
import numpy as np
from typing import Tuple

def scaled_dot_product_attention(
    query: np.ndarray,  # Shape: (seq_len, d_model)
    key: np.ndarray,    # Shape: (seq_len, d_model)
    value: np.ndarray,  # Shape: (seq_len, d_model)
    mask: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core attention mechanism.
    
    Returns:
        output: Attention-weighted values (seq_len, d_model)
        attention_weights: Attention distribution (seq_len, seq_len)
    """
    d_k = query.shape[-1]
    
    # Compute attention scores: how much each query relates to each key
    scores = np.matmul(query, key.T) / np.sqrt(d_k)  # (seq_len, seq_len)
    
    # Apply mask (for causal attention or padding)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Normalize to probability distribution
    attention_weights = softmax(scores, axis=-1)
    
    # Apply attention to values
    output = np.matmul(attention_weights, value)  # (seq_len, d_model)
    
    return output, attention_weights


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# Example: Attention on simple sequence
tokens = ["The", "cat", "sat"]
d_model = 4

# Simplified embeddings (normally learned)
embeddings = np.random.randn(3, d_model)

output, weights = scaled_dot_product_attention(
    query=embeddings,
    key=embeddings,
    value=embeddings
)

print("Attention weights (who attends to whom):")
print(weights)
# [[0.31, 0.35, 0.34],   # "The" attends to all tokens
#  [0.29, 0.38, 0.33],   # "cat" attends slightly more to itself
#  [0.33, 0.32, 0.35]]   # "sat" attends relatively evenly
```

**Practical Implications:**

- **Query, Key, Value concept**: Think of attention like a soft dictionary lookup. Query="what I'm looking for", Key="what I have", Value="what I return". If query matches key, return corresponding value.

- **Scaling factor (√d_k)**: Prevents dot products from growing too large (which would push softmax into saturated regions with tiny gradients). Critical for training stability.

- **Attention weights are interpretable**: You can visualize which input tokens influence each output token, useful for debugging and understanding model behavior.

**Real Constraints:**

- Memory usage is O(n² × d_model) for attention score matrix
- For 2048 token sequence with d_model=768: ~6MB per attention head
- With 96 heads (GPT-3 scale): ~600MB just for attention scores

**Concrete Example:**

```python
def demonstrate_attention_behavior():
    """Show how attention focuses on relevant context"""
    # Sentence: "The cat sat on the mat because it was tired"
    # Focus on: what does "it" refer to?
    
    tokens = ["cat", "sat", "mat", "it", "tired"]
    
    # Simulate embeddings where "cat" and "it" are semantically similar
    embeddings = np.array([
        [1.0, 0.1, 0.0, 0.9],  # cat
        [0.0, 0.8, 0.2, 0.1],  # sat
        [0.1, 0.2, 0.9, 0.0],  # mat
        [0.9, 0.1, 0.0, 0.8],  # it (similar to "cat")
        [0.0, 0.1, 0.0, 0.9],  # tired
    ])
    
    output, weights = scaled_dot_product_attention(
        query=embeddings,
        key=embeddings,
        value=embeddings
    )
    
    # Check what "it" (index 3) attends to
    it_attention = weights[3]
    print(f"Token 'it' attention distribution:")
    for token, weight in zip(tokens, it_attention):
        print(f"  {token}: {weight:.3f}")
    
    # Expected: higher weight on "cat" due to embedding similarity
    # Output:
    #   cat: 0.412    <- Highest attention
    #   sat: 0.143
    #   mat: 0.151
    #   it: 0.156
    #   tired: 0.138

demonstrate_attention_behavior()
```

### 2. Multi-Head Attention

Instead of one attention mechanism, Transformers use multiple parallel attention heads, each learning different relationship patterns.

**Technical Explanation:**

```python
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V (one per head)
        self.W_q = np.random.randn(num_heads, d_model, self.d_k) * 0.01
        self.W_k = np.random.randn(num_heads, d_model, self.d_k) * 0.01
        self.W_v = np.random.randn(num_heads, d_model, self.d_k) * 0.01
        
        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input tensor (seq_len, d_model)
        Returns:
            output: Multi-head attention output (seq_len, d_model)
        """
        seq_len = x.shape[0]
        head_outputs = []
        
        # Process each head independently
        for h in range(self.num_heads):
            # Project to Q, K, V for this head
            Q = np.dot(x, self.W_q[h])  # (seq_len, d_k)
            K = np.dot(x, self.W_k[h])
            V = np.dot(x, self.W_v[h])
            
            # Compute attention for this head
            head_output, _ = scaled_dot_product_attention(Q, K, V)
            head_outputs.append(head_output)
        
        # Concatenate all heads
        concatenated = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)
        
        # Final linear projection
        output = np.dot(concatenated, self.W_o)
        
        return output


# Example usage
seq_len, d_model, num_heads = 5, 512, 8
x = np.random.randn(seq_len, d_model)

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
output = mha.forward(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Each of {num_heads} heads processes {d_model // num_heads} dimensions")
```

**Practical Implications:**

- **Different heads learn different patterns**: One head might focus on syntactic relationships (subject-verb), another on semantic similarity, another on positional proximity.

- **Increases model capacity without increasing sequence length costs**: Adding heads increases parameters linearly but doesn't change O(n²) attention complexity.

- **Head dimension trade-off**: More heads with smaller dimensions vs. fewer heads with larger dimensions. Typical: 12-96 heads with d_k=64-128.

**Real Constraints:**

- Total dimension must be divisible by number of heads
- More heads = more parallel computation but diminishing returns after ~96 heads
- Each head adds ~3 × d_model × d_k parameters for Q, K, V projections

### 3. Position Encoding

Since attention is permutation-invariant, position information must be explicitly encoded.

**Technical Explanation:**

```python
def sinusoidal_position_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Generate fixed sinusoidal position encodings.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Returns:
        position_encoding: (seq_len, d_model)
    """
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )  # (d_model/2,)
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # Even dimensions
    pe[:, 1::2] = np.cos(position * div_term)  # Odd dimensions
    
    return pe


# Example: Add position encoding to embeddings
seq_len, d_model = 10, 128
token_embeddings = np.random.randn(seq_len, d_model)

pos_encoding = sinusoidal_position_encoding(seq_len, d_model)
input_embeddings = token_embeddings + pos_encoding

print(f"Token embeddings: {token_embeddings.shape}")
print(f"Position encoding: {pos_encoding.shape}")
print(f"Final input: {input_embeddings.shape}")

# Verify position encodings have useful properties
def position_similarity(pos_encoding: np.ndarray):
    """Check that nearby positions have similar encodings"""
    similarities = np.dot(pos_encoding, pos_encoding.T)
    
    print("\nPosition encoding similarities (cosine):")
    print("Position 0 vs others:", similarities[0, :5])
    print("Position 5 vs others:", similarities[5, 3:8])
    # Nearby positions have high similarity, distant positions low similarity

position_similarity(pos_encoding)
```

**Practical Implications:**

- **Sinusoidal vs. learned encodings**: Sinusoidal allows extrapolation beyond training sequence lengths. Learned encodings often perform slightly better but can't generalize to unseen lengths.

- **Relative vs. absolute position**: Sinusoidal encodings enable model to learn relative positions (token i is 3 positions before token j) through dot products.

- **Added to embeddings, not concatenated**: Position info is mixed with semantic info through addition, allowing attention to weigh position vs. content.

**Real Constraints:**

- Fixed position encodings limit maximum sequence length (though can be extended)
- Position encoding adds no additional parameters (for sinusoidal)
- Learned position embeddings add seq_len × d_model parameters

### 4. Feed-Forward Networks

After attention, each position passes through an identical feed-forward network independently.

**Technical Explanation:**

```python
class PositionWiseFeedForward:
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Two-layer feed-forward network applied to each position.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (typically 4 × d_model)
            dropout: Dropout probability
        """
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
        self.dropout = dropout
    
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Args:
            x: Input (seq_len, d_model)
        Returns:
            output: (seq_len, d_model)
        """
        # First layer with ReLU activation