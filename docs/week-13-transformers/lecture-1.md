# Week 13: The Transformer Architecture

## Theory: The Paradigm Shift to Parallel Attention

The "Attention is All You Need" paper (Vaswani et al., 2017) fundamentally changed AI by removing recurrence entirely. Transformers rely on **Self-Attention**, allowing the model to process all tokens in a sequence simultaneously, enabling massive parallelism.

### 1. Scaled Dot-Product Attention
The core mechanism computes a weighted sum of values $\mathbf{V}$, where the weights are determined by the compatibility of a query $\mathbf{Q}$ with a set of keys $\mathbf{K}$.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

*   **Query ($\mathbf{Q}$)**: Represents the current token seeking information.
*   **Key ($\mathbf{K}$)**: Represents the information offered by other tokens.
*   **Value ($\mathbf{V}$)**: The actual content to be extracted.
*   **$\sqrt{d_k}$ Scaling**: Prevents the dot product from growing too large, which would push the softmax into regions with extremely small gradients.

### 2. Multi-Head Attention (MHA)
Rather than one attention pass, MHA runs $h$ parallel "heads." This allows the model to attend to different types of relationships (e.g., one head for syntactic structure, another for semantic reference).
$\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O$

### 3. Positional Encoding
Since Transformers have no recurrence or convolution, they are permutation-invariant. To preserve order, we add a positional signal $\text{PE}$ to the embeddings:
$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$

:::tip Academic Note
The Transformer is essentially a sequence of "communication" (Attention) and "computation" (Feed-Forward Networks) steps. The Attention layer allows tokens to exchange information, and the FFN processes that information locally.
:::

## Modular PyTorch Implementation: Multi-Head Attention from Scratch

We implement the core MHA logic to understand the tensor reshaping and scaling required.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear Projections & Reshape for multi-head
        # (batch, seq, d_model) -> (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # scores: (batch, heads, seq_q, seq_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v) # (batch, heads, seq, d_k)
        
        # 3. Concat heads and final projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(context)

# Test: (Batch=1, Seq=10, Dim=512)
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(1, 10, 512)
output = mha(x, x, x)
print(f"MHA Output Shape: {output.shape}") # [1, 10, 512]
```

## Rigorous Assignment 13: Transformer Mechanics

1.  **Complexity Analysis**: Compare the time and space complexity of a Transformer layer vs an LSTM layer for a sequence of length $T$ and dimension $d$.
2.  **The Scaling Factor**: Prove why dividing by $\sqrt{d_k}$ is necessary. What happens to the softmax gradient as $d_k \rightarrow \infty$ without the scaling factor?
3.  **Masking Logic**: Implement a "Causal Mask" (Triangular mask) and explain how it prevents the model from "cheating" by looking at future tokens during training.
