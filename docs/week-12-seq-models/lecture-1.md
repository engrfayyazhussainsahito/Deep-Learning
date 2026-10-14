# Week 12: Encoder-Decoder Architectures & The Attention Mechanism

## Theory: Overcoming the Fixed-Length Vector

For tasks like Machine Translation, we use an **Encoder-Decoder (Seq2Seq)** architecture. However, the "Bottle Neck" problem arises when the encoder is forced to compress a whole sentence into a single fixed-length vector.

### 1. The Encoder-Decoder Framework
*   **Encoder**: Processes input sequence $\mathbf{x}$ into a context vector $\mathbf{c}$.
*   **Decoder**: Generates output sequence $\mathbf{y}$ conditioned on $\mathbf{c}$.

### 2. The Attention Mechanism: Dynamic Context
Attention allows the decoder to "look back" at all encoder hidden states $\mathbf{h}_1, \dots, \mathbf{h}_T$ and decide which ones are most relevant for the current output word.

#### The Mathematical Process (Bahdanau Attention)
1.  **Alignment Score**: Compute a score $e_{t,i}$ between the current decoder state $\mathbf{s}_{t-1}$ and encoder state $\mathbf{h}_i$:
    $e_{t,i} = v_a^T \tanh(\mathbf{W}_a [\mathbf{s}_{t-1}; \mathbf{h}_i])$
2.  **Attention Weights**: Normalize scores using softmax:
    $\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}$
3.  **Context Vector**: A weighted sum of encoder states:
    $\mathbf{c}_t = \sum_{i=1}^T \alpha_{t,i} \mathbf{h}_i$

### 3. Teacher Forcing
During training, instead of using the decoder's own (potentially wrong) prediction from step $t-1$ as input for step $t$, we use the **actual ground truth**. This stabilizes training and speeds up convergence.

:::tip Academic Note
The attention mechanism is the bridge between RNNs and Transformers. By replacing the recurrent state with a purely attention-based lookup, we can process sequences in parallel, leading to the Transformer revolution.
:::

## Modular PyTorch Implementation: The Attention Weighting Layer

We implement the core attention scoring and weighting logic as a standalone module.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, decoder_state, encoder_states):
        # decoder_state: (batch, hidden_dim)
        # encoder_states: (batch, seq_len, hidden_dim)
        
        batch_size = encoder_states.size(0)
        seq_len = encoder_states.size(1)
        
        # Expand decoder state to match seq_len
        # (batch, seq_len, hidden_dim)
        dec_expanded = decoder_state.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine decoder and encoder states
        # (batch, seq_len, hidden_dim * 2)
        combined = torch.cat([dec_expanded, encoder_states], dim=-1)
        
        # Compute scores: (batch, seq_len, 1)
        scores = self.attn(combined).squeeze(-1)
        
        # Weights via softmax: (batch, seq_len)
        weights = F.softmax(scores, dim=-1)
        
        # Context vector: (batch, hidden_dim)
        context = torch.matmul(weights.unsqueeze(1), encoder_states).squeeze(1)
        
        return context, weights

# Test with mock data
hidden_dim = 128
layer = AttentionLayer(hidden_dim)
d_state = torch.randn(1, hidden_dim)
e_states = torch.randn(1, 10, hidden_dim)

context, weights = layer(d_state, e_states)
print(f"Context Vector Shape: {context.shape}") # [1, 128]
print(f"Attention Weights Shape: {weights.shape}") # [1, 10]
```

## Rigorous Assignment 12: Sequence-to-Sequence Analysis

1.  **Complexity Analysis**: Compare the time complexity of a standard RNN decoder vs an Attention-based decoder per time step.
2.  **The Beam Search Algorithm**: Implement a simple Beam Search (width=3) to improve the quality of generated sequences compared to Greedy Search.
3.  **Cross-Attention vs Self-Attention**: Explain the difference between the attention used in the la-seq model (Decoder attending to Encoder) and the self-attention used in Transformers (Encoder attending to itself).
