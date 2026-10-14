# Week 10: Recurrent Neural Networks (RNNs) & LSTMs

## Theory: The Challenge of Temporal Dependencies

While CNNs excel at spatial data, sequential data (text, audio, time-series) requires a model that can maintain a **state** over time. Recurrent Neural Networks (RNNs) introduce the concept of "memory" by feeding the output of a time step back into the next.

### 1. The RNN Mathematical Framework
An RNN processes a sequence $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T$ by maintaining a hidden state $\mathbf{h}_t$:
$\mathbf{h}_t = \sigma(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b})$
The final output is then $\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$.

### 2. The Vanishing and Exploding Gradient Problem
The core weakness of RNNs is revealed during **Backpropagation Through Time (BPTT)**. The gradient of the loss with respect to the weights involves a product of the weight matrix $\mathbf{W}_{hh}$ over $T$ time steps:
$\frac{\partial L}{\partial \mathbf{W}} \propto \prod_{t=1}^T \mathbf{W}_{hh}^T$

*   **Vanishing Gradient**: If the eigenvalues of $\mathbf{W}_{hh} < 1$, the gradient decays exponentially, and the network "forgets" early inputs.
*   **Exploding Gradient**: If eigenvalues $> 1$, the gradient grows exponentially, causing numerical instability (solved via **Gradient Clipping**).

### 3. LSTMs: Gated Memory Control
Long Short-Term Memory (LSTM) networks solve the vanishing gradient problem by introducing a **Cell State** $\mathbf{C}_t$ (the "constant error carousel") and three regulatory gates:
*   **Forget Gate**: $f_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$ — decides what to discard.
*   **Input Gate**: $i_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$ — decides what to store.
*   **Output Gate**: $o_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$ — decides what to reveal to the next layer.

:::tip Academic Note
LSTMs allow gradients to flow through the cell state via addition rather than multiplication, significantly mitigating the vanishing gradient problem and allowing the model to capture dependencies across hundreds of time steps.
:::

## Modular PyTorch Implementation: RNN vs LSTM Performance

We implement a custom sequence processor to compare how RNNs and LSTMs handle long-term dependencies.

```python
import torch
import torch.nn as nn

class SequenceModel(nn.Module):
    def __init__(self, model_type='RNN', input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        if model_type == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.rnn(x) if isinstance(self.rnn, nn.LSTM) else (self.rnn(x)[0], (self.rnn(x)[1], None))
        # We only care about the final hidden state for classification
        last_hidden = out[:, -1, :] 
        return self.fc(last_hidden)

# Test Setup
seq_len, batch_size, in_dim = 50, 32, 10
data = torch.randn(batch_size, seq_len, in_dim)

rnn_model = SequenceModel('RNN')
lstm_model = SequenceModel('LSTM')

print(f"RNN Output: {rnn_model(data).shape}")
print(f"LSTM Output: {lstm_model(data).shape}")
```

## Rigorous Assignment 10: Temporal Analysis

1.  **BPTT Derivation**: Manually derive the gradient of the loss $L$ with respect to $\mathbf{W}_{hh}$ for a sequence of length $T=3$.
2.  **Gating Logic**: In an LSTM, what happens if the forget gate $f_t$ is permanently set to 0? What if it is permanently set to 1?
3.  **Gradient Clipping**: Implement a custom training loop that uses `torch.nn.utils.clip_grad_norm_` to stabilize an RNN trained on a highly volatile time-series dataset.
