# Week 4: Multilayer Perceptrons (MLP) & The Backpropagation Algorithm

## Theory: Breaking the Linear Barrier

The transition from a single perceptron to a **Multilayer Perceptron (MLP)** is what enables "Deep" learning. By stacking layers, we can approximate any continuous function, effectively warping the input space to separate complex data.

### 1. Architecture of an MLP
An MLP consists of an input layer, one or more **hidden layers**, and an output layer. Each layer is a collection of neurons that perform an affine transformation followed by a non-linear activation.
$\mathbf{a}^{(l)} = \sigma(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)})$

### 2. The Backpropagation Algorithm (The Mathematical Core)
Backpropagation is the efficient application of the **Chain Rule** to compute the gradient of the loss function with respect to every weight in the network.

#### The Four Fundamental Equations
Given a loss $L$ and activation $\sigma$:
1.  **Output Error**: $\delta^{(L)} = \nabla_a L \odot \sigma'(\mathbf{z}^{(L)})$
2.  **Error Backpropagation**: $\delta^{(l)} = ((\mathbf{W}^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(\mathbf{z}^{(l)})$
3.  **Weight Gradient**: $\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^T$
4.  **Bias Gradient**: $\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}$

### 3. The Universal Approximation Theorem
A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, provided the activation function is non-constant, bounded, and monotonically-increasing.

:::tip Deep Insight
While a single hidden layer *can* approximate any function, **depth** (adding more layers) is exponentially more efficient. Depth allows the network to learn a hierarchy of features: edges $\rightarrow$ shapes $\rightarrow$ objects.
:::

## Modular PyTorch Implementation: The MLP from Scratch

We implement a flexible MLP that allows us to specify the architecture and observe the gradients during training.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DeepMLP, self).__init__()
        layers = []
        curr_dim = input_dim
        
        # Build hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
            
        # Output layer
        layers.append(nn.Linear(curr_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Scenario: Solving a complex non-linear problem (e.g., Circle Classification)
# X: (1000, 2), Y: (1000, 1)
X = torch.randn(1000, 2)
Y = (torch.norm(X, dim=1, keepdim=True) < 1.0).float()

model = DeepMLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

print("Model trained. Ready for inference.")
```

## Rigorous Assignment 4: Backprop & Architecture

1.  **Manual Backprop**: For a network with one hidden layer (2 neurons) and one output neuron, manually derive the gradient $\frac{\partial L}{\partial W^{(1)}}$ using the chain rule.
2.  **Symmetry Breaking**: What happens if you initialize all weights in an MLP to zero? Explain using the concept of "symmetric neurons" and why this prevents learning.
3.  **Width vs Depth**: Train two models on the MNIST dataset: one "Wide" (1 hidden layer with 1000 neurons) and one "Deep" (5 hidden layers with 64 neurons each). Compare the training time and final accuracy.
