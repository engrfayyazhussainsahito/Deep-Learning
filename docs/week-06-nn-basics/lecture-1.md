# Week 6: Regularization, Dropout, and Advanced Optimization

## Theory: Generalization and Convergence

A model that performs perfectly on training data but fails on unseen data is **overfitting**. The goal of regularization is to constrain the model's complexity to improve generalization.

### 1. Regularization Techniques

#### L1 and L2 Weight Decay
We add a penalty term to the loss function based on the norm of the weights:
$L_{reg} = L_{original} + \lambda \Omega(W)$
*   **L2 Regularization (Ridge)**: $\Omega(W) = \frac{1}{2}\|W\|^2$. Encourages small weights, leading to smoother functions.
*   **L1 Regularization (Lasso)**: $\Omega(W) = \|W\|_1$. Encourages **sparsity** (many weights become exactly zero), acting as a built-in feature selector.

#### Dropout
A stochastic regularization technique. During training, neurons are randomly "dropped" (set to zero) with probability $p$. This prevents **co-adaptation**, forcing the network to learn redundant representations.

#### Batch Normalization (BatchNorm)
BatchNorm normalizes the activations of a layer to have zero mean and unit variance:
$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$
It allows for higher learning rates, accelerates convergence, and acts as a light regularizer.

### 2. Advanced Optimizers

While SGD is fundamental, modern DL relies on adaptive learning rates.

*   **Momentum**: Adds a fraction of the previous update to the current one to dampen oscillations.
*   **RMSProp**: Scales the learning rate by a running average of the magnitude of recent gradients.
*   **Adam (Adaptive Moment Estimation)**: Combines Momentum and RMSProp. It maintains separate learning rates for every single parameter in the network.

:::tip Industry Standard
Adam is the default choice for most architectures. However, for some state-of-the-art vision models, SGD with Momentum is still preferred for better final generalization.
:::

## Modular PyTorch Implementation: Regularization Suite

We implement a model that combines multiple regularization techniques and compare it against a "vanilla" MLP.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RegularizedMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128), # Batch Norm
            nn.ReLU(),
            nn.Dropout(0.3),      # Dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# Setup
X = torch.randn(1000, 20)
Y = torch.randn(1000, 1)

model = RegularizedMLP(20, 1)
# weight_decay implements L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 
criterion = nn.MSELoss()

# Training loop simplified for brevity
for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(X), Y)
    loss.backward()
    optimizer.step()

print("Regularized model trained successfully.")
```

## Rigorous Assignment 6: Generalization Analysis

1.  **Sparsity Analysis**: Train an MLP using L1 regularization. Plot a histogram of the weight values. Compare this to a model trained with L2.
2.  **Dropout Inference**: Explain why we must multiply the weights by $(1-p)$ or use `model.eval()` during inference when using Dropout.
3.  **BatchNorm Ablation**: Train a deep MLP (10+ layers) with and without Batch Normalization. Compare the number of epochs needed to reach 90% accuracy on MNIST.
