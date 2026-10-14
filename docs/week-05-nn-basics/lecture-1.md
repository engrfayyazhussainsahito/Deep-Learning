# Week 5: Activation Functions & Weight Initialization

## Theory: The Non-Linearity and the Signal

If we remove the activation functions from an MLP, the entire network collapses into a single linear transformation, regardless of depth. Activation functions introduce the non-linearity required to learn complex patterns.

### 1. The Taxonomy of Activation Functions

#### The Saturating Functions
*   **Sigmoid**: $\sigma(x) = \frac{1}{1+e^{-x}}$. Maps input to $(0, 1)$.
*   **Hyperbolic Tangent (tanh)**: Maps input to $(-1, 1)$. Centered at zero, which generally makes optimization easier than sigmoid.

#### The Non-Saturating Functions
*   **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$. The industry standard. It avoids vanishing gradients for positive values and is computationally efficient.
*   **Leaky ReLU**: $f(x) = \max(\alpha x, x)$. Prevents the "Dying ReLU" problem where neurons become permanently inactive.
*   **ELU (Exponential Linear Unit)**: Smoother than ReLU, can produce negative values, helping the mean activation stay closer to zero.

#### The Output Functions
*   **Softmax**: $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum e^{z_j}}$. Turns a vector of logits into a probability distribution.

### 2. The Weight Initialization Problem

The variance of activations should remain constant across layers to prevent signals from vanishing or exploding.

#### Xavier (Glorot) Initialization
Designed for symmetric activations (tanh). It sets weights from a distribution with variance:
$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$

#### He (Kaiming) Initialization
Designed specifically for ReLU. Since ReLU zeroes out half the input, the variance must be doubled:
$\text{Var}(W) = \frac{2}{n_{in}}$

:::tip Academic Note
Using the wrong initialization (e.g., Xavier for ReLU) often leads to the "Vanishing Gradient" problem, where the signal dies out before reaching the early layers, effectively stopping learning.
:::

## Modular PyTorch Implementation: Activation Analysis

We will implement a test suite to compare how different activation functions affect the gradient flow.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ActivationProbe(nn.Module):
    def __init__(self, act_fn):
        super().__init__()
        self.layer = nn.Linear(10, 10)
        self.act = act_fn
        
    def forward(self, x):
        return self.act(self.layer(x))

# Compare ReLU vs Sigmoid
activations = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "LeakyReLU": nn.LeakyReLU()
}

results = {}
for name, act in activations.items():
    model = ActivationProbe(act)
    x = torch.randn(1, 10, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    # Store the gradient of the weights
    results[name] = model.layer.weight.grad.abs().mean().item()

print("Average Weight Gradient Magnitude:")
for name, val in results.items():
    print(f"{name}: {val:.6f}")
```

## Rigorous Assignment 5: Stability and Signal

1.  **The Dying ReLU**: Create a network with a high learning rate and ReLU activations. Track the percentage of neurons that output 0 for all inputs. Explain why this happens.
2.  **Initialization Impact**: Train a 10-layer MLP using (a) Zero init, (b) Random Normal init, and (c) He init. Plot the loss for the first 20 iterations.
3.  **Softmax Proof**: Prove that the Softmax function is translation-invariant, i.e., $\text{Softmax}(z) = \text{Softmax}(z + c)$ for any scalar $c$.
