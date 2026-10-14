# Week 2: The Perceptron & Linear Separability

## Theory: The Atom of Deep Learning

The Perceptron, introduced by Frank Rosenblatt in 1958, is the simplest form of a neural network. It is a linear binary classifier.

### 1. The Mathematical Model
A perceptron takes an input vector $\mathbf{x} \in \mathbb{R}^n$, computes a weighted sum, and applies a Heaviside step function:
$y = \text{step}(\sum_{i=1}^n w_i x_i + b)$
Where $\text{step}(z) = 1$ if $z > 0$ else $0$.

### 2. The Decision Boundary
The equation $\sum w_i x_i + b = 0$ defines a **hyperplane** in $n$-dimensional space. All points on one side of the plane are classified as 1, and all on the other as 0.

### 3. The Perceptron Convergence Theorem
If the data is **linearly separable**, the perceptron learning algorithm is guaranteed to converge to a solution in a finite number of steps.

:::danger The XOR Limitation
The most famous failure of the single-layer perceptron is the XOR gate. Because XOR outputs are not linearly separable in 2D space, no single hyperplane can separate the classes. This realization by Minsky and Papert in 1969 led to the first "AI Winter."
:::

## Modular PyTorch Implementation: The Perceptron Classifier

We will implement the Perceptron as a modular class to follow industry standards.

```python
import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        # We initialize weights randomly using a small variance
        self.weights = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Linear combination: z = Wx + b
        z = torch.matmul(x, self.weights) + self.bias
        # Heaviside step function
        return (z > 0).float()

# Training Example: AND Gate
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
Y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

model = Perceptron(input_dim=2)
lr = 0.1

for epoch in range(100):
    for i in range(len(X)):
        pred = model(X[i])
        error = Y[i] - pred
        # Perceptron Update Rule: w = w + lr * error * x
        with torch.no_grad():
            model.weights += lr * error * X[i].unsqueeze(1)
            model.bias += lr * error

print(f"AND Gate Predictions: \n{model(X)}")
```

## Rigorous Assignment 2: Boundaries and Separation

1.  **Analytical Separation**: For the OR gate, provide a specific set of weights $w_1, w_2$ and bias $b$ that correctly classify all inputs. Prove it by testing all 4 input combinations.
2.  **The XOR Transformation**: If we map the 2D input $(x_1, x_2)$ to a 3D space using the feature $\phi(x_1, x_2) = [x_1, x_2, x_1 x_2]$, show that XOR becomes linearly separable in 3D.
3.  **Weight Convergence**: Write a script to track the norm of the weight vector $\|w\|$ over epochs. Does it grow indefinitely if the data is perfectly separable?
