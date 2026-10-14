# Week 3: Gradient Descent & The Optimization Landscape

## Theory: How Networks Navigate Loss

If the Perceptron is the "atom," Gradient Descent (GD) is the "engine" that allows a network to learn from data.

### 1. The Loss Surface
The Loss Function $L(\theta)$ creates a high-dimensional surface (the **Loss Landscape**). The goal is to find the global minimum.

### 2. First-Order Optimization
We use the first derivative (gradient) to determine the direction of steepest descent:
$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$
where $\eta$ is the **learning rate**.

### 3. Variance in Gradient Estimation
*   **Batch GD**: $\nabla L = \frac{1}{N} \sum \nabla L_i$. Stable but computationally expensive for large $N$.
*   **Stochastic GD (SGD)**: $\nabla L = \nabla L_i$. Fast and adds noise, which can help escape local minima.
*   **Mini-batch GD**: The optimal trade-off.

:::info The Saddle Point Problem
In high-dimensional spaces, true local minima are rare. Most critical points are **saddle points** (where the gradient is zero, but some directions go up and others go down). This is why simple GD can get stuck, necessitating momentum-based optimizers.
:::

## Modular PyTorch Implementation: Custom Training Loop for Linear Regression

We will avoid `nn.Linear` and `torch.optim` to implement the optimization loop from scratch.

```python
import torch
import matplotlib.pyplot as plt

# Generate synthetic data: y = 4x + 2 + noise
X = torch.randn(100, 1)
Y = 4 * X + 2 + torch.randn(100, 1) * 0.5

# Initialize parameters
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.01
epochs = 200
loss_history = []

for epoch in range(epochs):
    # Forward pass: Compute prediction and MSE loss
    pred = X * w + b
    loss = torch.mean((pred - Y)**2)
    
    # Backward pass: Compute gradients
    loss.backward()
    
    # Parameter update using SGD
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # CRITICAL: Zero the gradients for the next iteration
        w.grad.zero_()
        b.grad.zero_()
    
    loss_history.append(loss.item())

print(f"Optimized w: {w.item():.4f}, b: {b.item():.4f}")

# Visualization
plt.plot(loss_history)
plt.title("Loss Convergence (SGD)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
```

## Rigorous Assignment 3: Convergence & Stability

1.  **Learning Rate Divergence**: Train the above model with $\eta = 0.1$ and $\eta = 1.0$. Explain why the loss diverges.
2.  **L2 Regularization (Weight Decay)**: Modify the loss function to $L_{reg} = L + \lambda \|w\|^2$. Prove that this is equivalent to subtracting a fraction of the weight from itself during every update (Weight Decay).
3.  **Momentum Implementation**: Implement a "Momentum" optimizer from scratch:
    $v_{t+1} = \gamma v_t + \eta \nabla L$
    $\theta_{t+1} = \theta_t - v_{t+1}$
    Compare its convergence speed to vanilla SGD on the same dataset.
