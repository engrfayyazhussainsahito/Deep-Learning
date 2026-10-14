# Week 1: Linear Algebra & Calculus for DL

## The Mathematical Bedrock

Deep Learning is not merely an application of software libraries; it is the application of high-dimensional geometry and multi-variable calculus. To master the network, one must first master the space in which it operates.

### 1. Linear Algebra: The Geometry of Data

In deep learning, data is not stored as lists, but as **Tensors**. A tensor is a generalization of scalars (0D), vectors (1D), matrices (2D), and higher-dimensional arrays.

#### The Core Operations
*   **The Inner Product (Dot Product)**: $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i$. This represents the projection of one vector onto another—the fundamental operation of a neuron.
*   **Matrix Multiplication (GEMM)**: The transformation of a vector space. If $x \in \mathbb{R}^n$ and $W \in \mathbb{R}^{m \times n}$, then $Wx \in \mathbb{R}^m$. This is the "linear" part of a linear layer.
*   **Eigen-decomposition**: $Av = \lambda v$. Understanding the principal axes of data variance (crucial for PCA and understanding stability in RNNs).

:::tip Mathematical Rigor
The operation $Wx + b$ is an **affine transformation**. It rotates, scales, and translates the input space. The non-linearity (activation function) that follows is what allows the network to warp this space into a complex decision boundary.
:::

### 2. Multivariate Calculus: The Engine of Learning

Learning is an optimization problem: we seek to minimize a loss function $L(\theta)$ by adjusting parameters $\theta$.

#### The Gradient
The gradient $\nabla L(\theta)$ is a vector of partial derivatives:
$\nabla L(\theta) = \left[ \frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \dots, \frac{\partial L}{\partial \theta_n} \right]^T$
The gradient points in the direction of the steepest ascent. To minimize loss, we move in the **opposite** direction.

#### The Chain Rule (The Heart of Backpropagation)
For a composition of functions $z = f(g(x))$, the derivative is:
$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$
In a deep network, this allows us to propagate the error from the final output layer back to the first weight matrix.

## PyTorch Implementation: Tensors and Gradients from First Principles

Instead of relying on `nn.Linear`, let's implement the fundamental operations to understand the underlying memory and computation.

```python
import torch

def manual_linear_layer(x, W, b):
    """
    Implements y = Wx + b from first principles.
    x: Input tensor (batch_size, input_dim)
    W: Weight matrix (output_dim, input_dim)
    b: Bias vector (output_dim)
    """
    # Using torch.matmul for the matrix multiplication
    # W is (out, in), x is (batch, in) -> result is (batch, out)
    return torch.matmul(x, W.t()) + b

# Setup
batch_size, in_dim, out_dim = 32, 10, 5
x = torch.randn(batch_size, in_dim)
W = torch.randn(out_dim, in_dim, requires_grad=True)
b = torch.randn(out_dim, requires_grad=True)

# Forward pass
y = manual_linear_layer(x, W, b)

# Define a simple L2 loss: sum of squared errors
target = torch.randn(batch_size, out_dim)
loss = torch.mean((y - target)**2)

# Backward pass: compute gradients via autograd
loss.backward()

print(f"Weight gradient shape: {W.grad.shape}") # Should match W (5, 10)
print(f"Bias gradient shape: {b.grad.shape}")   # Should match b (5,)
```

## Rigorous Assignment 1: The Calculus of DL

1.  **Matrix Derivative**: Prove that for a scalar loss $L = \frac{1}{2}\|Wx - y\|^2$, the gradient with respect to $W$ is $\nabla_W L = (Wx - y)x^T$.
2.  **Hessian Analysis**: Research the Hessian matrix $\mathbf{H}$. Explain why the eigenvalues of the Hessian at a local minimum determine the curvature of the loss landscape.
3.  **Computational Graph**: Draw the computational graph for the operation $z = \text{ReLU}(Wx + b)$ and manually derive the gradient $\frac{\partial z}{\partial W}$.
