# Week 8: Advanced CNN Architectures & Residual Learning

## Theory: Overcoming the Degradation Problem

As we increase the depth of CNNs to capture more complex features, we encounter the **Degradation Problem**: accuracy saturates and then degrades rapidly. This is NOT caused by overfitting, but by the difficulty of optimizing very deep networks.

### 1. VGG-Net: The Power of Simplicity
VGG established a key principle: instead of using large filters (like $7 \times 7$), use stacks of small $3 \times 3$ filters. Two $3 \times 3$ filters have the same receptive field as one $5 \times 5$ filter but with fewer parameters and more non-linearity.

### 2. ResNet: The Residual Revolution
He et al. introduced **Residual Learning** to allow training of networks with hundreds of layers. The core idea is the **Skip Connection**.

Instead of learning a direct mapping $H(x)$, the network learns the residual $F(x) = H(x) - x$. The output is then $F(x) + x$.
If the optimal mapping is closer to an identity function, it is much easier for the network to push $F(x)$ to zero than to learn the identity mapping from scratch.

### 3. Inception and MobileNet
*   **Inception (GoogLeNet)**: Uses "Inception Modules" to perform $1 \times 1, 3 \times 3, 5 \times 5$ convolutions in parallel, capturing features at multiple scales.
*   **MobileNet**: Uses **Depthwise Separable Convolutions** to drastically reduce computation. It splits a standard convolution into a depthwise convolution (one filter per channel) and a pointwise convolution ($1 \times 1$).

:::tip Academic Note
The "Bottleneck Layer" (a $1 \times 1$ conv used to reduce dimensions before a $3 \times 3$ conv) is a critical optimization in ResNet-50 and beyond, reducing the parameter count while maintaining expressive power.
:::

## Modular PyTorch Implementation: The Residual Block

A professional implementation of a Residual Block must handle cases where the input and output dimensions differ.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions if stride > 1 or in != out
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity # The Residual Addition
        out = torch.relu(out)
        return out

# Test: Creating a mini-ResNet
res_block = ResidualBlock(64, 128, stride=2)
input_tensor = torch.randn(1, 64, 32, 32)
output = res_block(input_tensor)
print(f"Input: {input_tensor.shape} -> Output: {output.shape}") # Should be [1, 128, 16, 16]
```

## Rigorous Assignment 8: Architecture Analysis

1.  **Computational Complexity**: Calculate the number of parameters in a standard $3 \times 3$ conv layer with 64 input channels and 64 output channels. Compare this to a Depthwise Separable Convolution with the same dimensions.
2.  **Vanishing Gradient Proof**: Theoretically explain why the addition $F(x) + x$ in ResNet prevents the gradient from vanishing as it propagates backward through the layers.
3.  **Transfer Learning Project**: Use a pre-trained `torchvision.models.resnet50`. Freeze the backbone and implement a custom classifier head for a dataset of your choice (e.g., CIFAR-100).
