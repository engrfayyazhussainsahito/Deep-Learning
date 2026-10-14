# Week 7: Convolutional Neural Networks (CNNs)

## Theory: Exploiting Spatial Invariance

Traditional MLPs fail on image data because they treat pixels as independent features, ignoring the **spatial hierarchy** (edges $\rightarrow$ shapes $\rightarrow$ objects). CNNs solve this through **Local Receptive Fields** and **Weight Sharing**.

### 1. The Convolution Operation
Unlike a fully connected layer, a convolutional layer uses a small kernel (filter) $\mathbf{K}$ that slides across the input $\mathbf{X}$. The output (feature map) is the result of the cross-correlation operation:
$(X * K)_{i,j} = \sum_{m} \sum_{n} X_{i+m, j+n} K_{m,n}$

#### Key Architectural Parameters
*   **Stride ($s$):** The step size of the kernel. Higher strides reduce the output resolution.
*   **Padding ($p$):** Zero-padding around the input to control the output spatial dimensions. For a kernel $k$, padding $p = (k-1)/2$ preserves the input size.
*   **Channels:** Input images typically have 3 channels (RGB). A filter must match the input depth, producing one output channel per filter.

### 2. Pooling and Translation Invariance
Pooling layers (Max or Average) reduce the spatial dimensions, decreasing the computational load and making the network invariant to small translations of the object in the image.

### 3. The CNN Pipeline
A standard architecture follows a repeating pattern:
$\text{Input} \rightarrow [\text{Conv} \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{MaxPool}] \times N \rightarrow \text{Flatten} \rightarrow \text{FC Layers} \rightarrow \text{Softmax}$

:::tip Academic Note
The "Receptive Field" is the area of the original input image that a particular neuron in a deep layer "sees." As we go deeper, the receptive field grows, allowing the network to understand global context.
:::

## Modular PyTorch Implementation: Custom Conv-Block

Industry-standard CNNs don't just use `nn.Conv2d`; they wrap them in blocks for stability and reuse.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    A professional Convolutional Block: Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 16), # Input: 3x32x32 -> 16x32x32
            nn.MaxPool2d(2),    # -> 16x16x16
            ConvBlock(16, 32), # -> 32x16x16
            nn.MaxPool2d(2),    # -> 32x8x8
            ConvBlock(32, 64), # -> 64x8x8
            nn.MaxPool2d(2),    # -> 64x4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Test with a mock image (Batch=1, RGB=3, 32x32)
model = SimpleCNN()
img = torch.randn(1, 3, 32, 32)
output = model(img)
print(f"Output logits shape: {output.shape}")
```

## Rigorous Assignment 7: Spatial Analysis

1.  **Receptive Field Calculation**: In a network with three $3 \times 3$ conv layers (stride 1, padding 1), what is the receptive field of a single neuron in the final feature map?
2.  **Filter Design**: Manually define a $3 \times 3$ kernel that acts as a **Sobel Filter** for vertical edge detection. Apply it to an image using `F.conv2d` and visualize the result.
3.  **Dilation Experiment**: Implement a **Dilated Convolution** (Atrous Convolution). Explain how dilation increases the receptive field without increasing the number of parameters.
