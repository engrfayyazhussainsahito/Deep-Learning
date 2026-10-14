# Week 9: Object Detection & Semantic Segmentation

## Theory: Moving Beyond Global Classification

Image classification provides a single label for an entire image. To interact with the real world, we need **Spatial Localization**: knowing *where* an object is (Detection) and exactly *which pixels* belong to it (Segmentation).

### 1. Object Detection: Bounding Boxes and Anchors
The goal of object detection is to predict a set of bounding boxes $\mathbf{B} = \{x, y, w, h\}$ and a class label $C$ for every object in the image.

#### The Two Paradigms
*   **Two-Stage Detectors (e.g., Faster R-CNN)**:
    1.  **Region Proposal Network (RPN)**: Proposes potential object locations.
    2.  **R-CNN Head**: Classifies and refines the proposed boxes.
    *   *Trade-off*: High accuracy, but slow inference.
*   **One-Stage Detectors (e.g., YOLO, SSD)**:
    Treat detection as a regression problem. The image is divided into a grid; each cell predicts boxes and classes directly.
    *   *Trade-off*: Extreme speed (real-time), slightly lower accuracy on small objects.

#### Intersection over Union (IoU)
To evaluate detection, we use **IoU**, the ratio between the area of overlap and the area of union of the predicted box $B_p$ and the ground truth $B_{gt}$:
$\text{IoU} = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$

### 2. Image Segmentation: Pixel-Level Understanding
*   **Semantic Segmentation**: Assigns a class label to every pixel. All pixels of "car" are treated as one entity.
*   **Instance Segmentation**: Distinguishes between individual objects of the same class (e.g., Car 1, Car 2).

#### The U-Net Architecture
The gold standard for medical imaging, the U-Net uses a symmetric **Encoder-Decoder** structure:
1.  **Contracting Path (Encoder)**: Captures context via convolutions and pooling.
2.  **Expanding Path (Decoder)**: Enables precise localization using transposed convolutions.
3.  **Skip Connections**: Concatenate high-resolution features from the encoder directly to the decoder to recover spatial detail lost during pooling.

:::tip Academic Note
Modern detection systems like **DETR (Detection Transformer)** have replaced the manual design of anchor boxes with an end-to-end Transformer architecture, treating detection as a set-prediction problem.
:::

## Modular PyTorch Implementation: IoU and NMS Logic

Non-Maximum Suppression (NMS) is critical for removing duplicate detections of the same object.

```python
import torch

def calculate_iou(boxA, boxB):
    """
    Calculates IoU of two boxes: [x1, y1, x2, y2]
    """
    # Determine coordinates of intersection rectangle
    xA = torch.max(boxA[0], boxB[0])
    yA = torch.max(boxA[1], boxB[1])
    xB = torch.min(boxA[2], boxB[2])
    yB = torch.min(boxA[3], boxB[3])

    # Compute area of intersection
    interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    
    # Compute area of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU = Intersection / Union
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

# Example: Overlapping boxes
box1 = torch.tensor([10, 10, 50, 50], dtype=torch.float32)
box2 = torch.tensor([20, 20, 60, 60], dtype=torch.float32)

print(f"IoU of boxes: {calculate_iou(box1, box2).item():.4f}")
```

## Rigorous Assignment 9: Detection and Segmentation

1.  **The Anchor Box Problem**: Explain why the choice of anchor box sizes and aspect ratios is critical for YOLO. How does this relate to the distribution of object sizes in the COCO dataset?
2.  **U-Net Proof**: Prove that if we use a stride of 2 in the encoder, we must use a transposed convolution (or upsampling) of factor 2 in the decoder to restore the original image dimensions.
3.  **Implementation Challenge**: Use a pre-trained **Mask R-CNN** from `torchvision` to segment a complex image. Calculate the **mAP (mean Average Precision)** for your predictions.
