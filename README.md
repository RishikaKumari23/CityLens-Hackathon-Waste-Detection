## KPI Group 1: Cleanliness and Waste Management

Submission by Group 51 for the CityLens AI and Computer Vision Hackathon 2026,
organized by ARF and CDIS Tech Teams.

---

## Problem Statement

Detect civic cleanliness and waste management issues from CCTV images and video
frames with a minimum accuracy of 85% across all assigned categories.

---

## Models

| # | Category | Weights File | mAP50 | Priority Weight |
|---|----------|-------------|-------|-----------------|
| 1 | Construction and Demolition Waste | best-4_model1.pt | 0.961 | 50% |
| 2 | Garbage on Road | best-4model2.pt | 0.943 | 30% |
| 3 | Overflowing Dustbins | best-6.pt | 0.877 | 20% |

Weighted average mAP50: 0.934

---

## Architecture

All three models use YOLOv8n (nano variant) from Ultralytics, fine-tuned on
domain-specific datasets. YOLOv8n was chosen for its balance of speed and
accuracy, making it deployable in real-time CCTV inference scenarios.

- Framework: PyTorch 2.11.0
- Ultralytics version: 8.4.75
- GPU: Tesla T4 (Google Colab)
- Input resolution: 640x640 (Models 1 and 3), 256x256 (Model 2)

---

## Datasets

### Model 1 - Construction and Demolition Waste
- Dataset: construction-solid-waste
- Source: Roboflow Universe
- Size: 183 images
- License: CC BY 4.0
- URL: https://universe.roboflow.com/images-r8w7j/construction-solid-waste

### Model 2 - Garbage on Road
- Dataset: Garbage Classification v2
- Source: Kaggle (sumn2u)
- Size: 13,000 images across 7 classes
- Classes: cardboard, glass, trash, paper, plastic, metal, biological
- URL: https://kaggle.com/datasets/sumn2u/garbage-classification-v2
- Note: Dataset was in image classification format. Converted to YOLO detection
  format with full-image bounding boxes assigned per class label.

### Model 3 - Overflowing Dustbins
- Dataset: Waste Bin Fill Level Detect2
- Source: Roboflow Universe (finalyearproject-vvqft)
- Size: 460 images
- Classes: empty, full, half-full, overflowing
- License: CC BY 4.0
- URL: https://universe.roboflow.com/finalyearproject-vvqft/waste-bin-fill-level-detect2-lxsf4

---

## Training Details

### Model 1 - C&D Waste
- Base weights: yolov8n.pt (COCO pretrained)
- Epochs: 30
- Batch size: 16
- Image size: 640
- Final mAP50: 0.961

### Model 2 - Garbage on Road
- Base weights: yolov8n.pt (COCO pretrained)
- Epochs: 25
- Batch size: 32
- Image size: 256
- Final mAP50: 0.943

### Model 3 - Overflowing Dustbins
- Base weights: yolov8n.pt, trained in two rounds
- Round 1: 20 epochs, batch=16, lr=0.01, mAP50=0.808
- Round 2: 30 epochs, batch=8, lr=0.001, augment=True (fine-tuned from Round 1)
- Image size: 640
- Final mAP50: 0.877

---

## Output Format

Each label file contains one detection per line:
class_id x_center y_center width height

All values are normalized between 0 and 1 relative to image dimensions.

To recover pixel coordinates:
- x_pixel = x_center * image_width
- y_pixel = y_center * image_height
- w_pixel = width * image_width
- h_pixel = height * image_height

---

## Per-Class Results

### Model 1 - C&D Waste
| Class | mAP50 |
|-------|-------|
| Construction and demolition waste | 0.961 |

### Model 2 - Garbage on Road
| Class | mAP50 |
|-------|-------|
| cardboard | 0.979 |
| glass | 0.972 |
| trash | 0.830 |
| paper | 0.948 |
| plastic | 0.942 |
| metal | 0.939 |
| biological | 0.993 |

### Model 3 - Overflowing Dustbins
| Class | mAP50 |
|-------|-------|
| empty | 0.995 |
| full | 0.771 |
| half-full | 0.956 |
| overflowing | 0.786 |

---

## External References

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Roboflow Universe: https://universe.roboflow.com
- Kaggle Datasets: https://kaggle.com
- PyTorch: https://pytorch.org

---
