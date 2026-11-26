# Road Hazard Detection, Lane Estimation and Risk Assessment

A computer vision pipeline for detecting road hazards, estimating the vehicle's driving corridor and assessing the risk posed by detected objects. The project combines deep learning-based object detection with classical image processing and rule-based spatial reasoning.

Developed as part of the Image Processing and Pattern Recognition subject at the University of Technology Sydney (UTS), 2026.

## Overview

Object detection alone does not determine whether an object presents a risk to a vehicle. A pedestrian outside the driving corridor, for example, may present a different level of risk from an obstacle directly in the vehicle's path.

This project investigates this problem through three components:

- Road hazard detection using YOLOv8s and Faster R-CNN.
- Driving corridor estimation using classical computer vision techniques.
- Rule-based hazard risk classification using spatial features extracted from detection outputs.

## Dataset

The project uses CODA2022 (Corner Case Detection Dataset), which contains road scenes featuring conventional traffic objects and uncommon road hazards under different environmental conditions.

As CODA2022 does not provide an official training partition, its validation set was divided into training and internal validation subsets using an 80:20 split. The original test set was reserved for independent evaluation.

| Partition | Images | Bounding Boxes |
|---|---:|---:|
| Training | 3,907 | 32,660 |
| Internal Validation | 977 | 8,175 |
| Test | 4,884 | 39,345 |
| Total | 9,768 | 80,180 |

Both detectors were trained and evaluated using identical data partitions. Evaluation was restricted to the 29 shared object categories, excluding 14 categories exclusive to the test set.

## Methodology

### 1. Road Hazard Detection

Two object detection architectures were fine-tuned using pretrained COCO weights.

#### YOLOv8s

A single-stage object detector implemented using the Ultralytics framework.

- Initialised using pretrained COCO weights (`yolov8s.pt`).
- Fine-tuned on the CODA2022 training partition for 30 epochs.
- Uses a Task-Aligned Assigner to associate candidate predictions with ground-truth annotations.
- Best checkpoint selected using internal validation performance.

#### Faster R-CNN

A two-stage object detector implemented using PyTorch and Torchvision.

- ResNet-50 backbone for spatial feature extraction.
- Feature Pyramid Network (FPN) for multi-scale feature representation.
- Initialised using pretrained COCO weights.
- Classification head adapted to the CODA2022 categories.
- Fine-tuned for 5 epochs.

Both models were evaluated using COCO-style metrics, including:

- Mean Average Precision at IoU 0.50 (mAP50).
- Mean Average Precision across IoU thresholds 0.50–0.95 (mAP50–95).
- Average Recall at 100 detections (AR100).
- Per-class Average Precision.
- Fixed-threshold precision and recall.

Additional evaluation was conducted across individual object categories and daytime, nighttime, and dawn/dusk conditions.

### 2. Lane Detection and Driving Corridor Estimation

The lane detection component uses classical image processing to estimate the vehicle's driving corridor from individual road images.

The processing sequence consists of:

1. Grayscale conversion.
2. Gaussian smoothing.
3. Canny edge detection.
4. Trapezoidal region-of-interest masking.
5. Probabilistic Hough line detection.
6. Left and right lane boundary estimation.
7. Geometric validation and driving corridor construction.

Multiple Canny threshold configurations were examined to balance noise suppression against the preservation of weaker lane markings.

The estimated lane boundaries form a polygon representing the vehicle's approximate driving corridor. Geometrically invalid corridors are rejected rather than passed directly into the risk assessment component.

### 3. Rule-Based Risk Assessment

YOLOv8s detections with confidence scores of at least 0.25 are combined with the estimated driving corridor.

Three spatial features are extracted from each detected object's bounding box:

- Position relative to the driving corridor.
- Normalised vertical position within the image.
- Relative bounding-box area.

Vertical position and bounding-box area are used as approximate indicators of object proximity. These features are combined with corridor membership to classify hazards into three risk levels:

| Risk Level | Interpretation |
|---|---|
| High | Hazard presenting a potential immediate risk within the driving corridor |
| Medium | Hazard presenting a potential but less immediate risk |
| Low | Lower-priority hazard outside the immediate driving path |

The resulting classifications are visualised alongside object bounding boxes and the estimated driving corridor.

## Pipeline

```text
Input Road Image
        |
        +---------------------------+
        |                           |
        v                           v
Object Detection              Lane Detection
(YOLOv8s / Faster R-CNN)       (Canny / Hough)
        |                           |
        v                           v
Bounding Boxes                Driving Corridor
        |                           |
        +-------------+-------------+
                      |
                      v
             Spatial Feature Extraction
                      |
                      v
             Rule-Based Risk Assessment
                      |
                      v
              Annotated Output Image
```

YOLOv8s is used for the final integrated risk assessment pipeline, while Faster R-CNN serves as a comparative detection baseline.

## Results

### Object Detection

Results on the independent CODA2022 test set across the 29 shared categories:

| Metric | YOLOv8s | Faster R-CNN |
|---|---:|---:|
| mAP50 | **0.484** | 0.405 |
| Precision | **0.758** | 0.490 |
| Recall | 0.770 | **0.812** |

Precision and recall were calculated at a confidence threshold of 0.25 and an IoU threshold of 0.50.

YOLOv8s achieved higher mean average precision and precision, while Faster R-CNN achieved higher fixed-threshold recall. YOLOv8s was selected for integration with the downstream risk assessment component.

The evaluation also examined per-class detection performance and performance across different lighting conditions to identify weaknesses that may be concealed by aggregate metrics.

### Lane Detection

The lane detection component was evaluated on a sample of 500 test images.

| Metric | Result |
|---|---:|
| Images Evaluated | 500 |
| Geometrically Valid Corridors | 307 |
| Invalid Corridors | 193 |
| Valid Corridor Rate | 61.4% |

The valid corridor rate measures geometric validity rather than agreement with ground-truth lane annotations.

### Risk Classification

The risk assessment component was evaluated against 30 manually labelled hazards.

| Metric | Result |
|---|---:|
| Hazards Evaluated | 30 |
| Correct Classifications | 18 |
| Classification Accuracy | 60.0% |

The evaluation provides an initial indication of the effectiveness of spatial features for hazard risk classification. However, the limited sample size prevents broader conclusions about reliability.

## Technologies

- Python
- PyTorch
- Torchvision
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- pycocotools
- Matplotlib
- Jupyter Notebook
- Google Colab

## Running the Project

The implementation is contained within a Jupyter notebook and was originally executed using Google Colab.

### Dependencies

Install the required Python libraries:

```bash
pip install ultralytics torch torchvision opencv-python numpy pandas scikit-learn pycocotools matplotlib tqdm ipywidgets
```

### Dataset Preparation

1. Download the CODA2022 dataset.
2. Extract the validation and test partitions.
3. Preserve the original image and annotation directory structure.
4. Update the dataset and output paths within the notebook to match your environment.
5. Prepare the training and internal validation partitions.

### Execution

Run the notebook sections sequentially:

1. Dataset inspection and annotation preparation.
2. YOLO-format dataset generation.
3. YOLOv8s and Faster R-CNN training.
4. Detector inference and comparative evaluation.
5. Per-class and lighting-condition evaluation.
6. Lane detection and geometric validation.
7. Integrated hazard risk assessment.
8. Manual risk labelling and classification evaluation.

A CUDA-enabled GPU is recommended for model training and large-scale inference.

## Limitations

- Object detection performance varies across categories, particularly for uncommon hazards with limited training examples.
- The detectors were trained using different epoch budgets, limiting conclusions about their relative maximum achievable performance.
- Lane estimation relies on visible road markings and geometric assumptions that may not hold across curved roads, intersections or poorly illuminated scenes.
- Bounding-box size and vertical position provide approximate proximity estimates rather than physical distance measurements.
- The absence of ground-truth lane annotations prevents direct measurement of lane detection accuracy.
- The risk evaluation uses a limited manually labelled sample of 30 hazards.
- The pipeline processes individual images and does not incorporate temporal tracking, vehicle speed or time-to-collision estimation.

The system is an experimental computer vision prototype rather than a validated autonomous driving or safety-critical decision-making system.

## Future Work

- Replace classical lane detection with a learned lane segmentation model.
- Incorporate monocular depth estimation for improved proximity assessment.
- Extend the pipeline to video-based object tracking and temporal risk estimation.
- Evaluate risk classification against a larger independently annotated dataset.
- Investigate open-set detection for hazard categories not encountered during training.
- Benchmark detection architectures under equivalent computational and training budgets.

## Acknowledgements

Developed by Project Team 7 as part of the University of Technology Sydney's Image Processing and Pattern Recognition subject, 2026.

The project uses the CODA2022 dataset and pretrained object detection architectures provided by Ultralytics and Torchvision.
