# CS566 Final Project: Wall Designer Tool

**VIEW src/README for the latest run instructions**

## Research Questions

This project aims to implement a system for realistic, perspective-correct texture transfer onto indoor planar surfaces from a single image.

1.  Can modern computer vision methods accurately segment planar surfaces (walls, floors) from a single indoor image?
2.  How can we enable realistic, perspective-correct texture transfer onto these segmented planes?

## 🛠️ Implementation Stages

### I. Segmentation Stage

The goal of this stage is to isolate the major planar surfaces (walls and floors) in the input image. We will compare non-semantic and semantic approaches, ultimately focusing on semantic segmentation.

#### First Steps: Dataset & Code Setup

* **Dataset Preparation (Bala):**
    * **Primary:** ScanNet
    * **Secondary:** Matterport3D
    * **Custom:** Custom Mobile Capture
    * *Other suitable datasets to be researched by Bhinu, Bala, and Rain.*
* **Dataset Interface (Bala):** Create code/scripts to load and manage the prepared datasets.

#### Implementation Phases (Get the code running)

| Phase | Technique | Lead | Notes |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Edge Detection with Hough Transform | Bhinu | Baseline for detecting straight lines/boundaries. |
| **Phase 2** | Color-Based Clustering | Rain | Baseline for grouping visually similar pixels. |
| **Phase 3** | Semantic Segmentation (Mask2Former) | Bala | Core method for explicitly labeling walls and floors. |

#### Second Step: Benchmark & Comparison

The three segmentation methods will be rigorously evaluated against established metrics:

* **Segmentation Quality:** IoU, Dice Coefficient
* **Geometric Accuracy:** Corner Detection Quality (using depth/3D data)
* **Runtime Performance**

### II. Overlay Stage

The segmented planar surfaces will be processed to extract geometric information, correct for perspective distortion, and apply a new texture with realistic blending.

#### Third Step: Implementation

* Geometric Information Extraction (using depth/camera parameters)
* Perspective Correction (Homography calculation)
* Texture Warping
* Realistic Blending
* Post-processing

### III. Final Stage

* Website showcase the project
* Final Report and Presentation

## Repo Structure
- data folder contains documentation of how to get the data
- seg-clustering contains code of Segmentation stage phase two, where Rain is in charge of
