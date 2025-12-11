# This is the Benchmark Section
For clustering approach and Deep learning approach. Since the hough transform technique is not good enough

# Segmentation & Clustering Benchmark Implementation

This document outlines the tasks required to implement the comparative benchmark functionality for the deep learning (Mask2Former) and color-based clustering segmentation methods, focusing on segmentation quality and runtime performance.

## I. Setup and Data Pipeline

This section ensures the environment and data are consistently prepared for all tested methods.

| Status | Task | Details |
| :---: | :--- | :--- |
| $\square$ | **Virtual Environment Setup** | Create a dedicated environment (using `uv`, `conda`, or `venv`) and install all necessary libraries (`torch`, `transformers`, `scikit-learn`, `numpy`, `Pillow`, `opencv-python`). |
| $\square$ | **Data Preparation Script** | Implement a script to load, normalize, and manage a consistent subset of the chosen dataset (e.g., ScanNet or ADE20K) for benchmark testing. |
| $\square$ | **Ground Truth (GT) Handling** | Implement logic to correctly load the GT masks and map them to the specific target surface classes: **Wall** and **Floor**. |
| $\square$ | **Image Loading Loop** | Develop the iteration structure to sequentially process all images in the benchmark, passing the raw image to both segmentation strategies. |

---

## II. Method Integration and Standardization

Confirming that the output of each method is standardized (raw segmentation map) for consistent metric calculation.

### A. Deep Learning Method (Mask2Former)

| Status | Task | Details |
| :---: | :--- | :--- |
| $\square$ | **Model Initialization** | Verify that `Mask2FormerImageProcessor` and `Mask2FormerForUniversalSegmentation` are initialized once with the correct pre-trained weights. |
| $\square$ | **Segmentation Function Output** | Ensure the core segmentation function returns the raw class ID map (`prediction_np`) and the specific target IDs (e.g., `wall_ids`). |
| $\square$ | **Wall/Floor Binary Mask** | Implement a function to generate a unified **binary mask** for the aggregated target surfaces (walls and floors) for metric comparison with GT. |
| $\square$ | **GPU Optimization** | Verify use of `torch.no_grad()` and proper device transfer (`cuda` or `cpu`) for efficient runtime. |

### B. Clustering Method (e.g., Color-Based Clustering)

| Status | Task | Details |
| :---: | :--- | :--- |
| $\square$ | **Feature Extraction** | Define and extract the features (e.g., normalized RGB, L*a*b, or HSL values) used to cluster the image pixels. |
| $\square$ | **Clustering Algorithm** | Choose and implement the chosen clustering algorithm (e.g., `sklearn.cluster.KMeans` or Mean Shift). |
| $\square$ | **Post-Clustering Labeling** | Develop a method to reliably map the resulting cluster IDs to the target semantic classes (Wall/Floor). |
| $\square$ | **Standardized Output** | Ensure the clustering function outputs a segmented mask in the same shape, resolution, and data type as the deep learning output. |

---

## III. Evaluation and Benchmarking

Implementing the core logic for calculating segmentation quality and runtime metrics.

### A. Segmentation Quality Metrics

| Status | Task | Details |
| :---: | :--- | :--- |
| $\square$ | **IoU Calculation** | Implement a reliable function to calculate **Intersection over Union** (Jaccard Index) comparing prediction against GT.  |
| $\square$ | **Dice Coefficient Calculation** | Implement a function to calculate the **Dice Coefficient** (F1 Score) for both methods. |
| $\square$ | **Per-Class Metrics** | Calculate IoU/Dice specifically for the **Wall** class and the **Floor** class, in addition to the aggregated surface metric. |

### B. Runtime Performance Metrics

| Status | Task | Details |
| :---: | :--- | :--- |
| $\square$ | **Timer Implementation** | Use Python's `time` or `timeit` to accurately record the execution time for the core segmentation step (per image) for **each** method. |
| $\square$ | **Averaging Runtime** | Calculate and report the **average runtime** (e.g., in seconds per image) across the entire benchmark dataset. |
| $\square$ | **Hardware Logging** | Document the specific hardware configuration (CPU/GPU) used during benchmarking for reproducibility. |

---

## IV. Reporting and Saving

Ensuring results are aggregated, saved, and ready for analysis.

| Status | Task | Details |
| :---: | :--- | :--- |
| $\square$ | **Results Structure (DataFrame)** | Create a Pandas DataFrame to store the image-by-image results (file name, Method A IoU, Method B IoU, Runtimes, etc.). |
| $\square$ | **Aggregate Report Generation** | Compute and log the final aggregate results (Mean IoU, Mean Dice, Average Runtime) for each method. |
| $\square$ | **Output Mask Saving** | Implement saving of the final segmented masks (both Mask2Former and Clustering) as visual PNGs and/or raw NumPy files for qualitative analysis. |
| $\square$ | **Save Final Report** | Save the final aggregated results (DataFrame) as a persistent file (e.g., CSV or JSON). |