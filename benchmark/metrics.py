# This runs the benchmark for the 2 methods
#
# METRIC COMPARISON GUIDE:
# ========================
# All three metrics evaluate segmentation quality but focus on different aspects:
#
# 1. IoU (Intersection over Union):
#    - Formula: TP / (TP + FP + FN)
#    - Focus: Overall pixel overlap (area-based)
#    - Range: 0.0 to 1.0
#    - Sensitive to: Total coverage, false positives, false negatives
#    - Best for: General segmentation quality assessment
#
# 2. Dice Score (F1 Score - Area-based):
#    - Formula: 2 * TP / (2 * TP + FP + FN)
#    - Focus: Overall pixel classification accuracy (area-based F1)
#    - Range: 0.0 to 1.0
#    - Sensitive to: Total coverage (more forgiving than IoU for imbalanced classes)
#    - Best for: Binary segmentation with class imbalance
#    - Note: Always >= IoU (Dice penalizes errors less harshly)
#
# 3. Boundary F-Score (F1 Score - Edge-based):
#    - Formula: 2 * (Precision * Recall) / (Precision + Recall) on boundary pixels only
#    - Focus: Boundary/contour accuracy (edge-based F1)
#    - Range: 0.0 to 1.0
#    - Sensitive to: Edge alignment, boundary smoothness, contour precision
#    - Best for: Tasks requiring precise edge detection (e.g., wall boundaries)
#    - Tolerance: Configurable pixel distance (default: 2 pixels)
#
# Key Differences:
# - IoU & Dice: Evaluate ALL pixels in the mask (area-based)
# - Boundary F-Score: Evaluates ONLY boundary/edge pixels
# - Dice is typically higher than IoU for the same prediction
# - Boundary F-Score can be lower than both if edges are jagged/misaligned
#
# Example Interpretation:
# - High IoU + High Dice + Low Boundary F-Score = Good area coverage, but poor edge quality
# - High IoU + High Dice + High Boundary F-Score = Excellent overall segmentation
# - Low IoU + Low Dice + High Boundary F-Score = Poor area coverage, but edges are aligned

import numpy as np
from scipy import ndimage

def calculate_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) for two binary masks.
    
    IoU = Intersection / Union = TP / (TP + FP + FN)
    
    Args:
        prediction (np.ndarray): The predicted binary mask (0 or 1).
        ground_truth (np.ndarray): The ground truth binary mask (0 or 1).
        
    Returns:
        float: The IoU score. Returns 1.0 if both masks are empty (no union).
    """
    # Ensure inputs are flat arrays for easier calculation
    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()

    # Calculate Intersection (True Positives, TP): P AND GT
    # True Positives occur where both prediction and ground truth are 1.
    intersection = np.sum(prediction * ground_truth)
    
    # Calculate Union (TP + FP + FN): P OR GT
    # This is equivalent to np.sum(P) + np.sum(GT) - Intersection
    # Or, more simply, where either prediction or ground truth is 1.
    union = np.sum(np.logical_or(prediction, ground_truth))
    
    # Handle the edge case where the image contains none of the target class
    if union == 0:
        # If there is no union, and thus no intersection (assuming logic holds), 
        # the model correctly identified no instance. IoU is often defined as 1.0 here.
        if intersection == 0:
            return 1.0
        # Should not happen: intersection > 0 but union == 0
        return 0.0

    iou_score = intersection / union
    
    return float(iou_score)


def calculate_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates the Dice Coefficient (F1 Score) for two binary masks.
    
    Dice = 2 * Intersection / (Sum of Areas) = 2 * TP / (2 * TP + FP + FN)
    
    Args:
        prediction (np.ndarray): The predicted binary mask (0 or 1).
        ground_truth (np.ndarray): The ground truth binary mask (0 or 1).
        
    Returns:
        float: The Dice score. Returns 1.0 if both masks are empty (sum of areas is 0).
    """
    # Ensure inputs are flat arrays
    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()
    
    # Calculate Intersection (True Positives, TP)
    intersection = np.sum(prediction * ground_truth)
    
    # Calculate Sum of Areas (P + GT)
    # This is equivalent to 2 * TP + FP + FN
    sum_of_areas = np.sum(prediction) + np.sum(ground_truth)
    
    # Handle the edge case where the image contains none of the target class
    if sum_of_areas == 0:
        # If both masks are empty, the model correctly identified no instance.
        return 1.0

    dice_score = (2.0 * intersection) / sum_of_areas
    
    return float(dice_score)


def calculate_boundary_fscore(prediction: np.ndarray, ground_truth: np.ndarray, tolerance: int = 2) -> float:
    """
    Calculates Boundary F-Score for two binary masks.
    
    This metric evaluates segmentation quality by focusing on boundary/contour accuracy
    rather than overall pixel overlap. It's particularly useful for tasks where precise
    edge detection is critical (e.g., wall segmentation).
    
    Algorithm:
    1. Extract boundary pixels from both prediction and ground truth masks
    2. Dilate the ground truth boundary by tolerance pixels to create a tolerance zone
    3. Match prediction boundary pixels to the dilated ground truth boundary
    4. Calculate precision and recall based on boundary matches
    5. Compute F-score as the harmonic mean of precision and recall
    
    Args:
        prediction (np.ndarray): The predicted binary mask (0 or 1).
        ground_truth (np.ndarray): The ground truth binary mask (0 or 1).
        tolerance (int): Pixel distance tolerance for boundary matching (default: 2).
                        Higher values are more lenient (typically 2-5 pixels).
        
    Returns:
        float: The Boundary F-Score (0.0 to 1.0). Returns 1.0 if both masks are empty.
    """
    # Ensure masks are binary (0 or 1)
    prediction = (prediction > 0).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)
    
    # Handle edge case: both masks are empty
    if np.sum(prediction) == 0 and np.sum(ground_truth) == 0:
        return 1.0
    
    # Extract boundaries using morphological erosion
    # Boundary = original mask - eroded mask
    # Use a 3x3 structuring element (cross-shaped) for 4-connected boundaries
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.uint8)
    
    # Erode both masks
    pred_eroded = ndimage.binary_erosion(prediction, structure=structure).astype(np.uint8)
    gt_eroded = ndimage.binary_erosion(ground_truth, structure=structure).astype(np.uint8)
    
    # Extract boundaries
    pred_boundary = prediction - pred_eroded
    gt_boundary = ground_truth - gt_eroded
    
    # Handle edge case: no boundaries in either mask
    if np.sum(pred_boundary) == 0 and np.sum(gt_boundary) == 0:
        return 1.0
    
    # Create tolerance zone around ground truth boundary by dilating it
    # Use a disk-shaped structuring element for isotropic dilation
    if tolerance > 0:
        # Create a disk-shaped structuring element
        # For small tolerance values, we can approximate with a square or use iterations
        # Using multiple iterations of dilation with the cross structure
        gt_boundary_dilated = gt_boundary.copy()
        for _ in range(tolerance):
            gt_boundary_dilated = ndimage.binary_dilation(
                gt_boundary_dilated, structure=structure
            ).astype(np.uint8)
    else:
        gt_boundary_dilated = gt_boundary
    
    # Calculate matches: prediction boundary pixels within tolerance zone
    # True Positives: prediction boundary pixels that are in the dilated GT boundary
    tp = np.sum((pred_boundary > 0) & (gt_boundary_dilated > 0))
    
    # False Positives: prediction boundary pixels NOT in the tolerance zone
    fp = np.sum((pred_boundary > 0) & (gt_boundary_dilated == 0))
    
    # False Negatives: ground truth boundary pixels NOT matched by prediction
    # (GT boundary pixels that don't have a prediction boundary pixel within tolerance)
    # This is approximated as GT boundary pixels not covered by dilated prediction boundary
    if tolerance > 0:
        pred_boundary_dilated = pred_boundary.copy()
        for _ in range(tolerance):
            pred_boundary_dilated = ndimage.binary_dilation(
                pred_boundary_dilated, structure=structure
            ).astype(np.uint8)
    else:
        pred_boundary_dilated = pred_boundary
    
    fn = np.sum((gt_boundary > 0) & (pred_boundary_dilated == 0))
    
    # Calculate precision and recall
    # Precision: fraction of predicted boundary pixels that are correct
    if tp + fp == 0:
        precision = 1.0 if np.sum(pred_boundary) == 0 else 0.0
    else:
        precision = tp / (tp + fp)
    
    # Recall: fraction of ground truth boundary pixels that are matched
    if tp + fn == 0:
        recall = 1.0 if np.sum(gt_boundary) == 0 else 0.0
    else:
        recall = tp / (tp + fn)
    
    # Calculate F-score (harmonic mean of precision and recall)
    if precision + recall == 0:
        fscore = 0.0
    else:
        fscore = 2.0 * (precision * recall) / (precision + recall)
    
    return float(fscore)

# Optional: Example Usage
# if __name__ == '__main__':
#     # Perfect Match: IoU=1.0, Dice=1.0
#     A = np.array([[1, 1], [0, 0]])
#     B = np.array([[1, 1], [0, 0]])
#     print(f"Perfect Match IoU: {calculate_iou(A, B):.2f}, Dice: {calculate_dice(A, B):.2f}")

#     # Half Overlap: IoU=0.5, Dice=0.67
#     C = np.array([[1, 1], [0, 0]])
#     D = np.array([[1, 0], [1, 0]])
#     # TP=1, FP=1, FN=1. Union=3. Sum of Areas=4.
#     print(f"Half Overlap IoU: {calculate_iou(C, D):.2f}, Dice: {calculate_dice(C, D):.2f}")