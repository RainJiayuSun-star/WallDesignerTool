# This runs the benchmark for the 2 methods

import numpy as np

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