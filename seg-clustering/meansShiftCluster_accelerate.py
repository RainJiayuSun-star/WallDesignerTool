import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from pathlib import Path
import time
import torch # The new dependency for GPU acceleration

# --- Configuration ---
DATA_DIR = '../data/matterport_data/v1/scans/1LXtFkjw3qL/undistorted_color_images/1LXtFkjw3qL/_/undistorted_color_images/'
OUTPUT_DIR = './meansShift_output_torch_gpu'
BANDWIDTH = 20.0 # Mean Shift bandwidth (changed to float for PyTorch kernel)
MAX_ITER = 50 # Maximum iterations for convergence
TOL = 0.01 # Tolerance for cluster center convergence
# ---------------------

def mean_shift_torch(data, bandwidth, max_iter, tol):
    """
    Performs Mean Shift clustering using PyTorch for GPU acceleration.
    This implementation focuses purely on the color space (3 dimensions).

    Args:
        data (torch.Tensor): The input pixel data (N_pixels, 3), already on the CUDA device.
        bandwidth (float): The radius for the kernel.
        max_iter (int): Maximum iterations to run.
        tol (float): Convergence tolerance.

    Returns:
        tuple: (cluster_centers_cpu, labels_cpu)
    """
    if not data.is_cuda:
        raise ValueError("Input data must be on CUDA device.")

    # 1. Initialization
    # Each pixel starts as its own potential cluster center.
    # The 'centers' tensor will store the shifted points.
    centers = data.clone()
    n_samples, n_features = data.shape
    
    # Pre-calculate bandwidth squared and kernel normalization
    bandwidth_sq = bandwidth ** 2
    
    print(f"  -> Starting Mean Shift on {n_samples} samples...")
    
    # 2. Main Mean Shift Loop
    for i in range(max_iter):
        start_iter_time = time.time()
        
        # Keep track of the shift distance for convergence check
        max_shift = 0.0
        new_centers = torch.empty_like(centers)
        
        # Broadcast the centers tensor to enable parallel processing of all shifts
        # Reshape centers: (N_samples, 1, 3)
        # Reshape data: (1, N_samples, 3)
        # The difference is (N_samples, N_samples, 3)
        
        # Calculate the squared Euclidean distance: ||center_i - data_j||^2
        # (center_i - data_j)^2 -> sum over 3 features -> (N_samples, N_samples)
        # We need to compute the shift for *each* of the N_samples points.
        
        # The core idea: For each point P, find the weighted mean of its neighbors N.
        # The weight is based on the Gaussian kernel.

        # Pytorch's `cdist` (or manual calculation) is O(N^2), which is slow.
        # This implementation simplifies the O(N^2) color-space-only version.

        all_shifts = []
        
        # This loop is necessary to manage memory if N is very large, 
        # but it slows down the GPU benefit. For standard image sizes (N ~ 300k), 
        # a full O(N^2) distance matrix is too large for typical VRAM.
        # A more advanced implementation uses spatial trees (like KD-Tree/Ball-Tree) 
        # or a binning scheme, which is typically not available in pure PyTorch.
        # We'll stick to a simplified, small-batch loop to avoid OOM errors.
        
        batch_size = 5000 # Process centers in smaller batches
        for k in range(0, n_samples, batch_size):
            batch_centers = centers[k:k+batch_size].unsqueeze(1) # (batch_size, 1, 3)
            
            # Calculate squared distance from batch_centers to all data points
            # dist_sq: (batch_size, N_samples)
            diff = batch_centers - data.unsqueeze(0) # (batch_size, N_samples, 3)
            dist_sq = torch.sum(diff ** 2, dim=2)
            
            # Identify points within the bandwidth
            # mask: (batch_size, N_samples) -> boolean tensor
            mask = dist_sq < bandwidth_sq
            
            # Gaussian Kernel Weighting: exp(-d^2 / (2 * b^2))
            weights = torch.exp(-dist_sq / (2.0 * bandwidth_sq))
            
            # Apply mask to weights: only consider neighbors
            weights = weights * mask.float()
            
            # Calculate the shifted mean (new center)
            # new_center_k = (weights_k * data) / sum(weights_k)
            # data: (1, N_samples, 3)
            # weights: (batch_size, N_samples, 1) - expanded for element-wise multiply
            
            weighted_sum = (weights.unsqueeze(2) * data.unsqueeze(0)).sum(dim=1) # (batch_size, 3)
            weight_sum = weights.sum(dim=1).unsqueeze(1) # (batch_size, 1)
            
            # Handle division by zero (points with no neighbors, though unlikely)
            weight_sum[weight_sum == 0] = 1e-6 
            
            shift = weighted_sum / weight_sum # (batch_size, 3)
            
            # Calculate the shift distance for convergence check
            current_max_shift = torch.max(torch.linalg.norm(shift - centers[k:k+batch_size], dim=1)).item()
            max_shift = max(max_shift, current_max_shift)
            
            new_centers[k:k+batch_size] = shift

        # Update centers and check for convergence
        centers = new_centers
        
        end_iter_time = time.time()
        print(f"    - Iteration {i+1}/{max_iter}: Max shift = {max_shift:.4f} in {end_iter_time - start_iter_time:.2f}s")
        
        if max_shift < tol:
            print(f"  -> Converged after {i+1} iterations.")
            break
            
    # 3. Label Assignment
    # The final cluster centers are the unique, converged points.
    
    # Move centers to CPU for easy processing of unique values
    centers_cpu = centers.cpu().numpy()
    
    # Use a simpler, non-exact convergence approach for labeling:
    # Cluster points that are close to each other at the end.
    
    # NOTE: A proper Mean Shift implementation merges close centers throughout.
    # For a simple PyTorch implementation, we perform a final grouping on CPU.
    
    from sklearn.cluster import MeanShift as SklearnMeanShift # ONLY for final labeling/merging
    
    # We use a very small bandwidth for final grouping/merging on the converged points
    # This step is often necessary in simple MS implementations to merge nearby modes.
    # Since this is done on the small set of converged centers, it's fast on the CPU.
    ms_final = SklearnMeanShift(bandwidth=bandwidth/10.0) 
    ms_final.fit(centers_cpu)
    
    unique_centers = ms_final.cluster_centers_
    center_labels = ms_final.labels_
    
    # Map each of the N_samples initial points to its final merged cluster center
    # This involves calculating the distance from *data* to *unique_centers*.
    data_cpu = data.cpu().numpy()
    
    # Calculate all distances from original points to final merged unique centers
    from sklearn.metrics import pairwise_distances_argmin_min
    
    # labels: (N_samples,) - index of the closest unique center
    labels, _ = pairwise_distances_argmin_min(data_cpu, unique_centers)
    
    return unique_centers, labels


def process_image_meanshift_gpu(image_path, bandwidth):
    """
    Loads an image and performs GPU-accelerated Mean Shift segmentation using PyTorch.
    """
    # 1. Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image {image_path}. Skipping.")
        return None

    # 2. Convert to RGB and prepare data
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image for clustering (N_pixels, 3)
    # Mean Shift typically works on normalized data for better feature weighting
    pixel_vals_np = rgb_image.reshape((-1, 3)).astype(np.float32)

    # 3. Convert to PyTorch tensor and move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Error: CUDA not available. Cannot run GPU-accelerated code.")
        return None
        
    pixel_vals_torch = torch.from_numpy(pixel_vals_np).to(device)

    # 4. Apply PyTorch GPU-accelerated Mean Shift clustering
    print(f"  -> Applying PyTorch GPU Mean Shift with bandwidth={bandwidth}...")
    start_time = time.time()
    
    # Run the custom PyTorch Mean Shift
    centers_np, labels_np = mean_shift_torch(
        pixel_vals_torch, 
        bandwidth=bandwidth, 
        max_iter=MAX_ITER, 
        tol=TOL
    )
    
    end_time = time.time()
    n_clusters = len(np.unique(labels_np))
    print(f"  -> PyTorch Mean Shift complete. Found {n_clusters} clusters in {end_time - start_time:.2f} seconds.")

    # 5. Reconstruct the segmented image
    # The cluster centers are in float (0-255 range). Convert back to 8-bit integers
    centers_uint8 = np.clip(centers_np, 0, 255).astype(np.uint8)
    
    # Map the cluster centers (colors) back to the pixel labels
    segmented_data = centers_uint8[labels_np]
    
    # Reshape the data back into the original image dimensions
    segmented_image = segmented_data.reshape((rgb_image.shape))

    # The reconstructed image is in RGB format
    return segmented_image

# -----------------
# Main Execution
# -----------------
def main():
    if not torch.cuda.is_available():
        print("CUDA is NOT available. This GPU-accelerated code will not run.")
        return
    else:
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")

    # 1. Ensure the output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' ensured.")

    # 2. Find all images in the data directory
    search_path = os.path.join(DATA_DIR, '*.jpg')
    image_files = glob.glob(search_path)
    
    if not image_files:
        print(f"No .jpg files found in the directory '{DATA_DIR}'. Please check the path.")
        return

    print(f"Found {len(image_files)} image(s) to process.")

    j = 0 # to track the number of iterations
    # 3. Loop through and process each image
    for i, file_path in enumerate(image_files):
        
        # Only process a few images for quick testing
        if j == 10:
            break
            
        print(f"\n--- Processing image {i+1}/{len(image_files)}: {os.path.basename(file_path)} ---")

        # Use the new PyTorch GPU-accelerated function
        segmented_rgb = process_image_meanshift_gpu(file_path, BANDWIDTH) 

        if segmented_rgb is not None:
            # 4. Prepare for saving
            file_name = os.path.basename(file_path)
            output_file_name = f"meanshift_torch_gpu_{file_name}" 
            output_path = os.path.join(OUTPUT_DIR, output_file_name)

            # cv2.imwrite saves files in BGR format, so convert back from RGB
            segmented_bgr = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)

            # 5. Save the segmented image
            cv2.imwrite(output_path, segmented_bgr)
            print(f"  -> Saved segmented image to {output_path}")
        
        j += 1 # Increment counter
    
    print("\nBatch processing complete.")
    
if __name__ == "__main__":
    main()