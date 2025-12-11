import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from pathlib import Path
from sklearn.cluster import MeanShift, estimate_bandwidth
import time # Import time to measure performance difference

# --- Configuration ---
DATA_DIR = '../data/matterport_data/v1/scans/1LXtFkjw3qL/undistorted_color_images/1LXtFkjw3qL/_/undistorted_color_images/'
OUTPUT_DIR = './meansShift_output'
# Bandwidth is the most critical parameter for Mean Shift.
# Lower value -> more clusters (finer detail); Higher value -> fewer clusters (more smoothing).
# estimate_bandwidth can automatically guess a good starting value, but it's often slow.
# For images, a value between 5 and 20 usually works well for 8-bit color.
BANDWIDTH = 5 
# ---------------------

def process_image_meanshift(image_path, bandwidth):
    """
    Loads an image, performs Mean Shift segmentation, and returns the segmented image.
    """
    # 1. Load the image
    # Note: cv2.imread reads images in BGR format
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image {image_path}. Skipping.")
        return None

    # 2. Convert to RGB and prepare data
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image for clustering (N_pixels, 3)
    pixel_vals = rgb_image.reshape((-1, 3))
    
    # Optional: Convert to float for better numerical stability in Mean Shift, 
    # though MeanShift can often handle uint8 data directly.
    # pixel_vals = np.float32(pixel_vals)

    # 3. Apply Mean Shift clustering
    print(f"  -> Applying Mean Shift with bandwidth={bandwidth}...")
    start_time = time.time()
    
    # Initialize MeanShift model
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    
    # Fit model to data and predict labels
    ms.fit(pixel_vals)
    labels = ms.labels_
    centers = ms.cluster_centers_
    
    end_time = time.time()
    n_clusters = len(np.unique(labels))
    print(f"  -> Mean Shift complete. Found {n_clusters} clusters in {end_time - start_time:.2f} seconds.")

    # 4. Reconstruct the segmented image
    # Convert center values (which are floats from MeanShift) back to 8-bit integers (0-255)
    centers = np.uint8(centers)
    
    # Map the cluster centers (colors) back to the pixel labels
    segmented_data = centers[labels]
    
    # Reshape the data back into the original image dimensions
    segmented_image = segmented_data.reshape((rgb_image.shape))

    # The reconstructed image is in RGB format
    return segmented_image

def main():
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
        
        if j == 10:
            break

        print(f"\n--- Processing image {i+1}/{len(image_files)}: {os.path.basename(file_path)} ---")

        segmented_rgb = process_image_meanshift(file_path, BANDWIDTH)

        if segmented_rgb is not None:
            # 4. Prepare for saving
            file_name = os.path.basename(file_path)
            output_file_name = f"meanshift_{file_name}"
            output_path = os.path.join(OUTPUT_DIR, output_file_name)

            # cv2.imwrite saves files in BGR format, so convert back from RGB
            segmented_bgr = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)

            # 5. Save the segmented image
            cv2.imwrite(output_path, segmented_bgr)
            print(f"  -> Saved segmented image to {output_path}")

    print("\nBatch processing complete.")
    
if __name__ == "__main__":
    main()