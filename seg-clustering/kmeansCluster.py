import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from pathlib import Path

## Experiments
# # load the image and check if loaded successfully
# image = cv2.imread('sample.jpg')

# if image is None:
#     print("target image does not exist.")
#     exit()

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # --- Display Original Image ---
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1) # create the first subplot
# plt.title('Original Image')
# plt.imshow(image)
# # ------------------------------

# # reshape the image for k-means clustering
# pixel_vals = image.reshape((-1,3))

# pixel_vals = np.float32(pixel_vals)

# # simply use the kmeans function in cv2
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# k = 7
# retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# centers = np.uint8(centers)
# segmented_data = centers[labels.flatten()]

# segmented_image = segmented_data.reshape((image.shape))

# # --- Display Segmented Image ---
# plt.subplot(1, 2, 2) # Create the second subplot
# plt.title(f"Segmented Image (K={k})")
# plt.imshow(segmented_image)
# # -------------------------------

# # show graph
# plt.show()
# --- Configuration ---
DATA_DIR = '../data/matterport_data/v1/scans/1LXtFkjw3qL/undistorted_color_images/1LXtFkjw3qL/_/undistorted_color_images/'
OUTPUT_DIR = './kmeans_output'
K_CLUSTERS = 3 
# ---------------------

def process_image_kmeans(image_path, k=K_CLUSTERS):
    """
    Loads an image, performs K-means segmentation, and returns the segmented image.
    """
    # 1. Load the image
    # Note: cv2.imread reads images in BGR format
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image {image_path}. Skipping.")
        return None

    # 2. Convert to RGB for standard processing and visualization (Optional, but good practice)
    # The segmentation works on BGR, but we often prefer RGB for plotting/review.
    # We will convert it back to BGR for saving, as cv2.imwrite expects BGR.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. Reshape the image for k-means clustering (Pixel values, R-G-B/B-G-R)
    pixel_vals = rgb_image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # 4. Define K-means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    
    # 5. Apply K-means clustering
    # labels: The cluster index for each pixel
    # centers: The BGR color (mean) of each cluster
    retval, labels, centers = cv2.kmeans(
        pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # 6. Reconstruct the segmented image
    # Convert center values back to 8-bit integers (0-255)
    centers = np.uint8(centers)
    # Map the cluster centers (colors) back to the pixel labels
    segmented_data = centers[labels.flatten()]
    # Reshape the data back into the original image dimensions
    segmented_image = segmented_data.reshape((rgb_image.shape))

    # The reconstructed image is in RGB format because we started with rgb_image
    return segmented_image

def main():
    # 1. Ensure the output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' ensured.")

    # 2. Find all images in the data directory
    # glob.glob is used to find all files matching the pattern
    search_path = os.path.join(DATA_DIR, '*.jpg')
    image_files = glob.glob(search_path)
    
    if not image_files:
        print(f"No .jpg files found in the directory '{DATA_DIR}'. Please check the path.")
        return

    print(f"Found {len(image_files)} image(s) to process.")

    # 3. Loop through and process each image
    for i, file_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(file_path)}")

        segmented_rgb = process_image_kmeans(file_path, K_CLUSTERS)

        if segmented_rgb is not None:
            # 4. Prepare for saving
            # Get the original file name (e.g., 'sample.jpg')
            file_name = os.path.basename(file_path)
            # Create a new file name (e.g., 'segmented_sample.jpg')
            output_file_name = f"segmented_{file_name}"
            output_path = os.path.join(OUTPUT_DIR, output_file_name)

            # cv2.imwrite saves files in BGR format, so convert back from RGB
            segmented_bgr = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)

            # 5. Save the segmented image
            cv2.imwrite(output_path, segmented_bgr)
            print(f"Saved segmented image to {output_path}")

    print("\nBatch processing complete.")
    
if __name__ == "__main__":
    main()