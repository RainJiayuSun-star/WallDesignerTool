import numpy as np
import matplotlib.pyplot as plt
import cv2


# load the image and check if loaded successfully
image = cv2.imread('sample.jpg')

if image is None:
    print("target image does not exist.")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Display Original Image ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1) # create the first subplot
plt.title('Original Image')
plt.imshow(image)
# ------------------------------

# reshape the image for k-means clustering
pixel_vals = image.reshape((-1,3))

pixel_vals = np.float32(pixel_vals)

# simply use the kmeans function in cv2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

k = 7
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((image.shape))

# --- Display Segmented Image ---
plt.subplot(1, 2, 2) # Create the second subplot
plt.title(f"Segmented Image (K={k})")
plt.imshow(segmented_image)
# -------------------------------

# show graph
plt.show()