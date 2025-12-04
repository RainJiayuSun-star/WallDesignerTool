import torch
from PIL import Image
import requests
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define Model and Processor ---
# Use the instance segmentation model trained on COCO
MODEL_ID = "facebook/mask2former-swin-small-coco-instance"

# Load the image processor and model
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID)

# --- 2. Load and Prepare Image ---
# Replace this URL with the path to your own interior image, or load it with PIL.Image.open()
url = "http://images.cocodataset.org/val2017/000000039769.jpg" # Example image: a cat and two remotes on a couch
img = "data\matterport\sample\1e649cc84c9043b69e2367b7d5aeecf2_i1_4.jpg" 

print(f"Loading image from: {url}")
try:
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
except Exception as e:
    print(f"Error loading image: {e}")
    # Fallback to a local path if needed, e.g., image = Image.open("path/to/your/image.jpg").convert("RGB")
    exit()

# Prepare the image for the model (resizing, normalization, etc.)
# This outputs a dictionary of PyTorch tensors, including 'pixel_values' and 'pixel_mask'
inputs = image_processor(image, return_tensors="pt")

# --- 3. Run Inference ---
print("Running inference...")
with torch.no_grad():
    # Pass the processed inputs to the model
    outputs = model(**inputs)

# --- 4. Post-Process and Get Segmentation Map ---
# Post-process the raw model outputs to get the final instance segmentation map.
# target_sizes is required to resize the predicted mask back to the original image size.
target_sizes = [(image.height, image.width)]
pred_instance_map_list = image_processor.post_process_instance_segmentation(
    outputs, 
    target_sizes=target_sizes
)

# The result is a list of dictionaries (one per image in the batch). We take the first one.
pred_info = pred_instance_map_list[0]

# The 'segmentation' tensor is a map where each pixel value is a unique segment_id (instance ID)
# The 'segments_info' list contains details for each segment, like the predicted class label and score.
instance_segmentation_map = pred_info['segmentation']
segments_info = pred_info['segments_info']

# --- 5. Output Results ---
print("\n✅ Instance Segmentation Complete!")
print(f"Original image size: {image.size}")
print(f"Segmentation map shape: {instance_segmentation_map.shape}")
print(f"Found {len(segments_info)} object instances.")

print("\nIdentified Objects:")
# You can use the model's id2label mapping for more meaningful names, 
# but for simplicity, we just print the label ID and score here.
# Note: The COCO dataset class labels can be found in the model config's id2label.
for segment in segments_info:
    segment_id = segment['id']
    label_id = segment['label_id']
    score = segment['score']
    
    # You can apply a visual filter here:
    # Get the mask for a specific instance by checking where the map equals the segment_id
    # instance_mask = (instance_segmentation_map == segment_id)
    
    print(f"- Segment ID: {segment_id}, Class Label ID: {label_id}, Score: {score:.4f}")

# --- Optional: Display the results (requires matplotlib) ---
# import matplotlib.pyplot as plt
# import numpy as np

# Convert the map to a color-coded image for visualization (simple example)
# Create a random color for each unique instance ID
unique_ids = torch.unique(instance_segmentation_map)
colors = np.random.randint(0, 256, (len(unique_ids), 3))

color_map = np.zeros((*instance_segmentation_map.shape, 3), dtype=np.uint8)
for i, segment_id in enumerate(unique_ids):
    if segment_id.item() == 0: # Usually background is ID 0
        continue
    mask = (instance_segmentation_map == segment_id).cpu().numpy()
    color_map[mask] = colors[i]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
# Overlay the segmentation map on the original image for better context
blended_image = Image.fromarray(color_map).convert("RGB")
blended_image = Image.blend(image.convert("RGBA"), blended_image.convert("RGBA"), alpha=0.5)
plt.imshow(blended_image)
plt.title("Instance Segmentation")
plt.axis('off')
plt.show()