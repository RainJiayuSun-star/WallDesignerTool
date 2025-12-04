import torch
from PIL import Image
import requests
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy as np
import matplotlib.pyplot as plt

# check cuda availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Define Model and Processor ---
# MODEL_ID = "facebook/mask2former-swin-small-coco-instance" # trained on coco instance
MODEL_ID = "facebook/mask2former-swin-small-ade-semantic" # trained on ADE20K


# Load the image processor and model
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID)

# --- 1.5. Move Model to GPU ---
model.to(device)

# --- 2. Load and Prepare Image ---
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
# Note: Using forward slashes for Linux/Path compatibility, even on WSL
LOCAL_PATH = "../data/matterport/sample/1e649cc84c9043b69e2367b7d5aeecf2_i1_4.jpg" 

# Choose which image to use
image_source = LOCAL_PATH

print(f"Loading image from: {image_source}")
try:
    if image_source.startswith("http"):
        image = Image.open(requests.get(image_source, stream=True).raw).convert("RGB")
    else:
        # Load from local file path
        image = Image.open(image_source).convert("RGB")
except Exception as e:
    print(f"Error loading image from {image_source}: {e}")
    print("Exiting script.")
    exit()

# Prepare the image for the model (resizing, normalization, etc.)
inputs = image_processor(image, return_tensors="pt")

# <<< FIX APPLIED HERE: Move Input Tensors to the same device as the model >>>
inputs = {k: v.to(device) for k, v in inputs.items()}

# --- 3. Run Inference ---
print("Running inference...")
with torch.no_grad():
    # Pass the processed inputs to the model
    outputs = model(**inputs)

# --- 4. Post-Process and Get Segmentation Map ---
target_sizes = [(image.height, image.width)]
pred_instance_map_list = image_processor.post_process_instance_segmentation(
    outputs=outputs, 
    target_sizes=target_sizes
)

pred_info = pred_instance_map_list[0]
instance_segmentation_map = pred_info['segmentation']
segments_info = pred_info['segments_info']
id2label = model.config.id2label # Use id2label for clear output

# --- 5. Output Results ---
print("\n✅ Instance Segmentation Complete!")
print(f"Original image size: {image.size}")
# Move segmentation map back to CPU for NumPy/Matplotlib operations
seg_map_cpu = instance_segmentation_map.cpu() 
print(f"Segmentation map shape: {seg_map_cpu.shape}")
print(f"Found {len(segments_info)} object instances.")

print("\nIdentified Objects:")
for segment in segments_info:
    segment_id = segment['id']
    label_id = segment['label_id']
    score = segment['score']
    label_name = id2label.get(label_id, f"Unknown ID {label_id}")
    
    print(f"- Segment ID: {segment_id}, Object: **{label_name}** (Label ID: {label_id}), Score: {score:.4f}")

# ----------------------------------------------------
# --- 6. Visualization (with labels and blending) ---
# ----------------------------------------------------

unique_ids = torch.unique(seg_map_cpu)
np.random.seed(42) 
colors = np.random.randint(0, 256, (len(unique_ids), 3), dtype=np.uint8)

color_map = np.zeros((*seg_map_cpu.shape, 3), dtype=np.uint8)
for i, segment_id in enumerate(unique_ids):
    if segment_id.item() == 0:
        continue
    mask = (seg_map_cpu == segment_id).numpy()
    color_map[mask] = colors[i]

plt.figure(figsize=(10, 5))

# Plot 1: Original Image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# Plot 2: Instance Segmentation Overlay
plt.subplot(1, 2, 2)
image_np = np.array(image.convert("RGB")) 
blended_image_np = (image_np * 0.5 + color_map * 0.5).astype(np.uint8)

# Add text labels to the segmentation map
for segment in segments_info:
    mask = (seg_map_cpu == segment['id']).numpy()
    if np.any(mask):
        y, x = np.where(mask)
        center_x, center_y = int(np.mean(x)), int(np.mean(y))
        
        label_name = id2label.get(segment['label_id'], str(segment['label_id']))
        
        plt.text(
            center_x, center_y, 
            label_name, 
            color='white', 
            fontsize=8, 
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
        )

plt.imshow(blended_image_np)
plt.title("Instance Segmentation Overlay")
plt.axis('off')
plt.show()