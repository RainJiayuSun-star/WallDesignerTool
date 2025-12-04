import torch
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# --- 0. Setup Device & Model ID for SEMANTIC SEGMENTATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# !!! CHANGE THE MODEL ID for ADE20K Semantic Segmentation !!!
MODEL_ID = "facebook/mask2former-swin-small-ade-semantic"

# Load the image processor and model
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID)
model.to(device)

# --- 1. Load and Prepare Image ---
IMAGE_URL = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"

LOCAL_PATH = "../data/matterport/sample/1e649cc84c9043b69e2367b7d5aeecf2_i1_4.jpg" 

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

# Prepare and move inputs to GPU
inputs = image_processor(image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# --- 2. Run Inference ---
print("Running semantic inference...")
with torch.no_grad():
    outputs = model(**inputs)

# --- 3. Post-Process for SEMANTIC Segmentation ---
target_sizes = [(image.height, image.width)]

# NOTE: Using post_process_semantic_segmentation() instead of instance
pred_semantic_map_list = image_processor.post_process_semantic_segmentation(
    outputs, 
    target_sizes=target_sizes
)
# The result is a single tensor where pixel value = class ID
semantic_map = pred_semantic_map_list[0].cpu() 

# Get labels (ADE20K has many, including 'wall')
id2label = model.config.id2label

# --- 4. Output Results and Identify Wall ID ---
print("\n✅ Semantic Segmentation Complete!")
wall_id = None
print("\nIdentified Classes:")
for id, name in id2label.items():
    if "wall" in name.lower() or "ceiling" in name.lower() or "floor" in name.lower():
        print(f"- Class ID: {id}, Object: **{name}**")
    if "wall" == name.lower():
        wall_id = id # Find the specific wall class ID

# --- 5. Visualization (Highlighting the Wall) ---
if wall_id is not None:
    # Create a mask for all wall pixels
    wall_mask = (semantic_map == wall_id).numpy()
    
    # Create a highlight layer (e.g., red for the wall)
    highlight_color = np.array([255, 0, 0], dtype=np.uint8) # Red color
    color_map = np.zeros((*wall_mask.shape, 3), dtype=np.uint8)
    color_map[wall_mask] = highlight_color
    
    image_np = np.array(image.convert("RGB"))
    
    # Blend the original image with the red wall highlight
    # Use the mask to only blend where the wall is
    blended_image_np = image_np.copy()
    blended_image_np[wall_mask] = (image_np[wall_mask] * 0.5 + highlight_color * 0.5).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(blended_image_np)
    plt.title(f"Wall Segmentation (Class ID: {wall_id}, {id2label[wall_id]})")
    plt.axis('off')
    plt.show()
else:
    print("\nCould not find a specific 'wall' class ID in the model's labels.")