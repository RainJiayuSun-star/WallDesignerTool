import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
import os
import glob
import cv2
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from datasets import load_dataset
from PIL import Image

def get_ade20k_palette():
    """
    Returns a random color palette for visualizing segmentation maps.
    ADE20K has 150 semantic classes.
    """
    np.random.seed(69)
    # Generate 151 colors (0-150)
    palette = np.random.randint(0, 255, size=(151, 3), dtype=np.uint8)
    return palette

def visualize_prediction(image, pred_map, ax, title="Prediction", target_ids=None):
    """
    Visualizes the segmentation map by overlaying it on the original image.
    If target_ids is provided, only visualizes those classes.
    """
    # Create an RGB image from the class IDs using the palette
    palette = get_ade20k_palette()
    
    # pred_map is a 2D array of class IDs. 
    # We map these IDs to RGB colors.
    color_seg = np.zeros((pred_map.shape[0], pred_map.shape[1], 3), dtype=np.uint8)
    
    # Map each class ID to its color
    for label_id in np.unique(pred_map):
        # If filtering, skip if not in target_ids
        if target_ids is not None and label_id not in target_ids:
            continue

        # Ensure label_id is within palette range
        if label_id < len(palette):
            color_seg[pred_map == label_id] = palette[label_id]

    # Display
    ax.imshow(image)
    ax.imshow(color_seg, alpha=0.5)  # Overlay with transparency
    ax.set_title(title)
    ax.axis('off')

class WallRefiner:
    def __init__(self, min_wall_area=1000):
        self.min_wall_area = min_wall_area

    def refine_mask(self, binary_mask, original_image):
        """
        Takes a binary mask and the original RGB image.
        1. Finds vertical edges in the RGB content to split walls.
        2. Returns separated wall segments.
        """
        # 1. Morphological Cleanup (Fill small holes)
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # 2. Geometric Splitting (The "Corner Cutter")
        # We pass the original image now to find shadows/corners
        split_mask = self._split_joined_walls(cleaned_mask, original_image)

        # 3. Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(split_mask, connectivity=8)

        wall_segments = []
        wall_polygons = []

        for i in range(1, num_labels): # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_wall_area:
                continue

            segment_mask = (labels == i).astype(np.uint8) * 255
            poly = self._approximate_polygon(segment_mask)
            
            if poly is not None:
                wall_segments.append(segment_mask)
                wall_polygons.append(poly)

        return wall_segments, wall_polygons

    def _split_joined_walls(self, mask, image):
        split_mask = mask.copy()
        h, w = mask.shape

        # A. Pre-process Image for Edge Detection
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast (CLAHE) to help find corners in dark rooms (like Example 1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # B. Run Canny on the IMAGE, not the mask
        # Thresholds: low enough to catch shadows, high enough to ignore texture
        edges = cv2.Canny(gray, 30, 100) 

        # C. Filter Edges: Keep only edges that are INSIDE the wall mask
        # We erode the mask slightly so we don't accidentally detect the wall boundary itself
        mask_eroded = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=2)
        internal_edges = cv2.bitwise_and(edges, edges, mask=mask_eroded)

        # D. Hough Transform to find Vertical Lines
        # minLineLength: Needs to be long enough to be a corner (e.g., 10% of image height)
        # maxLineGap: Large enough to jump over a person sitting on a bench (Example 1)
        min_len = h // 8 
        lines = cv2.HoughLinesP(internal_edges, 1, np.pi/180, threshold=30, minLineLength=min_len, maxLineGap=50)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check for Verticality (+/- 15 degrees)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 75 < angle < 105: 
                    # EXTEND THE LINE:
                    # A corner usually goes floor-to-ceiling. 
                    # Even if we only detect the top half, we should cut the whole mask.
                    cv2.line(split_mask, (x1, 0), (x2, h), 0, thickness=3)

        return split_mask

    def _approximate_polygon(self, binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        cnt = max(contours, key=cv2.contourArea)
        
        # Dynamic epsilon based on arc length
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # If not 4 points, force Convex Hull
        if len(approx) != 4:
            hull = cv2.convexHull(cnt)
            approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
            
        if len(approx) != 4:
             rect = cv2.minAreaRect(cnt)
             box = cv2.boxPoints(rect)
             approx = box.astype(np.int32)

        return approx.reshape(-1, 2)

def run(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    examples = []
    num_examples = 0

    if args.local:
        # Load local images
        print(f"Loading local images from {args.dir}...")
        if not os.path.exists(args.dir):
            print(f"Directory {args.dir} does not exist.")
            return
            
        image_files = glob.glob(os.path.join(args.dir, "*.png"))
        # Sort for consistent order
        image_files.sort()
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                examples.append({"image": img, "filename": os.path.basename(img_path)})
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if not examples:
            print(f"No PNG images found in {args.dir}")
            return
            
        num_examples = len(examples)
    else:
        # 2. Load the Dataset (Streaming)
        # Using streaming=True allows us to load just the examples we need without 
        # downloading the full dataset (which is quite large).
        print("Loading ADE20K dataset stream...")
        try:
            dataset = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        # Process a few examples
        num_examples = 2
        # Use islice to grab just the first few items from the stream
        # Skip the first 10 to get different examples
        examples = list(itertools.islice(dataset, 10, 10 + num_examples))

    # 3. Load Model and Processor
    model_id = "facebook/mask2former-swin-large-ade-semantic"
    print(f"Loading model: {model_id}...")
    
    processor = Mask2FormerImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(device)
    model.eval()

    print(f"Processing {len(examples)} examples...")

    # Find wall ids
    wall_ids = [id for id, label in model.config.id2label.items() if 'wall' in label.lower()]
    print(f"Wall class IDs: {wall_ids}")

    # Setup plot
    # Adjust figsize based on number of examples
    fig, axes = plt.subplots(num_examples, 3, figsize=(18, 6 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for idx, example in enumerate(examples):
        image = example['image']
        
        # Check standard column names for ground truth. 
        # ADE20K on HF often uses 'annotation' or 'label'
        # For '1aurent/ADE20K', inspection usually reveals 'image' and 'annotation'
        ground_truth = None
        if 'annotation' in example:
            ground_truth = example['annotation']
        elif 'label' in example:
            ground_truth = example['label']

        # --- Inference ---
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process to get semantic map (H, W)
        # target_sizes is required to resize the output mask back to original image size
        target_sizes = [image.size[::-1]] # (H, W)
        prediction = processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )[0]
        
        prediction_np = prediction.cpu().numpy()

        # Extract the raw "Wall" class - combine ALL wall classes into one binary mask
        full_wall_mask = np.zeros(prediction_np.shape, dtype=np.uint8)
        for w_id in wall_ids:
            full_wall_mask[prediction_np == w_id] = 1

        # --- NEW: Refine and Split ---
        refiner = WallRefiner()
        segments, polygons = refiner.refine_mask(full_wall_mask, np.array(image))

        # --- Visualization ---
        # Column 1: Original Image
        axes[idx, 0].imshow(image)
        title_text = f"Example {idx+1}"
        if 'filename' in example:
            title_text += f": {example['filename']}"
        else:
            title_text += ": Original Image"
        axes[idx, 0].set_title(title_text)
        axes[idx, 0].axis('off')

        # Column 2: Raw Mask2Former Output (Wall Blobs)
        visualize_prediction(image, prediction_np, axes[idx, 1], title="Raw Semantic Output", target_ids=wall_ids)
        
        # Column 3: Geometric Segments (Polygons)
        # Create a visualization of the segments
        vis_img = np.array(image).copy()
        
        for i, poly in enumerate(polygons):
            # Draw the polygon outline
            cv2.polylines(vis_img, [poly], True, (0, 255, 0), 3)
            
            # Draw the corners
            for point in poly:
                cv2.circle(vis_img, tuple(point), 8, (255, 0, 0), -1)
                
            # Label the segment center
            M = cv2.moments(poly)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(vis_img, f"Wall {i+1}", (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        axes[idx, 2].imshow(vis_img)
        axes[idx, 2].set_title(f"Refined & Split ({len(segments)} walls)")
        axes[idx, 2].axis('off')
        
        # Optional: Print detected classes
        # The model config contains the mapping from ID to Label
        labels = [model.config.id2label[id] for id in np.unique(prediction_np) if id in model.config.id2label]
        print(f"Example {idx+1} detected classes: {', '.join(labels)}")

    plt.tight_layout()
    plt.show()
    print("Done! Check the generated plot.")

