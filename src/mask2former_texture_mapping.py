import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
import os
import glob
import cv2
import sys
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from datasets import load_dataset
from PIL import Image

def get_ade20k_palette():
    """
    Returns a random color palette for visualizing segmentation maps.
    ADE20K has 150 semantic classes.
    """
    np.random.seed(42)
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

class TextureMapper:
    def __init__(self, texture_path):
        # Load texture and ensure it's RGB
        img = cv2.imread(texture_path)
        if img is None:
            raise FileNotFoundError(f"Texture file not found at {texture_path}")
        self.texture = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    def apply_texture(self, original_image, wall_polygon):
        """
        original_image: The full RGB room image
        wall_polygon: Array of 4 points [[x,y],...] from WallRefiner
        """
        # 1. Order points consistently (top-left, top-right, bottom-right, bottom-left)
        pts_dst = self._order_points(wall_polygon)

        # 2. Calculate "Rectified" Wall Dimensions (Width/Height in flat space)
        #    We use Euclidean distance to estimate the real-world aspect ratio
        (tl, tr, br, bl) = pts_dst
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # 3. Create Tiled Texture Canvas
        #    Repeat the small texture to fill the estimated wall size
        tiled_texture = self._tile_texture(maxWidth, maxHeight)

        # 4. Compute Homography & Warp
        #    Source points are the corners of our flat tiled texture
        pts_src = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        #    Calculate the Perspective Transform Matrix
        M = cv2.getPerspectiveTransform(pts_src, pts_dst.astype("float32"))
        
        #    Warp the flat tiled texture to fit the wall polygon
        warped_texture = cv2.warpPerspective(tiled_texture, M, (original_image.shape[1], original_image.shape[0]))

        # 5. Blend with "Multiply" mode to preserve shadows
        #    Create a mask for the wall region
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts_dst.astype("int32"), 255)
        
        #    Convert to float for blending (0.0 - 1.0)
        img_float = original_image.astype(float) / 255.0
        tex_float = warped_texture.astype(float) / 255.0
        
        #    The Blend Logic: Output = Image * Texture
        #    This keeps the darkness of shadows because (Shadow=0.2 * Texture=0.9) = 0.18
        blended = img_float.copy()
        
        #    Only apply blend where the wall mask exists
        mask_indices = np.where(mask > 0)
        
        #    "Soft Light" or "Multiply" blending math:
        blended[mask_indices] = cv2.multiply(img_float[mask_indices], tex_float[mask_indices])
        
        #    Convert back to uint8
        output = (blended * 255).astype(np.uint8)
        
        return output

    def _tile_texture(self, w, h):
        """Repeat texture to fill w, h"""
        th, tw, _ = self.texture.shape
        # Calculate how many tiles we need
        nx = int(np.ceil(w / tw))
        ny = int(np.ceil(h / th))
        
        # Tile it
        tiled = np.tile(self.texture, (ny, nx, 1))
        
        # Crop to exact size
        return tiled[:h, :w]

    def _order_points(self, pts):
        """Orders points: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right has smallest difference
        rect[3] = pts[np.argmax(diff)] # Bottom-left has largest difference
        return rect

def run(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check texture file
    texture_path = args.texture
    if not os.path.exists(texture_path):
        print(f"Warning: Texture file '{texture_path}' not found. Please provide a valid texture path using --texture.")
        print("Creating a dummy texture for demonstration...")
        # Create a simple checkerboard texture
        dummy_texture = np.zeros((100, 100, 3), dtype=np.uint8)
        # Fill with white
        dummy_texture.fill(255)
        # Draw some black squares
        cv2.rectangle(dummy_texture, (0, 0), (50, 50), (0, 0, 255), -1) # Red square
        cv2.rectangle(dummy_texture, (50, 50), (100, 100), (0, 255, 0), -1) # Green square
        texture_path = "dummy_texture.png"
        cv2.imwrite(texture_path, dummy_texture)
        print(f"Created {texture_path}")

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
        print("Loading ADE20K dataset stream...")
        try:
            dataset = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        # Process a few examples
        num_examples = 2
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

    # Init TextureMapper
    try:
        texture_mapper = TextureMapper(texture_path)
        print(f"Loaded texture from {texture_path}")
    except Exception as e:
        print(f"Error initializing TextureMapper: {e}")
        return

    # Setup plot
    fig, axes = plt.subplots(num_examples, 3, figsize=(18, 6 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for idx, example in enumerate(examples):
        image = example['image']
        
        # --- Inference ---
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process
        target_sizes = [image.size[::-1]] # (H, W)
        prediction = processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )[0]
        
        prediction_np = prediction.cpu().numpy()

        # Extract the raw "Wall" class
        full_wall_mask = np.zeros(prediction_np.shape, dtype=np.uint8)
        for w_id in wall_ids:
            full_wall_mask[prediction_np == w_id] = 1

        # --- Refine and Split ---
        refiner = WallRefiner()
        image_np = np.array(image)
        segments, polygons = refiner.refine_mask(full_wall_mask, image_np)

        # --- Texture Mapping ---
        textured_image = image_np.copy()
        for poly in polygons:
            textured_image = texture_mapper.apply_texture(textured_image, poly)

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
        visualize_prediction(image, prediction_np, axes[idx, 1], title="Mask2Former Wall Segments", target_ids=wall_ids)
        
        # Column 3: Textured Result
        axes[idx, 2].imshow(textured_image)
        axes[idx, 2].set_title(f"Textured Walls ({len(segments)} segments)")
        axes[idx, 2].axis('off')
        
    plt.tight_layout()
    plt.show()
    print("Done! Check the generated plot.")

