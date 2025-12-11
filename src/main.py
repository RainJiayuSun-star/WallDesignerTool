import argparse
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import itertools
from datasets import load_dataset

# Import modular components
from segmentation import Mask2FormerSegmentation
from splitting import CannyHoughSplitting, WallRefinerSplitting
from mapping import HomographyMultiplyMapping, MaskedPerspectiveMapping

def get_ade20k_palette():
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(151, 3), dtype=np.uint8)
    return palette

def visualize_prediction(image, pred_map, ax, title="Prediction", target_ids=None):
    palette = get_ade20k_palette()
    color_seg = np.zeros((pred_map.shape[0], pred_map.shape[1], 3), dtype=np.uint8)
    for label_id in np.unique(pred_map):
        if target_ids is not None and label_id not in target_ids:
            continue
        if label_id < len(palette):
            color_seg[pred_map == label_id] = palette[label_id]
    ax.imshow(image)
    ax.imshow(color_seg, alpha=0.5)
    ax.set_title(title)
    ax.axis('off')

def load_examples(args):
    examples = []
    if args.local:
        print(f"Loading local images from {args.dir}...")
        if not os.path.exists(args.dir):
            print(f"Directory {args.dir} does not exist.")
            return []
        image_files = glob.glob(os.path.join(args.dir, "*.png"))
        image_files.sort()
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                examples.append({"image": img, "filename": os.path.basename(img_path)})
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        if not examples:
            print(f"No PNG images found in {args.dir}")
    else:
        print("Loading ADE20K dataset stream...")
        try:
            dataset = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
            examples = list(itertools.islice(dataset, 10, 12)) # Load 2 examples
        except Exception as e:
            print(f"Error loading dataset: {e}")
    return examples

def main():
    parser = argparse.ArgumentParser(description="Wall Designer Tool Pipeline")
    parser.add_argument("--segmentationMethod", type=str, default="mask2former", choices=["mask2former"], help="Method for wall segmentation")
    parser.add_argument("--splittingMethod", type=str, default="refiner", choices=["cannyhough", "refiner"], help="Method for splitting connected walls")
    parser.add_argument("--textureMappingMethod", type=str, default="maskedPerspective", choices=["homographyMultiply", "maskedPerspective"], help="Method for applying texture")
    
    parser.add_argument("--local", action="store_true", help="Use local images from --dir")
    parser.add_argument("--dir", type=str, default=os.path.join(os.path.dirname(__file__), "ourSet"), help="Directory containing local images")
    parser.add_argument("--texture", type=str, default=os.path.join(os.path.dirname(__file__), "dummy_texture.png"), help="Path to texture file")
    
    args = parser.parse_args()

    # 1. Initialize Components
    if args.segmentationMethod == "mask2former":
        segmenter = Mask2FormerSegmentation()
    
    if args.splittingMethod == "cannyhough":
        splitter = CannyHoughSplitting()
    elif args.splittingMethod == "refiner":
        splitter = WallRefinerSplitting()
        
    if args.textureMappingMethod == "homographyMultiply":
        mapper = HomographyMultiplyMapping(args.texture)
    elif args.textureMappingMethod == "maskedPerspective":
        mapper = MaskedPerspectiveMapping(args.texture)

    # 2. Load Data
    examples = load_examples(args)
    if not examples:
        return

    num_examples = len(examples)
    # 4 columns: Original, Segmentation, Splitting, Texture
    fig, axes = plt.subplots(num_examples, 4, figsize=(24, 6 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    print(f"Processing {num_examples} examples...")

    for idx, example in enumerate(examples):
        image = example['image']
        image_np = np.array(image)
        
        # A. Segmentation
        print(f"Segmenting example {idx+1}...")
        pred_map, wall_ids = segmenter.segment(image)
        
        # B. Splitting
        print(f"Splitting example {idx+1}...")
        full_wall_mask = np.zeros(pred_map.shape, dtype=np.uint8)
        for w_id in wall_ids:
            full_wall_mask[pred_map == w_id] = 1 # Keep 0/1 for now
            
        # Ensure mask is correct format if needed by splitter, but our splitter handles it.
        # WallRefinerSplitting expects 0-255 internally or handles 0-1.
        
        segments, polygons = splitter.split(full_wall_mask, image_np)
        
        # C. Mapping
        print(f"Mapping texture for example {idx+1}...")
        # Prepare full_wall_mask as 0-255/bool for mapper if needed
        full_wall_mask_255 = (full_wall_mask * 255).astype(np.uint8)
        
        textured_image = mapper.apply(image_np, polygons, full_mask=full_wall_mask_255)

        # D. Visualization
        # Col 1: Original
        axes[idx, 0].imshow(image)
        title = example.get('filename', f"Example {idx+1}")
        axes[idx, 0].set_title(title)
        axes[idx, 0].axis('off')

        # Col 2: Segmentation (Wall Blob)
        visualize_prediction(image, pred_map, axes[idx, 1], title="Segmentation (Wall Blob)", target_ids=wall_ids)

        # Col 3: Splitting (Wall Edges/Polygons)
        vis_split = image_np.copy()
        for i, poly in enumerate(polygons):
            cv2.polylines(vis_split, [poly], True, (0, 255, 0), 3)
            # Label center
            M = cv2.moments(poly)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(vis_split, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        axes[idx, 2].imshow(vis_split)
        axes[idx, 2].set_title(f"Splitting ({len(segments)} walls)")
        axes[idx, 2].axis('off')

        # Col 4: Texture Applied
        axes[idx, 3].imshow(textured_image)
        axes[idx, 3].set_title("Texture Applied")
        axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
