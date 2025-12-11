import argparse
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import itertools
from datasets import load_dataset # Required for ADE20K streaming
import pandas as pd
import time # Required for runtime metric

# Import modular components
from segmentation import Mask2FormerSegmentation
# from clustering import ClusteringSegmentation # Placeholder
from metrics import calculate_iou, calculate_dice # ASSUMPTION: You will implement these

# ==============================================================================
# DATASET AND VISUALIZATION UTILITIES (Copied from your original script)
# ==============================================================================

def get_ade20k_palette():
    np.random.seed(42)
    # 151 classes for ADE20K
    palette = np.random.randint(0, 255, size=(151, 3), dtype=np.uint8)
    return palette

def visualize_prediction(image, pred_map, ax, title="Prediction", target_ids=None):
    """Visualization function (primarily for qualitative checks)."""
    palette = get_ade20k_palette()
    color_seg = np.zeros((pred_map.shape[0], pred_map.shape[1], 3), dtype=np.uint8)
    for label_id in np.unique(pred_map):
        # We only visualize the target IDs here if specified
        if target_ids is not None and label_id not in target_ids:
            continue
        if label_id < len(palette):
            color_seg[pred_map == label_id] = palette[label_id]
            
    ax.imshow(image)
    ax.imshow(color_seg, alpha=0.5)
    ax.set_title(title)
    ax.axis('off')

def load_examples(args):
    """
    Loads images either locally or from the streamed ADE20K dataset,
    exactly as defined in your original script.
    """
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
                # Assuming you'll have GT masks that share a name base.
                img = Image.open(img_path).convert("RGB")
                examples.append({"image": img, "filename": os.path.basename(img_path)})
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        if not examples:
            print(f"No PNG images found in {args.dir}")
    else:
        print("Loading ADE20K dataset stream...")
        try:
            # Note: This streaming approach only provides the 'image' and 'annotation' keys
            # and may require an Internet connection.
            dataset = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
            # Load 2 examples (as in your original code)
            examples = list(itertools.islice(dataset, 10, 12)) 
        except Exception as e:
            print(f"Error loading dataset: {e}. Check internet connection or Hugging Face token.")
    return examples

# ==============================================================================
# BENCHMARK-SPECIFIC FUNCTIONS
# ==============================================================================

def load_ground_truth(example_data, gt_dir=None):
    """
    Extracts the ground truth mask from the loaded example data.
    
    If loading locally, it relies on the filesystem.
    If loading from ADE20K stream, the GT is in the 'annotation' field.
    """
    if 'annotation' in example_data:
        # Data loaded from ADE20K stream (Annotation is the PIL Image of segmentation map)
        gt_mask_img = example_data['annotation']
        gt_mask = np.array(gt_mask_img) # HxW array of class IDs
        
        # ADE20K class IDs for Wall (3) and Floor (5) - **Verify these IDs**
        ade20k_target_ids = [3, 5] 
        
        # Generate the target binary mask for the GT
        gt_binary_mask = np.isin(gt_mask, ade20k_target_ids).astype(np.uint8)
        
        return gt_binary_mask
        
    elif 'filename' in example_data and gt_dir:
        # Data loaded locally (Need to infer GT file name)
        image_filename = example_data['filename']
        base_name = os.path.splitext(image_filename)[0]
        # Assume GT masks are named identically but are stored as NumPy files (.npy)
        gt_path = os.path.join(gt_dir, base_name + ".npy")
        
        if os.path.exists(gt_path):
            gt_mask = np.load(gt_path)
            # You must define target IDs for your local dataset
            local_target_ids = [3, 5] 
            gt_binary_mask = np.isin(gt_mask, local_target_ids).astype(np.uint8)
            return gt_binary_mask
        else:
            print(f"Warning: Local GT mask not found for {image_filename} at {gt_path}")
            return None
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Wall Designer Tool Benchmark Runner")
    parser.add_argument("--segmentationMethod", type=str, default="mask2former", choices=["mask2former"], help="Method for wall segmentation")
    
    parser.add_argument("--local", action="store_true", help="Use local images from --dir")
    parser.add_argument("--dir", type=str, default=os.path.join(os.path.dirname(__file__), "benchmark_images"), help="Directory containing local images and GT masks")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Path to save the final benchmark CSV")
    
    args = parser.parse_args()

    # 1. Initialize Components
    if args.segmentationMethod == "mask2former":
        segmenter = Mask2FormerSegmentation() 
    
    # 2. Load Data (Uses your original function)
    examples = load_examples(args)
    if not examples:
        return

    benchmark_data = []
    print(f"Starting benchmark for {len(examples)} examples...")

    # 3. The Benchmark Loop 
    for idx, example in enumerate(examples):
        # The 'image' key is always a PIL Image (local or streaming)
        image = example['image']
        # The filename is available if loaded locally; otherwise, use a placeholder
        image_name = example.get('filename', f"ADE20K_Example_{idx+1}.png")
        
        print(f"--- Processing {image_name} ({idx+1}/{len(examples)}) ---")

        # A. Load Ground Truth Mask (Crucial Step)
        # Pass the full example data structure for context-aware GT loading
        gt_binary_mask = load_ground_truth(example, gt_dir=args.dir)
        if gt_binary_mask is None:
            continue # Skip images without GT

        # B. Run Segmentation and Time
        # segmenter.segment() returns (binary_mask_np, runtime)
        pred_binary_mask, runtime_m2f = segmenter.segment(image)
        
        # C. Calculate Metrics
        try:
            # IoU and Dice are calculated using the 0/1 predicted mask and 0/1 GT mask
            iou_m2f = calculate_iou(pred_binary_mask, gt_binary_mask)
            dice_m2f = calculate_dice(pred_binary_mask, gt_binary_mask)
        except Exception as e:
            print(f"Error calculating metrics for {image_name}: {e}")
            iou_m2f = dice_m2f = np.nan
            
        
        # D. Log Results
        result_entry = {
            'image_file': image_name,
            'segmentation_method': 'Mask2Former',
            'iou': iou_m2f,
            'dice_coefficient': dice_m2f,
            'runtime_sec': runtime_m2f,
        }
        benchmark_data.append(result_entry)
        
        print(f"  > IoU: {iou_m2f:.4f} | Dice: {dice_m2f:.4f} | Runtime: {runtime_m2f:.4f}s")


    # 4. Final Aggregation and Save
    if benchmark_data:
        # ... (Aggregation and Saving logic remains the same) ...
        results_df = pd.DataFrame(benchmark_data)
        
        # Calculate aggregate means
        mean_iou = results_df['iou'].mean()
        mean_dice = results_df['dice_coefficient'].mean()
        avg_runtime = results_df['runtime_sec'].mean()
        
        print("\n=============================================")
        print("✅ BENCHMARK SUMMARY (Mask2Former)")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Mean Dice: {mean_dice:.4f}")
        print(f"Average Runtime: {avg_runtime:.4f} seconds")
        print("=============================================")
        
        results_df.to_csv(args.output, index=False)
        print(f"Full results saved to {args.output}")
    else:
        print("No successful benchmark runs to report.")

if __name__ == "__main__":
    main()