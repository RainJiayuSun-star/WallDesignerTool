import argparse
import os
import sys
import glob
import numpy as np
from PIL import Image
import itertools
from datasets import load_dataset  # Required for ADE20K streaming
import pandas as pd
import time  # Required for runtime metric

# ------------------------------------------------------------------------------
# Import modular components
# Support both `python -m benchmark.main` and `python benchmark/main.py`
# ------------------------------------------------------------------------------
if __package__ is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from segmentation import Mask2FormerSegmentation  # type: ignore
    from metrics import calculate_iou, calculate_dice, calculate_boundary_fscore  # type: ignore
else:
    from .segmentation import Mask2FormerSegmentation
    from .metrics import calculate_iou, calculate_dice, calculate_boundary_fscore

# Clustering placeholder (not yet implemented for this benchmark focus)
# from clustering import ClusteringSegmentation


# ==============================================================================
# ADE20K DATA DECODING UTILITIES
# ==============================================================================

def get_ade20k_palette():
    """Returns a random palette for visualizing the 151 classes."""
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(151, 3), dtype=np.uint8)
    return palette

def decode_ade20k_seg_png(seg_img):
    """
    Decodes the color-coded ADE20K segmentation PNG (e.g., _seg.png)
    into a semantic class ID map. Local ADE20K _seg.png files may be
    single-channel label maps; the streamed dataset uses R + G * 256 encoding.
    """
    # If already single-channel label image, return directly
    if seg_img.mode in ("L", "I"):
        return np.array(seg_img, dtype=np.uint16)

    # Fallback for color-encoded masks: ADE20K encoding Class ID = R + G * 256
    seg_map_np = np.array(seg_img, dtype=np.uint16)
    R = seg_map_np[:, :, 0]
    G = seg_map_np[:, :, 1]
    semantic_map = R + G * 256
    return semantic_map.astype(np.uint16)


def load_examples(args):
    """
    Loads images either locally (looking for .jpg) or from the streamed ADE20K dataset.
    """
    examples = []
    if args.local:
        print(f"Loading local images from {args.dir}...")
        if not os.path.exists(args.dir):
            print(f"Directory {args.dir} does not exist.")
            return []
        
        # Look for .jpg files based on ADE20K structure
        image_files = glob.glob(os.path.join(args.dir, "*.jpg"))
        image_files.sort()
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                examples.append({"image": img, "filename": os.path.basename(img_path)})
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        if not examples:
            print(f"No JPG images found in {args.dir}")
    else:
        print("Loading ADE20K dataset stream...")
        try:
            # Load 2 examples (as in your original code)
            dataset = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
            examples = list(itertools.islice(dataset, 10, 12)) 
        except Exception as e:
            print(f"Error loading dataset: {e}. Check internet connection.")
    return examples


def load_ground_truth(example_data, gt_dir=None):
    """
    Extracts the ground truth mask, handling both streamed and local (_seg.png) formats.
    
    Returns: Binary mask (1=Wall, 0=Other).
    """
    # ADE20K class IDs for wall-only evaluation: wall=0
    ade20k_wall_id = 0

    if 'annotation' in example_data:
        # 1. Data loaded from ADE20K stream
        gt_mask_img = example_data['annotation']
        gt_mask = decode_ade20k_seg_png(gt_mask_img)
        
    elif 'filename' in example_data and gt_dir:
        # 2. Data loaded locally (load and decode the _seg.png file)
        image_filename = example_data['filename']
        base_name = os.path.splitext(image_filename)[0]

        # Try instance-mask-based wall union first (from JSON metadata)
        json_path = os.path.join(gt_dir, base_name + ".json")
        if os.path.exists(json_path):
            try:
                import json

                with open(json_path, "r") as f:
                    meta = json.load(f)
                wall_mask = None
                objects = meta.get("annotation", {}).get("object", [])
                for obj in objects:
                    name = (obj.get("name") or "").lower()
                    raw_name = (obj.get("raw_name") or "").lower()
                    if "wall" not in name and "wall" not in raw_name:
                        continue
                    mask_rel = obj.get("instance_mask")
                    if not mask_rel:
                        continue
                    mask_path = os.path.join(gt_dir, mask_rel)
                    if not os.path.exists(mask_path):
                        continue
                    mask_img = Image.open(mask_path).convert("L")
                    mask_arr = (np.array(mask_img) > 0).astype(np.uint8)
                    if wall_mask is None:
                        wall_mask = mask_arr
                    else:
                        wall_mask = np.logical_or(wall_mask, mask_arr).astype(np.uint8)
                if wall_mask is not None:
                    return wall_mask
            except Exception as e:
                print(f"Warning: Failed to build instance wall mask for {image_filename}: {e}")

        # Fallback: semantic seg png
        gt_path = os.path.join(gt_dir, base_name + "_seg.png")
        if os.path.exists(gt_path):
            gt_mask_img = Image.open(gt_path).convert("RGB")
            gt_mask = decode_ade20k_seg_png(gt_mask_img)
        else:
            print(f"Warning: Local GT mask not found for {image_filename} at {gt_path}")
            return None
    else:
        return None
    
    # Final step: Convert the semantic class ID map into a binary mask (wall only)
    gt_binary_mask = (gt_mask == ade20k_wall_id).astype(np.uint8)
    return gt_binary_mask


# ==============================================================================
# MAIN BENCHMARK EXECUTION LOGIC
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Wall Designer Tool Benchmark Runner")
    parser.add_argument("--segmentationMethod", type=str, default="mask2former", choices=["mask2former"], help="Method for wall segmentation")
    
    # Default to using the ADE20K child's room subset under validation/home_or_hotel
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_childs_room_dir = os.path.normpath(
        os.path.join(script_dir, "..", "data", "ADE20K", "validation", "home_or_hotel", "childs_room")
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=True,
        help="Use local images from --dir (defaults to ADE20K validation/home_or_hotel/childs_room)",
    )
    parser.add_argument("--dir", type=str, default=default_childs_room_dir, help="Directory containing local images and GT masks")
    default_output = os.path.normpath(os.path.join(script_dir, "benchmark_output", "benchmark_results_childs_room.csv"))
    parser.add_argument("--output", type=str, default=default_output, help="Path to save the final benchmark CSV")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print label stats for GT/pred and save binary masks (PNG) for inspection",
    )
    
    args = parser.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    debug_dir = None
    if args.debug:
        debug_dir = os.path.join(out_dir, "debug_masks")
        os.makedirs(debug_dir, exist_ok=True)

    # 1. Initialize Components
    if args.segmentationMethod == "mask2former":
        segmenter = Mask2FormerSegmentation() 
    # Add clustering initialization here later:
    # elif args.segmentationMethod == "clustering":
    #    segmenter = ClusteringSegmentation()

    # 2. Load Data 
    examples = load_examples(args)
    if not examples:
        return

    benchmark_data = []
    print(f"Starting benchmark for {len(examples)} examples...")

    # 3. The Benchmark Loop 
    for idx, example in enumerate(examples):
        image = example['image']
        image_name = example.get('filename', f"ADE20K_Example_{idx+1}.png")
        
        print(f"--- Processing {image_name} ({idx+1}/{len(examples)}) ---")

        # A. Load Ground Truth Mask 
        gt_binary_mask = load_ground_truth(example, gt_dir=args.dir)
        if gt_binary_mask is None:
            continue

        # B. Run Segmentation and Time
        # segmenter.segment() returns (binary_mask_np, runtime)
        pred_binary_mask, runtime_m2f = segmenter.segment(image)
        
        if args.debug:
            if idx == 0:
                print(f"GT unique labels for {image_name}: {np.unique(gt_binary_mask)}")
                print(f"Pred unique labels for {image_name}: {np.unique(pred_binary_mask)}")
            base_name = os.path.splitext(image_name)[0]
            # Save binary masks for quick visual inspection
            if debug_dir:
                gt_save = os.path.join(debug_dir, f"{base_name}_gt_mask.png")
                pred_save = os.path.join(debug_dir, f"{base_name}_pred_mask.png")
                Image.fromarray((gt_binary_mask * 255).astype(np.uint8)).save(gt_save)
                Image.fromarray((pred_binary_mask * 255).astype(np.uint8)).save(pred_save)
        
        # C. Calculate Metrics
        try:
            # IoU and Dice are calculated using the 0/1 predicted mask and 0/1 GT mask
            iou_m2f = calculate_iou(pred_binary_mask, gt_binary_mask)
            dice_m2f = calculate_dice(pred_binary_mask, gt_binary_mask)
            boundary_fscore_m2f = calculate_boundary_fscore(pred_binary_mask, gt_binary_mask, tolerance=2)
        except Exception as e:
            print(f"Error calculating metrics for {image_name}: {e}")
            iou_m2f = dice_m2f = boundary_fscore_m2f = np.nan
            
        
        # D. Log Results
        result_entry = {
            'image_file': image_name,
            'segmentation_method': 'Mask2Former',
            'iou': iou_m2f,
            'dice_coefficient': dice_m2f,
            'boundary_fscore': boundary_fscore_m2f,
            'runtime_sec': runtime_m2f,
            # Add placeholders for clustering results here when implemented
        }
        benchmark_data.append(result_entry)
        
        print(f"  > IoU: {iou_m2f:.4f} | Dice: {dice_m2f:.4f} | Boundary F-Score: {boundary_fscore_m2f:.4f} | Runtime: {runtime_m2f:.4f}s")


    # 4. Final Aggregation and Save
    if benchmark_data:
        results_df = pd.DataFrame(benchmark_data)
        
        # Calculate aggregate means
        mean_iou = results_df['iou'].mean()
        mean_dice = results_df['dice_coefficient'].mean()
        mean_boundary_fscore = results_df['boundary_fscore'].mean()
        avg_runtime = results_df['runtime_sec'].mean()
        
        print("\n=============================================")
        print("✅ BENCHMARK SUMMARY (Mask2Former)")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Mean Dice: {mean_dice:.4f}")
        print(f"Mean Boundary F-Score: {mean_boundary_fscore:.4f}")
        print(f"Average Runtime: {avg_runtime:.4f} seconds")
        print("=============================================")
        
        results_df.to_csv(args.output, index=False)
        print(f"Full results saved to {args.output}")
    else:
        print("No successful benchmark runs to report.")

if __name__ == "__main__":
    main()