import argparse
import os
import sys
import glob
import numpy as np
from PIL import Image
import pandas as pd
import time

# Support running as module or script
if __package__ is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from segmentation import (  # type: ignore
        Mask2FormerSegmentation,
        OneFormerSegmentation,
        SegFormerSegmentation,
        NyuSegmentation,
    )
    from metrics import calculate_iou, calculate_dice, calculate_boundary_fscore  # type: ignore
    from main import load_ground_truth  # type: ignore
else:
    from .segmentation import (
        Mask2FormerSegmentation,
        OneFormerSegmentation,
        SegFormerSegmentation,
        NyuSegmentation,
    )
    from .metrics import calculate_iou, calculate_dice, calculate_boundary_fscore
    from .main import load_ground_truth


def load_examples_recursive(root_dir: str):
    """Recursively load all .jpg images under root_dir."""
    image_files = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
    image_files.sort()
    examples = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            examples.append({"image": img, "filename": os.path.basename(img_path), "fullpath": img_path})
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    if not examples:
        print(f"No JPG images found under {root_dir}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Wall-only segmentation Benchmark on ADE20K validation (subset or full)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "ADE20K", "validation"))
    default_output = os.path.normpath(os.path.join(script_dir, "benchmark_output", "benchmark_results_ade20k_val.csv"))
    parser.add_argument(
        "--segmentationMethod",
        type=str,
        default="mask2former",
        choices=["mask2former", "oneformer", "segformer", "nyu"],
        help="Method for wall segmentation (single model fallback if --models not provided)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to run (mask2former, oneformer, segformer, nyu). Overrides --segmentationMethod.",
    )
    parser.add_argument(
        "--run_all_models",
        action="store_true",
        help="Run all supported models sequentially.",
    )
    parser.add_argument("--dir", type=str, default=default_dir, help="Root directory containing ADE20K validation data")
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Optional subdirectory under validation to limit the run (e.g., cultural, home_or_hotel)",
    )
    parser.add_argument("--output", type=str, default=default_output, help="Path to save the final benchmark CSV")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on number of images to process")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print label stats for first image and save binary masks (PNG) for inspection",
    )
    args = parser.parse_args()

    # Determine models to run
    supported_models = {
        "mask2former": Mask2FormerSegmentation,
        "oneformer": OneFormerSegmentation,
        "segformer": SegFormerSegmentation,
        "nyu": NyuSegmentation,
    }
    if args.run_all_models:
        model_list = list(supported_models.keys())
    elif args.models:
        model_list = [m.lower() for m in args.models if m.lower() in supported_models]
        if not model_list:
            model_list = [args.segmentationMethod]
    else:
        model_list = [args.segmentationMethod]

    # Derive output path including subset and model tag if using default
    if args.output == default_output:
        base_dir = os.path.dirname(default_output)
        subset_tag = args.subdir if args.subdir else "ade20k_val"
        if len(model_list) == 1:
            base_name = f"benchmark_results_{subset_tag}_{model_list[0]}.csv"
        else:
            base_name = f"benchmark_results_{subset_tag}_multi.csv"
        args.output = os.path.join(base_dir, base_name)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    debug_dir_base = None
    if args.debug:
        debug_dir_base = os.path.join(out_dir, "debug_masks_full")
        os.makedirs(debug_dir_base, exist_ok=True)

    # Resolve target root (optionally limited to a single top-level subdir)
    target_root = args.dir
    if args.subdir:
        target_root = os.path.join(args.dir, args.subdir)

    # Load data recursively
    examples = load_examples_recursive(target_root)
    if args.max_images:
        examples = examples[: args.max_images]
    if not examples:
        return

    benchmark_data = []
    print(f"Starting benchmark for {len(examples)} examples across models: {model_list}")

    for model_name in model_list:
        print(f"\n===== Running model: {model_name} =====")
        segmenter_cls = supported_models.get(model_name)
        if segmenter_cls is None:
            print(f"Unknown model {model_name}, skipping.")
            continue
        segmenter = segmenter_cls()

        debug_dir = None
        if debug_dir_base:
            debug_dir = os.path.join(debug_dir_base, model_name)
            os.makedirs(debug_dir, exist_ok=True)

        for idx, example in enumerate(examples):
            image = example["image"]
            image_name = example.get("filename", f"ADE20K_Example_{idx+1}.png")
            fullpath = example.get("fullpath", "")

            print(f"--- Processing {image_name} ({idx+1}/{len(examples)}) ---")

            # Load GT (instance-union preferred, semantic fallback handled inside)
            gt_dir = os.path.dirname(fullpath) if fullpath else args.dir
            gt_binary_mask = load_ground_truth({"filename": image_name}, gt_dir=gt_dir)
            if gt_binary_mask is None:
                continue

            # Segmentation + timing
            pred_binary_mask, runtime_model = segmenter.segment(image)

            if args.debug and idx == 0:
                print(f"GT unique labels for {image_name}: {np.unique(gt_binary_mask)}")
                print(f"Pred unique labels for {image_name}: {np.unique(pred_binary_mask)}")

            if args.debug and debug_dir:
                base_name = os.path.splitext(image_name)[0]
                gt_save = os.path.join(debug_dir, f"{base_name}_gt_mask.png")
                pred_save = os.path.join(debug_dir, f"{base_name}_pred_mask.png")
                Image.fromarray((gt_binary_mask * 255).astype(np.uint8)).save(gt_save)
                Image.fromarray((pred_binary_mask * 255).astype(np.uint8)).save(pred_save)

            # Metrics
            try:
                iou_score = calculate_iou(pred_binary_mask, gt_binary_mask)
                dice_score = calculate_dice(pred_binary_mask, gt_binary_mask)
                boundary_fscore = calculate_boundary_fscore(pred_binary_mask, gt_binary_mask, tolerance=2)
            except Exception as e:
                print(f"Error calculating metrics for {image_name}: {e}")
                iou_score = dice_score = boundary_fscore = np.nan

            # Log
            result_entry = {
                "image_file": image_name,
                "model_name": model_name,
                "segmentation_method": model_name,
                "iou": iou_score,
                "dice_coefficient": dice_score,
                "boundary_fscore": boundary_fscore,
                "runtime_sec": runtime_model,
            }
            benchmark_data.append(result_entry)

            if not np.isnan(iou_score) and not np.isnan(dice_score):
                boundary_str = f"{boundary_fscore:.4f}" if not np.isnan(boundary_fscore) else "N/A"
                print(f"  > IoU: {iou_score:.4f} | Dice: {dice_score:.4f} | Boundary F-Score: {boundary_str} | Runtime: {runtime_model:.4f}s")

    # Aggregate
    if benchmark_data:
        results_df = pd.DataFrame(benchmark_data)
        # Per-model summary
        grouped = results_df.groupby("model_name").agg(
            mean_iou=("iou", "mean"),
            mean_dice=("dice_coefficient", "mean"),
            mean_boundary_fscore=("boundary_fscore", "mean"),
            avg_runtime=("runtime_sec", "mean"),
            count=("image_file", "count"),
        )

        print("\n=============================================")
        print("✅ BENCHMARK SUMMARY (wall-only)")
        for model_name, row in grouped.iterrows():
            print(
                f"{model_name}: Mean IoU={row.mean_iou:.4f} | Mean Dice={row.mean_dice:.4f} | "
                f"Mean Boundary F-Score={row.mean_boundary_fscore:.4f} | "
                f"Avg Runtime={row.avg_runtime:.4f}s over {int(row['count'])} images"
            )
        print("=============================================")

        results_df.to_csv(args.output, index=False)
        print(f"Full results saved to {args.output}")
    else:
        print("No successful benchmark runs to report.")


if __name__ == "__main__":
    main()

