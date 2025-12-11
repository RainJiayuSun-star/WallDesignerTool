import argparse
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import itertools
import time
import pandas as pd
from datasets import load_dataset

# Import metrics from benchmark
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "benchmark"))
from metrics import calculate_iou, calculate_dice

# Import modular components
from segmentation import (
    Mask2FormerSegmentation,
    OneFormerSegmentation,
    SegFormerSegmentation,
    NyuSegmentation,
)
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
    ax.axis("off")


def load_examples(args):
    examples = []

    if args.use_ade20k:
        # Load from ADE20K validation set
        print("Loading ADE20K validation set...")
        try:
            dataset = load_dataset(
                "1aurent/ADE20K",
                split="validation",
                streaming=True,
            )
            # Load multiple examples for better comparison
            hf_examples = list(
                itertools.islice(
                    dataset,
                    args.ade20k_start_idx,
                    args.ade20k_start_idx + args.ade20k_count,
                )
            )
            for i, item in enumerate(hf_examples):
                if "image" in item and "annotation" in item:
                    # Handle image conversion
                    if isinstance(item["image"], Image.Image):
                        img = item["image"].convert("RGB")
                    else:
                        img = Image.fromarray(np.array(item["image"])).convert("RGB")

                    # Handle annotation
                    if isinstance(item["annotation"], Image.Image):
                        annotation = item["annotation"]
                    else:
                        annotation = Image.fromarray(np.array(item["annotation"]))

                    examples.append(
                        {
                            "image": img,
                            "annotation": annotation,  # Ground truth segmentation map
                            "filename": f"ADE20K_Val_{args.ade20k_start_idx + i}",
                            "source": "ADE20K",
                        }
                    )
            print(f"✅ Loaded {len(examples)} examples from ADE20K validation set")
        except Exception as e:
            print(f"❌ Error loading ADE20K dataset: {e}")
            print("Falling back to local images...")
            args.use_ade20k = False

    # Load local images if not using ADE20K or as fallback
    if not args.use_ade20k:
        print(f"Loading local images from {args.dir}...")
        if os.path.exists(args.dir):
            image_files = glob.glob(os.path.join(args.dir, "*.png"))
            image_files.extend(glob.glob(os.path.join(args.dir, "*.jpg")))
            image_files.sort()
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    examples.append(
                        {
                            "image": img,
                            "filename": os.path.basename(img_path),
                            "source": "Local",
                        }
                    )
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        else:
            print(f"Directory {args.dir} does not exist.")

    return examples


def run_pipeline_for_model(
    model_name, segmenter, examples, texture_path, output_filename
):
    print(f"\n--- Running Pipeline with {model_name} ---")

    splitter = WallRefinerSplitting()
    mapper = MaskedPerspectiveMapping(texture_path)

    num_examples = len(examples)
    if num_examples == 0:
        print("No examples to process.")
        return

    # Create figure: Rows = Models, Cols = (Original, Segmentation, Texture Result)
    # We will save ONE large chart for ALL models for the FIRST image.
    # OR, better: For each image, show all models.
    pass


def run_pipeline_for_all_models(models, examples, texture_path, output_filename):
    print(f"\n--- Running Pipeline Comparison with Metrics ---")

    splitter = WallRefinerSplitting()
    mapper = MaskedPerspectiveMapping(texture_path)

    num_examples = len(examples)
    num_models = len(models)

    if num_examples == 0:
        print("No examples to process.")
        return

    # Store metrics for all models and examples
    all_metrics = []

    # To keep the chart readable, we might want one chart per image.
    # Chart Structure:
    # Row 1: Original Image
    # Row 2..N+1: Model Result (Segmentation + Texture)

    for idx, example in enumerate(examples):
        image = example["image"]
        image_np = np.array(image)
        filename = example["filename"]
        print(f"Processing {filename}...")

        # Check for ground truth (from ADE20K stream)
        gt_mask = None
        if "annotation" in example:
            # ADE20K annotation available
            gt_mask_img = example["annotation"]
            if isinstance(gt_mask_img, Image.Image):
                gt_mask_array = np.array(gt_mask_img)
            else:
                gt_mask_array = np.array(gt_mask_img)

            # ADE20K class ID for wall is 0 (based on benchmark code)
            # Note: ADE20K uses 0-indexed class IDs where 0 = wall
            gt_mask = np.isin(gt_mask_array, [0]).astype(np.uint8)
            print(
                f"    Ground truth loaded: {np.sum(gt_mask > 0)} wall pixels ({np.sum(gt_mask > 0) / gt_mask.size * 100:.2f}%)"
            )

        # Figure: 1 row per model, 3 columns (Model Name, Segmentation, Texture Result)
        # Plus a row for Original? Or just put Original on top?

        fig, axes = plt.subplots(num_models + 1, 3, figsize=(15, 5 * (num_models + 1)))

        # Row 0: Original Image (Spanning all cols or just repeated/centered)
        # Let's just put Original in [0, 1]
        for c in range(3):
            axes[0, c].axis("off")

        axes[0, 1].imshow(image)
        axes[0, 1].set_title(f"Original: {filename}")

        for m_idx, (model_name, segmenter) in enumerate(models):
            row = m_idx + 1
            print(f"  Running {model_name}...")

            # A. Segmentation with timing
            start_time = time.time()
            try:
                pred_map, wall_ids = segmenter.segment(image)
                runtime = time.time() - start_time
            except Exception as e:
                print(f"    Segmentation failed: {e}")
                pred_map = np.zeros(image.size[::-1], dtype=int)
                wall_ids = []
                runtime = np.nan

            # B. Create binary wall mask
            full_wall_mask = np.zeros(pred_map.shape, dtype=np.uint8)
            for w_id in wall_ids:
                full_wall_mask[pred_map == w_id] = 1

            # Calculate wall coverage percentage
            wall_coverage = np.sum(full_wall_mask > 0) / full_wall_mask.size * 100

            # Calculate metrics if ground truth is available
            iou_score = np.nan
            dice_score = np.nan
            if gt_mask is not None:
                try:
                    # Ensure same shape
                    if gt_mask.shape != full_wall_mask.shape:
                        gt_resized = cv2.resize(
                            gt_mask.astype(np.uint8),
                            (full_wall_mask.shape[1], full_wall_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        gt_mask = gt_resized

                    iou_score = calculate_iou(full_wall_mask, gt_mask)
                    dice_score = calculate_dice(full_wall_mask, gt_mask)
                except Exception as e:
                    print(f"    Metrics calculation failed: {e}")

            # Store metrics
            all_metrics.append(
                {
                    "image": filename,
                    "model": model_name,
                    "iou": iou_score,
                    "dice": dice_score,
                    "wall_coverage_pct": wall_coverage,
                    "runtime_sec": runtime,
                    "num_wall_segments": len(wall_ids),
                }
            )

            # B. Splitting
            segments, polygons = splitter.split(full_wall_mask, image_np)

            # C. Mapping
            full_wall_mask_255 = (full_wall_mask * 255).astype(np.uint8)
            try:
                textured_image = mapper.apply(
                    image_np, polygons, full_mask=full_wall_mask_255
                )
            except Exception as e:
                print(f"    Mapping failed: {e}")
                textured_image = image_np

            # Visualization
            # Col 0: Model Name + Metrics
            metrics_text = f"{model_name}\n"
            metrics_text += f"Runtime: {runtime:.2f}s\n"
            metrics_text += f"Coverage: {wall_coverage:.1f}%\n"
            if not np.isnan(iou_score):
                metrics_text += f"IoU: {iou_score:.3f}\n"
                metrics_text += f"Dice: {dice_score:.3f}\n"
            metrics_text += f"Walls: {len(segments)}"

            axes[row, 0].text(
                0.5,
                0.5,
                metrics_text,
                fontsize=11,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            axes[row, 0].axis("off")

            # Col 1: Segmentation
            seg_title = "Segmentation"
            if not np.isnan(iou_score):
                seg_title += f"\nIoU: {iou_score:.3f}"
            visualize_prediction(
                image, pred_map, axes[row, 1], title=seg_title, target_ids=wall_ids
            )

            # Col 2: Result
            axes[row, 2].imshow(textured_image)
            axes[row, 2].set_title(f"Texture Applied\n({len(segments)} segments)")
            axes[row, 2].axis("off")

        plt.tight_layout()
        out_file = f"comparison_{filename}.png"
        print(f"Saving comparison to {out_file}")
        plt.savefig(out_file)
        plt.close(fig)

    # Save metrics to CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_csv = "segmentation_metrics_comparison.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"\n✅ Metrics saved to {metrics_csv}")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("METRICS SUMMARY")
        print("=" * 60)
        for model_name in metrics_df["model"].unique():
            model_data = metrics_df[metrics_df["model"] == model_name]
            print(f"\n{model_name}:")
            if not model_data["iou"].isna().all():
                print(
                    f"  Mean IoU: {model_data['iou'].mean():.4f} ± {model_data['iou'].std():.4f}"
                )
                print(
                    f"  Mean Dice: {model_data['dice'].mean():.4f} ± {model_data['dice'].std():.4f}"
                )
            print(
                f"  Mean Wall Coverage: {model_data['wall_coverage_pct'].mean():.2f}% ± {model_data['wall_coverage_pct'].std():.2f}%"
            )
            print(
                f"  Mean Runtime: {model_data['runtime_sec'].mean():.4f}s ± {model_data['runtime_sec'].std():.4f}s"
            )
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Chart Generator")
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "ourSet"),
        help="Directory containing local images",
    )
    parser.add_argument(
        "--texture",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "wood_texture.jpg"),
        help="Path to texture file",
    )
    parser.add_argument(
        "--use_ade20k",
        action="store_true",
        help="Use ADE20K validation set instead of local images",
    )
    parser.add_argument(
        "--ade20k_start_idx",
        type=int,
        default=0,
        help="Starting index for ADE20K validation set (default: 0)",
    )
    parser.add_argument(
        "--ade20k_count",
        type=int,
        default=5,
        help="Number of examples to load from ADE20K (default: 5)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.texture):
        # Fallback to dummy
        print(f"Texture {args.texture} not found, checking dummy...")
        args.texture = os.path.join(os.path.dirname(__file__), "dummy_texture.png")
        if not os.path.exists(args.texture):
            print("No texture found.")
            return

    # 1. Load Data (Both Local and HF)
    examples = load_examples(args)

    # 2. Define Models to Compare - All segmentation models from segmentation.py
    models = [
        ("Mask2Former (ADE20K)", Mask2FormerSegmentation()),
        ("OneFormer (ADE20K)", OneFormerSegmentation()),
        ("SegFormer (ADE20K)", SegFormerSegmentation()),
        ("SegFormer (NYU)", NyuSegmentation()),
    ]

    # 3. Run Pipeline
    run_pipeline_for_all_models(models, examples, args.texture, None)

    print("All comparison charts generated.")


if __name__ == "__main__":
    main()
