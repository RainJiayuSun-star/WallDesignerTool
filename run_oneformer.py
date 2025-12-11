import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
import os
import glob
import sys
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from datasets import load_dataset
from PIL import Image


def get_ade20k_palette():
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(151, 3), dtype=np.uint8)
    return palette


def visualize_prediction(image, pred_map, ax, title="Prediction", target_ids=None):
    palette = get_ade20k_palette()
    color_seg = np.zeros((pred_map.shape[0], pred_map.shape[1], 3), dtype=np.uint8)

    unique_labels = np.unique(pred_map)
    for label_id in unique_labels:
        if target_ids is not None and label_id not in target_ids:
            continue
        # Map label_id to palette range if valid
        if 0 <= label_id < len(palette):
            color_seg[pred_map == label_id] = palette[label_id]

    ax.imshow(image)
    ax.imshow(color_seg, alpha=0.5)
    ax.set_title(title)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Run OneFormer segmentation.")
    parser.add_argument("--local", action="store_true", help="Use local images")
    parser.add_argument(
        "--dir",
        type=str,
        default="src/ourSet",
        help="Image directory (default: src/ourSet)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="shi-labs/oneformer_ade20k_swin_large",
        help="Model ID",
    )
    parser.add_argument(
        "--output", type=str, default="oneformer_results.png", help="Output filename"
    )
    args = parser.parse_args()

    # Determine device
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    # MPS disabled for OneFormer due to float64 issues
    # elif torch.backends.mps.is_available():
    #     device_name = "mps"

    device = torch.device(device_name)
    print(f"Using device: {device}")

    # --- Load Data ---
    examples = []
    if args.local:
        print(f"Loading local images from {args.dir}...")
        if os.path.exists(args.dir):
            # Support multiple image formats
            extensions = ["*.png", "*.jpg", "*.jpeg"]
            image_files = []
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(args.dir, ext)))
                # Also try case-insensitive
                image_files.extend(glob.glob(os.path.join(args.dir, ext.upper())))

            image_files = sorted(list(set(image_files)))  # unique and sorted

            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    examples.append(
                        {"image": img, "filename": os.path.basename(img_path)}
                    )
                    print(f"Loaded {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        else:
            print(f"Directory {args.dir} does not exist.")
        num_examples = len(examples)
        print(f"Found {num_examples} images.")
    else:
        print("Loading ADE20K dataset stream (validation split)...")
        try:
            # Note: streaming=True avoids downloading the whole dataset
            dataset = load_dataset(
                "scene_parse_150",
                split="validation",
                streaming=True,
                trust_remote_code=True,
            )
            # Take a few examples
            examples_iter = itertools.islice(dataset, 0, 3)
            examples = []
            for item in examples_iter:
                if "image" in item:
                    examples.append(
                        {
                            "image": item["image"].convert("RGB"),
                            "filename": "ADE20K_sample",
                        }
                    )
            num_examples = len(examples)
            print(f"Loaded {num_examples} examples from ADE20K stream.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Try running with --local to use local images.")
            return

    if num_examples == 0:
        print("No images found to process.")
        return

    # --- Load OneFormer ---
    print(f"Loading model: {args.model}...")
    try:
        processor = OneFormerProcessor.from_pretrained(args.model)
        model = OneFormerForUniversalSegmentation.from_pretrained(args.model).to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model {args.model}: {e}")
        print(
            "If you are running out of memory/disk, try a smaller model like 'shi-labs/oneformer_ade20k_swin_tiny'"
        )
        return

    # Identify Wall ID
    # In ADE20K, 'wall' is usually index 0, but we check dynamically
    wall_ids = [
        id for id, label in model.config.id2label.items() if "wall" in label.lower()
    ]
    print(f"Target Wall IDs: {wall_ids}")

    # --- Setup Plot ---
    # Adjust figure size based on number of examples
    fig, axes = plt.subplots(num_examples, 3, figsize=(18, 6 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    print("Running inference...")
    for idx, example in enumerate(examples):
        image = example["image"]
        filename = example.get("filename", f"Image {idx+1}")

        # OneFormer requires explicit task input
        # We specify "semantic" to get the class map
        # Handle inputs based on device
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        prediction = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        prediction_np = prediction.cpu().numpy()

        # --- Visualization ---
        # 1. Original
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f"Original: {filename}")
        axes[idx, 0].axis("off")

        # 2. Full Segmentation
        visualize_prediction(
            image, prediction_np, axes[idx, 1], title="OneFormer Semantic"
        )

        # 3. Walls Only
        visualize_prediction(
            image,
            prediction_np,
            axes[idx, 2],
            title="Extracted Walls",
            target_ids=wall_ids,
        )

    plt.tight_layout()
    print(f"Saving results to {args.output}...")
    plt.savefig(args.output)
    # plt.show() # Commented out for headless environments, remove if running locally with GUI


if __name__ == "__main__":
    main()
