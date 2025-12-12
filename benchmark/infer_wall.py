"""
Quick inference script to visualize wall-only Mask2Former predictions on a single image.

Usage:
    python -m benchmark.infer_wall --image /path/to/image.jpg --output ./inference_debug
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation


def load_model(model_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {model_id} on {device} ...")
    processor = Mask2FormerImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(device)
    model.eval()

    id2label = model.config.id2label
    wall_ids = [i for i, lbl in id2label.items() if "wall" in lbl.lower()]
    print(f"Wall IDs: {wall_ids} ({[id2label[i] for i in wall_ids]})")
    return processor, model, wall_ids, device


def run_inference(
    image_path: str,
    processor,
    model,
    wall_ids,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [image.size[::-1]]  # (H, W)
    prediction = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
    prediction_np = prediction.cpu().numpy()

    # Wall-only binary mask
    wall_mask = np.isin(prediction_np, wall_ids).astype(np.uint8)
    return prediction_np, wall_mask


def save_outputs(
    image_path: str,
    prediction_np: np.ndarray,
    wall_mask: np.ndarray,
    output_dir: str,
    palette: np.ndarray,
):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    tag_base = f"{base}_inf"

    # Save raw label map (colorized)
    color_map = palette[prediction_np % len(palette)]
    Image.fromarray(color_map.astype(np.uint8)).save(os.path.join(output_dir, f"{tag_base}_seg_color.png"))

    # Save raw label map as npy
    np.save(os.path.join(output_dir, f"{tag_base}_seg.npy"), prediction_np)

    # Save wall binary mask
    Image.fromarray((wall_mask * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{tag_base}_wall_mask.png"))

    # Overlay wall mask on the original image for quick visual
    image = Image.open(image_path).convert("RGB").resize((wall_mask.shape[1], wall_mask.shape[0]))
    base_rgba = image.convert("RGBA")
    mask_alpha = Image.fromarray((wall_mask * 180).astype(np.uint8), mode="L")  # semi-opacity mask
    # Apply mask as alpha channel over a green highlight
    highlight = Image.new("RGBA", base_rgba.size, (0, 255, 0, 0))
    highlight.putalpha(mask_alpha)
    overlay = Image.alpha_composite(base_rgba, highlight)
    overlay.save(os.path.join(output_dir, f"{tag_base}_wall_overlay.png"))


def get_palette(num_classes: int = 151) -> np.ndarray:
    np.random.seed(42)
    return np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Mask2Former wall-only inference/visualization")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.normpath(os.path.join(os.path.dirname(__file__), "inference_output")),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/mask2former-swin-large-ade-semantic",
        help="HF model id",
    )
    args = parser.parse_args()

    processor, model, wall_ids, device = load_model(args.model_id)
    prediction_np, wall_mask = run_inference(args.image, processor, model, wall_ids, device)
    save_outputs(args.image, prediction_np, wall_mask, args.output, palette=get_palette())
    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    main()

