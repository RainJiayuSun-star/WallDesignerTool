import argparse
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(__file__))
from segmentation import Mask2FormerSegmentation, OneFormerSegmentation


def load_texture(texture_path):
    """Load texture image"""
    if not os.path.exists(texture_path):
        print(f"Texture not found at {texture_path}, creating dummy texture...")
        # Create a simple checkerboard pattern as dummy texture
        texture = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    texture[i : i + 32, j : j + 32] = [139, 69, 19]  # Brown
                else:
                    texture[i : i + 32, j : j + 32] = [160, 82, 45]  # Saddle brown
        return texture
    else:
        texture_img = Image.open(texture_path).convert("RGB")
        return np.array(texture_img)


def create_affine_transform(
    image_shape, texture_shape, scale=1.0, rotation=0, translation=(0, 0)
):
    """
    Create an affine transformation matrix.

    Args:
        image_shape: (height, width) of target image
        texture_shape: (height, width) of texture
        scale: Scaling factor
        rotation: Rotation angle in degrees
        translation: (tx, ty) translation in pixels

    Returns:
        2x3 affine transformation matrix
    """
    h, w = image_shape[:2]
    th, tw = texture_shape[:2]

    # Center of the image
    center_x, center_y = w / 2, h / 2

    # Create rotation matrix
    angle_rad = np.radians(rotation)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Scale the texture to fit better
    # Scale texture to cover a reasonable portion of the image
    scale_x = (w * scale) / tw
    scale_y = (h * scale) / th

    # Build transformation matrix
    # Order: Scale -> Rotate -> Translate
    M = np.array(
        [
            [
                scale_x * cos_a,
                -scale_x * sin_a,
                center_x
                + translation[0]
                - (tw * scale_x * cos_a) / 2
                + (th * scale_x * sin_a) / 2,
            ],
            [
                scale_x * sin_a,
                scale_x * cos_a,
                center_y
                + translation[1]
                - (tw * scale_x * sin_a) / 2
                - (th * scale_x * cos_a) / 2,
            ],
        ],
        dtype=np.float32,
    )

    return M


def affine_warp_texture(texture, image_shape, transform_matrix):
    """
    Warp texture using affine transformation.

    Args:
        texture: Input texture image (H, W, 3)
        image_shape: Target image shape (H, W) or (H, W, 3)
        transform_matrix: 2x3 affine transformation matrix

    Returns:
        Warped texture with same shape as target image
    """
    if len(image_shape) == 3:
        h, w = image_shape[:2]
    else:
        h, w = image_shape

    # Warp the texture
    warped = cv2.warpAffine(
        texture,
        transform_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return warped


def alpha_blend(base_image, overlay_image, mask, alpha=0.7):
    """
    Blend overlay image onto base image using mask and alpha transparency.

    Args:
        base_image: Base image (H, W, 3) uint8
        overlay_image: Overlay image to blend (H, W, 3) uint8
        mask: Binary mask (H, W) where 1 = apply overlay, 0 = keep base (0-1 or 0-255)
        alpha: Blending strength (0.0 to 1.0)

    Returns:
        Blended image (H, W, 3) uint8
    """
    # Normalize mask to 0-1 range
    if mask.dtype != np.float32 and mask.dtype != np.float64:
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = mask.astype(np.float32)

    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # Expand mask to 3 channels for broadcasting
    mask_3d = np.stack([mask] * 3, axis=2)

    # Apply alpha to the mask
    blend_mask = mask_3d * alpha

    # Blend: result = base * (1 - blend_mask) + overlay * blend_mask
    result = (
        base_image.astype(np.float32) * (1.0 - blend_mask)
        + overlay_image.astype(np.float32) * blend_mask
    )

    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Test Affine Warping + Alpha Blending")
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "ourSet", "Bhinu.png"),
        help="Path to input image",
    )
    parser.add_argument(
        "--texture",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "wood_texture.jpg"),
        help="Path to texture image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mask2former",
        choices=["mask2former", "oneformer"],
        help="Segmentation model to use",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.8,
        help="Texture scale factor (default: 0.8)",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=0.0,
        help="Texture rotation in degrees (default: 0.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Alpha blending strength 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_affine_blend_result.png",
        help="Output image filename",
    )

    args = parser.parse_args()

    # 1. Load input image
    print(f"Loading image: {args.image}")
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    print(f"Image shape: {image_np.shape}")

    # 2. Load texture
    print(f"Loading texture: {args.texture}")
    texture = load_texture(args.texture)
    print(f"Texture shape: {texture.shape}")

    # 3. Run segmentation
    print(f"\nRunning {args.model} segmentation...")
    if args.model == "mask2former":
        segmenter = Mask2FormerSegmentation()
    else:
        segmenter = OneFormerSegmentation()

    pred_map, wall_ids = segmenter.segment(image)

    # Create binary wall mask
    wall_mask = np.zeros(pred_map.shape, dtype=np.uint8)
    for w_id in wall_ids:
        wall_mask[pred_map == w_id] = 1

    wall_coverage = np.sum(wall_mask > 0) / wall_mask.size * 100
    print(f"Wall coverage: {wall_coverage:.2f}%")

    # 4. Create affine transformation
    print(
        f"\nCreating affine transformation (scale={args.scale}, rotation={args.rotation}°)..."
    )
    transform_matrix = create_affine_transform(
        image_np.shape,
        texture.shape,
        scale=args.scale,
        rotation=args.rotation,
        translation=(0, 0),
    )
    print(f"Transform matrix:\n{transform_matrix}")

    # 5. Warp texture
    print("Warping texture...")
    warped_texture = affine_warp_texture(texture, image_np.shape, transform_matrix)

    # 6. Alpha blend
    print(f"Blending with alpha={args.alpha}...")
    result = alpha_blend(image_np, warped_texture, wall_mask, alpha=args.alpha)

    # 7. Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Original, Wall Mask, Warped Texture
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(wall_mask, cmap="gray")
    axes[0, 1].set_title(f"Wall Mask ({wall_coverage:.1f}%)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(warped_texture)
    axes[0, 2].set_title("Warped Texture")
    axes[0, 2].axis("off")

    # Row 2: Original, Blended Result, Comparison
    axes[1, 0].imshow(image_np)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(result)
    axes[1, 1].set_title(f"Blended Result (α={args.alpha})")
    axes[1, 1].axis("off")

    # Show overlay on original for comparison
    comparison = np.hstack([image_np, result])
    axes[1, 2].imshow(comparison)
    axes[1, 2].set_title("Original | Blended")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n✅ Results saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
