import argparse
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import scipy.sparse
from scipy.sparse.linalg import spsolve

# Add src to path for imports
sys.path.append(os.path.dirname(__file__))
from segmentation import Mask2FormerSegmentation


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


def intrinsic_decomposition_simple(image):
    """
    Simple intrinsic decomposition using Retinex theory.
    Separates image into albedo (reflectance) and shading (illumination).

    I = A * S, where:
    - I: Input image
    - A: Albedo (color/texture)
    - S: Shading (lighting)

    We estimate shading as a smoothed version of the luminance,
    then compute albedo = I / S

    Args:
        image: Input image (H, W, 3) uint8 [0-255]

    Returns:
        albedo: Albedo image (H, W, 3) float32 [0-1]
        shading: Shading image (H, W, 3) float32 [0-1]
    """
    # Convert to float and normalize
    img_float = image.astype(np.float32) / 255.0

    # Compute luminance (grayscale)
    # Using standard luminance weights: 0.299*R + 0.587*G + 0.114*B
    luminance = (
        0.299 * img_float[:, :, 0]
        + 0.587 * img_float[:, :, 1]
        + 0.114 * img_float[:, :, 2]
    )

    # Estimate shading as a smoothed version of luminance
    # Using a large Gaussian filter to capture low-frequency lighting
    # This removes high-frequency texture details, leaving mostly lighting
    kernel_size = max(15, min(image.shape[:2]) // 20)  # Adaptive kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    shading_gray = cv2.GaussianBlur(
        luminance, (kernel_size, kernel_size), sigmaX=kernel_size / 3
    )

    # Add small epsilon to avoid division by zero
    shading_gray = np.clip(shading_gray, 0.01, 1.0)

    # Extend shading to 3 channels (assume uniform lighting across RGB)
    shading = np.stack([shading_gray] * 3, axis=2)

    # Compute albedo: A = I / S
    albedo = img_float / shading

    # Normalize albedo to reasonable range [0, 1]
    # Some pixels might exceed 1.0 due to bright highlights
    albedo = np.clip(albedo, 0.0, 1.0)

    return albedo, shading


def replace_albedo_with_texture(albedo, texture, mask, image_shape):
    """
    Replace albedo in masked regions with texture.

    Args:
        albedo: Original albedo (H, W, 3) float32
        texture: Texture to apply (H_texture, W_texture, 3) uint8
        mask: Binary mask (H, W) where 1 = replace albedo
        image_shape: Target image shape (H, W)

    Returns:
        new_albedo: Albedo with texture applied (H, W, 3) float32
    """
    h, w = image_shape[:2]

    # Resize texture to match image size
    texture_resized = cv2.resize(texture, (w, h), interpolation=cv2.INTER_LINEAR)
    texture_float = texture_resized.astype(np.float32) / 255.0

    # Normalize texture to match albedo characteristics
    # We want the texture to have similar brightness distribution as original albedo
    mask_3d = np.stack([mask] * 3, axis=2).astype(np.float32)
    if mask.max() > 1:
        mask_3d = mask_3d / 255.0

    # Replace albedo in masked regions
    new_albedo = albedo.copy()
    new_albedo = new_albedo * (1.0 - mask_3d) + texture_float * mask_3d

    return new_albedo


def poisson_blend(source, target, mask):
    """
    Poisson blending for seamless integration (simplified version).
    Uses boundary-preserving blending with smooth transitions.

    Args:
        source: Source image to blend in (H, W, 3) float32
        target: Target image (H, W, 3) float32
        mask: Binary mask (H, W) where 1 = blend region

    Returns:
        blended: Blended result (H, W, 3) float32
    """
    h, w = mask.shape
    mask_binary = (mask > 0.5).astype(np.uint8)

    # For each channel, apply boundary-preserving blend
    result = np.zeros_like(target)

    for c in range(3):
        # Create blended channel
        blended_channel = target[:, :, c].copy()

        # Inside mask: use source
        blended_channel[mask_binary == 1] = source[mask_binary == 1, c]

        # Smooth transition at boundary using distance transform
        # Distance from mask boundary
        dist = cv2.distanceTransform(1 - mask_binary, cv2.DIST_L2, 5)
        dist_norm = np.clip(dist / 5.0, 0, 1)  # Normalize to [0, 1]

        # At boundary region, blend between source and target
        boundary_mask = (dist_norm < 1.0) & (dist_norm > 0.0) & (mask_binary == 0)
        blended_channel[boundary_mask] = (
            dist_norm[boundary_mask] * target[boundary_mask, c]
            + (1 - dist_norm[boundary_mask]) * source[boundary_mask, c]
        )

        result[:, :, c] = blended_channel

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test Intrinsic Decomposition + Poisson Blending"
    )
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
        "--output",
        type=str,
        default="test_intrinsic_decomposition_result.png",
        help="Output image filename",
    )
    parser.add_argument(
        "--use_poisson",
        action="store_true",
        help="Use Poisson blending (slower but more seamless)",
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

    # 3. Run Mask2Former segmentation
    print("\nRunning Mask2Former segmentation...")
    segmenter = Mask2FormerSegmentation()
    pred_map, wall_ids = segmenter.segment(image)

    # Create binary wall mask
    wall_mask = np.zeros(pred_map.shape, dtype=np.uint8)
    for w_id in wall_ids:
        wall_mask[pred_map == w_id] = 1

    wall_coverage = np.sum(wall_mask > 0) / wall_mask.size * 100
    print(f"Wall coverage: {wall_coverage:.2f}%")

    # 4. Intrinsic Decomposition
    print("\nPerforming intrinsic decomposition...")
    albedo, shading = intrinsic_decomposition_simple(image_np)
    print(f"Albedo range: [{albedo.min():.3f}, {albedo.max():.3f}]")
    print(f"Shading range: [{shading.min():.3f}, {shading.max():.3f}]")

    # 5. Replace Albedo with Texture
    print("Replacing albedo with texture...")
    new_albedo = replace_albedo_with_texture(albedo, texture, wall_mask, image_np.shape)

    # 6. Re-combine: New Image = New Albedo * Original Shading
    print("Re-combining albedo and shading...")
    result_intrinsic = new_albedo * shading

    # 7. Optional: Apply Poisson Blending for seamless integration
    if args.use_poisson:
        print("Applying Poisson blending...")
        # Convert to float for Poisson
        result_float = result_intrinsic.astype(np.float32)
        original_float = image_np.astype(np.float32) / 255.0
        result_intrinsic = poisson_blend(result_float, original_float, wall_mask)

    # Clip and convert to uint8
    result_intrinsic = np.clip(result_intrinsic * 255.0, 0, 255).astype(np.uint8)

    # 8. Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Original, Albedo, Shading
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.clip(albedo, 0, 1))
    axes[0, 1].set_title("Albedo (Color/Texture)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.clip(shading, 0, 1))
    axes[0, 2].set_title("Shading (Lighting)")
    axes[0, 2].axis("off")

    # Row 2: Wall Mask, New Albedo, Final Result
    axes[1, 0].imshow(wall_mask, cmap="gray")
    axes[1, 0].set_title(f"Wall Mask ({wall_coverage:.1f}%)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.clip(new_albedo, 0, 1))
    axes[1, 1].set_title("New Albedo (Texture Applied)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(result_intrinsic)
    axes[1, 2].set_title(
        "Final Result\n(New Albedo × Original Shading"
        + (" + Poisson)" if args.use_poisson else ")")
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n✅ Results saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
