"""
Edge Detection with Hough Transform for Texture Overlay
Processes test_room.JPG and overlays brick_texture.jpg

Requirements:
    pip install opencv-python numpy matplotlib

Usage:
    python hough_edge_transform.py
"""

import sys

try:
    import cv2
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Error: Missing required dependencies.")
    print("Please install with: pip install opencv-python numpy matplotlib")
    print(f"Missing module: {e.name}")
    sys.exit(1)


def detect_edges(image):
    """Apply Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    return edges, gray


def hough_transform(edges):
    """Detect lines using Probabilistic Hough Transform."""
    # Detect line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,                    # Distance resolution in pixels
        theta=np.pi/180,          # Angular resolution in radians
        threshold=100,             # Minimum votes for a line
        minLineLength=100,         # Minimum line length
        maxLineGap=10              # Maximum gap between line segments
    )
    
    return lines


def cluster_lines(lines):
    """Cluster lines into horizontal and vertical groups."""
    if lines is None:
        return [], []
    
    horizontal = []
    vertical = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle in degrees
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Classify as horizontal or vertical
        if angle < 30 or angle > 150:  # Horizontal (±30°)
            horizontal.append(line[0])
        elif 60 < angle < 120:  # Vertical (±30° from 90°)
            vertical.append(line[0])
    
    return horizontal, vertical


def find_corners(horizontal_lines, vertical_lines, image_shape):
    """Find corner points from line intersections."""
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None
    
    # Sort lines
    h_sorted = sorted(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
    v_sorted = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
    
    # Take top/bottom horizontal and left/right vertical
    top_line = h_sorted[0]
    bottom_line = h_sorted[-1]
    left_line = v_sorted[0]
    right_line = v_sorted[-1]
    
    # Find intersections
    corners = []
    for h_line in [top_line, bottom_line]:
        for v_line in [left_line, right_line]:
            pt = line_intersection(h_line, v_line)
            if pt is not None:
                corners.append(pt)
    
    if len(corners) == 4:
        corners = np.array(corners)
        corners = order_points(corners)
        return corners
    
    return None


def line_intersection(line1, line2):
    """Find intersection point of two lines."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    
    return np.array([int(px), int(py)])


def order_points(pts):
    """Order points as: top-left, top-right, bottom-right, bottom-left."""
    # Sort by y-coordinate
    pts_sorted = pts[np.argsort(pts[:, 1])]
    
    # Top two points
    top_pts = pts_sorted[:2]
    top_pts = top_pts[np.argsort(top_pts[:, 0])]
    
    # Bottom two points
    bottom_pts = pts_sorted[2:]
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
    
    return np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]])


def create_mask_from_corners(corners, image_shape):
    """Create binary mask from corner points."""
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [corners], 255)
    return mask


def create_mask_from_edges(edges, image_shape):
    """Fallback: create mask from edges using morphological operations."""
    # Dilate edges to create regions
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    
    # Find largest contour
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        # Return full image mask
        return np.ones((image_shape[0], image_shape[1]), dtype=np.uint8) * 255
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create mask
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    return mask


def compute_homography(corners, texture_shape):
    """Compute homography matrix from corners to texture coordinates."""
    h, w = texture_shape[:2]
    
    # Destination points (texture corners)
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    # Source points (detected corners)
    src_pts = corners.astype(np.float32)
    
    # Compute homography
    H, _ = cv2.findHomography(dst_pts, src_pts)
    
    return H


def warp_texture(texture, H, image_shape):
    """Warp texture using homography matrix."""
    warped = cv2.warpPerspective(
        texture, H, (image_shape[1], image_shape[0])
    )
    return warped


def blend_texture(original, warped_texture, mask):
    """Blend warped texture with original image using mask."""
    # Ensure mask is binary
    mask = (mask > 127).astype(np.uint8)
    
    # Create 3-channel mask
    mask_3ch = cv2.merge([mask, mask, mask])
    
    # Blend: use texture where mask is white, original elsewhere
    result = np.where(mask_3ch > 0, warped_texture, original)
    
    # Optional: Add feathering at edges for smoother blend
    kernel = np.ones((5, 5), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=2)
    mask_border = mask - mask_eroded
    
    if np.any(mask_border):
        # Blend at borders with alpha
        alpha = 0.7
        mask_border_3ch = cv2.merge([mask_border, mask_border, mask_border])
        result = np.where(
            mask_border_3ch > 0,
            cv2.addWeighted(original, 1 - alpha, warped_texture, alpha, 0),
            result,
        )
    
    return result.astype(np.uint8)


def visualize_process(image, edges, lines, corners, mask, result):
    """Create visualization of the processing pipeline."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Edges
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Canny Edge Detection', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Lines detected
    img_with_lines = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    axes[0, 2].imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Hough Lines ({len(lines) if lines is not None else 0} detected)', 
                        fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Corners
    img_with_corners = image.copy()
    if corners is not None:
        for i, corner in enumerate(corners):
            cv2.circle(img_with_corners, tuple(corner), 10, (255, 0, 0), -1)
            cv2.putText(img_with_corners, str(i), tuple(corner + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # Draw polygon
        cv2.polylines(img_with_corners, [corners], True, (0, 0, 255), 3)
    axes[1, 0].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Detected Corners', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Mask
    axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title('Segmentation Mask', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Final result
    axes[1, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Final Result with Texture Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main processing pipeline."""
    print("=" * 60)
    print("Edge Detection with Hough Transform - Texture Overlay")
    print("=" * 60)
    
    # File paths
    image_path = "test_room.JPG"
    texture_path = "brick_texture.jpg"
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Load images
    print(f"\n[1/7] Loading images...")
    image = cv2.imread(image_path)
    texture = cv2.imread(texture_path)
    
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    if texture is None:
        raise FileNotFoundError(f"Could not load texture: {texture_path}")
    
    print(f"  ✓ Original image: {image.shape}")
    print(f"  ✓ Texture image: {texture.shape}")
    
    # Step 1: Edge detection
    print(f"\n[2/7] Applying Canny edge detection...")
    edges, gray = detect_edges(image)
    num_edges = np.sum(edges > 0)
    print(f"  ✓ Detected {num_edges} edge pixels")
    
    # Step 2: Hough Transform
    print(f"\n[3/7] Applying Hough Transform...")
    lines = hough_transform(edges)
    num_lines = len(lines) if lines is not None else 0
    print(f"  ✓ Detected {num_lines} line segments")
    
    if lines is None or num_lines == 0:
        print("  ⚠ Warning: No lines detected!")
        print("  Using edge-based fallback method...")
        mask = create_mask_from_edges(edges, image.shape)
        corners = None
    else:
        # Step 3: Cluster lines
        print(f"\n[4/7] Clustering lines...")
        horizontal_lines, vertical_lines = cluster_lines(lines)
        print(f"  ✓ Horizontal lines: {len(horizontal_lines)}")
        print(f"  ✓ Vertical lines: {len(vertical_lines)}")
        
        # Step 4: Find corners
        print(f"\n[5/7] Finding corners from line intersections...")
        corners = find_corners(horizontal_lines, vertical_lines, image.shape)
        
        if corners is not None:
            print(f"  ✓ Found 4 corners")
            mask = create_mask_from_corners(corners, image.shape)
        else:
            print("  ⚠ Could not find 4 corners, using edge-based fallback...")
            mask = create_mask_from_edges(edges, image.shape)
    
    # Step 5: Compute homography
    print(f"\n[6/7] Computing homography and warping texture...")
    if corners is not None:
        H = compute_homography(corners, texture.shape)
        warped_texture = warp_texture(texture, H, image.shape)
    else:
        # Use mask bounds as corners
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            H = compute_homography(corners, texture.shape)
            warped_texture = warp_texture(texture, H, image.shape)
        else:
            # Fallback: use full image
            corners = np.array([[0, 0], [image.shape[1], 0], 
                              [image.shape[1], image.shape[0]], [0, image.shape[0]]])
            H = compute_homography(corners, texture.shape)
            warped_texture = warp_texture(texture, H, image.shape)
    
    print(f"  ✓ Texture warped")
    
    # Step 6: Blend texture
    print(f"\n[7/7] Blending texture with original image...")
    result = blend_texture(image, warped_texture, mask)
    
    # Save results
    output_path = output_dir / "hough_transform_result.png"
    cv2.imwrite(str(output_path), result)
    print(f"  ✓ Result saved: {output_path}")
    
    # Save intermediate results
    cv2.imwrite(str(output_dir / "hough_edges.png"), edges)
    cv2.imwrite(str(output_dir / "hough_mask.png"), mask)
    cv2.imwrite(str(output_dir / "hough_warped_texture.png"), warped_texture)
    
    # Create visualization
    print(f"\n[8/8] Creating visualization...")
    fig = visualize_process(image, edges, lines, corners, mask, result)
    viz_path = output_dir / "hough_transform_visualization.png"
    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Visualization saved: {viz_path}")
    
    print(f"\n{'=' * 60}")
    print("Processing Complete!")
    print(f"{'=' * 60}")
    print(f"Results saved in: {output_dir}")
    print(f"  - Final result: hough_transform_result.png")
    print(f"  - Visualization: hough_transform_visualization.png")
    print(f"  - Edges: hough_edges.png")
    print(f"  - Mask: hough_mask.png")
    print(f"  - Warped texture: hough_warped_texture.png")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

