import cv2
import numpy as np
import os

class TextureMappingStrategy:
    def apply(self, original_image, wall_polygons, **kwargs):
        raise NotImplementedError

class HomographyMultiplyMapping(TextureMappingStrategy):
    def __init__(self, texture_path):
        self.texture_path = texture_path
        self._load_texture()

    def _load_texture(self):
        if not os.path.exists(self.texture_path):
            print(f"Warning: Texture file '{self.texture_path}' not found. Creating dummy texture.")
            self.texture = self._create_dummy_texture()
        else:
            img = cv2.imread(self.texture_path)
            if img is None:
                 print(f"Warning: Could not read texture file '{self.texture_path}'. Creating dummy texture.")
                 self.texture = self._create_dummy_texture()
            else:
                self.texture = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _create_dummy_texture(self):
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy.fill(255)
        cv2.rectangle(dummy, (0, 0), (50, 50), (0, 0, 255), -1)
        cv2.rectangle(dummy, (50, 50), (100, 100), (0, 255, 0), -1)
        return dummy

    def apply(self, original_image, wall_polygons, **kwargs):
        textured_image = original_image.copy()
        for poly in wall_polygons:
            textured_image = self._apply_single_polygon(textured_image, poly)
        return textured_image

    def _apply_single_polygon(self, original_image, wall_polygon):
        pts_dst = self._order_points(wall_polygon)

        # Calculate "Rectified" Wall Dimensions
        (tl, tr, br, bl) = pts_dst
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Create Tiled Texture Canvas
        tiled_texture = self._tile_texture(maxWidth, maxHeight)

        # Compute Homography & Warp
        pts_src = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts_src, pts_dst.astype("float32"))
        warped_texture = cv2.warpPerspective(tiled_texture, M, (original_image.shape[1], original_image.shape[0]))

        # Blend with "Multiply" mode
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts_dst.astype("int32"), 255)
        
        img_float = original_image.astype(float) / 255.0
        tex_float = warped_texture.astype(float) / 255.0
        
        blended = img_float.copy()
        mask_indices = np.where(mask > 0)
        
        blended[mask_indices] = cv2.multiply(img_float[mask_indices], tex_float[mask_indices])
        
        return (blended * 255).astype(np.uint8)

    def _tile_texture(self, w, h):
        th, tw, _ = self.texture.shape
        nx = int(np.ceil(w / tw))
        ny = int(np.ceil(h / th))
        tiled = np.tile(self.texture, (ny, nx, 1))
        return tiled[:h, :w]

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

class MaskedPerspectiveMapping(TextureMappingStrategy):
    def __init__(self, texture_path):
        self.texture_path = texture_path
        self._load_texture()

    def _load_texture(self):
        # Re-use loading logic or duplicate it. 
        # Duplicating for simplicity/independence unless I refactor to a base class helper.
        if not os.path.exists(self.texture_path):
            print(f"Warning: Texture file '{self.texture_path}' not found. Creating dummy texture.")
            self.texture = self._create_dummy_texture()
        else:
            img = cv2.imread(self.texture_path)
            if img is None:
                 print(f"Warning: Could not read texture file '{self.texture_path}'. Creating dummy texture.")
                 self.texture = self._create_dummy_texture()
            else:
                self.texture = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _create_dummy_texture(self):
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy.fill(255)
        cv2.rectangle(dummy, (0, 0), (50, 50), (0, 0, 255), -1)
        cv2.rectangle(dummy, (50, 50), (100, 100), (0, 255, 0), -1)
        return dummy

    def apply(self, original_image, wall_polygons, **kwargs):
        """
        1. Warps texture to the Geometric Polygons (Perspective)
        2. Masks the result using the Semantic Blob (Occlusion)
        3. Blends using Multiply (Lighting Preservation)
        """
        full_mask = kwargs.get('full_mask')
        if full_mask is None:
            # Fallback if no full_mask provided: create one from polygons
            print("Warning: full_mask not provided to MaskedPerspectiveMapping. Using polygons as mask.")
            full_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            for poly in wall_polygons:
                cv2.fillPoly(full_mask, [poly], 255)

        # Create a blank canvas to accumulate texture warps
        texture_layer = np.zeros_like(original_image)
        
        # A. Geometric Phase: Render texture onto the invisible rectangles
        for poly in wall_polygons:
            warped_segment = self._warp_texture_to_poly(original_image.shape, poly)
            # Combine segments (taking max handles slight overlaps cleanly)
            texture_layer = np.maximum(texture_layer, warped_segment)

        # B. Semantic Phase: The "Cookie Cutter"
        #    We only want the texture where Mask2Former says there is a wall.
        #    AND where we actually successfully rendered a texture (valid_tex)
        valid_tex = np.sum(texture_layer, axis=2) > 0
        final_mask = (full_mask > 0) & valid_tex
        
        # C. Blending Phase: Multiply
        img_float = original_image.astype(float) / 255.0
        tex_float = texture_layer.astype(float) / 255.0
        
        blended = img_float.copy()
        
        # Logic: Result = Wall_Color * Texture_Color
        # If the wall is dark (shadow), the result stays dark.
        blended[final_mask] = img_float[final_mask] * tex_float[final_mask]
        
        return (blended * 255).astype(np.uint8)

    def _warp_texture_to_poly(self, shape, poly):
        h, w = shape[:2]
        pts_dst = poly.astype("float32")
        
        # --- CRITICAL FIX: DO NOT RE-ORDER POINTS ---
        # The Splitting strategy explicitly returns [TL, TR, BR, BL].
        # Re-ordering them blindly (using sum/diff) causes twists on slanted walls.
        # We trust the input order.
        
        # Estimate width/height for Tiling
        # Top width (TL to TR)
        width_top = np.linalg.norm(pts_dst[0] - pts_dst[1])
        # Left height (TL to BL)
        height_left = np.linalg.norm(pts_dst[0] - pts_dst[3])
        
        # Safety check for degenerate polygons (width=0)
        if width_top < 1: width_top = 1
        if height_left < 1: height_left = 1
        
        # Create Tiled Source
        tiled = self._tile_texture(int(width_top), int(height_left))
        
        # Define Source Points [TL, TR, BR, BL] matching destination
        pts_src = np.array([
            [0, 0],                                  # TL
            [tiled.shape[1]-1, 0],                   # TR
            [tiled.shape[1]-1, tiled.shape[0]-1],    # BR
            [0, tiled.shape[0]-1]                    # BL
        ], dtype="float32")
        
        # Warp
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(tiled, M, (w, h))
        return warped

    def _tile_texture(self, w, h):
        if w <= 0 or h <= 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
            
        th, tw = self.texture.shape[:2]
        nx = int(np.ceil(w / tw))
        ny = int(np.ceil(h / th))
        tiled = np.tile(self.texture, (ny, nx, 1))
        return tiled[:h, :w]

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

class IntrinsicBlendingMapping(MaskedPerspectiveMapping):
    """
    Advanced mapping that uses Intrinsic Decomposition (Retinex) to separate
    Lighting (Shading) from Color (Albedo), allowing us to replace the wall's
    albedo with the texture while preserving the original lighting.
    """
    def apply(self, original_image, wall_polygons, **kwargs):
        full_mask = kwargs.get('full_mask')
        if full_mask is None:
            full_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            for poly in wall_polygons:
                cv2.fillPoly(full_mask, [poly], 255)

        # 1. Generate Perspective-Warped Texture (Same as parent)
        texture_layer = np.zeros_like(original_image)
        for poly in wall_polygons:
            warped_segment = self._warp_texture_to_poly(original_image.shape, poly)
            texture_layer = np.maximum(texture_layer, warped_segment)
            
        # 2. Extract Shading from Original Image
        _, shading = self._intrinsic_decomposition_simple(original_image)
        
        # 3. Create Final Mask
        valid_tex = np.sum(texture_layer, axis=2) > 0
        final_mask = (full_mask > 0) & valid_tex
        
        # 4. Blend: Result = Texture * Shading
        # We replace the original Albedo with the Texture, but keep the original Shading.
        img_float = original_image.astype(float) / 255.0 # For unmasked areas
        tex_float = texture_layer.astype(float) / 255.0
        
        blended = img_float.copy()
        
        # Apply texture * shading in the masked region
        # Note: 'shading' is already normalized [0-1] from the helper
        blended[final_mask] = tex_float[final_mask] * shading[final_mask]
        
        return (np.clip(blended, 0, 1) * 255).astype(np.uint8)

    def _intrinsic_decomposition_simple(self, image):
        """
        Simple Retinex-based decomposition.
        Returns (albedo, shading).
        """
        img_float = image.astype(np.float32) / 255.0
        
        # Luminance
        luminance = (
            0.299 * img_float[:, :, 0]
            + 0.587 * img_float[:, :, 1]
            + 0.114 * img_float[:, :, 2]
        )
        
        # Estimate shading via Gaussian Blur (Low-pass filter)
        kernel_size = max(15, min(image.shape[:2]) // 20)
        if kernel_size % 2 == 0: kernel_size += 1
        
        shading_gray = cv2.GaussianBlur(
            luminance, (kernel_size, kernel_size), sigmaX=kernel_size / 3
        )
        shading_gray = np.clip(shading_gray, 0.01, 1.0)
        
        shading = np.stack([shading_gray] * 3, axis=2)
        
        # Albedo = I / S
        albedo = img_float / shading
        albedo = np.clip(albedo, 0.0, 1.0)
        
        return albedo, shading
