import cv2
import numpy as np
import os

class TextureMappingStrategy:
    def apply(self, original_image, wall_polygons):
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

    def apply(self, original_image, wall_polygons):
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

