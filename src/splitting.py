import cv2
import numpy as np
from scipy.signal import find_peaks

class SplittingStrategy:
    def split(self, binary_mask, original_image):
        raise NotImplementedError

class CannyHoughSplitting(SplittingStrategy):
    def __init__(self, min_wall_area=1000):
        self.min_wall_area = min_wall_area

    def split(self, binary_mask, original_image):
        """
        Takes a binary mask and the original RGB image.
        Returns separated wall segments and their polygons.
        """
        # 1. Morphological Cleanup (Fill small holes)
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # 2. Geometric Splitting (The "Corner Cutter")
        split_mask = self._split_joined_walls(cleaned_mask, original_image)

        # 3. Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(split_mask, connectivity=8)

        wall_segments = []
        wall_polygons = []

        for i in range(1, num_labels): # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_wall_area:
                continue

            segment_mask = (labels == i).astype(np.uint8) * 255
            poly = self._approximate_polygon(segment_mask)
            
            if poly is not None:
                wall_segments.append(segment_mask)
                wall_polygons.append(poly)

        return wall_segments, wall_polygons

    def _split_joined_walls(self, mask, image):
        split_mask = mask.copy()
        h, w = mask.shape

        # A. Pre-process Image for Edge Detection
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # B. Run Canny on the IMAGE, not the mask
        edges = cv2.Canny(gray, 30, 100) 

        # C. Filter Edges: Keep only edges that are INSIDE the wall mask
        mask_eroded = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=2)
        internal_edges = cv2.bitwise_and(edges, edges, mask=mask_eroded)

        # D. Hough Transform to find Vertical Lines
        min_len = h // 8 
        lines = cv2.HoughLinesP(internal_edges, 1, np.pi/180, threshold=30, minLineLength=min_len, maxLineGap=50)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check for Verticality (+/- 15 degrees)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 75 < angle < 105: 
                    # EXTEND THE LINE
                    cv2.line(split_mask, (x1, 0), (x2, h), 0, thickness=3)

        return split_mask

    def _approximate_polygon(self, binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        cnt = max(contours, key=cv2.contourArea)
        
        # Dynamic epsilon based on arc length
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # If not 4 points, force Convex Hull
        if len(approx) != 4:
            hull = cv2.convexHull(cnt)
            approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
            
        if len(approx) != 4:
             rect = cv2.minAreaRect(cnt)
             box = cv2.boxPoints(rect)
             approx = box.astype(np.int32)

        return approx.reshape(-1, 2)

class WallRefinerSplitting(SplittingStrategy):
    def __init__(self, min_wall_area=1000):
        self.min_wall_area = min_wall_area

    def split(self, binary_mask, original_image):
        """
        Returns separated wall segments and their polygons.
        """
        # Ensure binary mask is 0 or 255
        if binary_mask.max() == 1:
            binary_mask = (binary_mask * 255).astype(np.uint8)
        else:
            binary_mask = binary_mask.astype(np.uint8)

        wall_polygons = self._get_structural_walls(binary_mask, original_image)
        
        wall_segments = []
        h, w = binary_mask.shape
        
        # Create masks from polygons for consistency with SplittingStrategy
        for poly in wall_polygons:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            # Intersect with original mask to respect some boundaries if needed,
            # but the Refiner logic implies these polygons ARE the structural walls.
            # However, to avoid segments outside the original mask (if that's desired),
            # we could AND it. But the Refiner seems to want to fix the mask.
            # Let's trust the Refiner's polygons.
            wall_segments.append(mask)
            
        return wall_segments, wall_polygons

    def _get_structural_walls(self, binary_mask, image):
        """
        Returns a list of 4-point polygons representing the PLANES of the walls.
        It converts the 2D image into a 1D signal to find dominant corners.
        """
        h, w = binary_mask.shape
        
        # 1. Bounding Box Optimization
        #    Only analyze the area where a wall actually exists to save time/noise.
        coords = cv2.findNonZero(binary_mask)
        if coords is None: return [] # Handle empty masks
        x, y, w_box, h_box = cv2.boundingRect(coords)
        
        # 2. Vertical Projection Analysis
        #    Crop to the wall area
        #    Ensure image is RGB (or at least has 3 channels or handle grayscale)
        if len(image.shape) == 2:
            roi_gray = image[y:y+h_box, x:x+w_box]
        else:
            roi_gray = cv2.cvtColor(image[y:y+h_box, x:x+w_box], cv2.COLOR_RGB2GRAY)
        
        #    Sobel X detects vertical changes (like corner shadows)
        sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=5)
        sobelx = np.abs(sobelx)
        
        #    COLLAPSE THE IMAGE: Sum the edge strength down each column.
        #    This is the "1D Signal" where corners become Peaks.
        vertical_score = np.sum(sobelx, axis=0)
        
        # 3. Peak Finding (The "Smart" Splitter)
        #    distance: Corners must be at least 10% of the width apart (prevents double lines)
        #    prominence: The peak must be significantly higher than the surrounding noise (blinds)
        min_dist = max(w_box // 10, 20)
        
        if np.max(vertical_score) == 0:
            return []

        peak_threshold = np.max(vertical_score) * 0.3
        peaks, _ = find_peaks(vertical_score, distance=min_dist, prominence=peak_threshold)
        
        #    Map local ROI coordinates back to global image coordinates
        cut_x_coords = [p + x for p in peaks]
        
        #    Define the vertical boundaries: Start of Box, Detected Corners, End of Box
        boundaries = [x, x + w_box]
        boundaries.extend(cut_x_coords)
        boundaries.sort()
        
        wall_polygons = []
        
        # 4. Construct Geometric Planes
        for i in range(len(boundaries) - 1):
            x_start = boundaries[i]
            x_end = boundaries[i+1]
            
            # Filter out tiny slivers (noise)
            if (x_end - x_start) < 50: 
                continue

            # Find the ceiling and floor for this specific segment
            # We look at the original mask within this vertical strip
            strip_mask = binary_mask[:, x_start:x_end]
            strip_points = cv2.findNonZero(strip_mask)
            
            if strip_points is not None:
                _, y_strip, _, h_strip = cv2.boundingRect(strip_points)
                
                # Define the Perfect Rectangle for this wall section
                # Note: y_strip is relative to the full image because strip_mask has full height
                # but sliced width. Wait, strip_mask = binary_mask[:, x_start:x_end] preserves height.
                # cv2.findNonZero returns (x, y). x is relative to strip (0..width), y is relative to image (0..h).
                
                poly = np.array([
                    [x_start, y_strip],           # Top-Left
                    [x_end, y_strip],             # Top-Right
                    [x_end, y_strip + h_strip],   # Bottom-Right
                    [x_start, y_strip + h_strip]  # Bottom-Left
                ], dtype=np.int32)
                wall_polygons.append(poly)

        return wall_polygons
