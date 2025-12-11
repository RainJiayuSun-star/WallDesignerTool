import cv2
import numpy as np

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

