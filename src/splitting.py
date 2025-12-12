import cv2
import numpy as np
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiLineString, GeometryCollection
from shapely.ops import unary_union, split

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
            wall_segments.append(mask)
            
        return wall_segments, wall_polygons

    def _get_structural_walls(self, binary_mask, image):
        """
        Returns a list of 4-point polygons representing the PLANES of the walls.
        NOW UPDATED to fit slanted lines for perspective (Trapezoids, not Rectangles).
        """
        h, w = binary_mask.shape
        
        # 1. Bounding Box Optimization
        coords = cv2.findNonZero(binary_mask)
        if coords is None: return [] 
        x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(coords)
        
        # 2. Vertical Projection Analysis
        if len(image.shape) == 2:
            roi_gray = image[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]
        else:
            roi_gray = cv2.cvtColor(image[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox], cv2.COLOR_RGB2GRAY)
        
        sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=5)
        vertical_score = np.sum(np.abs(sobelx), axis=0)
        
        # 3. Peak Finding
        min_dist = max(w_bbox // 10, 20)
        if np.max(vertical_score) == 0: return []
        peak_threshold = np.max(vertical_score) * 0.3
        peaks, _ = find_peaks(vertical_score, distance=min_dist, prominence=peak_threshold)
        
        cut_x_coords = [p + x_bbox for p in peaks]
        boundaries = [x_bbox, x_bbox + w_bbox]
        boundaries.extend(cut_x_coords)
        boundaries.sort()
        
        wall_polygons = []
        
        # 4. Construct Geometric Planes (Trapezoids)
        for i in range(len(boundaries) - 1):
            x_start = boundaries[i]
            x_end = boundaries[i+1]
            
            # Filter noise
            if (x_end - x_start) < 50: continue
            # Extract the strip for this wall segment
            # We want to find the equation y = mx + c for the top and bottom
            strip_mask = binary_mask[:, x_start:x_end]
            
            # Get all white pixels in this strip
            pts = cv2.findNonZero(strip_mask)
            if pts is None: continue
            
            # Shift points to be relative to the strip (for calculation)
            # pts is (1, N, 2) -> (N, 2)
            pts = pts.squeeze()
            if pts.ndim == 1: continue # Not enough points
            xs = pts[:, 0] + x_start # Global X
            ys = pts[:, 1]           # Global Y
            # --- KEY CHANGE: FIT LINES INSTEAD OF BOUNDING BOX ---
            
            # A. Find Top (Ceiling) and Bottom (Floor) Points
            # We iterate through X coordinates to find the min Y (ceiling) and max Y (floor) at each column
            unique_xs = np.unique(xs)
            
            top_points_x = []
            top_points_y = []
            bottom_points_x = []
            bottom_points_y = []
            
            for ux in unique_xs:
                # Get all Ys for this specific X column
                ys_at_x = ys[xs == ux]
                
                # Top is min Y, Bottom is max Y
                if len(ys_at_x) > 0:
                    top_points_x.append(ux)
                    top_points_y.append(np.min(ys_at_x))
                    
                    bottom_points_x.append(ux)
                    bottom_points_y.append(np.max(ys_at_x))
            # B. Fit Linear Regression (y = mx + b)
            # We use np.polyfit(x, y, 1) to fit a degree-1 polynomial (a line)
            
            # Fit Ceiling
            if len(top_points_x) > 2:
                m_top, b_top = np.polyfit(top_points_x, top_points_y, 1)
            else:
                m_top, b_top = 0, np.min(ys) # Fallback to flat
                
            # Fit Floor
            if len(bottom_points_x) > 2:
                m_bottom, b_bottom = np.polyfit(bottom_points_x, bottom_points_y, 1)
            else:
                m_bottom, b_bottom = 0, np.max(ys) # Fallback to flat
            # C. Calculate Trapezoid Corners
            # Y = m * X + b
            y_tl = int(m_top * x_start + b_top)
            y_tr = int(m_top * x_end + b_top)
            y_bl = int(m_bottom * x_start + b_bottom)
            y_br = int(m_bottom * x_end + b_bottom)
            
            # Clamp values to image dimensions
            y_tl = np.clip(y_tl, 0, h-1)
            y_tr = np.clip(y_tr, 0, h-1)
            y_bl = np.clip(y_bl, 0, h-1)
            y_br = np.clip(y_br, 0, h-1)
            poly = np.array([
                [x_start, y_tl],  # Top-Left
                [x_end, y_tr],    # Top-Right
                [x_end, y_br],    # Bottom-Right
                [x_start, y_bl]   # Bottom-Left
            ], dtype=np.int32)
            
            wall_polygons.append(poly)

        return wall_polygons

class ContourCornerSplitting(SplittingStrategy):
    """
    Implementation of the user's idea:
    1. Get the contour of the wall blob.
    2. Simplify it to find the main corners (using approxPolyDP).
    3. Identify the vertices that make up the 'top edge' (ceiling line).
    4. For each segment of the top edge, create a floor-to-ceiling trapezoid.
    """
    def __init__(self, epsilon_factor=0.005):
        # epsilon_factor controls corner detection sensitivity.
        # Smaller = more corners detected. 0.005 is a good starting point.
        self.epsilon_factor = epsilon_factor

    def split(self, binary_mask, original_image):
        h_img, w_img = binary_mask.shape
        wall_polygons = []
        
        # 1. Find Contour
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], []
        main_contour = max(contours, key=cv2.contourArea)
        
        # 2. Simplify
        epsilon = self.epsilon_factor * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        points = approx.squeeze() # (N, 2)
        if len(points.shape) < 2 or len(points) < 2: return [], []

        # 3. Robust Ceiling Finder
        # Split points into "Top Half" and "Bottom Half" based on the Leftmost and Rightmost indices.
        min_x_idx = np.argmin(points[:, 0])
        max_x_idx = np.argmax(points[:, 0])
        
        # Path A: Min -> Max (Clockwise or Counter-Clockwise depending on shape)
        if max_x_idx >= min_x_idx:
            path_a = points[min_x_idx:max_x_idx+1]
        else:
            path_a = np.concatenate((points[min_x_idx:], points[:max_x_idx+1]))
            
        # Path B: The rest of the points (Max -> Min)
        if min_x_idx >= max_x_idx:
            path_b = points[max_x_idx:min_x_idx+1]
        else:
            path_b = np.concatenate((points[max_x_idx:], points[:min_x_idx+1]))
            
        # Determine which path is the Ceiling (Lower Y values)
        # We check the average Y height of both paths.
        mean_y_a = np.mean(path_a[:, 1]) if len(path_a) > 0 else float('inf')
        mean_y_b = np.mean(path_b[:, 1]) if len(path_b) > 0 else float('inf')
        
        if mean_y_a < mean_y_b:
            ceiling_points = path_a
        else:
            ceiling_points = path_b
            
        # Sort by X to ensure Left-to-Right order for clean trapezoids
        ceiling_points = ceiling_points[np.argsort(ceiling_points[:, 0])]

        # OPTIONAL: Force the ceiling to be a perfect line (y = mx + b)
        # This fixes "wobbly" tops caused by hanging lights or bad segmentation.
        if len(ceiling_points) > 2:
            # Fit a line to the points
            line = cv2.fitLine(ceiling_points, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = line.flatten()

            # Recalculate the points based on this clean line
            # y = mx + b formulation
            slope = vy / (vx + 1e-6) # Avoid divide by zero
            intercept = y0 - slope * x0

            clean_ceiling = []
            for p in ceiling_points:
                new_y = int(slope * p[0] + intercept)
                # Clamp to image top (0)
                new_y = max(0, new_y)
                clean_ceiling.append([p[0], new_y])

            ceiling_points = np.array(clean_ceiling)

        # 4. Create Trapezoids
        for i in range(len(ceiling_points) - 1):
            p1 = ceiling_points[i]
            p2 = ceiling_points[i+1]
            
            # Skip tiny noise segments
            if abs(p2[0] - p1[0]) < 10: continue 

            # Create 4-point trapezoid (TL, TR, BR, BL)
            poly = np.array([
                [p1[0], p1[1]],           # TL (Ceiling Left)
                [p2[0], p2[1]],           # TR (Ceiling Right)
                [p2[0], h_img - 1],       # BR (Floor Right)
                [p1[0], h_img - 1]        # BL (Floor Left)
            ], dtype=np.int32)
            
            wall_polygons.append(poly)
            
        wall_segments = [binary_mask for _ in wall_polygons] # Dummy return for compatibility
        return wall_segments, wall_polygons

class RobustCornerSplitting(SplittingStrategy):
    """
    Advanced splitting that combines Geometric Analysis (Ceiling Kinks) 
    with Image Evidence (Vertical Gradients).
    """
    def __init__(self, sensitivity=0.002, angle_threshold=5, use_image_refinement=True):
        self.sensitivity = sensitivity        # Lower = keeps more points (0.002 is very sensitive)
        self.angle_threshold = angle_threshold # Min degrees to consider a "bend" a corner
        self.use_image_refinement = use_image_refinement

    def split(self, binary_mask, original_image):
        h_img, w_img = binary_mask.shape
        wall_polygons = []

        # 1. Get the Main Contour
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return [], []
        main_contour = max(contours, key=cv2.contourArea)

        # 2. Extract the "Ceiling Line" specifically
        # We start with a very low epsilon to capture subtle corners
        epsilon = self.sensitivity * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        points = approx.squeeze()
        
        if len(points) < 3:
             # Fallback: If blob is a simple quad/triangle, just return bounds
             return self._fallback_bbox(binary_mask)

        ceiling_points = self._extract_ceiling_path(points)
        
        # 3. Analyze Angles to find Split Candidates (X-coordinates)
        split_indices = [0] # Always include the start
        
        # We iterate through inner points: P_prev -> P_curr -> P_next
        for i in range(1, len(ceiling_points) - 1):
            p_prev = ceiling_points[i-1]
            p_curr = ceiling_points[i]
            p_next = ceiling_points[i+1]
            
            # Calculate angle deviation from straight line (180 deg)
            angle_deg = self._calculate_angle(p_prev, p_curr, p_next)
            deviation = abs(180 - angle_deg)
            
            # A. Geometric Check: Is the bend sharp enough?
            is_geometric_corner = deviation > self.angle_threshold
            
            # B. Image Check: Is there a vertical line in the image here?
            # (Only performed if we have the image and flag is set)
            is_visual_edge = True 
            if self.use_image_refinement and original_image is not None and is_geometric_corner:
                is_visual_edge = self._verify_vertical_edge(original_image, p_curr[0])

            if is_geometric_corner and is_visual_edge:
                split_indices.append(i)

        split_indices.append(len(ceiling_points) - 1) # Always include the end

        # 4. Construct Trapezoids from Splits
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]
            
            # Get the segment of the ceiling for this wall
            segment_points = ceiling_points[start_idx : end_idx + 1]
            
            # We approximate this segment as a single straight line (y=mx+b)
            # to make the final texture mapping clean
            vx, vy, x0, y0 = cv2.fitLine(segment_points, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Get X boundaries
            x_start = segment_points[0][0]
            x_end = segment_points[-1][0]
            
            if (x_end - x_start) < 10: continue # Skip noise
            
            # Project y = mx + b
            slope = vy / (vx + 1e-6)
            intercept = y0 - slope * x0
            
            y_tl = int(slope * x_start + intercept)
            y_tr = int(slope * x_end + intercept)
            
            # Clamp Y (Ceiling shouldn't be below floor, or above image)
            y_tl = max(0, min(y_tl, h_img - 1))
            y_tr = max(0, min(y_tr, h_img - 1))

            # Create Trapezoid
            poly = np.array([
                [x_start, y_tl],      # TL
                [x_end, y_tr],        # TR
                [x_end, h_img - 1],   # BR
                [x_start, h_img - 1]  # BL
            ], dtype=np.int32)
            
            wall_polygons.append(poly)

        # Create dummy masks for compatibility
        wall_segments = [binary_mask for _ in wall_polygons]
        return wall_segments, wall_polygons

    def _extract_ceiling_path(self, points):
        """ Separates the top path of the contour from the bottom. """
        # Sort by X to make finding left/right easy
        # Note: This simple heuristic assumes the wall blob is somewhat convex-ish
        # For complex concave blobs, this might need logic adjustment
        
        # 1. Find Left-most and Right-most points
        min_x_idx = np.argmin(points[:, 0])
        max_x_idx = np.argmax(points[:, 0])
        
        # 2. Split into two paths
        if max_x_idx >= min_x_idx:
            path_a = points[min_x_idx:max_x_idx+1]
        else:
            path_a = np.concatenate((points[min_x_idx:], points[:max_x_idx+1]))
            
        if min_x_idx >= max_x_idx:
            path_b = points[max_x_idx:min_x_idx+1]
        else:
            path_b = np.concatenate((points[max_x_idx:], points[:min_x_idx+1]))
            
        # 3. Keep the path with lower average Y (Higher in image = Ceiling)
        mean_y_a = np.mean(path_a[:, 1]) if len(path_a) > 0 else float('inf')
        mean_y_b = np.mean(path_b[:, 1]) if len(path_b) > 0 else float('inf')
        
        ceiling = path_a if mean_y_a < mean_y_b else path_b
        
        # Sort strictly by X for easy line processing
        return ceiling[np.argsort(ceiling[:, 0])]

    def _calculate_angle(self, p1, p2, p3):
        """ Calculates inner angle at p2 in degrees """
        v1 = p1 - p2
        v2 = p3 - p2
        # Normalize
        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2)
        if m1 == 0 or m2 == 0: return 180.0
        
        cos_angle = np.dot(v1, v2) / (m1 * m2)
        # Numerical stability clip
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    def _verify_vertical_edge(self, image, x_coord, window=10):
        """ 
        Checks a small window around x_coord for a vertical edge.
        Returns True if edge strength is significant.
        """
        h, w = image.shape[:2]
        x_start = max(0, int(x_coord - window))
        x_end = min(w, int(x_coord + window))
        
        if x_end - x_start < 2: return False
        
        roi = image[:, x_start:x_end]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Sobel X (Vertical edges)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        mag = np.abs(sobelx)
        
        # Sum vertically
        col_sums = np.sum(mag, axis=0)
        
        # If the max peak in this window is significant compared to the mean
        peak = np.max(col_sums)
        mean_noise = np.mean(col_sums)
        
        return peak > (mean_noise * 1.5) # Threshold factor

    def _fallback_bbox(self, mask):
        x, y, w, h = cv2.boundingRect(mask)
        poly = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
        return [mask], [poly]

class CeilingKinkSplitting(SplittingStrategy):
    """
    Implements the 'Smart Kink' logic with Perspective Floor Inference:
    1. Extracts Top Profile (Ceiling Line).
    2. Identifies kinks (geometric corners).
    3. Reconstructs trapezoidal polygons where the floor line mirrors the ceiling slope
       but sits 'outside' the image to ensure full coverage.
    """
    def __init__(self, epsilon_factor=0.003, bend_threshold=10, margin_top=5):
        self.epsilon_factor = epsilon_factor
        self.bend_threshold = bend_threshold
        self.margin_top = margin_top

    def split(self, binary_mask, original_image):
        h_img, w_img = binary_mask.shape
        wall_polygons = []

        # --- Phase 1: Top Profile Extraction ---
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return [], []
        main_contour = max(contours, key=cv2.contourArea)

        if main_contour.shape[0] < 3:
             return self._fallback_bbox(binary_mask)

        ceiling_path = self._extract_ceiling_path(main_contour.squeeze())
        if len(ceiling_path) < 2:
            return self._fallback_bbox(binary_mask)
        
        ceiling_path = ceiling_path[np.argsort(ceiling_path[:, 0])]

        # --- Phase 2: The "Smart Kink" Logic ---
        arc_len = cv2.arcLength(ceiling_path.reshape(-1, 1, 2), False)
        epsilon = self.epsilon_factor * arc_len
        simplified_path = cv2.approxPolyDP(ceiling_path.reshape(-1, 1, 2), epsilon, False)
        simplified_path = simplified_path.squeeze()

        if len(simplified_path.shape) < 2:
             simplified_path = np.array([ceiling_path[0], ceiling_path[-1]])
        
        valid_split_indices = [0]
        
        if len(simplified_path) > 2:
            for i in range(1, len(simplified_path) - 1):
                p_prev = simplified_path[i-1]
                p_curr = simplified_path[i]
                p_next = simplified_path[i+1]

                angle_deg = self._calculate_angle(p_prev, p_curr, p_next)
                deviation = abs(180 - angle_deg)
                y_val = p_curr[1]
                
                is_significant_bend = deviation > self.bend_threshold
                is_not_boundary = y_val > self.margin_top
                
                if is_significant_bend and is_not_boundary:
                    valid_split_indices.append(i)
        
        valid_split_indices.append(len(simplified_path) - 1)
        valid_split_indices = sorted(list(set(valid_split_indices)))

        # --- Phase 3: Polygon Reconstruction (With Perspective Floor) ---
        split_points = simplified_path[valid_split_indices]

        for i in range(len(split_points) - 1):
            p_start = split_points[i]
            p_end = split_points[i+1]
            
            x_start = p_start[0]
            x_end = p_end[0]
            
            if (x_end - x_start) < 10: continue

            # 1. Fit the Ceiling Line
            segment_mask = (ceiling_path[:, 0] >= x_start) & (ceiling_path[:, 0] <= x_end)
            segment_points = ceiling_path[segment_mask]

            if len(segment_points) >= 2:
                vx, vy, x0, y0 = cv2.fitLine(segment_points, cv2.DIST_L2, 0, 0.01, 0.01)
                slope = vy / (vx + 1e-6)
                intercept = y0 - slope * x0
                
                y_ceil_start = int(slope * x_start + intercept)
                y_ceil_end = int(slope * x_end + intercept)
            else:
                y_ceil_start = p_start[1]
                y_ceil_end = p_end[1]

            # 2. Calculate Perspective Floor Line (Mirrored Slope)
            # Calculate the vertical change of the ceiling
            delta_y_ceiling = y_ceil_end - y_ceil_start
            
            # The floor should change in the OPPOSITE direction to create vanishing point
            delta_y_floor = -delta_y_ceiling 
            
            # Determine Floor Anchor
            # We want the HIGHEST point of the floor line to be exactly at the image bottom (h_img).
            # This ensures the line is technically "outside" or at the edge, covering all pixels.
            if delta_y_floor >= 0:
                # Floor goes DOWN (or flat) from left to right.
                # So the Left point is higher (smaller Y). Set Left to h_img.
                y_floor_start = h_img
                y_floor_end = h_img + delta_y_floor
            else:
                # Floor goes UP from left to right.
                # So the Right point is higher (smaller Y). Set Right to h_img.
                y_floor_end = h_img
                y_floor_start = h_img - delta_y_floor # delta is negative, so we subtract to add

            # Clamp Ceiling (Visual Safety only, math stays pure)
            y_c_s = max(0, min(y_ceil_start, h_img - 1))
            y_c_e = max(0, min(y_ceil_end, h_img - 1))

            # Create Trapezoid
            poly = np.array([
                [x_start, y_c_s],        # TL
                [x_end, y_c_e],          # TR
                [x_end, int(y_floor_end)],    # BR (Calculated Perspective)
                [x_start, int(y_floor_start)] # BL (Calculated Perspective)
            ], dtype=np.int32)
            
            wall_polygons.append(poly)

        wall_segments = [binary_mask for _ in wall_polygons]
        return wall_segments, wall_polygons

    def _extract_ceiling_path(self, points):
        min_x_idx = np.argmin(points[:, 0])
        max_x_idx = np.argmax(points[:, 0])
        if max_x_idx >= min_x_idx:
            path_a = points[min_x_idx:max_x_idx+1]
        else:
            path_a = np.concatenate((points[min_x_idx:], points[:max_x_idx+1]))
        if min_x_idx >= max_x_idx:
            path_b = points[max_x_idx:min_x_idx+1]
        else:
            path_b = np.concatenate((points[max_x_idx:], points[:min_x_idx+1]))
        mean_y_a = np.mean(path_a[:, 1]) if len(path_a) > 0 else float('inf')
        mean_y_b = np.mean(path_b[:, 1]) if len(path_b) > 0 else float('inf')
        return path_a if mean_y_a < mean_y_b else path_b

    def _calculate_angle(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2)
        if m1 == 0 or m2 == 0: return 180.0
        cos_angle = np.dot(v1, v2) / (m1 * m2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _fallback_bbox(self, mask):
        x, y, w, h = cv2.boundingRect(mask)
        poly = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
        return [mask], [poly]

class CeilingAndFloorKinkSplitting(SplittingStrategy):
    """
    Splits walls based on geometric kinks in BOTH the ceiling and floor profiles.
    
    1. Extracts Ceiling and Floor profiles from the mask contour.
    2. Detects valid geometric corners (kinks) in both profiles.
    3. Merges split points from both top and bottom.
    4. Reconstructs trapezoids by fitting lines to both ceiling and floor segments.
    """
    def __init__(self, epsilon_factor=0.003, bend_threshold=16, margin=5):
        self.epsilon_factor = epsilon_factor
        self.bend_threshold = bend_threshold
        self.margin = margin # Used for both top (0+margin) and bottom (h-margin)

    def split(self, binary_mask, original_image):
        h_img, w_img = binary_mask.shape
        wall_polygons = []

        # --- Phase 1: Profile Extraction ---
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return [], []
        main_contour = max(contours, key=cv2.contourArea)

        if main_contour.shape[0] < 3:
             return self._fallback_bbox(binary_mask)
        
        # Get both paths
        ceiling_path, floor_path = self._extract_profiles(main_contour.squeeze())
        
        if len(ceiling_path) < 2 or len(floor_path) < 2:
            return self._fallback_bbox(binary_mask)

        # Sort by X
        ceiling_path = ceiling_path[np.argsort(ceiling_path[:, 0])]
        floor_path = floor_path[np.argsort(floor_path[:, 0])]

        # --- Phase 2: Kink Detection ---
        
        # Helper to find splits in a path
        def find_kinks(path, is_ceiling):
            arc_len = cv2.arcLength(path.reshape(-1, 1, 2), False)
            epsilon = self.epsilon_factor * arc_len
            simplified = cv2.approxPolyDP(path.reshape(-1, 1, 2), epsilon, False).squeeze()
            
            if len(simplified.shape) < 2: return []
            
            kinks = []
            if len(simplified) > 2:
                for i in range(1, len(simplified) - 1):
                    p_prev = simplified[i-1]
                    p_curr = simplified[i]
                    p_next = simplified[i+1]

                    angle = self._calculate_angle(p_prev, p_curr, p_next)
                    if abs(180 - angle) > self.bend_threshold:
                        # Boundary check
                        y = p_curr[1]
                        if is_ceiling:
                            if y > self.margin: kinks.append(p_curr[0])
                        else: # Floor
                            if y < (h_img - self.margin): kinks.append(p_curr[0])
            return kinks

        ceiling_kinks = find_kinks(ceiling_path, is_ceiling=True)
        floor_kinks = find_kinks(floor_path, is_ceiling=False)
        
        # Combine splits
        # We also need the start and end of the blob
        min_x = min(ceiling_path[0][0], floor_path[0][0])
        max_x = max(ceiling_path[-1][0], floor_path[-1][0])
        
        all_splits = [min_x] + ceiling_kinks + floor_kinks + [max_x]
        all_splits = sorted(list(set(all_splits)))
        
        # Filter close splits (e.g. within 20px)
        filtered_splits = [all_splits[0]]
        for x in all_splits[1:]:
            if x - filtered_splits[-1] > 20: # Min segment width
                filtered_splits.append(x)
        # Ensure last point is included if it wasn't too close, or update last point
        if filtered_splits[-1] != max_x:
             if max_x - filtered_splits[-1] > 20:
                 filtered_splits.append(max_x)
             else:
                 filtered_splits[-1] = max_x

        # --- Phase 3: Reconstruction ---
        for i in range(len(filtered_splits) - 1):
            x_start = filtered_splits[i]
            x_end = filtered_splits[i+1]
            
            # Helper to fit line
            def get_y_at_x(path, x_s, x_e):
                mask = (path[:, 0] >= x_s) & (path[:, 0] <= x_e)
                pts = path[mask]
                if len(pts) < 2: 
                    return None, None 
                
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                slope = vy / (vx + 1e-6)
                intercept = y0 - slope * x0
                
                y_s = int(slope * x_s + intercept)
                y_e = int(slope * x_e + intercept)
                return y_s, y_e

            y_c_s, y_c_e = get_y_at_x(ceiling_path, x_start, x_end)
            y_f_s, y_f_e = get_y_at_x(floor_path, x_start, x_end)
            
            # Fallbacks
            if y_c_s is None:
                 idx_s = np.searchsorted(ceiling_path[:, 0], x_start)
                 idx_s = min(idx_s, len(ceiling_path)-1)
                 y_c_s = ceiling_path[idx_s][1]
                 
                 idx_e = np.searchsorted(ceiling_path[:, 0], x_end)
                 idx_e = min(idx_e, len(ceiling_path)-1)
                 y_c_e = ceiling_path[idx_e][1]

            if y_f_s is None:
                 idx_s = np.searchsorted(floor_path[:, 0], x_start)
                 idx_s = min(idx_s, len(floor_path)-1)
                 y_f_s = floor_path[idx_s][1]
                 
                 idx_e = np.searchsorted(floor_path[:, 0], x_end)
                 idx_e = min(idx_e, len(floor_path)-1)
                 y_f_e = floor_path[idx_e][1]

            # Clamp
            y_c_s = max(0, min(y_c_s, h_img - 1))
            y_c_e = max(0, min(y_c_e, h_img - 1))
            y_f_s = max(0, min(y_f_s, h_img - 1))
            y_f_e = max(0, min(y_f_e, h_img - 1))
            
            # Safety: Ceiling must be above Floor
            if y_c_s >= y_f_s: y_c_s = y_f_s - 1
            if y_c_e >= y_f_e: y_c_e = y_f_e - 1

            poly = np.array([
                [x_start, y_c_s],
                [x_end, y_c_e],
                [x_end, y_f_e],
                [x_start, y_f_s]
            ], dtype=np.int32)
            wall_polygons.append(poly)
            
        wall_segments = [binary_mask for _ in wall_polygons]
        return wall_segments, wall_polygons

    def _extract_profiles(self, points):
        min_x_idx = np.argmin(points[:, 0])
        max_x_idx = np.argmax(points[:, 0])
        
        if max_x_idx >= min_x_idx:
            path_a = points[min_x_idx:max_x_idx+1]
        else:
            path_a = np.concatenate((points[min_x_idx:], points[:max_x_idx+1]))
            
        if min_x_idx >= max_x_idx:
            path_b = points[max_x_idx:min_x_idx+1]
        else:
            path_b = np.concatenate((points[max_x_idx:], points[:min_x_idx+1]))
            
        mean_y_a = np.mean(path_a[:, 1]) if len(path_a) > 0 else float('inf')
        mean_y_b = np.mean(path_b[:, 1]) if len(path_b) > 0 else float('inf')
        
        if mean_y_a < mean_y_b:
            return path_a, path_b
        else:
            return path_b, path_a

    def _calculate_angle(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2)
        if m1 == 0 or m2 == 0: return 180.0
        cos_angle = np.dot(v1, v2) / (m1 * m2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _fallback_bbox(self, mask):
        x, y, w, h = cv2.boundingRect(mask)
        poly = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
        return [mask], [poly]

class TrapezoidalDecompositionSplitting(SplittingStrategy):
    def __init__(self, epsilon_factor=0.02, debug_dir="out/debug_splitting"):
        self.epsilon_factor = epsilon_factor
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)
        self.step_count = 0

    def _save_debug_plot(self, title, image, polygons=None, lines=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.title(title)
        
        if polygons:
            for poly in polygons:
                if isinstance(poly, Polygon):
                    x, y = poly.exterior.xy
                    plt.plot(x, y, 'r-', linewidth=2)
                elif isinstance(poly, np.ndarray):
                    poly_reshaped = poly.reshape(-1, 2)
                    poly_closed = np.vstack([poly_reshaped, poly_reshaped[0]])
                    plt.plot(poly_closed[:,0], poly_closed[:,1], 'r-', linewidth=2)

        if lines:
            for line in lines:
                x, y = line.xy
                plt.plot(x, y, 'g--', linewidth=1)

        plt.axis('off')
        plt.savefig(os.path.join(self.debug_dir, f"step_{self.step_count:02d}_{title.replace(' ', '_')}.png"))
        plt.close()
        self.step_count += 1

    def split(self, binary_mask, original_image):
        self.step_count = 0
        h, w = binary_mask.shape
        
        # 1. Morphological Closing (Fill Bites)
        kernel = np.ones((15, 15), np.uint8) # Large kernel
        closed_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        self._save_debug_plot("1_Morphological_Closing", closed_mask)

        # 2. Contour Approximation
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return [], []
        main_contour = max(contours, key=cv2.contourArea)
        
        epsilon = self.epsilon_factor * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        approx = approx.squeeze()
        
        self._save_debug_plot("2_Contour_Approximation", closed_mask, polygons=[approx])

        # 3. Create Shapely Polygon
        if len(approx) < 3: return [], []
        poly_shapely = Polygon(approx)
        if not poly_shapely.is_valid:
            poly_shapely = poly_shapely.buffer(0)

        # 4. Vertical Decomposition
        # Create vertical lines at every vertex
        # Note: A pure trapezoidal decomposition also shoots rays from edge intersections,
        # but for this wall application, shooting from vertices is usually sufficient to handle concave shapes.
        
        minx, miny, maxx, maxy = poly_shapely.bounds
        unique_xs = sorted(list(set([p[0] for p in approx])))
        
        cut_lines = []
        for x in unique_xs:
            # Vertical line spanning height
            line = LineString([(x, miny - 1), (x, maxy + 1)])
            # Intersect with polygon to find the internal segments
            intersection = poly_shapely.intersection(line)
            
            if intersection.is_empty:
                continue
                
            if isinstance(intersection, LineString):
                cut_lines.append(intersection)
            elif isinstance(intersection, (MultiLineString, GeometryCollection)):
                for geom in intersection.geoms:
                    if isinstance(geom, LineString):
                        cut_lines.append(geom)

        self._save_debug_plot("3_Vertical_Cuts", closed_mask, polygons=[poly_shapely], lines=cut_lines)

        # 5. Split the Polygon
        splitter = unary_union(cut_lines)
        result_collection = split(poly_shapely, splitter)
        
        trapezoids = []
        for geom in result_collection.geoms:
            if isinstance(geom, Polygon):
                trapezoids.append(geom)

        self._save_debug_plot("4_Trapezoids", closed_mask, polygons=trapezoids)

        # 6. Convert back to OpenCV format
        wall_polygons = []
        wall_segments = []
        
        for trap in trapezoids:
            x, y = trap.exterior.xy
            pts = np.array([list(zip(x, y))], dtype=np.int32).squeeze()

            # Remove the closing point if it's duplicated (Shapely adds it)
            if len(pts) > 0 and np.array_equal(pts[0], pts[-1]):
                pts = pts[:-1]
            
            if len(pts) == 3:
                # Triangle: Duplicate the last point to make it a degenerate quad
                pts = np.vstack([pts, pts[-1]])
            
            elif len(pts) > 4:
                 # Simplify to 4 points
                 epsilon = 0.01 * cv2.arcLength(pts, True)
                 max_iter = 10
                 iter_count = 0
                 pts_simplified = pts
                 while len(pts_simplified) > 4 and iter_count < max_iter:
                     epsilon *= 1.5
                     approx = cv2.approxPolyDP(pts, epsilon, True)
                     pts_new = approx.squeeze()
                     if len(pts_new) < 3: # Collapsed too much
                         break
                     pts_simplified = pts_new
                     iter_count += 1
                 pts = pts_simplified
                 
                 # If we over-collapsed to triangle or invalid, handle it
                 if len(pts) == 3:
                      pts = np.vstack([pts, pts[-1]])
                 elif len(pts) < 3:
                      # Last resort: Bounding Box
                      rect = cv2.minAreaRect(np.array(trap.exterior.coords).astype(np.int32))
                      box = cv2.boxPoints(rect)
                      pts = box.astype(np.int32)

            if len(pts) == 4:
                wall_polygons.append(pts)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                wall_segments.append(mask)

        return wall_segments, wall_polygons
