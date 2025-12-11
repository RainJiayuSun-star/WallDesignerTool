import torch
import numpy as np
from PIL import Image
import time
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# Define abstract base class (remains the same)
class SegmentationStrategy:
    def segment(self, image):
        # Must return the prediction mask and the time taken (float)
        raise NotImplementedError

# Revised Mask2Former Segmentation Class
class Mask2FormerSegmentation(SegmentationStrategy):
    """
    Implements Mask2Former segmentation revised for benchmarking.
    The segment method now returns the wall/floor binary mask and runtime.
    """
    def __init__(self, model_id="facebook/mask2former-swin-large-ade-semantic"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading segmentation model: {model_id} on {self.device}...")
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # --- Define Target Class IDs for Benchmarking ---
        id2label = self.model.config.id2label
        
        # Identify wall classes (assuming 'wall' in label)
        self.wall_ids = [id for id, label in id2label.items() if 'wall' in label.lower()]
        
        # Identify floor classes (Need to manually check/confirm ADE20K IDs for 'floor')
        # Common ADE20K floor IDs: 5 (floor), 8 (carpet), 20 (rug/mat), 21 (tile), 27 (ceiling - optional)
        self.floor_ids = [id for id, label in id2label.items() if 'floor' in label.lower()]
        
        # Combine target IDs for the aggregated surface benchmark
        self.target_ids = self.wall_ids + self.floor_ids
        
        print(f"Wall Class IDs: {self.wall_ids}")
        print(f"Floor Class IDs: {self.floor_ids}")
        print(f"Total Target IDs: {len(self.target_ids)}")

    def segment(self, image):
        """
        Performs segmentation, times the process, and generates the target binary mask.
        
        Returns:
            - binary_mask_np (np.array): HxW array where 1=Target Surface (Wall/Floor), 0=Other.
            - runtime (float): Time taken for core segmentation (seconds).
        """
        # 1. Start Timer
        start_time = time.time()
        
        # 2. Core Segmentation Logic
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = [image.size[::-1]] # (H, W)
        prediction = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )[0]
        
        prediction_np = prediction.cpu().numpy()
        
        # 3. Stop Timer
        end_time = time.time()
        runtime = end_time - start_time
        
        # 4. Generate Target Binary Mask for Benchmarking
        # Check if each pixel's class ID is in the list of target IDs (walls/floors)
        binary_mask_np = np.isin(prediction_np, self.target_ids).astype(np.uint8)
        
        # Return the standardized benchmark output
        return binary_mask_np, runtime

    # The save method can remain as a utility, but we will focus on the new segment output
    # def save_raw_segmentation(self, ...):
    #     ... (code for saving raw segmentation is fine)