import torch
import numpy as np
from PIL import Image
import time
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    OneFormerProcessor,
    OneFormerForUniversalSegmentation,
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

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
        
        # For wall-only evaluation
        self.target_ids = self.wall_ids
        
        wall_labels = [id2label[id] for id in self.wall_ids]
        floor_labels = [id2label[id] for id in self.floor_ids]

        print(f"Wall Class IDs: {self.wall_ids} (Labels: {wall_labels})")
        print(f"Floor Class IDs: {self.floor_ids} (Labels: {floor_labels})")
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


class OneFormerSegmentation(SegmentationStrategy):
    """
    OneFormer segmentation for ADE20K (wall-only). Avoids MPS due to float64 issues.
    """

    def __init__(self, model_id="shi-labs/oneformer_ade20k_swin_large"):
        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = "cuda"
        self.device = torch.device(device_name)
        print(f"Loading segmentation model: {model_id} on {self.device}...")
        self.processor = OneFormerProcessor.from_pretrained(model_id)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()

        id2label = self.model.config.id2label
        self.wall_ids = [i for i, lbl in id2label.items() if "wall" in lbl.lower()]
        self.target_ids = self.wall_ids
        print(f"Wall Class IDs: {self.wall_ids} (Labels: {[id2label[i] for i in self.wall_ids]})")

    def segment(self, image):
        start_time = time.time()
        inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [image.size[::-1]]
        prediction = self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
        prediction_np = prediction.cpu().numpy()
        runtime = time.time() - start_time
        binary_mask_np = np.isin(prediction_np, self.target_ids).astype(np.uint8)
        return binary_mask_np, runtime


class SegFormerSegmentation(SegmentationStrategy):
    """
    SegFormer segmentation for ADE20K (wall-only). Falls back to CPU on buffer/MPS errors.
    """

    def __init__(self, model_id="nvidia/segformer-b0-finetuned-ade-512-512"):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Loading segmentation model: {model_id} on {self.device}...")
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_id)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device)
        except Exception as e:
            print(f"Error loading SegFormer model {model_id}: {e}")
            print("Falling back to default ADE20K SegFormer...")
            fallback_id = "nvidia/segformer-b0-finetuned-ade-512-512"
            self.processor = SegformerImageProcessor.from_pretrained(fallback_id)
            self.model = SegformerForSemanticSegmentation.from_pretrained(fallback_id).to(self.device)

        self.model.eval()

        id2label = self.model.config.id2label
        self.wall_ids = [i for i, lbl in id2label.items() if "wall" in lbl.lower()]
        self.target_ids = self.wall_ids
        print(f"Wall Class IDs: {self.wall_ids} (Labels: {[id2label[i] for i in self.wall_ids]})")

    def segment(self, image):
        start_time = time.time()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            prediction = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]
        except Exception as e:
            if "buffer size" in str(e) or "MPS" in str(e):
                print(f"Error on {self.device}: {e}. Retrying on CPU...")
                self.model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                prediction = self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[image.size[::-1]]
                )[0]
                self.device = torch.device("cpu")
            else:
                raise e

        prediction_np = prediction.cpu().numpy()
        runtime = time.time() - start_time
        binary_mask_np = np.isin(prediction_np, self.target_ids).astype(np.uint8)
        return binary_mask_np, runtime


class NyuSegmentation(SegFormerSegmentation):
    """
    Wrapper for SegFormer model optionally trained on NYU Depth V2.
    Defaults to ADE20K SegFormer if a NYU checkpoint is not provided.
    """

    def __init__(self, model_id=None):
        if model_id is None:
            model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        super().__init__(model_id=model_id)

    # The save method can remain as a utility, but we will focus on the new segment output
    # def save_raw_segmentation(self, ...):
    #     ... (code for saving raw segmentation is fine)