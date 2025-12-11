import torch
import numpy as np
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

class SegmentationStrategy:
    def segment(self, image):
        raise NotImplementedError

class Mask2FormerSegmentation(SegmentationStrategy):
    def __init__(self, model_id="facebook/mask2former-swin-large-ade-semantic"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading segmentation model: {model_id} on {self.device}...")
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # Identify wall classes
        self.wall_ids = [id for id, label in self.model.config.id2label.items() if 'wall' in label.lower()]
        print(f"Wall class IDs: {self.wall_ids}")

    def segment(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = [image.size[::-1]] # (H, W)
        prediction = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )[0]
        
        prediction_np = prediction.cpu().numpy()
        return prediction_np, self.wall_ids

