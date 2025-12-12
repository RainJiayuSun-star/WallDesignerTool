import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# define an abstract base class for any segmentation approach
class SegmentationStrategy:
    def segment(self, image):
        raise NotImplementedError

# define an abstract base class for segmentation approach
class Mask2FormerSegmentation(SegmentationStrategy):
    """
    Implements wall segmentation using a pre-trained Mask2Former model
    """
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

    # method to save the output as a file
    def save_raw_segmentation(self, prediction_np, file_path):
        """
        Saves the raw class ID Numpy array to a file
        """
        np.save(file_path, prediction_np)
        print(f"Raw segmentation saved to {file_path}")
        