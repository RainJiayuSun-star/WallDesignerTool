import torch
import numpy as np
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    OneFormerProcessor,
    OneFormerForUniversalSegmentation,
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


class SegmentationStrategy:
    def segment(self, image):
        raise NotImplementedError


class Mask2FormerSegmentation(SegmentationStrategy):
    def __init__(self, model_id="facebook/mask2former-swin-large-ade-semantic"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading segmentation model: {model_id} on {self.device}...")
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(
            self.device
        )
        self.model.eval()

        # Identify wall classes
        self.wall_ids = [
            id
            for id, label in self.model.config.id2label.items()
            if "wall" in label.lower()
        ]
        print(f"Wall class IDs: {self.wall_ids}")

    def segment(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [image.size[::-1]]  # (H, W)
        prediction = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )[0]

        prediction_np = prediction.cpu().numpy()
        return prediction_np, self.wall_ids


class OneFormerSegmentation(SegmentationStrategy):
    def __init__(self, model_id="shi-labs/oneformer_ade20k_swin_large"):
        # Explicitly avoid MPS for OneFormer due to known issues with float64 support
        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = "cuda"

        self.device = torch.device(device_name)
        print(f"Loading segmentation model: {model_id} on {self.device}...")
        self.processor = OneFormerProcessor.from_pretrained(model_id)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_id).to(
            self.device
        )
        self.model.eval()

        # Identify wall classes
        self.wall_ids = [
            id
            for id, label in self.model.config.id2label.items()
            if "wall" in label.lower()
        ]
        print(f"Wall class IDs: {self.wall_ids}")

    def segment(self, image):
        # OneFormer requires explicit task input
        inputs = self.processor(
            images=image, task_inputs=["semantic"], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [image.size[::-1]]  # (H, W)
        prediction = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )[0]

        prediction_np = prediction.cpu().numpy()
        return prediction_np, self.wall_ids


class SegFormerSegmentation(SegmentationStrategy):
    def __init__(self, model_id="nvidia/segformer-b0-finetuned-ade-512-512"):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Loading segmentation model: {model_id} on {self.device}...")
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_id)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(
                self.device
            )
        except Exception as e:
            print(f"Error loading SegFormer model {model_id}: {e}")
            print("Falling back to default ADE20K SegFormer...")
            fallback_id = "nvidia/segformer-b0-finetuned-ade-512-512"
            self.processor = SegformerImageProcessor.from_pretrained(fallback_id)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                fallback_id
            ).to(self.device)

        self.model.eval()

        # Identify wall classes
        self.wall_ids = [
            id
            for id, label in self.model.config.id2label.items()
            if "wall" in label.lower()
        ]
        print(f"Wall class IDs: {self.wall_ids}")

    def segment(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = [image.size[::-1]]  # (H, W)
            prediction = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )[0]
        except Exception as e:
            if "buffer size" in str(e) or "MPS" in str(e):
                print(f"Error on {self.device}: {e}. Retrying on CPU...")
                # Move to CPU
                self.model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)

                target_sizes = [image.size[::-1]]
                prediction = self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes
                )[0]

                # Move back to original device if needed, or stay on CPU?
                # For now, let's just leave it on CPU to prevent ping-pong if next image is also large
                self.device = torch.device("cpu")
            else:
                raise e

        prediction_np = prediction.cpu().numpy()
        return prediction_np, self.wall_ids


class NyuSegmentation(SegFormerSegmentation):
    """
    Wrapper for SegFormer model trained on NYU Depth V2.
    Defaults to ADE20K fallback if no specific NYU ID is provided, but intended for use with NYU checkpoints.
    """

    def __init__(self, model_id=None):
        # Default to a generic SegFormer if none provided, but we warn in main.py
        # Ideally we would use "nvidia/segformer-b0-finetuned-nyu" if it existed publicly.
        if model_id is None:
            model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        super().__init__(model_id=model_id)
