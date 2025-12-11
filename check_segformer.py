from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch

try:
    model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    print("✅ SegFormer ADE found")
except Exception as e:
    print(f"❌ SegFormer ADE failed: {e}")

try:
    model_id = "nvidia/segformer-b0-finetuned-nyu"  # Guess
    processor = SegformerImageProcessor.from_pretrained(model_id)
    print("✅ SegFormer NYU found")
except Exception as e:
    print(f"❌ SegFormer NYU failed: {e}")
