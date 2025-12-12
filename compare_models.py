import torch
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    OneFormerProcessor,
    OneFormerForUniversalSegmentation,
)
from PIL import Image
import numpy as np
import os
import gc

# Check for CUDA/MPS
# MPS often has buffer limits for large tensors, so we might fallback to CPU for large models/images
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Initial device selection: {device}")


def get_palette(num_classes=256):
    state = np.random.RandomState(0)
    palette = state.randint(0, 256, (num_classes, 3))
    return palette


def overlay_segmentation(image, segmentation, palette):
    color_segmentation = np.zeros(
        (segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8
    )
    # Resize segmentation to match image if needed (though they should match)
    if segmentation.shape != image.size[::-1]:
        # This case shouldn't happen with post_process_semantic_segmentation but good to be safe
        pass

    unique_labels = np.unique(segmentation)
    for label in unique_labels:
        if label == -1:
            continue  # Ignore index
        color_segmentation[segmentation == label] = palette[label % len(palette)]

    color_image = Image.fromarray(color_segmentation)
    return Image.blend(image.convert("RGB"), color_image, alpha=0.5)


def resize_image(image, max_size=1024):
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"Resizing image from {image.size} to ({new_w}, {new_h})")
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image


def run_mask2former(image_path, model_id="facebook/mask2former-swin-tiny-ade-semantic"):
    print(f"\nRunning Mask2Former ({model_id})...")
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(model_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(device)
        model.eval()

        image = Image.open(image_path).convert("RGB")
        image = resize_image(image)  # Resize to avoid OOM

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        segmentation = prediction.cpu().numpy()

        # Identify wall pixels
        wall_ids = [
            id for id, label in model.config.id2label.items() if "wall" in label.lower()
        ]
        wall_mask = np.isin(segmentation, wall_ids)
        print(f"Mask2Former Wall coverage: {np.sum(wall_mask) / wall_mask.size:.2%}")

        return segmentation, model.config.id2label, image
    except Exception as e:
        print(f"Error running Mask2Former on {device}: {e}")
        if device.type == "mps":
            # pass because we are just printing
            pass

            original_device = device
            local_device = torch.device("cpu")  # switch to CPU locally
            try:
                # Need to reload model to CPU
                model = model.to("cpu")
                inputs = inputs.to("cpu")
                with torch.no_grad():
                    outputs = model(**inputs)
                prediction = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[image.size[::-1]]
                )[0]
                segmentation = prediction.cpu().numpy()
                wall_ids = [
                    id
                    for id, label in model.config.id2label.items()
                    if "wall" in label.lower()
                ]
                wall_mask = np.isin(segmentation, wall_ids)
                print(
                    f"Mask2Former Wall coverage (CPU): {np.sum(wall_mask) / wall_mask.size:.2%}"
                )
                # device = original_device # No need to restore global var
                return segmentation, model.config.id2label, image
            except Exception as e2:
                print(f"Failed on CPU as well: {e2}")
                # device = original_device
                raise e2
        else:
            raise e


def run_oneformer(image_path, model_id="shi-labs/oneformer_ade20k_swin_tiny"):
    print(f"\nRunning OneFormer ({model_id})...")
    try:
        processor = OneFormerProcessor.from_pretrained(model_id)
        model = OneFormerForUniversalSegmentation.from_pretrained(model_id).to(device)
        model.eval()

        image = Image.open(image_path).convert("RGB")
        image = resize_image(image)

        # OneFormer requires task inputs.
        inputs = processor(
            images=image, task_inputs=["semantic"], return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        segmentation = prediction.cpu().numpy()

        # Identify wall pixels
        wall_ids = [
            id for id, label in model.config.id2label.items() if "wall" in label.lower()
        ]
        wall_mask = np.isin(segmentation, wall_ids)
        print(f"OneFormer Wall coverage: {np.sum(wall_mask) / wall_mask.size:.2%}")

        return segmentation, model.config.id2label, image

    except OSError as e:
        if "disk space" in str(e).lower() or "no space" in str(e).lower():
            print("Not enough disk space for large model.")
            if "swin_large" in model_id:
                tiny_model = "shi-labs/oneformer_ade20k_swin_tiny"
                print(f"Trying smaller model: {tiny_model}")
                return run_oneformer(image_path, model_id=tiny_model)
        raise e
    except Exception as e:
        print(f"Error running OneFormer on {device}: {e}")
        if device.type == "mps":
            # pass because we are just printing
            pass
            # global device removed because it causes syntax error if used after read

            original_device = device
            local_device = torch.device("cpu")
            try:
                # Clean up GPU memory if possible
                if "model" in locals():
                    del model
                if "inputs" in locals():
                    del inputs
                gc.collect()
                torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None

                # Reload on CPU
                processor = OneFormerProcessor.from_pretrained(model_id)
                model = OneFormerForUniversalSegmentation.from_pretrained(model_id).to(
                    "cpu"
                )
                model.eval()

                image = Image.open(image_path).convert("RGB")
                image = resize_image(image)
                inputs = processor(
                    images=image, task_inputs=["semantic"], return_tensors="pt"
                ).to("cpu")

                with torch.no_grad():
                    outputs = model(**inputs)

                prediction = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[image.size[::-1]]
                )[0]
                segmentation = prediction.cpu().numpy()

                wall_ids = [
                    id
                    for id, label in model.config.id2label.items()
                    if "wall" in label.lower()
                ]
                wall_mask = np.isin(segmentation, wall_ids)
                print(
                    f"OneFormer Wall coverage (CPU): {np.sum(wall_mask) / wall_mask.size:.2%}"
                )
                # device = original_device
                return segmentation, model.config.id2label, image
            except Exception as e2:
                print(f"Failed on CPU as well: {e2}")
                # device = original_device
                raise e2
        else:
            raise e


if __name__ == "__main__":
    image_path = "src/ourSet/Bhinu.png"
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}, please check path.")
        exit(1)

    print(f"Processing {image_path}")
    palette = get_palette()

    # Mask2Former
    try:
        m2f_seg, m2f_labels, m2f_img = run_mask2former(image_path)
        m2f_vis = overlay_segmentation(m2f_img, m2f_seg, palette)
        m2f_vis.save("comparison_mask2former.png")
        print("Saved comparison_mask2former.png")

        # Clean up
        del m2f_seg, m2f_labels, m2f_img, m2f_vis
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    except Exception as e:
        print(f"Mask2Former failed: {e}")

    # OneFormer
    try:
        of_seg, of_labels, of_img = run_oneformer(image_path)
        of_vis = overlay_segmentation(of_img, of_seg, palette)
        of_vis.save("comparison_oneformer.png")
        print("Saved comparison_oneformer.png")
    except Exception as e:
        print(f"OneFormer failed: {e}")

    print("Done.")
