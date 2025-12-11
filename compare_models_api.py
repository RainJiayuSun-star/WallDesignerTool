import requests
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import os
import json

# You can assume a valid HF_TOKEN is in the environment for now,
# or we can ask the user if it's missing.
HF_TOKEN = os.environ.get("HF_TOKEN")

# Models
MASK2FORMER_ID = "facebook/mask2former-swin-large-ade-semantic"
# Note: OneFormer large might not be on the free API, but we'll try.
# Sometimes the inference API URL convention differs.
ONEFORMER_ID = "shi-labs/oneformer_ade20k_swin_large"


def query_hf_api(image_path, model_id):
    if not HF_TOKEN:
        print(
            "Warning: HF_TOKEN not found. API might be rate limited or fail for gated models."
        )
        headers = {}
    else:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Updated API URL based on recent changes/error message
    api_url = f"https://router.huggingface.co/models/{model_id}"

    with open(image_path, "rb") as f:
        data = f.read()

    print(f"Querying {model_id}...")
    response = requests.post(api_url, headers=headers, data=data)

    if response.status_code != 200:
        # Try to print a concise error if it's HTML
        if "html" in response.headers.get("content-type", "").lower():
            raise Exception(
                f"API Request failed: {response.status_code} - (HTML response, likely 404/410/Auth error)"
            )
        raise Exception(f"API Request failed: {response.status_code} - {response.text}")

    return response.json()


def get_palette(num_classes=256):
    state = np.random.RandomState(0)
    palette = state.randint(0, 256, (num_classes, 3))
    return palette


def decode_mask(mask_data):
    # The API might return different formats.
    # Often for semantic segmentation it returns a list of labels with masks.
    # The mask itself is often base64 encoded.
    if isinstance(mask_data, str):
        # Base64 string
        img_data = base64.b64decode(mask_data)
        img = Image.open(BytesIO(img_data))
        return np.array(img)
    return None


def process_api_response(response_json, image_size):
    # Response is typically a list of dicts: [{'label': 'wall', 'score': 0.99, 'mask': 'base64...'}]
    # We need to reconstruct the full segmentation map.
    # Initialize empty map
    h, w = image_size
    full_segmentation = np.zeros((h, w), dtype=int) - 1  # -1 for background/unknown

    # Sort by score or process in order? Usually higher score overwrites?
    # Or maybe the API returns non-overlapping masks?
    # For semantic segmentation, they might be layers.

    print(f"Received {len(response_json)} segments from API.")

    # We need a mapping from label string to integer ID for visualization
    label_to_id = {}
    next_id = 0

    # Reverse order might be better if they are ordered by confidence?
    # Actually, let's just iterate.
    for segment in response_json:
        label = segment.get("label")
        mask_str = segment.get("mask")  # This might be the base64 string

        if not mask_str:
            print(f"Skipping segment {label} (no mask data)")
            continue

        if label not in label_to_id:
            label_to_id[label] = next_id
            next_id += 1

        mask_array = decode_mask(mask_str)
        if mask_array is not None:
            # Resize if needed (API might resize input)
            if mask_array.shape != (h, w):
                # Simple resize
                m_img = Image.fromarray(mask_array)
                m_img = m_img.resize((w, h), Image.Resampling.NEAREST)
                mask_array = np.array(m_img)

            # Where mask is active (usually white/255)
            # mask_array might be boolean or 0-255
            mask_bool = mask_array > 128
            full_segmentation[mask_bool] = label_to_id[label]

    return full_segmentation, label_to_id


def overlay_segmentation(image, segmentation, palette):
    color_segmentation = np.zeros(
        (segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8
    )
    unique_labels = np.unique(segmentation)
    for label in unique_labels:
        if label == -1:
            continue
        color_segmentation[segmentation == label] = palette[label % len(palette)]

    color_image = Image.fromarray(color_segmentation)
    return Image.blend(image.convert("RGB"), color_image, alpha=0.5)


if __name__ == "__main__":
    image_path = "src/ourSet/Bhinu.png"
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        exit(1)

    image = Image.open(image_path).convert("RGB")
    # Resize for API speed/limits
    max_size = 1024
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        # Save temp file for upload
        image.save("temp_api_upload.jpg")
        upload_path = "temp_api_upload.jpg"
    else:
        upload_path = image_path

    palette = get_palette()

    # 1. Mask2Former
    try:
        res = query_hf_api(upload_path, MASK2FORMER_ID)
        # Check if response is list (success) or dict (error/loading)
        if isinstance(res, dict) and "error" in res:
            print(f"Mask2Former API Error: {res['error']}")
        else:
            seg, labels = process_api_response(res, image.size[::-1])  # h, w
            vis = overlay_segmentation(image, seg, palette)
            vis.save("comparison_mask2former_api.png")
            print("Saved comparison_mask2former_api.png")
            print("Labels found:", labels.keys())

            # Check for 'wall'
            wall_labels = [l for l in labels.keys() if "wall" in l.lower()]
            print(f"Wall labels identified: {wall_labels}")

    except Exception as e:
        print(f"Mask2Former API failed: {e}")

    # 2. OneFormer
    try:
        res = query_hf_api(upload_path, ONEFORMER_ID)
        if isinstance(res, dict) and "error" in res:
            print(f"OneFormer API Error: {res['error']}")
        else:
            seg, labels = process_api_response(res, image.size[::-1])
            vis = overlay_segmentation(image, seg, palette)
            vis.save("comparison_oneformer_api.png")
            print("Saved comparison_oneformer_api.png")
            print("Labels found:", labels.keys())
            # Check for 'wall'
            wall_labels = [l for l in labels.keys() if "wall" in l.lower()]
            print(f"Wall labels identified: {wall_labels}")

    except Exception as e:
        print(f"OneFormer API failed: {e}")

    if os.path.exists("temp_api_upload.jpg"):
        os.remove("temp_api_upload.jpg")
