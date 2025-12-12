import requests


def check_model(model_id):
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    print(f"Checking {model_id}...")
    response = requests.head(url)
    if response.status_code == 200:
        print(f"✅ Model {model_id} exists!")
        return True
    else:
        print(f"❌ Model {model_id} not found (status {response.status_code}).")
        return False


models_to_check = [
    "facebook/mask2former-swin-large-scannet-semantic",
    "facebook/mask2former-swin-base-scannet-semantic",
    "facebook/mask2former-swin-small-scannet-semantic",
    "facebook/mask2former-swin-large-nyu-semantic",  # Unlikely
    "facebook/mask2former-swin-tiny-scannet-semantic",
    "nvidia/segformer-b0-finetuned-ade-512-512",  # Standard
    "nvidia/segformer-b5-finetuned-ade-640-640",
]

for m in models_to_check:
    check_model(m)
