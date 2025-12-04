# Mask2Former deployment for Segmentation
consider: https://huggingface.co/saninmohammedn/mask2former-deployment
The Mask2Former explanation: https://huggingface.co/docs/transformers/en/model_doc/mask2former

## Current Files
The followings are drafts/examples to run mask2former models:
- facebookModelGPU_walls.py uses the model trained on ADE20K, using pytoch for GPU acceleration
- facebookModel.py is the initial draft using cpu, the model is trained on coco, not optimized for wall segmentation
- facebookModelGPU_instances.py uses the same model trained on coco, but using GPU acceleration 