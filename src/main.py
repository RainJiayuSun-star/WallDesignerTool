import argparse
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from datasets import load_dataset

# Import modular components
from segmentation import Mask2FormerSegmentation, OneFormerSegmentation
from splitting import CannyHoughSplitting, WallRefinerSplitting, ContourCornerSplitting, RobustCornerSplitting, CeilingKinkSplitting, CeilingAndFloorKinkSplitting, TrapezoidalDecompositionSplitting
from mapping import HomographyMultiplyMapping, MaskedPerspectiveMapping


def get_ade20k_palette():
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(151, 3), dtype=np.uint8)
    return palette


def visualize_prediction(image, pred_map, ax, title="Prediction", target_ids=None):
    palette = get_ade20k_palette()
    color_seg = np.zeros((pred_map.shape[0], pred_map.shape[1], 3), dtype=np.uint8)
    for label_id in np.unique(pred_map):
        if target_ids is not None and label_id not in target_ids:
            continue
        if label_id < len(palette):
            color_seg[pred_map == label_id] = palette[label_id]
    ax.imshow(image)
    ax.imshow(color_seg, alpha=0.5)
    ax.set_title(title)
    ax.axis("off")


def check_ade20k_criteria(example):
    # 1. Check scene sequence
    # scene has [ indoor, home or hotel ] both
    scene_labels = example.get('scene', [])
    if not scene_labels:
        return False
    
    # scene_labels might be a list of strings or list of ints (ClassLabel)
    # We convert to string and lower case to be safe if they are strings.
    # If they are ints, this check will fail unless we have the mapping, 
    # but based on the prompt we assume they are strings or we match string representation.
    scene_str = [str(s).lower() for s in scene_labels]
    
    has_indoor = False
    has_home_or_hotel = False
    
    for s in scene_str:
        if 'indoor' in s:
            has_indoor = True
        if 'home' in s or 'hotel' in s:
            has_home_or_hotel = True
            
    if not (has_indoor and has_home_or_hotel):
        return False
        
    # 2. Check objects list
    # The user provided example shows a list of dicts.
    # HF datasets might return dict of lists. We handle both.
    objects = example.get('objects', [])
    if not objects:
        return False
        
    has_valid_wall = False
    
    if isinstance(objects, list):
        # List of dicts
        for obj in objects:
            raw_name = obj.get('raw_name')
            occluded = obj.get('occluded')
            crop = obj.get('crop')
            
            if raw_name == 'wall' and (occluded is False) and (crop is False):
                has_valid_wall = True
                break
    elif isinstance(objects, dict):
        # Dict of lists (HF Sequence)
        raw_names = objects.get('raw_name', [])
        occluded_list = objects.get('occluded', [])
        cropped_list = objects.get('crop', [])
        
        # Ensure lists are same length
        if len(raw_names) == len(occluded_list) == len(cropped_list):
            for i in range(len(raw_names)):
                if (raw_names[i] == 'wall' and 
                    occluded_list[i] is False and 
                    cropped_list[i] is False):
                    has_valid_wall = True
                    break
            
    return has_valid_wall

def load_examples(args):
    examples = []
    if args.filelist:
        # Process images from filelist - search in HuggingFace dataset
        if not os.path.exists(args.filelist):
            print(f"Filelist {args.filelist} does not exist.")
            return []
        
        print(f"Loading images from filelist {args.filelist}...")
        print(f"Searching for files in ADE20K dataset...")
        
        # Read filenames from file
        with open(args.filelist, 'r') as f:
            target_filenames = [line.strip() for line in f if line.strip()]
        
        # Create a set for faster lookup
        target_filenames_set = set(target_filenames)
        found_filenames = set()
        
        try:
            # Load dataset in streaming mode
            dataset = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
            iterator = iter(dataset)
            
            print(f"Searching for {len(target_filenames)} files in dataset...")
            checked = 0
            
            # Iterate through dataset to find matching filenames
            while len(found_filenames) < len(target_filenames):
                try:
                    ex = next(iterator)
                    checked += 1
                    if checked % 1000 == 0:
                        print(f"Checked {checked} examples... Found {len(found_filenames)}/{len(target_filenames)}")
                    
                    ex_filename = ex.get('filename', '')
                    if ex_filename in target_filenames_set:
                        examples.append(ex)
                        found_filenames.add(ex_filename)
                        print(f"Found: {ex_filename}")
                        
                except StopIteration:
                    print("End of dataset reached.")
                    break
            
            # Check which files were not found
            not_found = target_filenames_set - found_filenames
            for filename in not_found:
                print(f"File not found: {filename}")
            
            if not examples:
                print(f"No images found from filelist {args.filelist}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
    elif args.local:
        print(f"Loading local images from {args.dir}...")
        if not os.path.exists(args.dir):
            print(f"Directory {args.dir} does not exist.")
            return []
        image_files = glob.glob(os.path.join(args.dir, "*.png"))
        image_files.sort()
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                filename = os.path.basename(img_path)
                examples.append({"image": img, "filename": filename})
                print(f"Loaded image: {filename}")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        if not examples:
            print(f"No PNG images found in {args.dir}")
    else:
        print(f"Loading ADE20K dataset stream with filtering for {args.num_examples} examples...")
        try:
            # Load dataset in streaming mode
            dataset = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
            
            # Shuffle with buffer to get random samples
            dataset = dataset.shuffle(seed=42, buffer_size=1000)
            
            iterator = iter(dataset)
            found_count = 0
            target_count = args.num_examples # Sample N random photos
            
            print("Searching for matching examples...")
            checked = 0
            while found_count < target_count:
                try:
                    ex = next(iterator)
                    checked += 1
                    if checked % 100 == 0:
                        print(f"Checked {checked} examples...")
                except StopIteration:
                    print("End of dataset reached.")
                    break
                
                if check_ade20k_criteria(ex):
                     examples.append(ex)
                     print(f"Found matching example: {ex.get('filename', 'Unknown')}")
                     found_count += 1
            
            if not examples:
                print("No matching examples found.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Wall Designer Tool Pipeline")
    parser.add_argument(
        "--segmentationMethod",
        type=str,
        default="mask2former",
        choices=["mask2former", "oneformer", "segformer", "nyu"],
        help="Method for wall segmentation",
    )
    parser.add_argument(
        "--splittingMethod",
        type=str,
        default="ceilingKink",
        choices=["cannyhough", "refiner", "contourCorner", "ceilingKink", "ceilingFloorKink", "trapezoidal"],
        help="Method for splitting connected walls",
    )
    parser.add_argument(
        "--textureMappingMethod",
        type=str,
        default="maskedPerspective",
        choices=["homographyMultiply", "maskedPerspective"],
        help="Method for applying texture",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="HuggingFace Model ID (overrides default for selected method)",
    )
    parser.add_argument(
        "--local", action="store_true", help="Use local images from --dir"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "ourSet"),
        help="Directory containing local images",
    )
    parser.add_argument(
        "--texture",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dummy_texture.png"),
        help="Path to texture file",
    )
    parser.add_argument("--num_examples", type=int, default=2, help="Number of random examples to fetch from ADE20K")
    parser.add_argument("--output_dir", type=str, default="out", help="Directory to save results")
    parser.add_argument("--save", action="store_true", help="Save results to output directory")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--filelist",
        type=str,
        default=None,
        help="Path to text file containing filenames (one per line) to process. Files will be searched in the HuggingFace ADE20K dataset by filename field.",
    )

    args = parser.parse_args()

    # 1. Initialize Components
    if args.segmentationMethod == "mask2former":
        kwargs = {"model_id": args.model_id} if args.model_id else {}
        segmenter = Mask2FormerSegmentation(**kwargs)
    elif args.segmentationMethod == "oneformer":
        kwargs = {"model_id": args.model_id} if args.model_id else {}
        segmenter = OneFormerSegmentation(**kwargs)
    elif args.segmentationMethod == "segformer":
        from segmentation import SegFormerSegmentation

        kwargs = {"model_id": args.model_id} if args.model_id else {}
        segmenter = SegFormerSegmentation(**kwargs)
    elif args.segmentationMethod == "nyu":
        print("--- NYU Depth V2 Specialist ---")
        print(
            "Note: Official semantic segmentation checkpoints for NYU are rare on HF Hub."
        )
        print(
            "Using SegFormer architecture. If you have a specific NYU checkpoint, pass it via --model_id."
        )
        from segmentation import NyuSegmentation

        kwargs = {"model_id": args.model_id} if args.model_id else {}
        segmenter = NyuSegmentation(**kwargs)

    if args.splittingMethod == "cannyhough":
        splitter = CannyHoughSplitting()
    elif args.splittingMethod == "refiner":
        splitter = WallRefinerSplitting()
    elif args.splittingMethod == "contourCorner":
        # Updated to use RobustCornerSplitting as requested
        # sensitivity=0.002 catches shallow corners
        # angle_threshold=5 ignores minor wiggly lines
        splitter = RobustCornerSplitting(sensitivity=0.002, angle_threshold=5)
    elif args.splittingMethod == "ceilingKink":
        splitter = CeilingKinkSplitting(epsilon_factor=0.003, bend_threshold=10, margin_top=5)
    elif args.splittingMethod == "ceilingFloorKink":
        splitter = CeilingAndFloorKinkSplitting(epsilon_factor=0.003, bend_threshold=10, margin=5)
    elif args.splittingMethod == "trapezoidal":
        splitter = TrapezoidalDecompositionSplitting()
    if args.textureMappingMethod == "homographyMultiply":
        mapper = HomographyMultiplyMapping(args.texture)
    elif args.textureMappingMethod == "maskedPerspective":
        mapper = MaskedPerspectiveMapping(args.texture)

    # 2. Load Data
    examples = load_examples(args)
    if not examples:
        return

    # Create output directory
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    num_examples = len(examples)
    print(f"Processing {num_examples} examples...")

    for idx, example in enumerate(examples):
        filename_raw = example.get('filename', f"example_{idx+1}")
        # Sanitize filename if needed
        filename_base = os.path.splitext(os.path.basename(filename_raw))[0]
        
        image = example['image']
        image_np = np.array(image)

        # A. Segmentation
        print(f"[{idx+1}/{num_examples}] Segmenting {filename_raw}...")
        pred_map, wall_ids = segmenter.segment(image)

        # B. Splitting
        print(f"[{idx+1}/{num_examples}] Splitting {filename_raw}...")
        full_wall_mask = np.zeros(pred_map.shape, dtype=np.uint8)
        for w_id in wall_ids:
            full_wall_mask[pred_map == w_id] = 1  # Keep 0/1 for now

        # Ensure mask is correct format if needed by splitter, but our splitter handles it.
        # WallRefinerSplitting expects 0-255 internally or handles 0-1.
        
        # RobustCornerSplitting can use the image for verification
        segments, polygons = splitter.split(full_wall_mask, image_np)

        # C. Mapping
        print(f"[{idx+1}/{num_examples}] Mapping texture for {filename_raw}...")
        # Prepare full_wall_mask as 0-255/bool for mapper if needed
        full_wall_mask_255 = (full_wall_mask * 255).astype(np.uint8)

        textured_image = mapper.apply(image_np, polygons, full_mask=full_wall_mask_255)

        # D. Visualization & Saving
        
        # Create a figure for this example
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # Col 1: Original
        axes[0].imshow(image)
        axes[0].set_title(f"{filename_raw}")
        axes[0].axis('off')

        # Col 2: Segmentation (Wall Blob)
        visualize_prediction(image, pred_map, axes[1], title="Segmentation (Wall Blob)", target_ids=wall_ids)

        # Col 3: Splitting (Wall Edges/Polygons)
        vis_split = image_np.copy()
        debug_img = np.array(image).copy()
        for i, poly in enumerate(polygons):
            # Draw the full trapezoid in Green
            cv2.polylines(vis_split, [poly], True, (0, 255, 0), 2)
            cv2.polylines(debug_img, [poly], True, (0, 255, 0), 2)
            # Draw JUST the ceiling line in Red (to confirm logic)
            cv2.line(debug_img, tuple(poly[0]), tuple(poly[1]), (255, 0, 0), 4)
            # Label center
            M = cv2.moments(poly)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(vis_split, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(debug_img, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        axes[2].imshow(debug_img)
        axes[2].set_title(f"Splitting (Red=Ceiling)")
        axes[2].axis('off')

        # Col 4: Texture Applied
        axes[3].imshow(textured_image)
        axes[3].set_title("Texture Applied")
        axes[3].axis('off')

        plt.tight_layout()
        
        if args.save:
            # Save plot
            plot_path = os.path.join(args.output_dir, f"{filename_base}_visualization.png")
            plt.savefig(plot_path)
            print(f"Saved visualization to {plot_path}")
            plt.close(fig) # Close figure to free memory
            
            # Save textured image separately
            tex_path = os.path.join(args.output_dir, f"{filename_base}_textured.png")
            # Convert RGB to BGR for cv2
            cv2.imwrite(tex_path, cv2.cvtColor(textured_image, cv2.COLOR_RGB2BGR))
            print(f"Saved textured image to {tex_path}")

    if not args.save:
        plt.show()
    
    print("Done.")


if __name__ == "__main__":
    main()