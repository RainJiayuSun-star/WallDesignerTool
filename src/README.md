# Wall Designer Tool

A Python pipeline for segmenting walls in interior images, splitting them into distinct surfaces (e.g., left wall, right wall), and applying perspective-correct textures.

## Usage

Run the main script from the `src` directory:

```bash
python main.py [options]
```

## Options

### General
- `--save`: Save results to the output directory. If not provided, results will be displayed interactively using matplotlib.
- `--output_dir`: Directory to save results (default: `out`).
- `--num_examples`: Number of random examples to fetch from the ADE20K dataset (default: 2).
- `--show`: (Deprecated/Unused) Interactive show mode is now the default if `--save` is not provided.

### Input Data
- `--local`: Use local images from the directory specified by `--dir` instead of downloading from ADE20K.
- `--dir`: Directory containing local images (default: `ourSet`). Only used if `--local` is set.
- `--filelist`: Path to a text file containing filenames (one per line). These files will be searched for in the HuggingFace ADE20K dataset.
- `--texture`: Path to the texture image file to apply to the walls (default: `dummy_texture.png`).

### Pipeline Methods

#### Segmentation
Control how walls are identified in the image.
- `--segmentationMethod`: 
  - `mask2former` (default): Uses Mask2Former model.
  - `oneformer`: Uses OneFormer model.
  - `segformer`: Uses SegFormer model.
  - `nyu`: Uses a model specialized for NYU Depth V2 (often SegFormer based).
- `--model_id`: Specify a custom HuggingFace Model ID to override the default for the selected segmentation method.

#### Wall Splitting
Control how the detected wall "blob" is split into individual planar surfaces.
- `--splittingMethod`:
  - `ceilingKink` (default): Splits based on kinks in the ceiling line.
  - `ceilingFloorKink`: Splits based on kinks in both ceiling and floor lines.
  - `cannyhough`: Uses Canny edge detection and Hough transform.
  - `refiner`: Uses a refinement based splitting approach.
  - `contourCorner`: Uses robust corner detection on the wall contour.

#### Texture Mapping
Control how the texture is projected onto the wall surfaces.
- `--textureMappingMethod`:
  - `maskedPerspective` (default): Applies perspective transformation masked to the wall shape.
  - `homographyMultiply`: Applies homography multiplication.

## Examples

**Run on 2 random ADE20K images and display results:**
```bash
python main.py
```

**Run on local images in `my_images/` and save results to `results/`:**
```bash
python main.py --local --dir my_images --save --output_dir results
```

**Use specific methods:**
```bash
python main.py --segmentationMethod oneformer --splittingMethod contourCorner
```
