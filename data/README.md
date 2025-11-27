# This File contains information of the data and instruction to attain it

We are making use of 2 datasets:
- Scannet
- Matterport3D

## Scannet
full Scannet documentation refers to: https://github.com/scannet/scannet#data-organization
### Space
The dataset storage 1.3TB is demanding for typical commodity hardware that we students use. Thus Rain added max amount of data limit functionality on orignal script and saved as download-scannet.py

#### Command to download the data and set max download limit to 5gb: 
```{bash}
python3 data/download-scannet.py -o ~/cs566FinalProject/data/scannet_data -s 5.0
```

#### Command to download specific scene [here scene0000_01]
```{bash}
python3 data/download-scannet.py -o ~/cs566FinalProject/data/scannet_data --id scene0000_01
```

#### Each scene
- scene0000_00: 3.88GB




## Matterport3D
Script to download the Matterport3D data: http://kaldir.vc.cit.tum.de/matterport/download_mp.py. 

Some useful info:
Scan data is named by a house hash id. The list of house hash ids is at http://kaldir.vc.cit.tum.de/matterport/v1/scans.txt. 
Script usage:
- To download the entire Matterport3D release (1.3TB): download-mp.py -o [directory in which to download] 
- To download a specific scan (e.g., 17DRP5sb8fy): download-mp.py -o [directory in which to download] --id 17DRP5sb8fy. This is listed in the directory as matterport-scans.txt
- To download a specific file type (e.g., *.sens, valid file suffixes listed here): download-mp.py -o [directory in which to download] --type .sens
- *.sens files can be read using the sens-File reader (it's a bit easier to handle than a larger number of separate image files)
### File types needed for this project

#### Summary
We need the following types of files from this dataset: 
- undistorted_color_images (Input image)
- undistorted_depth_images (Input geometry)
- undistorted_camera_parameters (Input camera matrices)
- house_segmentations (Ground truth for walls/floors)

#### color images (the visual input)
- Data type: undistorted_color_images
- Why: This is your primary input for all stages (Segmentation, Texture Transfer). These images are tone-mapped (good color) and, crucially, have radial distortion removed, making them geometrically superior for the later perspective correction stage. They are in JPG format.

#### depth images (the geometric constraint)
- Data Type: undistorted_depth_images 
- Why: Depth is absolutely critical for your "Geometric Accuracy" metric and for extracting the 3D plane equation. The undistorted depth images are approximately aligned pixel-for-pixel with the undistorted_color_images. You need this to calculate the 3D coordinates $(X, Y, Z)$ for every pixel in a segmented planar region.

#### camera parameters (the projection keys)
- Data Type: undistorted_camera_parameters
- Why: This file provides the undistorted intrinsic and extrinsic matrices for every image.Intrinsics ($K$): Needed to convert a pixel coordinate and its depth value into a 3D point in the camera frame.Extrinsics (Camera-to-World Matrix): Needed to transform the camera-frame 3D points into a consistent global coordinate system for geometric evaluation.

#### semantic segmentation ground truth
- Data Type: house_segmentations
- Why: To train and evaluate your Phase 3 (Mask2Former) and your Segmentation Quality metrics (IoU, Dice Coefficient), you need the ground truth labels for 'wall' and 'floor' regions. This information is contained in the semantic annotation files (like xxx.house, xxx.ply, and associated JSON files) which map 3D mesh faces to semantic categories. You'll likely need to process this 3D ground truth to generate 2D ground truth masks aligned with your undistorted_color_images.

### Commands
be aware that this script is using the python2 syntax!
This forum discussed how to install python2.7 on ubuntu22.04 and 24.04: https://askubuntu.com/questions/1527867/python-2-7-12-install-on-ubuntu-22-04

#### Commad to download selected data types for all houses
<YOUR_OUTPUT_BASE_DIR> = data/matterport_data
```{bash}
python2.7 original-download-matterport.py \
    -o <YOUR_OUTPUT_BASE_DIR> \ 
    --id ALL \
    --type undistorted_color_images undistorted_depth_images undistorted_camera_parameters house_segmentations
```

#### Command to Download Selected Data Types for a Single House
```{bash}
python2.7 original-download-matterport.py \
    -o <YOUR_OUTPUT_BASE_DIR> \
    --id <SCAN_ID> \
    --type undistorted_color_images undistorted_depth_images undistorted_camera_parameters house_segmentations
```
For example, to get the 17DRP5sb8fy house (run inside the data directory):
```{bash}
python2.7 original-download-matterport.py \
    -o matterport_data \
    --id 1LXtFkjw3qL \
    --type undistorted_color_images undistorted_depth_images undistorted_camera_parameters house_segmentations
```