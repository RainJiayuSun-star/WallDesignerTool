import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run Wall Segmentation Pipeline.")
    parser.add_argument("-v", "--version", type=str, default="1", help="Version of the pipeline to run (default: 1)")
    parser.add_argument("--local", action="store_true", help="Use local images from --dir instead of streaming dataset")
    parser.add_argument("--dir", type=str, default="ourSet", help="Directory containing local images (default: ourSet)")
    parser.add_argument("--texture", type=str, default="texture.jpg", help="Path to texture file for wall mapping (default: texture.jpg)")
    
    # Allow passing unknown args just in case specific versions have their own args (optional, but good practice if extensions differ)
    # For now we'll just use parse_args
    args = parser.parse_args()

    version = args.version
    
    # Default to version 1 if unknown or explicitly 1
    if version == "1":
        print(f"Running version {version} (Canny-Hough Wall Splitting)...")
        import mask2former_cannyhoughwallsplitting
        mask2former_cannyhoughwallsplitting.run(args)
    elif version == "2":
        print(f"Running version {version} (Texture Mapping)...")
        import mask2former_texture_mapping
        mask2former_texture_mapping.run(args)
    else:
        # Current default is also version 1 logic
        print(f"Version '{version}' not explicitly found. Defaulting to latest (Version 1)...")
        import mask2former_cannyhoughwallsplitting
        mask2former_cannyhoughwallsplitting.run(args)

if __name__ == "__main__":
    main()
