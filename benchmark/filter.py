import pandas as pd
import os
import glob
from pathlib import Path

# --- Configuration ---
BENCHMARK_DIR = Path(__file__).resolve().parent
METADATA_FILE = BENCHMARK_DIR / "metadata_filtered.csv"
BENCHMARK_OUTPUT_DIR = BENCHMARK_DIR / "benchmark_output"

# CORRECTED: The column in metadata_filtered.csv that contains the filenames to match
METADATA_FILENAME_COLUMN = "filename" 
# CORRECTED: The column that identifies the original benchmark result in the large CSV
RESULT_FILENAME_COLUMN = "image_file" 

def filter_benchmark_results():
    """
    Reads metadata_filtered.csv, filters the benchmark results from 
    four models based on the filenames in the metadata, and saves 
    the filtered results to new preprocessed CSV files.
    """
    
    print(f"--- Starting benchmark result filtering ---")

    # 1. Load the metadata file and extract target filenames
    if not METADATA_FILE.exists():
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return
        
    try:
        metadata_df = pd.read_csv(METADATA_FILE)
        print(f"Loaded metadata from {METADATA_FILE}. Total rows: {len(metadata_df)}")
    except Exception as e:
        print(f"Error loading metadata CSV: {e}")
        return

    # Extract the list of filenames to keep from the metadata file
    if METADATA_FILENAME_COLUMN not in metadata_df.columns:
        print(f"Error: The required column '{METADATA_FILENAME_COLUMN}' was not found in {METADATA_FILE}.")
        return
        
    # Create a set for fast lookup of target filenames
    target_filenames = set(metadata_df[METADATA_FILENAME_COLUMN].astype(str).tolist())
    print(f"Extracted {len(target_filenames)} unique target filenames from metadata.")
    
    # 2. Identify and Process Benchmark Output Files
    
    # Use glob to find all relevant benchmark result files in the output directory
    search_pattern = str(BENCHMARK_OUTPUT_DIR / "benchmark_results_*.csv")
    benchmark_files = glob.glob(search_pattern)

    if not benchmark_files:
        print(f"Warning: No benchmark result files found in {BENCHMARK_OUTPUT_DIR}.")
        return

    # Process each benchmark file
    processed_count = 0
    for file_path in benchmark_files:
        path = Path(file_path)
        print(f"\nProcessing file: {path.name}")
        
        try:
            # Load the benchmark results
            benchmark_df = pd.read_csv(path)
            print(f"  Total rows in benchmark results: {len(benchmark_df)}")
            
            # Ensure the necessary column for filtering is present
            if RESULT_FILENAME_COLUMN not in benchmark_df.columns:
                print(f"  Skipping: Required column '{RESULT_FILENAME_COLUMN}' not found in {path.name}.")
                continue

            # 3. Perform the filtering
            # Filter where the 'image_file' column matches any filename in our target set
            filtered_df = benchmark_df[
                benchmark_df[RESULT_FILENAME_COLUMN].astype(str).isin(target_filenames)
            ]
            
            print(f"  Filtered rows: {len(filtered_df)}")
            
            # 4. Save the preprocessed results
            # Create the new filename: {original_name}_preprocessed.csv
            new_filename = f"{path.stem}_preprocessed.csv"
            new_path = path.parent / new_filename
            
            # Save the new filtered dataframe
            filtered_df.to_csv(new_path, index=False)
            print(f"  Successfully saved preprocessed results to: {new_path.name}")
            processed_count += 1

        except Exception as e:
            print(f"  An error occurred while processing {path.name}: {e}")

    print(f"\n--- Filtering complete. {processed_count} files successfully preprocessed. ---")

if __name__ == "__main__":
    filter_benchmark_results()