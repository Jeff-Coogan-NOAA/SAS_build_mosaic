#  Example crop all files in directory and subdirectories

from src.crop_edges import crop_geotiff_left_right
import os
import glob
import shutil
from pathlib import Path

downsample_dir = r"XX:\Data"

# Create common crop output directory
common_crop_dir = os.path.join(downsample_dir, "crop")
os.makedirs(common_crop_dir, exist_ok=True)

# Find all .tif files recursively in all subdirectories
all_tif_files = glob.glob(os.path.join(downsample_dir, "**", "*.tif"), recursive=True)

# Filter out files with "motions" in the filename
tif_files = [f for f in all_tif_files if "motions" not in os.path.basename(f).lower()]

print(f"Found {len(all_tif_files)} total .tif files, filtering out {len(all_tif_files) - len(tif_files)} files with 'motions' in name")
print(f"Processing {len(tif_files)} .tif files")

# Loop through each .tif file and crop it
for tif_file in tif_files:
    print(f"Processing: {os.path.relpath(tif_file, downsample_dir)}")
    try:
        # Crop the file (this will create it in the local crop subdirectory)
        temp_output_path = crop_geotiff_left_right(tif_file, nadir_crop_m=45, outer_edge_crop_m=3)
        
        # Move the cropped file to the common crop directory
        filename = os.path.basename(temp_output_path)
        final_output_path = os.path.join(common_crop_dir, filename)
        
        # Handle duplicate filenames by adding a counter
        counter = 1
        base_name, ext = os.path.splitext(filename)
        while os.path.exists(final_output_path):
            new_filename = f"{base_name}_{counter}{ext}"
            final_output_path = os.path.join(common_crop_dir, new_filename)
            counter += 1
        
        shutil.move(temp_output_path, final_output_path)
        
        # Clean up the temporary local crop directory if it's empty
        temp_crop_dir = os.path.dirname(temp_output_path)
        if os.path.exists(temp_crop_dir) and not os.listdir(temp_crop_dir):
            os.rmdir(temp_crop_dir)
            
        print(f"Cropped SAS saved to: {os.path.relpath(final_output_path, downsample_dir)}")
    except Exception as e:
        print(f"Error processing {os.path.basename(tif_file)}: {e}")

print(f"Batch cropping complete! All cropped files saved to: {common_crop_dir}")
