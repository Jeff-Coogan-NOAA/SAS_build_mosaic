#  Example crop all 10cm in directory

from src.crop_edges import crop_geotiff_left_right
import os
import glob

downsample_dir = r"XX:\Data"

# Find all .tif files in the directory
tif_files = glob.glob(os.path.join(downsample_dir, "*.tif"))

print(f"Found {len(tif_files)} .tif files to process")

# Loop through each .tif file and crop it
for tif_file in tif_files:
    print(f"Processing: {os.path.basename(tif_file)}")
    try:
        output_path = crop_geotiff_left_right(tif_file, nadir_crop_m=45, outer_edge_crop_m=3)
        print(f"Cropped SAS: {output_path}")
    except Exception as e:
        print(f"Error processing {os.path.basename(tif_file)}: {e}")

print("Batch cropping complete!")
