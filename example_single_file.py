#  Example of processing a single SAS file with motion data

from src.geotiff_downscale import resample_geotiff
from src.crop_edges import crop_geotiff_left_right
import src.confidence_mapper as dpc
import src.sas_masker as csm


high_res_SAS_filepath = r"XX:\Data.tif"
output_dir = r"XX:\Data"
motion_file_path=r"XX:\Data.motions.hdf5"

# Downsample the high-resolution SAS file
downsample_filepath =resample_geotiff(high_res_SAS_filepath, output_dir, 0.10)

output_path = crop_geotiff_left_right(downsample_filepath, nadir_crop_m=45, outer_edge_crop_m=3) 

# # Convert motion data to geotif
tif_output_filename=dpc.main_confidence_to_coverage(motion_file_path, downsample_filepath, goal_resolution_m=1.0)

# Mask the SAS mask tif using the confidence values
print("Masking SAS with confidence values...")
masked_sas_filename = csm.mask_sas_with_confidence(
    sas_file_path=downsample_filepath,
    confidence_file_path=tif_output_filename,
    threshold=0.5
)



print(f"Original SAS: {high_res_SAS_filepath}")
print(f"Downsampled SAS: {downsample_filepath}")
print(f"Cropped SAS: {output_path}")
# print(f"Confidence map: {tif_output_filename}")
# print(f"Masked SAS: {masked_sas_filename}")
