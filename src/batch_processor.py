"""
Batch SAS Processing Functions
Functions for processing multiple SAS .tif files with their corresponding motion files.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional

# Import the required modules
from .geotiff_downscale import resample_geotiff
from . import confidence_mapper as dpc
from . import sas_masker as csm


def find_tif_files(input_directory: str) -> List[str]:
    """
    Find all .tif files in subdirectories of the input directory.
    
    Args:
        input_directory (str): Path to the input directory
        
    Returns:
        List[str]: List of .tif file paths
    """
    tif_files = []
    input_path = Path(input_directory)
    
    # Search for .tif files recursively
    for tif_file in input_path.rglob("*.tif"):
        tif_files.append(str(tif_file))
        
    return sorted(tif_files)


def find_motion_file(tif_filepath: str) -> Optional[str]:
    """
    Find corresponding motion file for a .tif file.
    Motion file is expected to be in a 'logs' subdirectory with the same base name
    but with '.motions.hdf5' extension instead of '.tif'.
    
    Args:
        tif_filepath (str): Path to the .tif file
        
    Returns:
        Optional[str]: Path to the motion file if found, None otherwise
    """
    tif_path = Path(tif_filepath)
    
    # Get the base name without extension
    base_name = tif_path.stem
    
    # Look for logs subdirectory in the same directory as the .tif file
    logs_dir = tif_path.parent / "logs"
    
    # Construct expected motion file path
    motion_file = logs_dir / f"{base_name}.motions.hdf5"
    
    if motion_file.exists():
        return str(motion_file)
    else:
        return None


def get_valid_file_pairs(input_directory: str) -> List[Tuple[str, str]]:
    """
    Get all valid .tif and motion file pairs from the input directory.
    
    Args:
        input_directory (str): Path to the input directory
        
    Returns:
        List[Tuple[str, str]]: List of (tif_file, motion_file) pairs
    """
    tif_files = find_tif_files(input_directory)
    valid_pairs = []
    
    for tif_file in tif_files:
        motion_file = find_motion_file(tif_file)
        if motion_file:
            valid_pairs.append((tif_file, motion_file))
    
    return valid_pairs


def process_single_file_pair(tif_file: str, motion_file: str, output_dir: str,
                           resolution: float = 0.10, goal_resolution: float = 1.0,
                           threshold: float = 0.5, verbose: bool = True) -> dict:
    """
    Process a single .tif and motion file pair.
    
    Args:
        tif_file (str): Path to the .tif file
        motion_file (str): Path to the motion file
        output_dir (str): Output directory for processed files
        resolution (float): Downsampling resolution in meters (default: 0.10)
        goal_resolution (float): Motion goal resolution in meters (default: 1.0)
        threshold (float): Confidence threshold (default: 0.5)
        verbose (bool): Whether to print progress messages
        
    Returns:
        dict: Dictionary containing paths to output files
    """
    if verbose:
        print(f"Processing: {os.path.basename(tif_file)}")
    
    # Use main output directory directly (no subfolders)
    file_output_dir = output_dir
    os.makedirs(file_output_dir, exist_ok=True)
    
    results = {
        'original_tif': tif_file,
        'motion_file': motion_file,
        'output_dir': file_output_dir
    }
    
    try:
        # Step 1: Downsample the SAS file
        if verbose:
            print(f"  Downsampling to {resolution}m resolution...")
        downsample_filepath = resample_geotiff(tif_file, file_output_dir, resolution)
        results['downsampled_tif'] = downsample_filepath
        if verbose:
            print(f"  Downsampled file: {os.path.basename(downsample_filepath)}")
        
        # Step 2: Build motion data geotiff
        if verbose:
            print("  Building confidence coverage map...")
        tif_output_filename = dpc.main_confidence_to_coverage(
            motion_file, downsample_filepath, goal_resolution_m=goal_resolution)
        results['confidence_map'] = tif_output_filename
        if verbose:
            print(f"  Confidence map: {os.path.basename(tif_output_filename)}")
        
        # Step 3: Mask the SAS tif using confidence values
        if verbose:
            print("  Creating masked SAS file...")
        masked_sas_filename = csm.mask_sas_with_confidence(
            sas_file_path=downsample_filepath,
            confidence_file_path=tif_output_filename,
            threshold=threshold
        )
        results['masked_sas'] = masked_sas_filename
        if verbose:
            print(f"  Masked SAS: {os.path.basename(masked_sas_filename)}")
        
        results['success'] = True
        results['error'] = None
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        if verbose:
            print(f"  Error: {str(e)}")
    
    return results


def batch_process_sas_files(input_directory: str, output_directory: str,
                          resolution: float = 0.10, goal_resolution: float = 1.0,
                          threshold: float = 0.5, verbose: bool = True) -> List[dict]:
    """
    Process all valid .tif and motion file pairs in the input directory.
    
    Args:
        input_directory (str): Path to the input directory containing .tif files
        output_directory (str): Path to the output directory for processed files
        resolution (float): Downsampling resolution in meters (default: 0.10)
        goal_resolution (float): Motion goal resolution in meters (default: 1.0)
        threshold (float): Confidence threshold (default: 0.5)
        verbose (bool): Whether to print progress messages
        
    Returns:
        List[dict]: List of results for each processed file
    """
    # Get all valid file pairs
    file_pairs = get_valid_file_pairs(input_directory)
    
    if not file_pairs:
        if verbose:
            print("No valid .tif and motion file pairs found")
        return []
    
    if verbose:
        print(f"Found {len(file_pairs)} valid file pairs to process")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Process each file pair
    results = []
    for i, (tif_file, motion_file) in enumerate(file_pairs):
        if verbose:
            print(f"\n--- Processing {i+1}/{len(file_pairs)} ---")
        
        result = process_single_file_pair(
            tif_file, motion_file, output_directory,
            resolution, goal_resolution, threshold, verbose
        )
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    if verbose:
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Output saved to: {output_directory}")
    
    return results


# Example usage function
def example_usage():
    """
    Example of how to use the batch processing functions.
    """
    # Example paths (modify these for your use case)
    input_dir = r"V:\EN2501_AzureSent\SAS\DIVE009_SN402\processing\geotiff\En2501_Dive009_03cm"
    output_dir = r"V:\EN2501_AzureSent\SAS\DIVE009_SN402\processing\geotiff\batch_processed"
    
    # Preview files that will be processed
    print("Previewing files...")
    file_pairs = get_valid_file_pairs(input_dir)
    for tif_file, motion_file in file_pairs:
        print(f"  {os.path.basename(tif_file)} -> {os.path.basename(motion_file)}")
    
    # Process all files
    results = batch_process_sas_files(
        input_directory=input_dir,
        output_directory=output_dir,
        resolution=0.10,  # 10cm resolution
        goal_resolution=1.0,  # 1m resolution for motion
        threshold=0.5,  # Confidence threshold
        verbose=True
    )
    
    return results


if __name__ == "__main__":
    # Run example usage
    example_usage()
