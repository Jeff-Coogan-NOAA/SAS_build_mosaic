import os

def mask_sas_with_confidence(sas_file_path, confidence_file_path, threshold=0.65, output_path=None):
    """
    Mask a SAS GeoTIFF using confidence values from another GeoTIFF.
    
    Args:
        sas_file_path (str): Path to the original SAS GeoTIFF file
        confidence_file_path (str): Path to the confidence GeoTIFF file
        threshold (float): Confidence threshold value (default: 0.65)
        output_path (str): Output path for masked GeoTIFF (optional)
    
    Returns:
        str: Path to the output masked GeoTIFF file
    """
    try:
        import rasterio
        import numpy as np
        
        # Generate output filename if not provided
        if output_path is None:
            output_path = sas_file_path.replace('.tif', '_masked.tif')
        
        # Read the SAS GeoTIFF
        with rasterio.open(sas_file_path) as sas_src:
            sas_data = sas_src.read(1)  # Read first band
            sas_profile = sas_src.profile.copy()
            sas_transform = sas_src.transform
            
        # Read the confidence GeoTIFF
        with rasterio.open(confidence_file_path) as conf_src:
            conf_data = conf_src.read(1)  # Read first band
            conf_transform = conf_src.transform
            
        print(f"SAS data shape: {sas_data.shape}, dtype: {sas_data.dtype}")
        print(f"Confidence data shape: {conf_data.shape}, dtype: {conf_data.dtype}")
        print(f"SAS transform: {sas_transform}")
        print(f"Confidence transform: {conf_transform}")
        
        # Check if the datasets have the same dimensions and transform
        if sas_data.shape != conf_data.shape:
            print("Warning: SAS and confidence data have different shapes. Attempting to resize...")
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            zoom_y = sas_data.shape[0] / conf_data.shape[0]
            zoom_x = sas_data.shape[1] / conf_data.shape[1]
            
            # Resize confidence data to match SAS data
            conf_data = zoom(conf_data, (zoom_y, zoom_x), order=1)
            print(f"Resized confidence data to: {conf_data.shape}")
        
        # Create mask: True where confidence >= threshold
        mask = conf_data >= threshold
        
        # Convert SAS data to float32 to handle NaN values
        masked_sas = sas_data.astype(np.float32)
        
        # Apply mask to SAS data
        masked_sas[~mask] = np.nan  # Set low confidence areas to NaN
        
        # Alternatively, you could set them to a specific value like 0:
        # masked_sas[~mask] = 0
        
        print(f"Applied mask with threshold {threshold}")
        print(f"Pixels above threshold: {np.sum(mask)} / {mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")
        
        # Update profile for output - ensure float32 and set nodata value
        sas_profile.update(
            dtype=rasterio.float32,
            nodata=np.nan
        )
        
                
    
        mask_dir = os.path.join(os.path.dirname(sas_file_path), "mask")
        os.makedirs(mask_dir, exist_ok=True)
        output_path = os.path.join(mask_dir, os.path.basename(sas_file_path).replace('.tif', '_masked.tif'))
        
        # Write the masked GeoTIFF
        with rasterio.open(output_path, 'w', **sas_profile) as dst:
            dst.write(masked_sas, 1)
        
        print(f"Masked SAS GeoTIFF saved to: {output_path}")
        return output_path
        
    except ImportError:
        print("Rasterio not available. Please install it: pip install rasterio")
        return None
    except Exception as e:
        print(f"Error masking SAS file: {e}")
        import traceback
        traceback.print_exc()
        return None

def mask_sas_with_confidence_simple(sas_file_path, confidence_file_path, threshold=0.65, output_path=None):
    """
    Simple version using tifffile if rasterio is not available.
    
    Args:
        sas_file_path (str): Path to the original SAS GeoTIFF file
        confidence_file_path (str): Path to the confidence GeoTIFF file  
        threshold (float): Confidence threshold value (default: 0.65)
        output_path (str): Output path for masked GeoTIFF (optional)
    
    Returns:
        str: Path to the output masked GeoTIFF file
    """
    try:
        import tifffile
        import numpy as np
        
        # Generate output filename if not provided
        if output_path is None:
            output_path = sas_file_path.replace('.tif', '_masked.tif')
        
        # Read the SAS GeoTIFF
        sas_data = tifffile.imread(sas_file_path)
        
        # Read the confidence GeoTIFF
        conf_data = tifffile.imread(confidence_file_path)
        
        print(f"SAS data shape: {sas_data.shape}")
        print(f"Confidence data shape: {conf_data.shape}")
        
        # Resize if needed
        if sas_data.shape != conf_data.shape:
            print("Warning: Different shapes. Consider using the rasterio version for proper resampling.")
            # Simple resize using numpy (not recommended for geospatial data)
            from scipy.ndimage import zoom
            zoom_y = sas_data.shape[0] / conf_data.shape[0]
            zoom_x = sas_data.shape[1] / conf_data.shape[1]
            conf_data = zoom(conf_data, (zoom_y, zoom_x), order=1)
        
        # Create mask and apply
        mask = conf_data >= threshold
        masked_sas = sas_data.copy().astype(np.float32)
        masked_sas[~mask] = np.nan
        
        print(f"Applied mask with threshold {threshold}")
        print(f"Pixels above threshold: {np.sum(mask)} / {mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")
        
        # Save using tifffile
        tifffile.imwrite(output_path, masked_sas, compress='lzw')
        
        print(f"Masked SAS GeoTIFF saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error masking SAS file: {e}")
        return None