# Python code based on Dan Plotnick's MATLAB code for reducing the resolution of GeoTIFF files.
###############################################################################################################
###############################################################################################################

# NNNNNNNN        NNNNNNNN     OOOOOOOOO                 AAA                              AAA               
# N:::::::N       N::::::N   OO:::::::::OO              A:::A                            A:::A              
# N::::::::N      N::::::N OO:::::::::::::OO           A:::::A                          A:::::A             
# N:::::::::N     N::::::NO:::::::OOO:::::::O         A:::::::A                        A:::::::A            
# N::::::::::N    N::::::NO::::::O   O::::::O        A:::::::::A                      A:::::::::A           
# N:::::::::::N   N::::::NO:::::O     O:::::O       A:::::A:::::A                    A:::::A:::::A          
# N:::::::N::::N  N::::::NO:::::O     O:::::O      A:::::A A:::::A                  A:::::A A:::::A         
# N::::::N N::::N N::::::NO:::::O     O:::::O     A:::::A   A:::::A                A:::::A   A:::::A        
# N::::::N  N::::N:::::::NO:::::O     O:::::O    A:::::A     A:::::A              A:::::A     A:::::A       
# N::::::N   N:::::::::::NO:::::O     O:::::O   A:::::AAAAAAAAA:::::A            A:::::AAAAAAAAA:::::A      
# N::::::N    N::::::::::NO:::::O     O:::::O  A:::::::::::::::::::::A          A:::::::::::::::::::::A     
# N::::::N     N:::::::::NO::::::O   O::::::O A:::::AAAAAAAAAAAAA:::::A        A:::::AAAAAAAAAAAAA:::::A    
# N::::::N      N::::::::NO:::::::OOO:::::::OA:::::A             A:::::A      A:::::A             A:::::A   
# N::::::N       N:::::::N OO:::::::::::::OOA:::::A               A:::::A    A:::::A               A:::::A  
# N::::::N        N::::::N   OO:::::::::OO A:::::A                 A:::::A  A:::::A                 A:::::A 
# NNNNNNNN         NNNNNNN     OOOOOOOOO  AAAAAAA                   AAAAAAAAAAAAAA                   AAAAAAA

###############################################################################################################
###############################################################################################################                                                                                                          
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve
import rasterio
import os
from logging import info
import warnings
from pathlib import Path
from rasterio.transform import Affine
from rasterio.enums import ColorInterp
import numpy as np


def resample_geotiff(filepath, output_dir, goal_resolution_m=0.1):
    """
    Resample a GeoTIFF file to a new resolution.

    Parameters:
    -----------
    filepath : path to the GeoTIFF file full size
    goal_resolution_m : float
        Target resolution in meters (default is 0.1m).
    """
    

    try:
        info = imread_with_geotiff_info(filepath)
    except Exception as e:
        print(f"Error reading GeoTIFF file {filepath}: {e}")
        return

    
    # MATLAB: if ~isfield(info,'ModelTransformationTag') warningNotAnAffineGeotiff(file); end
    if 'ModelTransformationTag' not in info:
        print(f"Warning: The file {filepath} does not contain 'ModelTransformationTag'. It may not be an affine GeoTIFF.")
        return
    
    # === Data load in ===
    A = info['ModelTransformationTag']
    A = np.array(A).reshape(4, 4).T  


    with rasterio.open(filepath) as ds:
        I = ds.read()
        cmap_ = ds.colormap(1) if ds.count == 1 and ds.colormap(1) else None


    I2, A2 = resample_affine_image(I, A, goal_resolution_m)

    # === Data save out ===
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    res_cm = int(round(goal_resolution_m * 100))
    save_name = f"{base_name}_{res_cm}cm.tif"
    output_path = os.path.join(output_dir, save_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    output_filepath=write_affine_geotiff(output_path, I2, A2, cmap_, filepath)

    print(f"Resampled GeoTIFF saved to {output_path}")
    return output_filepath



def write_affine_geotiff(dst_path, image_data, affine_matrix, colormap=None, original_src=None):
    """
    Write a GeoTIFF file with affine transformation and colormap.
    
    Parameters:
    -----------
    dst_path : str
        Output file path for the GeoTIFF
    image_data : numpy.ndarray
        Image data to write (can be 2D or 3D array)
    affine_matrix : numpy.ndarray
        4x4 affine transformation matrix
    colormap : dict or matplotlib.colors.Colormap, optional
        Colormap to apply to the image
    original_src : str, optional
        Path to original file to copy metadata from
    """
    
    # Ensure image_data is properly shaped
    if image_data.ndim == 2:
        # Single band image
        height, width = image_data.shape
        count = 1
        image_data = image_data[np.newaxis, :, :]  # Add band dimension
    elif image_data.ndim == 3:
        # Multi-band image
        if image_data.shape[0] > image_data.shape[2]:
            # Assume bands are last dimension, transpose to bands-first
            image_data = np.transpose(image_data, (2, 0, 1))
        count, height, width = image_data.shape
    else:
        raise ValueError("Image data must be 2D or 3D array")
    
    transform = Affine(
        affine_matrix[0, 0],  # a: pixel width
        affine_matrix[1, 0],  # b: row rotation
        affine_matrix[3, 0],  # c: x-coordinate of upper-left corner
        affine_matrix[0, 1],  # d: column rotation  
        affine_matrix[1, 1],  # e: pixel height (negative)
        affine_matrix[3, 1]   # f: y-coordinate of upper-left corner
    )
    
    # Default metadata
    meta = {
        'driver': 'GTiff',
        'dtype': image_data.dtype,
        'nodata': None,
        'width': width,
        'height': height,
        'count': count,
        'crs': None,  # Will be updated from original if available
        'transform': transform,
        'compress': 'lzw',  # Good compression for most data
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512
    }
    
    # Copy metadata from original file if provided
    if original_src and os.path.exists(original_src):
        try:
            with rasterio.open(original_src) as src_ds:
                # Copy important metadata
                if src_ds.crs:
                    meta['crs'] = src_ds.crs
                if src_ds.nodata is not None:
                    meta['nodata'] = src_ds.nodata
                    
                # Copy other metadata tags
                meta.update(src_ds.tags())
                
        except Exception as e:
            warnings.warn(f"Could not read metadata from original file: {e}")
    
    # Ensure data type compatibility
    if image_data.dtype == np.float64:
        image_data = image_data.astype(np.float32)
        meta['dtype'] = np.float32
    elif image_data.dtype not in [np.uint8, np.uint16, np.int16, np.uint32, np.int32, np.float32]:
        # Convert unsupported types to appropriate format
        if image_data.min() >= 0 and image_data.max() <= 255:
            image_data = image_data.astype(np.uint8)
            meta['dtype'] = np.uint8
        else:
            image_data = image_data.astype(np.float32)
            meta['dtype'] = np.float32
    
    # Write the GeoTIFF
    with rasterio.open(dst_path, 'w', **meta) as dst_ds:
        # Write image data
        if count == 1:
            dst_ds.write(image_data[0], 1)
        else:
            for i in range(count):
                dst_ds.write(image_data[i], i + 1)
        
        # Apply colormap if provided
        if colormap is not None and count == 1:
            if isinstance(colormap, dict):
                # Convert rasterio colormap format to the expected format
                cmap_dict = {}
                for idx, rgba in colormap.items():
                    if isinstance(rgba, (list, tuple)) and len(rgba) >= 3:
                        # Ensure RGBA values are integers 0-255 and clamp them
                        if all(0 <= c <= 1 for c in rgba[:3]):  # Normalized values
                            r, g, b = [max(0, min(255, int(c * 255))) for c in rgba[:3]]
                        else:  # Already 0-255 values
                            r, g, b = [max(0, min(255, int(c))) for c in rgba[:3]]
                        a = max(0, min(255, int(rgba[3] * 255))) if len(rgba) > 3 else 255
                        
                        # Ensure idx is within valid range (0-65535 for short)
                        if 0 <= idx <= 65535:
                            cmap_dict[idx] = (r, g, b, a)
                
                if cmap_dict:
                    try:
                        dst_ds.write_colormap(1, cmap_dict)
                        dst_ds.colorinterp = [ColorInterp.palette]
                    except (OverflowError, ValueError) as e:
                        print(f"Warning: Could not write colormap: {e}")
                        print("Continuing without colormap...")
            
        # Add some custom tags to identify this as a resampled image
        dst_ds.update_tags(
            RESAMPLED='true',
            RESAMPLING_METHOD='affine_transformation',
            SOFTWARE='Python rasterio resample_geotiff'
        )
    
    print(f"Successfully wrote resampled GeoTIFF to: {dst_path}")
    return dst_path


def imread_with_geotiff_info(src):
    """
    Reads a GeoTIFF file and returns its metadata, image array, and colormap (if any).
    Prints all metadata keys and their types.
    """
    import tifffile

    with tifffile.TiffFile(src) as tif:
        info = {tag.name: tag.value for tag in tif.pages[0].tags.values()}
    return info


def resample_affine_image(I, A, goal_resolution_m):
    resolution_m = get_affine_resolution(A)
    R, U = get_resampling_factors(goal_resolution_m, resolution_m)
    I = filter_image(I, U)
    A2 = scale_affine(A, R)
    pA = pinv_affine(A)
    pA2 = pinv_affine(A2)
    n_rows, n_cols = get_target_size(I, pA2, A)
    I2 = interpolate_image(I, A2, pA, n_rows, n_cols)
    return I2, A2


# === Data Processing Subfunctions ===
# --- Image Processing ---
def filter_image(I, R):
    """
    Python equivalent of MATLAB filterImage(I,R)
    Handles both 2D and 3D (multi-band) images.
    """
    L = 2 * R + 1
    if I.ndim == 2:
        Itmp = pad_image(I, R)
        Itmp = hann_filter_2d(Itmp, L)
        I = unpad_image(Itmp, R)
    elif I.ndim == 3:
        bands = []
        for b in range(I.shape[0]):
            band = pad_image(I[b], R)
            band = hann_filter_2d(band, L)
            band = unpad_image(band, R)
            bands.append(band)
        I = np.stack(bands, axis=0)
    else:
        raise ValueError("Unsupported image dimensions for filtering.")
    return I


def pad_image(I, n_pad):

    pad = np.fliplr(I[:, :n_pad])
    I = np.concatenate([pad, I], axis=1)

    pad = np.fliplr(I[:, -n_pad:])
    I = np.concatenate([I, pad], axis=1)

    pad = np.flipud(I[:n_pad, :])
    I = np.concatenate([pad, I], axis=0)

    pad = np.flipud(I[-n_pad:, :])
    I = np.concatenate([I, pad], axis=0)
    
    return I


def unpad_image(I, n_pad):
    return I[n_pad:-n_pad, n_pad:-n_pad]


def hann_filter_2d(im, L):
    k = hann_kernel_2d(L)
    im = convolve(im, k, mode='constant')
    return im


def hann_kernel_2d(L):
    n = np.arange(L)
    k = 0.5 * (1 - np.cos(2 * np.pi * n / (L - 1)))
    k = np.outer(k, k)
    k = k / np.sum(k)
    
    return k


def get_resampling_factors(resolution_new, resolution_old):
    R = resolution_new / resolution_old

    U = int(np.ceil(R))
    
    return R, U

def interpolate_image(I, A2, pA, n_rows, n_cols):

    rows = np.arange(n_rows)
    cols = np.arange(n_cols)

    Cols, Rows = np.meshgrid(cols, rows, indexing='xy')

    Rows, Cols = remap_via_affine(A2, pA, Rows, Cols)

    # Handle both 2D and 3D arrays (squeeze singleton dimensions for single-band images)
    if I.ndim == 3 and I.shape[0] == 1:
        # Single band image with singleton first dimension
        Id = I[0, :, :].astype(float)  # Extract the 2D slice
        rows1 = np.arange(Id.shape[0])
        cols1 = np.arange(Id.shape[1])
    elif I.ndim == 2:
        # Already 2D
        Id = I.astype(float)
        rows1 = np.arange(Id.shape[0])
        cols1 = np.arange(Id.shape[1])
    else:
        raise ValueError(f"Unsupported image dimensions: {I.shape}")
    

    from scipy.interpolate import interpn
    points = (rows1, cols1)

    xi = np.column_stack([Rows.ravel(), Cols.ravel()])

    Inew_flat = interpn(points, Id, xi, method='linear', bounds_error=False, fill_value=0)
    Inew = Inew_flat.reshape(Rows.shape)    

    I2 = np.floor(Inew).astype(np.uint8)
    
    return I2

def remap_via_affine(A, pA2, Rows, Cols):
    A = pA2 @ A

    A1 = A[0, 0]
    A2 = A[0, 1] 
    A4 = A[0, 3]
    
    A5 = A[1, 0]
    A6 = A[1, 1]
    A8 = A[1, 3]
    
    Cols2 = Cols * A1 + Rows * A2 + A4
    Rows2 = Cols * A5 + Rows * A6 + A8
    
    return Rows2, Cols2


# --- Geodesy ---

def get_affine_resolution(A):

    d_lat, d_lon = d_ll(A[3, 1])

    del_lon = A[0, 0] + A[1, 0]
    del_lat = A[0, 1] + A[1, 1]

    del_n = del_lat / d_lat
    del_e = del_lon / d_lon

    resolution_m = np.sqrt(del_n**2 + del_e**2) * np.sqrt(2) / 2
    
    return resolution_m


def d_ll(lat0_deg):
    def h_sin(x):
        return np.sin(np.radians(x / 2))**2

    def ah_sin(x):
        return 2 * np.degrees(np.arcsin(np.sqrt(x)))

    R = 6371e3  # average earth radius
    d_lat_deg = 1/R * 180 / np.pi
    d_lon_deg = ah_sin(h_sin(1/R) / (np.cos(np.radians(lat0_deg))**2)) * 180 / np.pi
    
    return d_lat_deg, d_lon_deg


def pinv_affine(A):
    zeds = [A[2, 0], A[2, 1], A[0, 2], A[1, 2], A[2, 2], A[3, 2], A[0, 3], A[1, 3], A[2, 3]]

    assert all(np.isclose(zeds, 0, atol=1e-10)), "Invalid Affine Matrix"
    assert np.isclose(A[3,3], 1, atol=1e-10), "Invalid Affine Matrix"

    a = A[0,0]
    b = A[1,0]
    d = A[3,0]
    e = A[0,1]
    f = A[1,1]
    h = A[3,1]

    t1 = a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + ((d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)))*(b*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - d + a*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)))))/((b*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - d + a*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + (f*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - h + e*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + 1)
    t2 = e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + ((d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)))*(f*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - h + e*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)))))/((b*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - d + a*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + (f*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - h + e*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + 1)
    t3 = 0
    t4 = -(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)))/((b*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - d + a*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + (f*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - h + e*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + 1)
    t5 = (b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))*(b*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - d + a*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)))))/((b*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - d + a*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + (f*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - h + e*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + 1)
    t6 = (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))*(f*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - h + e*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)))))/((b*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - d + a*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + (f*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - h + e*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + 1)
    t7 = 0
    t8 = -((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))/((b*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - d + a*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + (f*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - h + e*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + 1)
    t9 = 0
    t10  = 0
    t11  = 0
    t12  = 0
    t13 = 0
    t14 = 0
    t15 = 0
    t16 = 1/((b*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - d + a*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + (f*((d*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2) + (h*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) - h + e*(d*(a/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2)) + h*(e/(a**2 + e**2) - (((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))*(f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2))))/((b - a*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2 + (f - e*((a*b)/(a**2 + e**2) + (e*f)/(a**2 + e**2)))**2))))**2 + 1)

    pA = np.zeros((4,4))
    pA[0,0] = t1
    pA[1,0] = t2
    pA[2,0] = t3
    pA[3,0] = t4
    pA[0,1] = t5
    pA[1,1] = t6
    pA[2,1] = t7
    pA[3,1] = t8
    pA[0,2] = 0
    pA[1,2] = 0
    pA[2,2] = 0
    pA[3,2] = 0
    pA[0,3] = 0
    pA[1,3] = 0
    pA[2,3] = 0
    pA[3,3] = 1

    return pA


def scale_affine(A, R):
    
    A2 = A.copy()
    A2[0:2, 0:2] = A2[0:2, 0:2] * R
    corner_old = A @ np.array([0.5, 0.5, 0, 1])
    corner_new = A2 @ np.array([0.5, 0.5, 0, 1])
    A2[3, 0] = A2[3, 0] + corner_old[0] - corner_new[0]
    A2[3, 1] = A2[3, 1] + corner_old[1] - corner_new[1]
    
    return A2


def get_target_size(I, A, pA2):

    sz = I.shape

    br_pixel = pA2 @ A @ np.array([sz[2], sz[1], 0, 1])

    n_rows = int(np.ceil(br_pixel[1]))
    n_cols = int(np.ceil(br_pixel[0]))
    
    return n_rows, n_cols