import numpy as np

def get_side_from_filename(filepath):
    filename = str(filepath).lower()
    if "starboard" in filename:
        return "S"
    elif "port" in filename:
        return "P"
    else:
        return None  # or raise an error if needed

def crop_geotiff_left_right(input_path, nadir_crop_m=40, outer_edge_crop_m=3, output_suffix='_crop'):
    """
    Crop a GeoTIFF by left_crop_m meters from the left and right_crop_m meters from the right, then save as new file.
    Args:
        input_path (str): Path to input GeoTIFF.
        left_crop_m (float): Meters to crop from left.
        right_crop_m (float): Meters to crop from right.
        output_suffix (str): Suffix for output filename.
    """
    
    side = get_side_from_filename(input_path)
    if side=='S':
        left_crop_m = nadir_crop_m
        right_crop_m = outer_edge_crop_m
    elif side=='P':
        left_crop_m = outer_edge_crop_m
        right_crop_m = nadir_crop_m
    else:
        raise ValueError("side must be 'port' or 'starboard'")
    import rasterio
    from rasterio.transform import Affine
    import os
    import math

    with rasterio.open(input_path) as src:
        transform = src.transform

        d_lat, d_lon = d_ll(transform.f)
        # diagonal pixel spacing in longitude and latitude
        del_lon = transform.a + transform.b
        del_lat = transform.d + transform.e
        # convert diagonal spacing to meters
        del_n = del_lat / d_lat
        del_e = del_lon / d_lon

        resolution_m = np.sqrt(del_n**2 + del_e**2) * np.sqrt(2) / 2
        print(f"Resolution from rasterio (m): {resolution_m}")

        crs = src.crs
        width = src.width
        height = src.height

        left_pixels = round(left_crop_m / resolution_m)
        right_pixels = round(right_crop_m / resolution_m)
        new_width = width - left_pixels - right_pixels

        window = rasterio.windows.Window(left_pixels, 0, new_width, height)
        data = src.read(window=window)
        # Update transform for new origin
        new_transform = rasterio.windows.transform(window, transform)
        
        # Prepare output path with crop subdirectory
        base, ext = os.path.splitext(input_path)
        input_dir = os.path.dirname(base)
        filename = os.path.basename(base)
        crop_dir = os.path.join(input_dir, "crop")
        os.makedirs(crop_dir, exist_ok=True)
        output_path = os.path.join(crop_dir, f"{filename}{output_suffix}{ext}")
        
        # Save cropped image
        profile = src.profile.copy()
        profile.update({
            'width': new_width,
            'height': height,
            'transform': new_transform
        })
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
    return output_path



def d_ll(lat0_deg):
    def h_sin(x):
        return np.sin(np.radians(x / 2))**2

    def ah_sin(x):
        return 2 * np.degrees(np.arcsin(np.sqrt(x)))

    R = 6371e3  # average earth radius
    d_lat_deg = 1/R * 180 / np.pi
    d_lon_deg = ah_sin(h_sin(1/R) / (np.cos(np.radians(lat0_deg))**2)) * 180 / np.pi
    
    return d_lat_deg, d_lon_deg
