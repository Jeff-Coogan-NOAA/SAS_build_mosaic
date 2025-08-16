import numpy as np
from scipy.interpolate import interp1d
import h5py
import matplotlib.pyplot as plt
from . import affine_utils as make_scaled_affine 
import tifffile
from rasterio.transform import Affine

def dpc_confidence_to_coverage_map(input_data, goal_resolution_m=1.0):
    """
    Uses an averaging method to map single ping DPC Coherence/Confidence to
    the image frame.
    
    Args:
        input_data (dict): Input data structure containing required fields
        goal_resolution_m (float): Goal resolution in meters (default: 1.0)
    
    Returns:
        tuple: (C_out, count) - Coverage map and count array
    """
    # Validate input
    if not validate_input(input_data):
        raise ValueError("Invalid input data structure")
    
    threshold = 0.1  # Minimum confidence value to consider valid

    # --- Extract and describe variables from input structure ---
    offset = input_data['ImagingGrid']['Offset']  # dict: {'X', 'Y'} - grid offset in meters
    step = input_data['ImagingGrid']['Step']      # dict: {'X', 'Y'} - grid step size in meters
    size = input_data['ImagingGrid']['Size']      # dict: {'X', 'Y'} - grid size (number of steps)

    az_limit = input_data['Parameters']['AzLimit']  # float: sensor azimuth field of view (radians)
    side = input_data['Side']                       # str: 'port' or 'starboard'

    alt = input_data['Nav']['Altitude']             # np.array: vehicle altitude for each ping (meters)
    pos = input_data['Nav']['Position']             # dict: {'X', 'Y'} arrays of vehicle positions (meters)
    yaw = input_data['Nav']['Yaw']                  # np.array: vehicle yaw for each ping (radians)

    meters_per_column = input_data['TimeDelays']['MetersPerColumn']  # float: meters per range column
    C = np.abs(input_data['TimeDelays']['Confidence'])  # np.array: confidence matrix (range x ping)

    # --- Create imaging grid and range vector ---
    Y, X = make_imaging_grid(step, offset, size, goal_resolution_m)  # Y, X: meshgrid arrays (meters)
    print(f"Imaging grid created with shape: {Y.shape}, {X.shape}")
    print(f"Offset: {offset}, Step: {step}, Size: {size}")

    R = make_range_vector(C, meters_per_column)                      # np.array: range vector (meters)
    sd = make_side_sign(side)                                       # int: 1 for port, 0 for starboard  

    n_pings = C.shape[0]  # int: number of pings (MATLAB: size(C,2))

    # --- Initialize mean and count arrays for output ---
    count = np.zeros(X.shape, dtype=C.dtype)  # np.array: number of valid pings per grid cell
    mn = np.zeros(X.shape, dtype=C.dtype)     # np.array: sum of confidence values per grid cell

    # --- Main loop over pings ---
    for n in range(n_pings):

        r = np.sqrt(R**2 + alt[n]**2)  # Ground range

        # Shift grid to be system referenced
        if n == n_pings - 1:  # Last ping
            del_x = (pos['X'][n] - pos['X'][n-1]) / 2
            del_y = (pos['Y'][n] - pos['Y'][n-1]) / 2
            Xg = X - pos['X'][n] + del_x
            Yg = Y - pos['Y'][n] + del_y
        else:
            Xg = X - (pos['X'][n] + pos['X'][n+1]) / 2
            Yg = Y - (pos['Y'][n] + pos['Y'][n+1]) / 2


        # Convert to polar coordinates
        # Xg, Yg: grid points in meters relative to image course
        # tht: polar angle of grid points rel. image course
        # Polar angle of grid points rel. image course
        tht = np.arctan2(Xg, Yg)
        rg = np.sqrt(Xg**2 + Yg**2)

        # Polar angle of grid rel. port/stbd image boresight. Unwrapped.
        tht = np.angle(np.exp(1j * (tht + sd * np.pi)))

        # Account for sensor yaw rel. image course
        if n == n_pings - 1:  # Last ping
            del_tht = np.angle(np.exp(1j * (yaw[n] - yaw[n-1]))) / 2
        else:
            del_tht = np.angle(np.exp(1j * (yaw[n+1] - yaw[n]))) / 2
        tht_b = tht + yaw[n] + del_tht

        # Check if points are in the FOV of the sensor
        good = np.abs(tht_b) < az_limit / 2

        # Use only good points, and make sure we are centered on DPC range
        # cells, not on edges
        rg_i = rg[good] + meters_per_column / 2

        # rgi = rg(good(:)) + MetersPerColumn/2;

        # % Main interpolation
        # tmp = interp1(r,C(:,n).',rgi,'nearest',0);

        # Main interpolation
        # issue rg_i is off the chart
        if len(rg_i) > 0:
            # Create interpolation function - MATLAB: C(:,n)
            interp_func = interp1d(r, C[n, :], kind='nearest', 
                                 bounds_error=False, fill_value=0)


            tmp = interp_func(rg_i)

            # Aggregate onto our sum and count grids
            count_up = tmp >= threshold
            sum_up = tmp * count_up

            # Update count and mean arrays
            count[good] += count_up
            mn[good] += sum_up
        # else:
        #     # print("No good points for this ping.")
    
    # Now get our mean, and remove edge cases
    bad = count == 0
    norm = np.ones_like(count, dtype=float)
    norm[~bad] = 1.0 / count[~bad]  # Inverse here for stability
    norm[bad] = 0
    
    C_out = mn * norm
    # print(mn)
    # print(norm)
    # plt.contourf(Y,X,C_out, levels=20, cmap='hot')
    # plt.colorbar()
    # plt.title("DPC Confidence to Coverage Map")
    # plt.xlabel("Range (m)")
    # plt.ylabel("Cross-range (m)")
    # plt.show()


    return C_out, count


def make_side_sign(side):
    """Convert side string to sign value"""
    if side.lower() == 'port':
        return 1
    else:
        return 0


def make_range_vector(C, meters_per_column):
    """Create range vector from confidence matrix"""
    return np.arange(C.shape[1]) * meters_per_column


def make_imaging_grid(step, offset, size, goal_resolution_m):
    """Create imaging grid"""
    sz_x = float(size['X'])
    sz_y = float(size['Y'])
    
    x_start = offset['X']
    x_stop = (sz_x - 1) * step['X'] + offset['X']
    y_start = offset['Y']
    y_stop = (sz_y - 1) * step['Y'] + offset['Y']
    
    goal_resolution_y = goal_resolution_m * np.sign(step['Y'])
    print(f"goal_resolution_y: {goal_resolution_y}")
    x = np.arange(x_start, x_stop , goal_resolution_m)
    y = np.arange(y_start, y_stop , goal_resolution_y)
    
    if np.sign(step['Y']) == -1:  # port side data
        y = np.flip(y)
    
    # Create meshgrid (note: MATLAB meshgrid has different convention)
    Y, X = np.meshgrid(y, x)
    return Y, X


def validate_input(input_data):
    """Validate input data structure"""
    if not isinstance(input_data, dict):
        return False
    
    # Check main fields
    required_fields = ['ImagingGrid', 'Nav', 'Parameters', 'Side', 'TimeDelays']
    for field in required_fields:
        if field not in input_data:
            return False
    
    # Validate ImagingGrid
    if not validate_imaging_grid(input_data['ImagingGrid']):
        return False
    
    # Validate Nav
    if not validate_nav(input_data['Nav']):
        return False
    
    # Validate Parameters
    if not validate_parameters(input_data['Parameters']):
        return False
    
    # Validate Side
    if not validate_side(input_data['Side']):
        return False
    
    # Validate TimeDelays
    if not validate_time_delays(input_data['TimeDelays']):
        return False
    
    return True


def validate_imaging_grid(imaging_grid):
    """Validate ImagingGrid structure"""
    required_fields = ['Offset', 'Step', 'Size']
    for field in required_fields:
        if field not in imaging_grid:
            return False
    return True


def validate_nav(nav):
    """Validate Nav structure"""
    required_fields = ['Altitude', 'Position', 'Yaw']
    for field in required_fields:
        if field not in nav:
            return False
    return True


def validate_parameters(parameters):
    """Validate Parameters structure"""
    return 'AzLimit' in parameters


def validate_side(side):
    """Validate Side field"""
    return side.lower() in ['starboard', 'port']


def validate_time_delays(time_delays):
    """Validate TimeDelays structure"""
    required_fields = ['Confidence', 'MetersPerColumn']
    for field in required_fields:
        if field not in time_delays:
            return False
    return True


# Example usage and test function
def load_hdf5_data(file_path):
    """
    Load data from HDF5 file and convert to expected dictionary structure
    
    Args:
        file_path (str): Path to the HDF5 file
        
    Returns:
        dict: Data structure compatible with dpc_confidence_to_coverage_map
    """
    with h5py.File(file_path, 'r') as f:
        print("Available groups in HDF5 file:")
        def print_structure(name, obj):
            print(name)
        f.visititems(print_structure)
        input_data = {}
        # Offset
        offset_arr = f["ImagingGrid/Offset"][:][0]
        input_data['ImagingGrid'] = {}
        input_data['ImagingGrid']['Offset'] = {'X': float(offset_arr[0]), 'Y': float(offset_arr[1])}
        # Step
        step_arr = f["ImagingGrid/Step"][:][0]
        input_data['ImagingGrid']['Step'] = {'X': float(step_arr[0]), 'Y': float(step_arr[1])}
        # Size
        size_arr = f["ImagingGrid/Size"][:][0]
        input_data['ImagingGrid']['Size'] = {'X': float(size_arr[0]), 'Y': float(size_arr[1])}
        print("ImagingGrid:", input_data['ImagingGrid'])
        print("ImagingGrid Offset:", input_data['ImagingGrid']['Offset'])
        print("ImagingGrid Step:", input_data['ImagingGrid']['Step'])
        print("ImagingGrid Size:", input_data['ImagingGrid']['Size'])
        # Parameters
        input_data['Parameters'] = {}
        az_arr = f["Parameters/AzLimit"][:]
        input_data['Parameters']['AzLimit'] = float(az_arr.item() if az_arr.size == 1 else az_arr[0])
        # Side (decode if bytes)
        side_val = f["Side"][()]
        print("Side value:", side_val)
        print("Side type:", type(side_val))
        if isinstance(side_val, np.ndarray):
            side_val = side_val.item()
        # Map enum to string
        side_map = {1: 'port', 2: 'starboard', 3: 'down', 4: 'forward'}
        side_val = side_map.get(int(side_val), str(side_val))
        input_data['Side'] = side_val
        # Nav
        input_data['Nav'] = {}
        input_data['Nav']['Altitude'] = np.array(f["Nav/Altitude"][:])
        # Position: shape (N,2) or (N,3) -> split to X, Y
        pos_arr = np.array(f["Nav/Position"][:])
        print("Nav/Position shape:", pos_arr.shape)
        print("Nav/Position dtype:", pos_arr.dtype)
        print("Nav/Position first 10:", pos_arr[:10])

        if hasattr(pos_arr, 'dtype') and pos_arr.dtype.fields is not None:
            # Structured array, extract by field name
            if 'X' in pos_arr.dtype.fields and 'Y' in pos_arr.dtype.fields:
                input_data['Nav']['Position'] = {
                    'X': np.array(pos_arr['X']),
                    'Y': np.array(pos_arr['Y'])
                }
            else:
                raise ValueError(f"Nav/Position structured dtype fields: {pos_arr.dtype.fields}")
        elif pos_arr.ndim == 1:
            if pos_arr.size % 2 == 0:
                pos_arr2 = pos_arr.reshape(-1, 2)
                input_data['Nav']['Position'] = {
                    'X': pos_arr2[:,0],
                    'Y': pos_arr2[:,1]
                }
            elif pos_arr.size % 3 == 0:
                pos_arr2 = pos_arr.reshape(-1, 3)
                input_data['Nav']['Position'] = {
                    'X': pos_arr2[:,0],
                    'Y': pos_arr2[:,1]
                }
            else:
                raise ValueError(f"Nav/Position array shape not recognized: {pos_arr.shape}, dtype: {pos_arr.dtype}, first 10: {pos_arr[:10]}")
        elif pos_arr.ndim == 2:
            input_data['Nav']['Position'] = {
                'X': pos_arr[:,0],
                'Y': pos_arr[:,1]
            }
        else:
            raise ValueError(f"Nav/Position array shape not recognized: {pos_arr.shape}, dtype: {pos_arr.dtype}")
        input_data['Nav']['Yaw'] = np.array(f["Nav/Yaw"][:])
        # TimeDelays
        input_data['TimeDelays'] = {}
        input_data['TimeDelays']['Confidence'] = np.array(f["TimeDelays/Confidence"][:])


        # MetersPerColumn is an attribute, not a dataset
        input_data['TimeDelays']['MetersPerColumn'] = float(f["TimeDelays"].attrs['MetersPerColumn'])

    return input_data


def main_confidence_to_coverage(motion_file_path, SAS_file_path, goal_resolution_m=1.0):

    print(f"Attempting to load data from: {motion_file_path}")
    hdf5_data = load_hdf5_data(motion_file_path)
    print("Sample data loaded successfully.")

    C_out, count = dpc_confidence_to_coverage_map(hdf5_data, 1.0)
    # print(f"Success! Output shape: {C_out.shape}")
    # print(f"Count shape: {count.shape}")
    # print(f"C_out shape: {C_out.shape}, count shape: {count.shape}")
    # print("C_out:", C_out)
    # print("Count:", count)
    
    # plt.imshow(C_out, cmap='hot', interpolation='nearest')

    try:
        info = make_scaled_affine.imread_with_geotiff_info(SAS_file_path)
    except Exception as e:
        print(f"Error reading GeoTIFF file {SAS_file_path}: {e}")
        return

    if 'ModelTransformationTag' not in info:
        print(f"Warning: The file {SAS_file_path} does not contain 'ModelTransformationTag'. It may not be an affine GeoTIFF.")
        return
    
    # === Data load in ===
    A = info['ModelTransformationTag']
    A = np.array(A).reshape(4, 4).T  
    Aout=make_scaled_affine.make_scaled_affine(A, goal_resolution_m=1.0)

    # C_out_180 = np.rot90(C_out, 2)
    # C_out = np.fliplr(C_out) 
    C_out = np.flipud(C_out) 
    # replace motion_file_path .hdf5 with .tif
    tif_filename = motion_file_path.replace('.hdf5', '20.tif')
    save_geotiff_rasterio(tif_filename, C_out, Aout)
    return tif_filename

def save_geotiff(filename, data, affine):
    """
    Save a 2D numpy array as a GeoTIFF with a 4x4 affine transformation.

    Args:
        filename (str): Output file path.
        data (np.ndarray): 2D array to save.
        affine (np.ndarray): 4x4 affine transformation matrix.
    """
    # The 4x4 affine matrix should be flattened in row-major order for ModelTransformationTag
    model_transformation_tag = tuple(affine.flatten().astype(np.float64))
    
    print(f"4x4 Affine matrix:\n{affine}")
    print(f"ModelTransformationTag: {model_transformation_tag}")
    
    # Create proper GeoTIFF metadata
    metadata = {
        'ModelTransformationTag': model_transformation_tag,
        'GTModelTypeGeoKey': 1,  # ModelTypeProjected
        'GTRasterTypeGeoKey': 1,  # RasterPixelIsArea
    }
    
    # Write GeoTIFF with proper metadata
    tifffile.imwrite(
        filename,
        data.astype(np.float32),
        metadata=metadata,
        compress='lzw',  # Add compression
        photometric='minisblack'
    )
    print(f"GeoTIFF saved to {filename} with shape {data.shape}")


def save_geotiff_rasterio(filename, data, affine_4x4):
    """
    Save a 2D numpy array as a GeoTIFF using rasterio (alternative method).
    
    Args:
        filename (str): Output file path.
        data (np.ndarray): 2D array to save.
        affine_4x4 (np.ndarray): 4x4 affine transformation matrix.
    """
    try:
        import rasterio
        from rasterio.transform import Affine
        
        print(f"Affine 4x4 matrix:\n{affine_4x4}")
        transform = Affine(
            affine_4x4[0, 0],  # a: pixel width (x-scale)
            affine_4x4[1, 0],  # b: rotation/skew
            affine_4x4[3, 0],  # c: x-coordinate of upper-left corner
            affine_4x4[0, 1],  # d: rotation/skew
            affine_4x4[1, 1],  # e: pixel height (y-scale, usually negative)
            affine_4x4[3, 1]   # f: y-coordinate of upper-left corner
        )
        
        print(f"Rasterio Affine transform: {transform}")
        
        # Write GeoTIFF using rasterio
        with rasterio.open(
            filename,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",  # You might want to set a proper CRS
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(data, 1)
            
        print(f"GeoTIFF saved using rasterio to {filename} with shape {data.shape}")
        
    except ImportError:
        print("Rasterio not available, falling back to tifffile method")
        save_geotiff(filename, data, affine_4x4)
