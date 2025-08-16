import numpy as np
import math
import matplotlib.pyplot as plt
import tifffile
import h5py

def make_scaled_affine(A, goal_resolution_m=1.0):
    resolution_m = get_affine_resolution(A)
    ratio = get_ratio(goal_resolution_m, resolution_m)
    Aout = scale_affine(A, ratio)
    Aout = shift_affine_after_scaling(Aout, A, ratio)
    return Aout

def get_ratio(goal_resolution_m, resolution_m):
    return goal_resolution_m / resolution_m

def scale_affine(A, ratio):
    A = A.copy()
    # MATLAB: A([1,2,5,6]) = A([1,2,5,6]) * ratio;
    # In Python, flatten and index, or use 2D indices:
    A[0,0] *= ratio
    A[0,1] *= ratio
    A[1,0] *= ratio
    A[1,1] *= ratio
    return A

def shift_affine_after_scaling(A2, A1, ratio):
    # delX = (ratio-1) * (A1(1,1) + A1(1,2));
    # delY = (ratio-1) * (A1(2,1) + A1(2,2));
    delX = (ratio - 1) * (A1[0,0] + A1[1,0])
    delY = (ratio - 1) * (A1[0,1] + A1[1,1])
    A2 = A2.copy()
    A2[3,0] = A1[3,0] + delX
    A2[3,1] = A1[3,1] + delY
    return A2

def get_affine_resolution(A):
    print(A)
    dLat, dLon = dLL(A[1,3])
    delLon = A[0,0] + A[1,0]
    delLat = A[0,1] + A[1,1]
    delN = delLat / dLat
    delE = delLon / dLon
    resolution_m = math.sqrt(delN**2 + delE**2) * math.sqrt(2)/2
    resolution_m = round(resolution_m * 1000) / 1000
    return resolution_m

def dLL(lat0_deg):
    # h_sin  = @(x) sind(x/2).^2;
    # ah_sin = @(x) 2*asind(sqrt(x));
    R = 6371e3
    dLat_deg = 1/R * 180 / math.pi
    h_sin = math.sin(1/(2*R))**2
    cos_lat = math.cos(math.radians(lat0_deg))
    ah_sin = 2 * math.degrees(math.asin(math.sqrt(h_sin / (cos_lat**2))))
    dLon_deg = ah_sin
    return dLat_deg, dLon_deg

def imread_with_geotiff_info(src):
    """
    Reads a GeoTIFF file and returns its metadata, image array, and colormap (if any).
    Prints all metadata keys and their types.
    """
    import tifffile

    with tifffile.TiffFile(src) as tif:
        info = {tag.name: tag.value for tag in tif.pages[0].tags.values()}
    return info
