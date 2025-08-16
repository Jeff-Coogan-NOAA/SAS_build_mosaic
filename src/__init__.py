"""
SAS Mosaic Builder

A Python package for batch processing Synthetic Aperture Sonar (SAS) GeoTIFF files.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .batch_processor import batch_process_sas_files, process_single_file_pair
from .geotiff_downscale import resample_geotiff
from .confidence_mapper import main_confidence_to_coverage
from .sas_masker import mask_sas_with_confidence

__all__ = [
    'batch_process_sas_files',
    'process_single_file_pair', 
    'resample_geotiff',
    'main_confidence_to_coverage',
    'mask_sas_with_confidence'
]
