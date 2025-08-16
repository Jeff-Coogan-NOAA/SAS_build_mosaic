# SAS Mosaic Builder

A Python application for batch processing Synthetic Aperture Sonar (SAS) GeoTIFF files with motion data to create downsampled, confidence-mapped, and masked outputs.

## Features

- **Batch Processing**: Process multiple SAS .tif files automatically
- **GUI Interface**: User-friendly interface for selecting directories and parameters
- **Motion Data Integration**: Uses motion files (.motions.hdf5) to generate confidence maps
- **Flexible Output**: Configurable resolution and confidence thresholds
- **Progress Tracking**: Real-time progress updates and detailed logging

## Usage GUI Application
- python run_gui.py

The GUI allows you to:
1. Select an input directory containing .tif files
2. Choose an output directory for processed files
3. Configure processing parameters
4. Preview files before processing
5. Monitor progress in real-time

## Contributions 
Code is based on Dan Plotnick's matlab SAS mosaic builder and has benn converted to python for V1.0
