"""
Example of batch processing multiple SAS files
"""

from src.batch_processor import batch_process_sas_files, get_valid_file_pairs

def main():
    """
    Example of batch processing SAS files
    """

    # Define input and output directories
    input_dir = r"XX:\KRAKEN"
    output_dir = r"XX:\KRAKEN\batch_processed"

    print("SAS Mosaic Builder - Batch Processing Example")
    print("=" * 50)
    
    # Preview files that will be processed
    print("Scanning for valid file pairs...")
    file_pairs = get_valid_file_pairs(input_dir)
    
    if not file_pairs:
        print("No valid file pairs found.")
        return
    
    print(f"Found {len(file_pairs)} valid file pairs:")
    for i, (tif_file, motion_file) in enumerate(file_pairs, 1):
        print(f"  {i}. {tif_file.split('\\')[-1]} -> {motion_file.split('\\')[-1]}")
    
    # Confirm processing
    response = input(f"\nProceed with processing {len(file_pairs)} files? (y/n): ")
    if response.lower() != 'y':
        print("Processing cancelled.")
        return
    
    # Process files
    print("\nStarting batch processing...")
    results = batch_process_sas_files(
        input_directory=input_dir,
        output_directory=output_dir,
        resolution=0.10,        # 10cm resolution
        goal_resolution=1.0,    # 1m resolution for motion
        threshold=0.5,          # Confidence threshold
        verbose=True
    )
    
    # Report results
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for result in results:
            if not result['success']:
                print(f"  - {result['original_tif'].split('\\')[-1]}: {result['error']}")
    
    print(f"\nOutput saved to: {output_dir}")

if __name__ == "__main__":
    main()
