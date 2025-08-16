"""
Batch SAS Processing GUI
A GUI application for processing multiple SAS .tif files with their corresponding motion files.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import glob
from pathlib import Path
import threading
import queue
import sys

# Import the required modules
from .geotiff_downscale import resample_geotiff
from . import confidence_mapper as dpc
from . import sas_masker as csm


class SASProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch SAS Processor")
        self.root.geometry("800x600")
        
        # Set custom window icon (.ico file)
        icon_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'NOAA_Emblem.ico')
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception as e:
                print(f"Could not set window icon: {e}")

        # Queue for thread communication
        self.queue = queue.Queue()
        
        # Variables
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.resolution = tk.DoubleVar(value=0.10)  # 10cm default
        self.goal_resolution = tk.DoubleVar(value=1.0)  # 1m default for motion
        self.threshold = tk.DoubleVar(value=0.65)  # Confidence threshold
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Input directory selection
        ttk.Label(main_frame, text="Input Directory (containing .tif files):").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(input_frame, textvariable=self.input_dir, width=50).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(input_frame, text="Browse", command=self.browse_input_dir).grid(
            row=0, column=1)
        
        # Output directory selection
        ttk.Label(main_frame, text="Output Directory:").grid(
            row=2, column=0, sticky=tk.W, pady=5)
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).grid(
            row=0, column=1)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Processing Parameters", padding="10")
        params_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        params_frame.columnconfigure(1, weight=1)
        
        # Downsampling resolution
        ttk.Label(params_frame, text="Downsampling Resolution (m):").grid(
            row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(params_frame, textvariable=self.resolution, width=10).grid(
            row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # Motion file goal resolution
        ttk.Label(params_frame, text="Motion Goal Resolution (m):").grid(
            row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(params_frame, textvariable=self.goal_resolution, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # Confidence threshold
        ttk.Label(params_frame, text="Confidence Threshold:").grid(
            row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(params_frame, textvariable=self.threshold, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        self.process_button = ttk.Button(button_frame, text="Process Files", 
                                       command=self.start_processing)
        self.process_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.preview_button = ttk.Button(button_frame, text="Preview Files", 
                                       command=self.preview_files)
        self.preview_button.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Log output
        ttk.Label(main_frame, text="Processing Log:").grid(
            row=7, column=0, sticky=tk.W, pady=(10, 5))
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=70)
        self.log_text.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights for log area
        main_frame.rowconfigure(8, weight=1)
        
    def browse_input_dir(self):
        """Browse for input directory"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir.set(directory)
            
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
    
    def log_message(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def find_tif_files(self, input_directory):
        """Find all .tif files in subdirectories of the input directory"""
        tif_files = []
        input_path = Path(input_directory)
        
        # Search for .tif files recursively
        for tif_file in input_path.rglob("*.tif"):
            tif_files.append(str(tif_file))
            
        return sorted(tif_files)
    
    def find_motion_file(self, tif_filepath):
        """Find corresponding motion file for a .tif file"""
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
    
    def preview_files(self):
        """Preview which files will be processed"""
        if not self.input_dir.get():
            messagebox.showerror("Error", "Please select an input directory")
            return
            
        self.log_text.delete(1.0, tk.END)
        self.log_message("Scanning for files...")
        
        try:
            tif_files = self.find_tif_files(self.input_dir.get())
            
            if not tif_files:
                self.log_message("No .tif files found in the specified directory")
                return
                
            self.log_message(f"Found {len(tif_files)} .tif files:")
            
            valid_pairs = 0
            for tif_file in tif_files:
                motion_file = self.find_motion_file(tif_file)
                if motion_file:
                    self.log_message(f"✓ {os.path.basename(tif_file)} -> {os.path.basename(motion_file)}")
                    valid_pairs += 1
                else:
                    self.log_message(f"✗ {os.path.basename(tif_file)} (no motion file found)")
            
            self.log_message(f"\nSummary: {valid_pairs} valid file pairs found")
            
        except Exception as e:
            self.log_message(f"Error during preview: {str(e)}")
    
    def start_processing(self):
        """Start the processing in a separate thread"""
        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please select both input and output directories")
            return
            
        # Disable the process button
        self.process_button.config(state='disabled')
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Start processing thread
        thread = threading.Thread(target=self.process_files)
        thread.daemon = True
        thread.start()
        
        # Start checking for updates
        self.check_queue()
    
    def check_queue(self):
        """Check queue for updates from processing thread"""
        try:
            while True:
                message = self.queue.get_nowait()
                if message[0] == "log":
                    self.log_message(message[1])
                elif message[0] == "progress":
                    self.progress['value'] = message[1]
                elif message[0] == "done":
                    self.process_button.config(state='normal')
                    self.progress['value'] = 0
                    return
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def process_files(self):
        """Process all files (runs in separate thread)"""
        try:
            # Find all .tif files
            tif_files = self.find_tif_files(self.input_dir.get())
            
            if not tif_files:
                self.queue.put(("log", "No .tif files found in the specified directory"))
                self.queue.put(("done", None))
                return
            
            # Filter files that have corresponding motion files
            valid_files = []
            for tif_file in tif_files:
                motion_file = self.find_motion_file(tif_file)
                if motion_file:
                    valid_files.append((tif_file, motion_file))
                else:
                    self.queue.put(("log", f"Skipping {os.path.basename(tif_file)} - no motion file found"))
            
            if not valid_files:
                self.queue.put(("log", "No valid file pairs found"))
                self.queue.put(("done", None))
                return
            
            self.queue.put(("log", f"Processing {len(valid_files)} file pairs..."))
            
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir.get(), exist_ok=True)
            
            # Process each file pair
            for i, (tif_file, motion_file) in enumerate(valid_files):
                try:
                    self.queue.put(("log", f"\n--- Processing {i+1}/{len(valid_files)}: {os.path.basename(tif_file)} ---"))
                    
                    # Update progress
                    progress = (i / len(valid_files)) * 100
                    self.queue.put(("progress", progress))
                    
                    # Use main output directory directly (no subfolders)
                    file_output_dir = self.output_dir.get()
                    os.makedirs(file_output_dir, exist_ok=True)
                    
                    # Step 1: Downsample the SAS file
                    self.queue.put(("log", f"Downsampling to {self.resolution.get()}m resolution..."))
                    downsample_filepath = resample_geotiff(tif_file, file_output_dir, self.resolution.get())
                    self.queue.put(("log", f"Downsampled file: {os.path.basename(downsample_filepath)}"))
                    
                    # Step 2: Build motion data geotiff
                    self.queue.put(("log", "Building confidence coverage map..."))
                    tif_output_filename = dpc.main_confidence_to_coverage(
                        motion_file, downsample_filepath, goal_resolution_m=self.goal_resolution.get())
                    self.queue.put(("log", f"Confidence map: {os.path.basename(tif_output_filename)}"))
                    
                    # Step 3: Mask the SAS tif using confidence values
                    self.queue.put(("log", "Creating masked SAS file..."))
                    masked_sas_filename = csm.mask_sas_with_confidence(
                        sas_file_path=downsample_filepath,
                        confidence_file_path=tif_output_filename,
                        threshold=self.threshold.get()
                    )
                    self.queue.put(("log", f"Masked SAS: {os.path.basename(masked_sas_filename)}"))
                    
                    self.queue.put(("log", f"Completed processing {os.path.basename(tif_file)}"))
                    
                except Exception as e:
                    self.queue.put(("log", f"Error processing {os.path.basename(tif_file)}: {str(e)}"))
                    continue
            
            # Final progress update
            self.queue.put(("progress", 100))
            self.queue.put(("log", f"\n=== Processing Complete ==="))
            self.queue.put(("log", f"Processed {len(valid_files)} files"))
            self.queue.put(("log", f"Output saved to: {self.output_dir.get()}"))
            
        except Exception as e:
            self.queue.put(("log", f"Fatal error: {str(e)}"))
        finally:
            self.queue.put(("done", None))


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SASProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
