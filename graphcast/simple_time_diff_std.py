#!/usr/bin/env python3

"""
Different from compute_time_diff_std.py:

No DALI: Reads HDF5 files directly instead of using DALI datapipe
No Multiprocessing: Runs in single process to avoid subprocess issues
Direct File Access: Uses h5py instead of DALI's external source
"""

import torch
import numpy as np
import h5py
import os
from glob import glob
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

def normalized_grid_cell_area(lat_coords, unit="deg"):
    """
    Compute normalized grid cell area weights for latitude coordinates.
    This is identical to the one used in the original compute_time_diff_std.py
    """
    if unit == "deg":
        lat_coords = np.deg2rad(lat_coords)
    
    # Create 2D latitude grid
    if lat_coords.ndim == 1:
        lat_coords_2d = lat_coords[:, np.newaxis]
    else:
        lat_coords_2d = lat_coords
    
    # Compute area weights based on cosine of latitude
    area_weights = np.cos(lat_coords_2d)
    
    # Normalize so the mean is 1
    area_weights = area_weights / np.mean(area_weights)
    
    return area_weights

def compute_time_diff_std_exact(data_dir, save_dir=".", num_channels=34, latlon_res=(721, 1440)):
    """
    Exact computation of time difference std matching the original algorithm
    """
    
    # Find all HDF5 files
    h5_files = glob(os.path.join(data_dir, "train", "*.h5"))
    
    if not h5_files:
        print("No HDF5 files found!")
        return
    
    # Create area weights identical to original algorithm
    lat_coords = np.linspace(-90, 90, num=latlon_res[0])
    area = normalized_grid_cell_area(lat_coords, unit="deg")  # Shape: [721, 1]
    area = area[:, 0]  # Make it 1D: [721]
    
    # Initialize accumulators (exactly like the original)
    mean = np.zeros((1, num_channels))  # [1, num_channels]
    mean_sqr = np.zeros((1, num_channels))  # [1, num_channels]
    total_samples = 0
    
    for h5_file in h5_files:
        
        with h5py.File(h5_file, 'r') as f:
            # Get the data - assuming it's stored as 'fields'
            if 'fields' in f:
                data = f['fields'][:]  # Shape: [time, channel, lat, lon]
                
                # Process each consecutive pair of time steps
                for t in range(data.shape[0] - 1):
                    # Get consecutive time steps
                    invar = data[t]      # [channel, lat, lon]
                    outvar = data[t + 1] # [channel, lat, lon]
                    
                    # Compute difference (exactly like original)
                    diff = outvar - invar  # [channel, lat, lon]
                    
                    # Apply area weighting (exactly like original)
                    # area shape: [721], diff shape: [channel, 721, 1440]
                    # Broadcasting: area[np.newaxis, :, np.newaxis] -> [1, 721, 1]
                    weighted_diff = area[np.newaxis, :, np.newaxis] * diff  # [channel, lat, lon]
                    weighted_diff_sqr = np.square(weighted_diff)  # [channel, lat, lon]
                    
                    # Compute spatial means (exactly like original)
                    # mean over dimensions (lat, lon) = (1, 2)
                    spatial_mean = np.mean(weighted_diff, axis=(1, 2))      # [channel]
                    spatial_mean_sqr = np.mean(weighted_diff_sqr, axis=(1, 2))  # [channel]
                    
                    # Accumulate (exactly like original)
                    mean[0] += spatial_mean
                    mean_sqr[0] += spatial_mean_sqr
                    total_samples += 1
                    
            else:
                print(f"Warning: 'fields' not found in {h5_file}, available keys: {list(f.keys())}")
        
    # Normalize by number of samples (exactly like original)
    mean = mean / total_samples        # [1, channel]
    mean_sqr = mean_sqr / total_samples  # [1, channel]
    
    # Compute variance and std (exactly like original)
    variance = mean_sqr - mean**2  # [1, channel]
    std = np.sqrt(variance)        # [1, channel]
    
    # Reshape to expected format [1, channel, 1, 1] (exactly like original)
    mean = mean.reshape(1, -1, 1, 1)
    std = std.reshape(1, -1, 1, 1)
    
    # Save results with exact same names as original
    mean_file = os.path.join(save_dir, "time_diff_mean.npy")
    std_file = os.path.join(save_dir, "time_diff_std.npy")
    
    np.save(mean_file, mean)
    np.save(std_file, std)
    
    return mean, std

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function using Hydra config"""
    
    # Get data directory from config
    data_dir = to_absolute_path(cfg.dataset_path)
    save_dir = os.path.join(data_dir, "stats")
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found!")
        print("Please check your dataset_path in config.yaml")
        return
    
    compute_time_diff_std_exact(data_dir, save_dir, cfg.num_channels_climate, tuple(cfg.latlon_res))

if __name__ == "__main__":
    main()

# Legacy standalone function for manual use
def run_standalone():
    """Standalone function if you want to run without Hydra"""
    # Adjust this path to your data directory
    data_dir = "./hdf5_data"  # or "/workspace/hdf5_data" if running in Docker
    
    if not os.path.exists(data_dir):
        data_dir = "/workspace/hdf5_data"
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found!")
        print("Please adjust the data_dir path in the script")
    else:
        compute_time_diff_std_exact(data_dir, ".")
