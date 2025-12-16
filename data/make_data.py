"""
Simulate fMRI data using BrainIAK's fmrisim module.

Parameters
----------
directory : str
    The directory where the simulated data will be saved.
volume : tuple
    The dimensions of the fMRI volume (x, y, z, time).

    
Returns
-------
None
-------
"""

from brainiak.utils import fmrisim #type: ignore

import numpy as np # type: ignore
import pandas as pd # type: ignore
import nibabel as nib # type: ignore
from argparse import ArgumentParser
from typing import Union
import os, ast, json

def load_optseq_timing(par_file, exclude_null=True):
    """
    Load optseq2 .par file and convert to format usable by fmrisim.
    
    Parameters
    ----------
    par_file : str
        Path to the .par file from optseq2
    exclude_null : bool
        Whether to exclude NULL (baseline) events (condition ID = 0)
    
    Returns
    -------
    onsets : list
        List of event onset times in seconds
    durations : list
        List of event durations in seconds
    conditions : list
        List of condition IDs (integers)
    weights : list
        List of event weights
    """
    # Read the par file
    df = pd.read_csv(par_file, sep='\s+', header=None, 
                     names=['onset', 'condition', 'duration', 'weight', 'label'])
    
    # Optionally exclude NULL events
    if exclude_null:
        df = df[df['condition'] != 0]
    
    return (df['onset'].tolist(), 
            df['duration'].tolist(), 
            df['condition'].tolist(),
            df['weight'].tolist())

def parse_args():
    parser = ArgumentParser(description="Simulate fMRI data")
    parser.add_argument('-d', '--directory', type=str, required=True,
                        help='Directory to save the simulated data')
    parser.add_argument('--seed', type=int, default=42, required=True,
                        help='Random seed for reproducibility')
    parser.add_argument('-v', '--volume', type=Union[tuple, np.ndarray], default=(64, 64, 36, 1000),
                        help='Dimensions of the fMRI volume (x, y, z, time). Make sure time aligns with timing file if provided.')
    parser.add_argument('-n', '--noise_dict', type=dict, 
                        default={
                            'snr': 25.0, 
                            'sfnr': 70.0, 
                            'fwhm': 4.0,
                            'max_activity': 1.0,
                            'voxel_size': [1.0, 1.0, 1.0],
                            'auto_reg_rho': [0.3],
                            'auto_reg_sigma': 1.0,
                            'ma_rho': [0.0],
                            'physiological_sigma': 0,
                            'task_sigma': 0,
                            'drift_sigma': 0.0,
                            },
                        help='Dictionary specifying noise parameters for fmrisim')
    parser.add_argument('-tr', '--tr_duration', type=float, default=2.0,
                        help='Repetition time (TR) duration in seconds')
    parser.add_argument('-c', '--coordinates', type=str, default="[21, 21, 21]",
                        help='Coordinates for signal placement (e.g., "[32, 32, 18]")')
    parser.add_argument('-s', '--size', type=int, default=3,
                        help='Size of the signal region (in voxels)')
    parser.add_argument('-t', '--timing_file', type=str, default=None,
                        help='Path to timing file (.par from optseq2 or 3-column format)')
    
    parser.add_argument('--config', type=Union[str, dict])

    return parser.parse_args()
    
args = parse_args()
np.random.seed(args.seed)

# Parse coordinates
if args.coordinates is not None:
    if isinstance(args.coordinates, str):
        args.coordinates = ast.literal_eval(args.coordinates)

# Volume dimensions
if isinstance(args.volume, tuple):
    volume = np.ones(args.volume)
    mask, template = fmrisim.mask_brain(volume=volume, mask_self=False)
elif isinstance(args.volume, np.ndarray):
    volume = args.volume
    mask, template = fmrisim.mask_brain(volume=volume, mask_self=True)

# Generate stimulus function
if args.timing_file is not None:

    # Load timing from file
    onsets, durations, conditions, weights = load_optseq_timing(args.timing_file)
    total_time = volume.shape[3] * args.tr_duration
    
    # Get unique conditions
    unique_conditions = np.unique(conditions)
    
    # Generate separate stimulus functions for each condition
    stimfunctions = []
    for cond in unique_conditions:
        # Get onsets for this condition
        cond_idx = np.where(np.array(conditions) == cond)[0]
        cond_onsets = [onsets[i] for i in cond_idx]
        cond_durations = [durations[i] for i in cond_idx]
        cond_weights = [weights[i] for i in cond_idx]
        
        # Generate stimulus function for this condition
        stimfunc = fmrisim.generate_stimfunction(
            onsets=cond_onsets,
            event_durations=cond_durations,
            total_time=total_time,
            weights=cond_weights,
            temporal_resolution=100.0
        )
        stimfunctions.append(stimfunc)
    
    # Combine all conditions into one matrix (timepoints x conditions)
    stimfunction_combined = np.hstack(stimfunctions)
    
    # Convolve with HRF and downsample to TR resolution
    signal_function = fmrisim.convolve_hrf(
        stimfunction=stimfunction_combined,
        tr_duration=args.tr_duration,
        temporal_resolution=100.0
    )
    
    # For generate_noise, we need a single timecourse
    # Sum across conditions or use the first condition
    stimfunction_tr = signal_function[:, 0]
else:
    # No timing file provided, create baseline
    stimfunction_tr = np.zeros(volume.shape[3])

noise = fmrisim.generate_noise(dimensions=volume.shape[0:3], # spatial dimensions
                                tr_duration=args.tr_duration,
                                stimfunction_tr=stimfunction_tr,
                                mask=mask,
                                template=template,
                                noise_dict=args.noise_dict,
                                )

# Generate signal volumes for different ROIs
if args.coordinates is not None:
    if isinstance(args.coordinates, (list, tuple)):
        if isinstance(args.coordinates[0], (int, float)):
            coordinates_array = np.array([args.coordinates])
        else:
            coordinates_array = np.array(args.coordinates)
    else:
        raise ValueError("Coordinates must be a list or tuple")
    
    # Prepare feature_size as list
    if isinstance(args.size, int):
        size_list = [args.size] * len(coordinates_array)
    else:
        size_list = args.size
    
    signal_volume = fmrisim.generate_signal(
        dimensions=volume.shape[0:3],
        feature_type=['cube'] * len(coordinates_array),
        feature_coordinates=coordinates_array,
        feature_size=size_list,
        signal_magnitude=[1] * len(coordinates_array)
    )
else:
    signal_volume = np.zeros(volume.shape[0:3])
    coordinates_array = None

# Calculate number of voxels in signal ROI
if isinstance(args.size, int):
    voxels = args.size ** 3
else:
    voxels = sum([s ** 3 for s in args.size])

# Generate multivariate patterns for each condition
if args.timing_file is not None and np.sum(signal_volume) > 0:
    patterns = {}
    for cond in unique_conditions:
        patterns[cond] = np.random.rand(voxels, 1)
    
    # Create weighted stimulus functions for each voxel
    # Each voxel responds differently to each condition
    signal_idxs = np.where(signal_volume != 0)
    n_signal_voxels = len(signal_idxs[0])
    
    # Initialize the weighted stimfunction matrix
    stimfunc_weighted = np.zeros((stimfunction_combined.shape[0], n_signal_voxels))
    
    # For each condition, weight the stimulus function by the pattern
    for idx, cond in enumerate(unique_conditions):
        # Get the stimulus function for this condition
        cond_stimfunc = stimfunctions[idx]
        
        # Weight it by the pattern for each voxel
        for voxel_idx in range(n_signal_voxels):
            pattern_weight = patterns[cond][voxel_idx % voxels, 0]
            stimfunc_weighted[:, voxel_idx] += cond_stimfunc[:, 0] * pattern_weight
    
    # Convolve the weighted stimulus functions with HRF
    signal_func = fmrisim.convolve_hrf(
        stimfunction=stimfunc_weighted,
        tr_duration=args.tr_duration,
        temporal_resolution=100.0,
        scale_function=1
    )
    
    # Compute signal magnitude
    noise_func = noise[signal_idxs[0], signal_idxs[1], signal_idxs[2], :].T
    signal_func_scaled = fmrisim.compute_signal_change(
        signal_func,
        noise_func,
        args.noise_dict,
        magnitude=[1.0],  # Adjust this for signal strength
        method='CNR_Amp/Noise-SD'
    )
    
    # Apply the signal 
    signal = fmrisim.apply_signal(
        signal_function=signal_func_scaled,
        volume_signal=signal_volume
    )
else:
    # No timing file or no signal volume, create empty signal
    signal = np.zeros(volume.shape)

# Combine signal and noise
brain = signal + noise

# Save
os.makedirs(args.directory, exist_ok=True)
brain_nii = nib.Nifti1Image(brain, np.eye(4))
nib.save(brain_nii, os.path.join(args.directory, 'simulated_brain.nii.gz'))

# Save mask and template as NIfTI files
mask_nii = nib.Nifti1Image(mask.astype(np.float32), np.eye(4))
nib.save(mask_nii, os.path.join(args.directory, 'mask.nii.gz'))

template_nii = nib.Nifti1Image(template, np.eye(4))
nib.save(template_nii, os.path.join(args.directory, 'template.nii.gz'))

# Save metadata as JSON (without large arrays)
metadata = {
    'volume_shape': list(volume.shape),
    'tr_duration': args.tr_duration,
    'noise_parameters': args.noise_dict,
    'signal_coordinates': args.coordinates,
    'signal_size': args.size,
    'conditions': unique_conditions.tolist() if args.timing_file is not None else [],
    'n_signal_voxels': int(n_signal_voxels) if args.timing_file is not None and np.sum(signal_volume) > 0 else 0
}

with open(os.path.join(args.directory, 'simulation_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Simulation complete! Data saved to {args.directory}")
