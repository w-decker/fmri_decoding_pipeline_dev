import nibabel
import pickle
import deepdish
import numpy as np

def _infer_file_type(file_path):
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        return 'nifti'
    elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        return 'pickle'
    elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
        return 'hdf5'
    elif file_path.endswith('.npy'):
        return 'npy'
    else:
        raise ValueError(f"Unsupported file type for file: {file_path}")
    
def load_nifti(file_path):
    img = nibabel.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_hdf5(file_path):
    data = deepdish.io.load(file_path)
    return data

def load_npy(file_path):
    data = np.load(file_path)
    return data

def load_data(file_path):
    file_type = _infer_file_type(file_path)
    
    if file_type == 'nifti':
        return load_nifti(file_path)
    elif file_type == 'pickle':
        return load_pickle(file_path)
    elif file_type == 'hdf5':
        return load_hdf5(file_path)
    elif file_type == 'npy':
        return load_npy(file_path)
