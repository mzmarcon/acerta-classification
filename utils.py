import os
import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiLabelsMasker  
import numpy as np 
from nilearn import plotting 
from nilearn.connectome import ConnectivityMeasure
import torch

def get_region_timeseries(file_list, atlas):
    timeseries_list = []
    for file in file_list:
        # masked_data = apply_mask(file,mask) #full masked-timeseries - no atlas(features x samples)
        masker = NiftiLabelsMasker(labels_img=atlas, standardize=True, 
                                    memory='nilearn_cache', verbose=5)
        timeseries = masker.fit_transform(file) 
        timeseries_list.append(timeseries)

    return np.array(timeseries_list)

def get_correlation_matrix(region_timeseries):
    correlation_matrix = []
    correlation_measure = ConnectivityMeasure(kind='correlation') 

    for item in region_timeseries:
        matrix = correlation_measure.fit_transform([item])[0]
        correlation_matrix.append(matrix)

    return np.array(correlation_matrix)

def plot_correlation_matrix(correlation_matrix, atlas_data):
    # Make a large figure 
    # Mask the main diagonal for visualization: 
    np.fill_diagonal(correlation_matrix, 0) 
    # The labels we have start with the background (0), hence we skip the 
    # first label 
    # matrices are ordered for block-like representation 
    labels = np.unique(atlas_data)
    plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels, 
                         vmax=0.8, vmin=-0.8, reorder=True) 
    plotting.show()


def preprocess_dataset(filenames,labels,ids, mask_data):
    '''
    Split images in slices.

    portion: portion of the axial slices that enter the dataset.
    first  (x) = Left-to-Right -- Sagital
    second (y) = Posterior-to-Anterior --Coronal
    third  (z) = Inferior-to-Superior  --Axial [-orient LPI]
    '''

    slices_data = []
    portion = 0.8
    for i, filename in enumerate(filenames):
        if i % 50 == 0:
            print('Loading %d/%d' % (i, len(filenames)))
        input_image = nib.load(filename).get_fdata()
        input_image = torch.FloatTensor(input_image * mask_data)

        input_image = input_image.permute(2, 0, 1)

        start = int((1.-portion)*input_image.shape[0])
        end = int(portion*input_image.shape[0])
        input_image = input_image[start:end,:,:]
        for slice_idx in range(input_image.shape[0]):
            slice = input_image[slice_idx,:,:]
            slice = slice.unsqueeze(0)
            slices_data.append({
                'image': slice,
                'label': labels[i],
                'id': ids[i]
            })
    
    return slices_data