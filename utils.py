import os
import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiLabelsMasker  
import numpy as np 
from nilearn import plotting 
from nilearn.connectome import ConnectivityMeasure
import torch

def preprocess_slices_dataset(image_data,labels,ids):
    '''
    Split images in slices.

    portion: portion of the axial slices that enter the dataset.
    first  (x) = Left-to-Right -- Sagital
    second (y) = Posterior-to-Anterior --Coronal
    third  (z) = Inferior-to-Superior  --Axial [-orient LPI]
    '''
    slices_data = []
    portion = 0.8
    for i, input_image in enumerate(image_data):
        input_image = torch.FloatTensor(input_image)
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


def load_dataset(filenames, atlas_data, discard=False):

    if discard:
        discarded_regions = [0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23, 9, 10, 11, 14, 
                            33, 34, 35, 36, 37, 38, 39, 45, 44, 47, 71, 78, 79, 
                            81, 96, 94, 62, 60]

        atlas_data[np.isin(atlas_data, discarded_regions)] = 0
        atlas_data[np.isin(atlas_data, discarded_regions, invert=True)] = 1

    else:
        atlas_data[np.isin(atlas_data, 0, invert=True)] = 1

    image_data = []
    for i, filename in enumerate(filenames):
        if i % 10 == 0:
            print('Loading %d/%d' % (i, len(filenames)))
        data = nib.load(filename).get_fdata()
        data = data * atlas_data
        image_data.append(data)

    #zscore padronization
    image_data = (image_data - np.mean(image_data)) / (np.std(image_data) + 1e-6)

    #scaling normalization
    # image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    return image_data


def preprocess_volume_dataset(image_data,labels,ids):
    '''
    Split images in slices.

    portion: portion of the axial slices that enter the dataset.
    first  (x) = Left-to-Right -- Sagital
    second (y) = Posterior-to-Anterior --Coronal
    third  (z) = Inferior-to-Superior  --Axial [-orient LPI]
    '''
    dataset = []
    for i, input_image in enumerate(image_data):
        input_image = torch.FloatTensor(input_image)
        input_image = input_image.permute(2, 0, 1)

        dataset.append({
            'image': input_image,
            'label': labels[i],
            'id': ids[i]
        })
    
    return dataset
