import os
import numpy as np
import nibabel as nib
from glob import glob
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils import *
                         
def preprocess_dataset():
    '''
    split: train/val split
    portion: portion of the axial slices that enter the dataset

    first  (x) = Left-to-Right -- Sagital
    second (y) = Posterior-to-Anterior --Coronal
    third  (z) = Inferior-to-Superior  --Axial [-orient LPI]
    '''
    portion = 0.8
    for i, data in enumerate(dataset):
        if i % 50 == 0:
            print('Loading %d/%d' % (i, len(dataset)))
        filename = os.path.join(ctx["dataset_path"], 'gm', data['subject'] + '_gm.nii.gz')
        input_image = torch.FloatTensor(nib.load(filename).get_fdata())
        input_image = input_image.permute(2, 0, 1)
        # if not data['subject'] == 'sub2394':
        #     filename_wm = os.path.join(ctx["dataset_path"], 'wm', data['subject'] + '_wm.nii.gz')
        #     input_image_wm = torch.FloatTensor(nib.load(filename_wm).get_fdata())
        #     input_image_wm = input_image_wm.permute(2, 0, 1)

        start = int((1.-portion)*input_image.shape[0])
        end = int(portion*input_image.shape[0])
        input_image = input_image[start:end,:,:]
        for slice_idx in range(input_image.shape[0]):
            slice = input_image[slice_idx,:,:]
            slice = slice.unsqueeze(0)
            slices.append({
                'image': slice,
                'age': data['age']
            })

        # start_wm = int((1.-portion)*input_image_wm.shape[0])
        # end_wm = int(portion*input_image_wm.shape[0])
        # input_image_wm = input_image_wm[start_wm:end_wm,:,:]
        # for slice_idx in range(input_image_wm.shape[0]):
        #     slice = input_image_wm[slice_idx,:,:]
        #     slice = slice.unsqueeze(0)
        #     slices.append({
        #         'image': slice,
        #         'age': data['age']
        #     })


# data_path = '/home/marcon/Documents/Data/acerta_RST/'
data_path = '/home/marcon/Documents/Data/acerta_data/acerta_TASK/'
dataset = []

control_paths = glob(data_path + '/SCHOOLS/visit1/' + '*nii.gz')
condition_paths = glob(data_path + '/AMBAC/visit1/' + '*nii.gz')
mask_path = '/home/marcon/Documents/Data/SCHOOLS/Masks/HaskinsPeds_NL_template_3x3x3_maskRESAMPLED.nii'
atlas_path = '/home/marcon/Documents/Data/niftis/HaskinsPeds_NL_atlasRESAMPLED1.0.nii'
cc200_path = '/home/marcon/Documents/Data/SCHOOLS/Templates/rm_group_mean_tcorr_cluster_200.nii.gz'

#set the same number of files for each class
condition_paths = condition_paths[:len(control_paths)]

#atlas_data = nib.load(atlas_path).get_fdata()
cc200_data = nib.load(cc200_path).get_fdata()

#TODO read data from csv
ids = [i for i in range(len(control_paths+condition_paths))]
labels = [0] * len(control_paths) + [1] * len(condition_paths)
num_classes = 2

# #one-hot encoding for labels
# labels = np.zeros([len(ids), num_classes])
# for i in range(len(ids)):
#     labels[i, int(label_list[ids[i]])] = 1

if os.path.isfile('region_timeseries.npz'):
    timeseries = np.load('region_timeseries.npz', allow_pickle=True)['timeseries']
else:
    timeseries = get_region_timeseries(control_paths+condition_paths, cc200_path)
    np.savez('region_timeseries.npz', timeseries=timeseries)

#cut time series to match smaller instance
sizes = [] 
for item in timeseries: 
    sizes.append(item.shape[0])
min_size = np.min(sizes)
for n in range(len(timeseries)):
    timeseries[n] = timeseries[n][:min_size]

split=0.8
training_set, val_set, train_ids, val_ids = train_test_split(timeseries, ids,
                                                            train_size=split, random_state=42, stratify=labels)

set = 'training'                                                            
if set == 'training':
    for n in range(len(train_ids)):
        dataset.append({
            'subject': train_ids[n],
            'timeseries': training_set[n],
            'label': labels[train_ids[n]]
        })

if set == 'val':
    for n in range(len(val_ids)):
        dataset.append({
            'subject': val_ids[n],
            'timeseries': training_set[n],
            'label': labels[val_ids[n]]
        })
