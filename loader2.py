import os
import numpy as np
import nibabel as nib
from glob import glob
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils import *
                         

data_path = '/home/marcon/Documents/Data/acerta_RST/'
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
