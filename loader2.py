import os
import numpy as np
import nibabel as nib
from glob import glob
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils import *
                         
# data_path = '/home/marcon/Documents/Data/acerta_RST/'
data_path = '/home/marcon/Documents/Data/acerta_data/acerta_TASK/'
# dataset = []

control_paths = glob(data_path + '/SCHOOLS/visit1/' + '*nii.gz')
condition_paths = glob(data_path + '/AMBAC/visit1/' + '*nii.gz')
mask_path = '/home/marcon/Documents/Data/SCHOOLS/Masks/HaskinsPeds_NL_template_3x3x3_maskRESAMPLED.nii'
atlas_path = '/home/marcon/Documents/Data/niftis/HaskinsPeds_NL_atlasRESAMPLED1.0.nii'
cc200_path = '/home/marcon/Documents/Data/SCHOOLS/Templates/rm_group_mean_tcorr_cluster_200.nii.gz'

#set the same number of files for each class
if len(control_paths) < len(condition_paths):
    condition_paths = condition_paths[:len(control_paths)]
else:
    control_paths = control_paths[:len(condition_paths)]

#atlas_data = nib.load(atlas_path).get_fdata()
cc200_data = nib.load(cc200_path).get_fdata()

#TODO read data from csv
ids = [i for i in range(len(control_paths+condition_paths))]
labels = [0] * len(control_paths) + [1] * len(condition_paths)
num_classes = 2

#load data
file_paths = control_paths + condition_paths

split=0.8
train_set, val_set, train_ids, val_ids, \
train_labels, val_labels  = train_test_split(file_paths, ids, labels,
                                             train_size=split, random_state=42, stratify=labels)

set = 'training'                                                            
if set == 'training':
    dataset = preprocess_dataset(train_set, train_labels, train_ids)
if set == 'val':
    dataset = preprocess_dataset(val_set, val_labels, val_ids)



# set = 'training'                                                            
# if set == 'training':
#     for n in range(len(train_ids)):
#         dataset.append({
#             'subject': train_ids[n],
#             'input_image': train_set[n],
#             'label': labels[train_ids[n]]
#         })

# if set == 'val':
#     for n in range(len(val_ids)):
#         dataset.append({
#             'subject': val_ids[n],
#             'input_image': training_set[n],
#             'label': labels[val_ids[n]]
#         })
