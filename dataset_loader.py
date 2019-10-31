import os
import numpy as np
import nibabel as nib
from glob import glob
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils import *
                         

class ACERTA_data(object):
    def __init__(self, set, split=0.8):
        # data_path = '/home/marcon/Documents/Data/acerta_data/acerta_RST/'
        data_path = '/home/marcon/Documents/Data/acerta_data/acerta_TASK/'
        self.dataset = []

        control_paths = glob(data_path + '/SCHOOLS/visit1/' + '*nii.gz')
        condition_paths = glob(data_path + '/AMBAC/visit1/' + '*nii.gz')
        mask_path = '/home/marcon/Documents/Data/SCHOOLS/Masks/HaskinsPeds_NL_template_3x3x3_maskRESAMPLED.nii'
        atlas_path = '/home/marcon/Documents/Data/niftis/HaskinsPeds_NL_atlasRESAMPLED1.0.nii'
        cc200_path = '/home/marcon/Documents/Data/SCHOOLS/Templates/rm_group_mean_tcorr_cluster_200.nii.gz'

        #set the same number of files for each class
        condition_paths = condition_paths[:len(control_paths)]
        # atlas_data = nib.load(atlas_path).get_fdata()
        cc200_data = nib.load(cc200_path).get_fdata()
        
        #TODO read data from csv
        ids = [i for i in range(len(control_paths+condition_paths))]
        labels = [0] * len(control_paths) + [1] * len(condition_paths)
        
        # #one-hot encoding for labels
        # num_classes = 2
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
            # # turn half of condition to 0 for testing classification
            # if n>48:
            #     timeseries[n][int(0.5*len(timeseries[n])):] = 0

        training_set, val_set, training_ids, val_ids = train_test_split(timeseries, ids,
                                                                    train_size=split, random_state=42, stratify=labels)

        if set == 'training':
            for n in range(len(training_ids)):
                self.dataset.append({
                    'subject': training_ids[n],
                    'timeseries': training_set[n],
                    'label': labels[training_ids[n]]
                })

        if set == 'validation':
            for n in range(len(val_ids)):
                self.dataset.append({
                    'subject': val_ids[n],
                    'timeseries': val_set[n],
                    'label': labels[val_ids[n]]
                })

    def __getitem__(self, idx):
        data = self.dataset[idx]

        return {
            'input': torch.FloatTensor(data['timeseries']),
            'label': data['label']
        }

    def __len__(self):
        return len(self.dataset)

    def preprocess_dataset(self):
        for i, data in enumerate(self.dataset):
            if i % 50 == 0:
                print('Loading %d/%d' % (i, len(self.dataset)))
            filename = os.path.join(self.ctx["dataset_path"], 'gm', data['subject'] + '_gm.nii.gz')
            input_image = torch.FloatTensor(nib.load(filename).get_fdata())
            input_image = input_image.permute(2, 0, 1)
            # if not data['subject'] == 'sub2394':
            #     filename_wm = os.path.join(self.ctx["dataset_path"], 'wm', data['subject'] + '_wm.nii.gz')
            #     input_image_wm = torch.FloatTensor(nib.load(filename_wm).get_fdata())
            #     input_image_wm = input_image_wm.permute(2, 0, 1)

            start = int((1.-self.portion)*input_image.shape[0])
            end = int(self.portion*input_image.shape[0])
            input_image = input_image[start:end,:,:]
            for slice_idx in range(input_image.shape[0]):
                slice = input_image[slice_idx,:,:]
                slice = slice.unsqueeze(0)
                self.slices.append({
                    'image': slice,
                    'age': data['age']
                })

            # start_wm = int((1.-self.portion)*input_image_wm.shape[0])
            # end_wm = int(self.portion*input_image_wm.shape[0])
            # input_image_wm = input_image_wm[start_wm:end_wm,:,:]
            # for slice_idx in range(input_image_wm.shape[0]):
            #     slice = input_image_wm[slice_idx,:,:]
            #     slice = slice.unsqueeze(0)
            #     self.slices.append({
            #         'image': slice,
            #         'age': data['age']
            #     })