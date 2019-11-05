import os
import numpy as np
import nibabel as nib
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import *
                         

class ACERTA_data(Dataset):
    def __init__(self, set, split=0.8, transform=None):
        # data_path = '/home/marcon/Documents/Data/acerta_data/acerta_TASK/'
        data_path = '/home/marcon/datasets/acerta_data/acerta_TASK/'

        self.dataset = []

        control_paths = glob(data_path + 'SCHOOLS/visit1/' + '*nii.gz')
        condition_paths = glob(data_path + 'AMBAC/visit1/' + '*nii.gz')

        #set the same number of files for each class
        if len(control_paths) < len(condition_paths):
            condition_paths = condition_paths[:len(control_paths)]
        else:
            control_paths = control_paths[:len(condition_paths)]
        
        #TODO read data from csv
        ids = [i for i in range(len(control_paths+condition_paths))]
        labels = [0] * len(control_paths) + [1] * len(condition_paths)
        
        #load data
        file_paths = control_paths + condition_paths

        train_set, val_set, train_ids, val_ids, \
        train_labels, val_labels  = train_test_split(file_paths, ids, labels,
                                                    train_size=split, random_state=42, stratify=labels)

        if set == 'training':
            self.dataset = preprocess_dataset(train_set, train_labels, train_ids)
        if set == 'validation':
            self.dataset = preprocess_dataset(val_set, val_labels, val_ids)

        # self.transform = transforms.Compose([transforms.Normalize(mean=0.5,
        #                                                           std=0.5)])


    def __getitem__(self, idx):
        data = self.dataset[idx]

        return {
            'input' : torch.FloatTensor(data['image']),
            'label' : data['label'],
            'id'    : data['id']
        }

        # transformed = {
        #     'input': data['image']
        # }

        # transformed = self.transform(transformed['input'])

        # return {
        #     'input': torch.FloatTensor(transformed['image']),
        #     'label': data['label']
        # }

    def __len__(self):
        return len(self.dataset)