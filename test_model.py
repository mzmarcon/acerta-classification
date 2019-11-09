import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from dataset_loader import TEST_data
from model import VGGBased13
import sys 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)

# data_path = '/home/marcon/Documents/Data/acerta_data/acerta_TASK/'
data_path = '/home/marcon/datasets/acerta_data/acerta_TASK/'

training_set = TEST_data(data_path=data_path)
val_loader = DataLoader(training_set, shuffle=False, drop_last=False,
                        num_workers=8, batch_size=1)

model = VGGBased13().to(device)
model.load_state_dict(torch.load('checkpoints/best_fullvolume.pth'))
model.eval()

accuracy = []
labels = []
predictions = defaultdict(list)
iterations = 0
control_result = []
condition_result = []
for i, data in enumerate(val_loader):
    input_data = Variable(data['input']).to(device)
    output = model(input_data)
    label = Variable(data['label']).float().to(device)

    iterations += 1 

    prediction = (output > 0).float()
    correct = (prediction == label.reshape(-1,1)).float()
    total_correct = correct.sum()
    accuracy.append(total_correct / label.shape[0])

    if int(label) == 0:
        control_result.append(int(total_correct))
    else:
        condition_result.append(int(total_correct))

print('Accuracy: {:.3f}'.format(torch.mean(torch.stack(accuracy))))

print(control_result)
print(condition_result)
