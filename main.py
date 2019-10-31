import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset_loader import ACERTA_data 
from model import VGGBasedModel2D
import sys 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)

params = { 'batch_size': 4,
           'learning_rate': 1e-4,
           'weight_decay': 6e-2,
           'epochs': 5 }


training_set = ACERTA_data(set='training', split=0.8)
validation_set = ACERTA_data(set='validation', split=0.8)

train_loader = DataLoader(training_set, shuffle=True, drop_last=True,
                             num_workers=8, batch_size=params['batch_size'])

val_loader = DataLoader(validation_set, shuffle=False, drop_last=False,
                        num_workers=8, batch_size=1)

# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

model = VGGBasedModel2D()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                             weight_decay=params['weight_decay'])

for e in range(params['epochs']):
    print('Epoch:',e)
    model.train()
    total_loss = 0
    iterations = 0
    losses = []
    for i, data in enumerate(train_loader):
        input_data = Variable(data['input']).to(device)
        input_data = input_data.unsqueeze(1)
        # print('Input:',input_data.shape)
        output = model(input_data)
        label = Variable(data['label']).float().to(device)

        # print('Label:',label)
        # print('Output:',output)
        # print('Output:',output.shape)

        loss = criterion(output, label.unsqueeze(1))
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterations += 1          

        # if iterations % 5 == 0:
            # print('Training Loss:', np.mean(losses[-5]))

        print('Training Loss: {:.3f}'.format(loss.item()))

    print('Validation...')
    model.eval()
    val_loss = []
    accuracy = []
    iterations = 0
    for i, data in enumerate(val_loader):
        input_data = Variable(data['input']).to(device)
        input_data = input_data.unsqueeze(1)
        # print('Input:',input_data.shape)
        output = model(input_data)
        label = Variable(data['label']).float().to(device)

        loss = criterion(output, label.unsqueeze(1))
        val_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterations += 1          

        prediction = (output > 0).float()
        correct = (prediction == label).float().sum()
        accuracy.append(correct / label.shape[0])

        print('Output:',output[0])

        print('Label:',label)
        print('Prediction:',prediction)
        print('Acc:',accuracy[-1])
        # if iterations % 5 == 0:
        #     print('Accuracy:', np.mean(accuracy))

    print('Validation Loss: {:.3f}'.format(np.mean(val_loss)))
