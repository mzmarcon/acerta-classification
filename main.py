import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from dataset_loader import ACERTA_data 
from model import VGGBasedModel2D
import sys 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)

checkpoint = 'checkpoints/checkpoint.pth'

params = { 'train_batch_size': 40,
           'val_batch_size': 1,
           'learning_rate': 1e-5,
           'weight_decay': 1e-1,
           'epochs': 100 }


training_set = ACERTA_data(set='training', split=0.8)
validation_set = ACERTA_data(set='validation', split=0.8)

train_loader = DataLoader(training_set, shuffle=True, drop_last=True,
                             num_workers=8, batch_size=params['train_batch_size'])

val_loader = DataLoader(validation_set, shuffle=False, drop_last=False,
                        num_workers=8, batch_size=params['val_batch_size'])

# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

model = VGGBasedModel2D()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                             weight_decay=params['weight_decay'])

for e in range(params['epochs']):
    print('Epoch:',e)
    model.train()
    losses = []
    accuracy = []
    predictions = defaultdict(list)
    iterations = 0
    for i, data in enumerate(train_loader):
        input_data = Variable(data['input']).float().to(device)
        output = model(input_data)
        label = Variable(data['label']).float().to(device)

        loss = criterion(output, label.unsqueeze(1))
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterations += 1

        prediction = (output > 0).float()
        correct = (prediction == label.reshape(-1,1)).float()
        total_correct = correct.sum()
        accuracy.append((total_correct / label.shape[0]))

        #voting classification
        for n in range(params['train_batch_size']): 
            predictions[int(data['id'][n])].append(int(correct[n])) 

        print('Training Loss: {:.3f}'.format(loss.item()))

    voting_acc = []
    for subject in list(predictions.keys()):
        sub_score = np.array(predictions[subject]).sum() / len(predictions[subject])
        voting_acc.append([1 if sub_score > 0.5 else 0])

    print('Accuracy:', torch.mean(torch.stack(accuracy)))
    print('Voting Accuracy:',np.mean(voting_acc))

    print('Validation...')
    model.eval()
    val_loss = []
    accuracy = []
    predictions = defaultdict(list)
    iterations = 0
    for i, data in enumerate(val_loader):
        input_data = Variable(data['input']).to(device)
        output = model(input_data)
        label = Variable(data['label']).float().to(device)

        loss = criterion(output, label.unsqueeze(1))
        val_loss.append(loss.item())
        iterations += 1 

        prediction = (output > 0).float()
        correct = (prediction == label.reshape(-1,1)).float()
        total_correct = correct.sum()
        accuracy.append(total_correct / label.shape[0])

        #voting classification
        for n in range(params['val_batch_size']): 
            predictions[int(data['id'][n])].append(int(correct[n])) 

        if iterations % 50 == 0:
            print('Validation Loss: {:.3f}'.format(loss.item()))

    voting_acc = []
    for subject in list(predictions.keys()):
        sub_score = np.array(predictions[subject]).sum() / len(predictions[subject])
        voting_acc.append([1 if sub_score > 0.5 else 0])
    
    print('Accuracy:', torch.mean(torch.stack(accuracy)))
    print('Voting Accuracy:',np.mean(voting_acc))
    torch.save(model.state_dict(), checkpoint)