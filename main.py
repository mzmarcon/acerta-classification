import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from dataset_loader import ACERTA_data 
from model import VGGBased13
import sys 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)

checkpoint = 'checkpoints/checkpoint.pth'

params = { 'train_batch_size': 4,
           'val_batch_size': 1,
           'learning_rate': 2e-5,
           'weight_decay': 1e-1,
           'epochs': 100,
           'early_stop': 10 }


training_set = ACERTA_data(set='training', split=0.8)
validation_set = ACERTA_data(set='validation', split=0.8)

train_loader = DataLoader(training_set, shuffle=True, drop_last=True,
                             num_workers=8, batch_size=params['train_batch_size'])

val_loader = DataLoader(validation_set, shuffle=False, drop_last=False,
                        num_workers=8, batch_size=params['val_batch_size'])

# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

model = VGGBased13()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                             weight_decay=params['weight_decay'])

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,10], gamma=0.1)

best_loss = np.inf  
loss_list = []
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

        # if iterations % 5 == 0:
            # print('Training Loss: {:.3f}'.format(np.mean(losses)))

    # print('Training Loss: {:.3f}'.format(np.mean(losses)))
    # print('Training Accuracy: {:.3f}'.format(torch.mean(torch.stack(accuracy))))

    print('Training Loss:   {:.3f} - Accuracy: {:.3f}'.format(np.mean(losses),torch.mean(torch.stack(accuracy))))

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

        # if iterations % 4 == 0:
        #     print('Validation Loss: {:.3f}'.format(np.mean(val_loss)))

    #early stop
    loss_list.append(np.mean(val_loss))
    if loss_list[-1] < best_loss:
        best_loss = loss_list[-1]
        loss_list = []
        # print('New best:',best_loss)
    if len(loss_list) == params['early_stop']:
        print('Early stopping.')
        print('Validation Loss: {:.3f} - Accuracy: {:.3f}'.format(np.mean(val_loss),
                                                                  torch.mean(torch.stack(accuracy))))
        torch.save(model.state_dict(), checkpoint)    
        break

    # print('List len:',len(loss_list))
    # print('Validation Accuracy: {:.3f}'.format(torch.mean(torch.stack(accuracy))))
    print('Validation Loss: {:.3f} - Accuracy: {:.3f}'.format(np.mean(val_loss),
                                                              torch.mean(torch.stack(accuracy))))
    torch.save(model.state_dict(), checkpoint)
