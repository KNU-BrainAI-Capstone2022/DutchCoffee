import random
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchmetrics
import torch.nn as nn

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


def accuracy_function(real, pred):

    accuracies = torch.eq(real, torch.argmax(pred, dim=1))
    #print(torch.argmax(pred,dim=1))
    mask = torch.logical_not(torch.eq(real, 0))
    accuracies = torch.logical_and(mask, accuracies)
    accuracies = accuracies.clone().detach()
    mask = mask.clone().detach()

    return torch.sum(accuracies)/torch.sum(mask)



def train_step(batch_item, epoch, batch, training, model, optimizer):
    
    loss_fn = nn.MSELoss()    
    if training is True:
        model.train()
        optimizer.zero_grad()
        label = batch_item['label'].to(device)
        #label2 = batch_item['label2'].to(device)
        input_seq = batch_item['inputs'].to(device)
        
        #print(label)
        with torch.cuda.amp.autocast():
            output = model(input_seq)
            
          

        loss = loss_fn(output, label)
        
        loss.backward()
        optimizer.step()

        return loss
    else:
        model.eval()
        label = batch_item['label'].to(device)
        with torch.no_grad():
            output = model(batch_item['inputs'].to(device))
        loss = loss_fn(output, label)

        return loss
    
def pretrain_step(batch_item, epoch, batch, training, model, optimizer):
    
    loss_fn = nn.CrossEntropyLoss()
    if training is True:
        model.train()
        optimizer.zero_grad()
        label = batch_item['label'].to(device)
        #label2 = batch_item['label2'].to(device)
        input_seq = batch_item['inputs'].to(device)
        
        #print(label)
        with torch.cuda.amp.autocast():
            output = model(input_seq)

        loss = loss_fn(output, input_seq) #+ loss_fn(secondoutput,label2)
        
        loss.backward()
        optimizer.step()

        return loss
    else:
        model.eval()
        label = batch_item['label'].to(device)
        input_seq = batch_item['inputs'].to(device)
        with torch.no_grad():
            output = model(input_seq)
        loss = loss_fn(output, input_seq) #+ loss_fn(secondoutput,label2)

        return loss
    
