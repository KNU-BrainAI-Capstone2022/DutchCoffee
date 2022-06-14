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

    if training is True:
        model.train()
        optimizer.zero_grad()
        loss_fn = torch.nn.MSELoss()
        #print(batch_item['inputs'])
        with torch.cuda.amp.autocast():
            output = model(batch_item['inputs'].to(device))
            
        loss = loss_fn(output, batch_item['labels'][0:batch_item.size].to(device))
        accuracy = torchmetrics.functional.accuracy(output,batch_item['labels'][0:batch_item.size].to(device))
        
        loss.backward()
        optimizer.step()

        return loss, accuracy
    else:
        model.eval()
        #with torch.no_grad():

        return loss, accuracy