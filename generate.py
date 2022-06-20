import argparse
import os
import dataset
from torch.utils.data import DataLoader
import pandas as pd
import trainstep
import gc
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
import models
import random
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


eval_data = pd.read_pickle('testdata3.pkl')
model = models.LSTMModel().to(device)
checkpoint = torch.load("default_final_best.ckpt")
model.load_state_dict(checkpoint)
labels = []
outputs =[]
for i in range(len(eval_data)):
    input_seq = torch.tensor(eval_data['Z'][i]).unsqueeze(0).to(device)
    label = eval_data['Label'][i][0]
    output = float(model(input_seq))*eval_data['Stddev'][i]+eval_data['Mean'][i]
    outputs.append(output)
    labels.append(label)

x=np.arange(len(labels))    
plt.plot(x,labels,outputs)


labels = []
outputs =[]
for i in range(2500,4200):
    input_seq = torch.tensor(eval_data['Z'][i]).unsqueeze(0).to(device)
    label = eval_data['ZL'][i][0]
    output = float(model(input_seq))
    outputs.append(output)
    labels.append(label)

x=np.arange(len(labels))
plt.plot(x,labels,outputs)

labels = []
outputs =[]
for i in range(2500,4200):
    input_seq = torch.tensor(eval_data['Z'][i]).unsqueeze(0).to(device)
    label = eval_data['Label'][i][0]
    output = float(model(input_seq))*eval_data['Stddev'][i]+eval_data['Mean'][i]
    outputs.append(output)
    labels.append(label)

x=np.arange(len(labels))
plt.plot(x,labels,outputs)

labels = []
outputs =[]
for i in range(2700,2900):
    input_seq = torch.tensor(eval_data['Z'][i]).unsqueeze(0).to(device)
    label = eval_data['Label'][i][0]
    output = float(model(input_seq))*eval_data['Stddev'][i]+eval_data['Mean'][i]
    outputs.append(output)
    labels.append(label)

x=np.arange(len(labels))
plt.plot(x,labels,outputs)









eval_data = pd.read_pickle('testdata3.pkl')
model = models.LSTMModel().to(device)
checkpoint = torch.load("default_final_best.ckpt")
model.load_state_dict(checkpoint)
    

inputs = torch.tensor(eval_data['Z'][2]).unsqueeze(0)
model(inputs.to(device))
label = eval_data['Label'][2]
mean = eval_data['Mean'][2]
stddev = eval_data['Stddev'][2]
outputs = []
input_seq = inputs.detach().to(device)
for i in range(10):
    output = model(input_seq)
    foutput = float(output)*stddev+mean
    outputs.append(foutput)
    output = output.unsqueeze(0)
    input_seq = torch.cat([input_seq[0][1:],output]).unsqueeze(0)
labels = eval_data['Price'][2] + label
outputss = eval_data['Price'][2] +outputs
x = np.arange(110)
plt.plot(x,labels,outputss)


eval_data = pd.read_pickle('testdata3.pkl')
model = models.LSTMModel().to(device)
checkpoint = torch.load("default_final_best.ckpt")
model.load_state_dict(checkpoint)
    

inputs = torch.tensor(eval_data['Z'][3]).unsqueeze(0)
model(inputs.to(device))
label = eval_data['Label'][3]
mean = eval_data['Mean'][3]
stddev = eval_data['Stddev'][3]
outputs = []
input_seq = inputs.detach().to(device)
for i in range(10):
    output = model(input_seq)
    foutput = float(output)*stddev+mean
    outputs.append(foutput)
    output = output.unsqueeze(0)
    input_seq = torch.cat([input_seq[0][1:],output]).unsqueeze(0)
labels = eval_data['Price'][3] + label
outputss = eval_data['Price'][3] +outputs
x = np.arange(110)
plt.plot(x,labels,outputss)

inputs = torch.tensor(eval_data['Z'][10]).unsqueeze(0)
model(inputs.to(device))
label = eval_data['Label'][10]
mean = eval_data['Mean'][10]
stddev = eval_data['Stddev'][10]
outputs = []
input_seq = inputs.detach().to(device)
for i in range(10):
    output = model(input_seq)
    foutput = float(output)*stddev+mean
    outputs.append(foutput)
    output = output.unsqueeze(0)
    input_seq = torch.cat([input_seq[0][1:],output]).unsqueeze(0)
labels = eval_data['Price'][10] + label
outputss = eval_data['Price'][10] +outputs
x = np.arange(110)
plt.plot(x,labels,outputss)
