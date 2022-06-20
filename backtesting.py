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
import time

#%% Setting

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

#%%
ballance = 1000000
crypto_ea = 0
crypto_asset=  0
origin =0
total_revenue = 0
eval_data = pd.read_pickle('testdata.pkl')
model = models.LSTMModel().to(device)
checkpoint = torch.load("default_final_best.ckpt")
model.load_state_dict(checkpoint)
for i in range(len(eval_data)):
    current = eval_data['Price'][i][59]
    input_seq = torch.tensor(eval_data['Z'][i]).unsqueeze(0).to(device)
    label = eval_data['ZL'][i]
    output = model(input_seq)*eval_data['Stddev'][i]+eval_data['Mean'][i]
    crypto_asset = current*crypto_ea
    asset = ballance + crypto_asset
    if crypto_ea > 0 :
        avg = origin/crypto_ea
    else:
        avg = 0
    
    print(f'current BTC price is {current}')
    print(f'our prediction of BTC price is {output}')
    print(f'your avg is {avg}')
    print(f'your ballance is {ballance} and your BTC is {crypto_asset}:{crypto_ea}EA')
    print(f'your total asset is {asset} and your total revenue is {total_revenue}')
    if current > avg :
        """
        if current > 1.05*avg :
            if crypto_ea > 1:
                print(f"sell all BTC your Revenue is {(current-avg)*crypto_ea}")
                ballance = ballance + crypto_asset
                print(crypto_ea)
                print(current)
                print(avg)
                total_revenue = total_revenue + float((current-avg)*crypto_ea)
                crypto_ea = 0
                origin = 0
        """
        if output < current :
            if crypto_ea > 1:
                print(f"sell all BTC your Revenue is {(current-avg)*crypto_ea}")
                ballance = ballance + float(crypto_ea*current-0.001*current*crypto_ea)
                total_revenue = total_revenue + float((current-avg)*crypto_ea-0.001*current*crypto_ea)
                origin = 0
                crypto_ea = 0
        elif output > current :
            if ballance > current:
                ea = int(ballance/(2*current))
                crypto_ea = crypto_ea +  ea
                #crypto_asset = crypto_asset + float(ea*current)
                origin = origin + float(ea*current)
                print(f'buy {ea}BTC and value is {ea*current}')
                ballance = ballance - float(ea*current+0.001*current*ea)
                total_revenue = total_revenue - float(0.001*current*ea)
    elif current < avg:
        if current < avg*0.9:
            print(f"LOSS sell all BTC your Revenue is {(current-avg)*crypto_ea}")
            ballance = ballance + float(crypto_ea*current-0.001*current*crypto_ea)
            total_revenue = total_revenue + float((current-avg)*crypto_ea-0.001*current*crypto_ea)
            crypto_ea = 0
            origin = 0
        elif ballance > current:
            if output> current*1.02:
                ea = int(ballance/(2*current))
                crypto_ea = crypto_ea +  ea
                #crypto_asset = crypto_asset + float(ea*current)
                print(f'buy {ea}BTC and value is {ea*current}')
                origin = origin + float(ea*current)
                ballance = ballance - float(ea*current+0.001*current*ea)
                total_revenue = total_revenue - float(0.001*current*ea)
    print(' ')
    
    time.sleep(0.2)
                
  