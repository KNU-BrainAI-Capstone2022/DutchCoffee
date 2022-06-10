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

parser = argparse.ArgumentParser(prog="train", description="BART for price prediction")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--method", type=str, choices=["default", "pretrain","finetuning"], default="default", help="training method")
g.add_argument("--pretrained-ckpt-path", type=str, help="pretrained BART model path or name")
g.add_argument("--batch-size", type=int, default=4, help="training batch size")
g.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
g.add_argument("--max-seq-len", type=int, default=60, help="Input Sequence's max length")

def main(args: argparse.Namespace):

    if args.method == "pretrain":
        data = pd.read_pickle('pretrain.pkl')
    elif args.method == "finetuning":
        data = pd.read_pickle('finetuning.pkl')
        eval_data = pd.read_pickle('validata.pkl')
        
    train_step = {"pretrain": trainstep.pretrain_step,"finetuning": trainstep.finetuning_step}[args.method]
    datasets = {"pretrain": dataset.PretrainDataset,"finetuning": dataset.FinetuningDataset1}[args.method]
    
    train_dataset = datasets(dataframe = data ,max_seq_len=args.max_seq_len)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    
    eval_dataset = datasets(dataframe = eval_data ,max_seq_len=args.max_seq_len)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=args.batch_size)
        
    
    epochs = args.epochs
    model = models.LSTMModel()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 
    
    
    for epoch in range(epochs): 
        gc.collect()
        total_train_loss, total_val_loss = 0, 0
        total_train_acc, total_val_acc = 0, 0
        
        
        tqdm_dataset = tqdm(train_dataloader)
        training = True
        for batch, batch_item in enumerate(tqdm_dataset):
            
            batch_loss, batch_acc= train_step(batch_item, epoch, batch, training, model, optimizer)
            total_train_loss += batch_loss.item()
            total_train_acc += batch_acc
            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Total Loss' : '{:06f}'.format(total_train_loss/(batch+1)),
                'Total ACC' : '{:06f}'.format(total_train_acc/(batch+1)),
                'learning rawte' : '{:06f}'.format(optimizer.param_groups[0]['lr'])
            })
            
        training = False
        tqdm_dataset = tqdm(eval_dataloader)
        for batch, batch_item in enumerate(tqdm_dataset):
            
            batch_loss, batch_acc= train_step(batch_item, epoch, batch, training, model, optimizer)
            total_val_loss += batch_loss.item()
            total_val_acc += batch_acc
            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Total Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                'Total Val ACC' : '{:06f}'.format(total_val_acc/(batch+1))
            })
        
        torch.save(model.state_dict(), f'{args.method}_{epoch+1}_epoch.ckpt')

if __name__ == "__main__":
    exit(main(parser.parse_args()))