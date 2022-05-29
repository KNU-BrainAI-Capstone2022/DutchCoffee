import argparse
import os
from glob import glob
import dataset
from torch.utils.data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration
import pandas as pd
import trainstep
import gc
from tqdm import tqdm
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
# fmt: off
parser = argparse.ArgumentParser(prog="train", description="BART for price prediction")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--method", type=str, choices=["default", "pretrain","finetuning"], default="default", help="training method")
g.add_argument("--pretrained-ckpt-path", type=str, help="pretrained BART model path or name")
g.add_argument("--batch-size", type=int, default=4, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=256, help="validation batch size")
g.add_argument("--epochs", type=int, default=10, help="the numnber of training epochs")
g.add_argument("--max-learning-rate", type=float, default=2e-4, help="max learning rate")
g.add_argument("--min-learning-rate", type=float, default=1e-5, help="min Learning rate")
g.add_argument("--warmup-rate", type=float, default=0.05, help="warmup step rate")
g.add_argument("--max-seq-len", type=int, default=60, help="dialogue max sequence length")
g.add_argument("--pred-max-seq-len", type=int, default=64, help="summary max sequence length")
g.add_argument("--all-dropout", type=float, help="override all dropout")
g.add_argument("--logging-interval", type=int, default=100, help="logging interval")
g.add_argument("--evaluate-interval", type=int, default=500, help="validation interval")
g.add_argument("--masking-rate", type=float, default=0.3, help="pretrain parameter (only used with `pretrain` method)")

def accuracy_function(real, pred):

    accuracies = torch.eq(real, pred)
    #print(torch.argmax(pred,dim=1))
    mask = torch.logical_not(torch.eq(real, 0))
    accuracies = torch.logical_and(mask, accuracies)
    accuracies = accuracies.clone().detach()
    mask = mask.clone().detach()

    return torch.sum(accuracies)/torch.sum(mask)


def main(args: argparse.Namespace):

    #os.makedirs(args.output_dir)

    #if args.method == "pretrain":
    #    train_dataset = dataset.PretrainDataset(dataframe = data ,max_seq_len=args.max_seq_len)
    
    data = pd.read_pickle('test2.pkl')
    
    train_dataset = dataset.FinetuningDataset(dataframe = data ,max_seq_len=args.max_seq_len)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    
    override_args = (
        {
            "dropout": args.all_dropout,
            "attention_dropout": args.all_dropout,
            "activation_dropout": args.all_dropout,
            "classifier_dropout": args.all_dropout,
        }
        if args.all_dropout
        else {}
    )
    
    model = BartForConditionalGeneration(BartConfig.from_pretrained('default.json', **override_args)).to(device)
    model.load_state_dict(torch.load('finetuning_30_epoch.ckpt'))
    
    
    tqdm_dataset = tqdm(train_dataloader)
    total_acc = 0
    for batch, batch_item in enumerate(tqdm_dataset):
        
        res_ids = model.generate(input_ids = batch_item['input_ids'].to(device), max_length=61,
                                 num_beams=5,eos_token_id=3)
        accuracy = accuracy_function(batch_item['labels'][:,:].to(device),res_ids)
        print(batch_item['input_ids'])
        print(res_ids)
        print(batch_item['labels'])
        
        total_acc += accuracy
        
        tqdm_dataset.set_postfix({
            'Total ACC' : '{:06f}'.format(total_acc/(batch+1))
        })
    

        

    

    #model_dir = os.path.join(args.output_dir, "models")



if __name__ == "__main__":
    exit(main(parser.parse_args()))