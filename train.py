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


class LinearWarmupLR(LambdaLR):
    """LR Scheduling function which is increase lr on warmup steps and decrease on normal steps"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        reduce_rate: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            optimizer: torch optimizer
            num_warmup_steps: number of warmup steps
            num_training_steps: number of whole training steps
            reduce_rate: min Learning Rate / max Learning Rate
        """
        self.num_warmup_steps = num_warmup_steps
        self.decrement = (1.0 - reduce_rate) / (num_training_steps - num_warmup_steps)
        self.reduce_rate = reduce_rate

        super().__init__(optimizer, self._get_lr, last_epoch=last_epoch, verbose=verbose)

    def _get_lr(self, current_step: int) -> float:
        if current_step < self.num_warmup_steps:
            return current_step / self.num_warmup_steps
        return max(1.0 - self.decrement * (current_step - self.num_warmup_steps), self.reduce_rate)

def main(args: argparse.Namespace):

    #os.makedirs(args.output_dir)

    #if args.method == "pretrain":
    #    train_dataset = dataset.PretrainDataset(dataframe = data ,max_seq_len=args.max_seq_len)
    if args.method == "pretrain":
        data = pd.read_pickle('data.pkl')
    elif args.method == "finetuning":
        data = pd.read_pickle('finetuning.pkl')
        
    train_step = {"pretrain": trainstep.pretrain_step,"finetuning": trainstep.finetuning_step}[args.method]
    datasets = {"pretrain": dataset.PretrainDataset,"finetuning": dataset.FinetuningDataset}[args.method]
    train_dataset = datasets(dataframe = data ,max_seq_len=args.max_seq_len)
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
    print(model.config)
    if args.method == "finetuning":
        model.load_state_dict(torch.load('Pretrain_60_epoch.ckpt'))
        
    
    epochs = args.epochs
    learning_rate = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    total_steps = len(train_dataloader) * epochs
    scheduler = LinearWarmupLR(optimizer,
            int(total_steps * 0.05),
            total_steps,
            1e-5 / 2e-4)
    
    
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
                'learning rate' : '{:06f}'.format(optimizer.param_groups[0]['lr'])
            })
            
        torch.save(model.state_dict(), f'{args.method}_{epoch+1}_epoch.ckpt')
        scheduler.step()
            
    

    #model_dir = os.path.join(args.output_dir, "models")



if __name__ == "__main__":
    exit(main(parser.parse_args()))