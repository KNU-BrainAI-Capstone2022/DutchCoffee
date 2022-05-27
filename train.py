import argparse
import os
from glob import glob

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration

from summarizer.data import DialogueSummarizationDataset, PretrainDataset
from summarizer.method import DefaultModule, R3FModule, RDropModule, ReinforceLearningModule
from summarizer.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser(prog="train", description="BART for price prediction")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--method", type=str, choices=["default", "pretrain"], default="default", help="training method")
g.add_argument("--pretrained-ckpt-path", type=str, help="pretrained BART model path or name")
g.add_argument("--batch-size", type=int, default=128, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=256, help="validation batch size")
g.add_argument("--epochs", type=int, default=10, help="the numnber of training epochs")
g.add_argument("--max-learning-rate", type=float, default=2e-4, help="max learning rate")
g.add_argument("--min-learning-rate", type=float, default=1e-5, help="min Learning rate")
g.add_argument("--warmup-rate", type=float, default=0.05, help="warmup step rate")
g.add_argument("--max-seq-len", type=int, default=256, help="dialogue max sequence length")
g.add_argument("--pred-max-seq-len", type=int, default=64, help="summary max sequence length")
g.add_argument("--all-dropout", type=float, help="override all dropout")
g.add_argument("--logging-interval", type=int, default=100, help="logging interval")
g.add_argument("--evaluate-interval", type=int, default=500, help="validation interval")
g.add_argument("--masking-rate", type=float, default=0.3, help="pretrain parameter (only used with `pretrain` method)")


def main(args: argparse.Namespace):

    os.makedirs(args.output_dir)

    if args.method == "pretrain":
        train_dataset = PretrainDataset(
            paths=glob(args.train_dataset_pattern),
            dialogue_max_seq_len=args.max_seq_len,
            masking_rate=args.masking_rate,
        )
        valid_dataset = PretrainDataset(
            paths=glob(args.valid_dataset_pattern),
            dialogue_max_seq_len=args.dialogue_max_seq_len,
            masking_rate=args.masking_rate,
        )

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)


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
    model = BartForConditionalGeneration(BartConfig.from_pretrained(args.model_config_path, **override_args))
    

    model_dir = os.path.join(args.output_dir, "models")



if __name__ == "__main__":
    exit(main(parser.parse_args()))