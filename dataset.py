from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch
import random
import numpy  as np
import os

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)

class Dataset(Dataset):
    def __init__( self,dataframe)->None:
        super().__init__()
        self.dataframe = dataframe

    def __len__(self) -> int:
        return self.dataframe.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        inputs = self.dataframe['Z'][index]
        Mean = self.dataframe['Mean'][index]
        Stddev = self.dataframe['Stddev'][index]
        labels = torch.tensor(self.dataframe['ZL'][index], dtype=torch.float16)
        original_input = self.dataframe['Price'][index]
        original_label = self.dataframe['Label'][index]
        
        return {
            "inputs": torch.tensor(inputs),
            'label' : labels[0],
            'label2' : labels[1],
            "original_input": original_input,
            "original label": original_label,
            'Mean' : Mean,
            'Stddev':Stddev
        }
