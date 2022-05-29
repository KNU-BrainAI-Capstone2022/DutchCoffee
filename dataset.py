from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch
import random
import numpy  as np


class PretrainDataset(Dataset):
    """Dataset for pretraining of BART with dialogue
    Attributes:
        tokenizer: tokenizer to tokenize dialogue and summary string
        dialogue_max_seq_len: max sequence length of dialouge
        masking_rate: rate of the number of masked token / sequence length
        bos_token: bos token
        eos_token: eos token
        sep_token: turn seperation token to divide each utterances
        mask_token_id: mask token id for text infilling
        ids: id of each example
        dialogues: dialogue of each example
    """

    def __init__( self,max_seq_len, dataframe,masking_rate: float = 0.15)->None:
        """
        Args:
            paths: list of dataset paths (tsv or json)
            tokenizer: tokenizer to tokenize dialogue and summary string
            dialogue_max_seq_len: max sequence length of dialouge
            masking_rate: rate of the number of masked token / sequence length
        Returns:
            original ids, dialogues, summaries and input ids and attention masks for dialogues and summaries
        """
        
        """
        data = pd.DataFrame(columns=['Price'])
        train2 = []


        for i in range(int((df2['Close'].size)/30 -1 )):
            print(i)
            temp = []
            for k in range(60):
                temp.append(df2['Close'][i*30+k])
            train2.append(temp)
        """
        
        super().__init__()

        self.max_seq_len = max_seq_len
        self.masking_rate = masking_rate
        self.mask_token_id = 6
        self.bos_token = 2
        self.eos_token = 3
        self.dataframe = dataframe

    def __len__(self) -> int:
        return self.dataframe.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        #print(index) 
        #target_row = self.dataframe[index]
        input_ids = self.dataframe['Price'][index]
        num_masking = int(self.max_seq_len * self.masking_rate)
        
        decoder_input_ids = torch.from_numpy(np.array([self.bos_token] + 
                                                      input_ids +[self.eos_token],dtype=np.int64))
        # bos + 인풋 + eos  size 62
        decoder_attention_mask = torch.from_numpy(np.array([1] * len(decoder_input_ids),dtype=np.float_))
        encoder_input_ids = torch.from_numpy(np.array(input_ids,dtype=np.int64))
        indices = torch.randperm(self.max_seq_len)[:num_masking]
        encoder_input_ids[indices] = self.mask_token_id
        encoder_input_ids = torch.cat([encoder_input_ids,
                                       torch.from_numpy(np.array([self.eos_token]+[0],dtype=np.int64))])
        # 마스킹된 인풋 + eos + pad size 62 마스킹률 15% ( 60*0.15 = 9)
        encoder_attention_mask = decoder_attention_mask.clone()
        labels = torch.from_numpy(np.array(input_ids+[self.eos_token]+[0], dtype=np.int64))
        # 인풋 + eos + pad size 62
        
        return {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            'labels' : labels
        }
    
class NormPretrainDataset(Dataset):
    def __init__( self,max_seq_len, dataframe,masking_rate: float = 0.15)->None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.masking_rate = masking_rate
        self.mask_token_id = 6
        self.bos_token = 2
        self.eos_token = 3
        self.dataframe = dataframe

    def __len__(self) -> int:
        return self.dataframe.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        #print(index) 
        #target_row = self.dataframe[index]
        input_ids = self.dataframe['Price'][index]
        
        num_masking = int(self.max_seq_len * self.masking_rate)
        
        decoder_input_ids = torch.from_numpy(np.array([self.bos_token] + 
                                                      input_ids +[self.eos_token],dtype=np.int64))
        # bos + 인풋 + eos  size 62
        decoder_attention_mask = torch.from_numpy(np.array([1] * len(decoder_input_ids),dtype=np.float_))
        encoder_input_ids = torch.from_numpy(np.array(input_ids,dtype=np.int64))
        indices = torch.randperm(self.max_seq_len)[:num_masking]
        encoder_input_ids[indices] = self.mask_token_id
        encoder_input_ids = torch.cat([encoder_input_ids,
                                       torch.from_numpy(np.array([self.eos_token]+[0],dtype=np.int64))])
        # 마스킹된 인풋 + eos + pad size 62 마스킹률 15% ( 60*0.15 = 9)
        encoder_attention_mask = decoder_attention_mask.clone()
        labels = torch.from_numpy(np.array(input_ids+[self.eos_token]+[0], dtype=np.int64))
        # 인풋 + eos + pad size 62
        
        return {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            'labels' : labels
        }
    
    
class FinetuningDataset(Dataset):

    def __init__( self,max_seq_len, dataframe)->None:        
        super().__init__()

        self.max_seq_len = max_seq_len
        self.mask_token_id = 6
        self.bos_token = 2
        self.eos_token = 3
        self.dataframe = dataframe
        

    def __len__(self) -> int:
        return self.dataframe.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        #print(index) 
        #target_row = self.dataframe[index]
        input_ids = self.dataframe['Price'][index]
        label = self.dataframe['label'][index]
        minimum = self.dataframe['Minimum'][index]
        
        decoder_input_ids = torch.from_numpy(np.array([self.bos_token] + 
                                                      label +[self.eos_token],dtype=np.int64))
        # bos + input + eos  size 62
        decoder_attention_mask = torch.from_numpy(np.array([1] * len(decoder_input_ids),dtype=np.float_))
        encoder_input_ids = torch.from_numpy(np.array(input_ids+[self.eos_token]+[0],dtype=np.int64))
        # input + eos + pad size 62
        encoder_attention_mask = decoder_attention_mask.clone()
        labels = torch.from_numpy(np.array(label+[self.eos_token]+[0], dtype=np.int64))
        # label + eos + pad size 62
        
        return {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            'labels' : labels,
            'Minimum' : minimum
        }
    
class NormFinetuningDataset(Dataset):
    def __init__( self,max_seq_len, dataframe,masking_rate: float = 0.15)->None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.masking_rate = masking_rate
        self.mask_token_id = 6
        self.bos_token = 2
        self.eos_token = 3
        self.dataframe = dataframe

    def __len__(self) -> int:
        return self.dataframe.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        #print(index) 
        #target_row = self.dataframe[index]
        input_ids = self.dataframe['Price'][index]
        minimum = self.dataframe['Mninimum'][index]
        
        num_masking = int(self.max_seq_len * self.masking_rate)
        
        decoder_input_ids = torch.from_numpy(np.array([self.bos_token] + 
                                                      input_ids +[self.eos_token],dtype=np.int64))
        # bos + 인풋 + eos  size 62
        decoder_attention_mask = torch.from_numpy(np.array([1] * len(decoder_input_ids),dtype=np.float_))
        encoder_input_ids = torch.from_numpy(np.array(input_ids,dtype=np.int64))
        indices = torch.randperm(self.max_seq_len)[:num_masking]
        encoder_input_ids[indices] = self.mask_token_id
        encoder_input_ids = torch.cat([encoder_input_ids,
                                       torch.from_numpy(np.array([self.eos_token]+[0],dtype=np.int64))])
        # 마스킹된 인풋 + eos + pad size 62 마스킹률 15% ( 60*0.15 = 9)
        encoder_attention_mask = decoder_attention_mask.clone()
        labels = torch.from_numpy(np.array(input_ids+[self.eos_token]+[0], dtype=np.int64))
        # 인풋 + eos + pad size 62
        
        return {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            'labels' : labels,
            'Minimum' : minimum
        }
    