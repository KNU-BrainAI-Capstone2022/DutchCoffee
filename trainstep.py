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



def pretrain_step(batch_item, epoch, batch, training, model, optimizer):

    if training is True:
        model.train()
        model.model.encoder.config.gradient_checkpointing = True
        model.model.decoder.config.gradient_checkpointing = True
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input_ids = batch_item['input_ids'].to(device),
                          attention_mask = batch_item['attention_mask'].to(device),
                          decoder_input_ids = batch_item['decoder_input_ids'].to(device),
                          decoder_attention_mask = batch_item['decoder_attention_mask'].to(device),
                           use_cache=False, return_dict=True)
        
        labels = batch_item["labels"][:,:-1].reshape(-1).to(device)
        # 레이블 =  인풋 + eos + pad size 62 우리가 볼 건 61개
        logits = output["logits"][:,:-1].reshape([labels.shape[0], -1])
        # 생성된 녀석 우리가 볼 건 앞의 61개

        loss = F.cross_entropy(logits, labels,ignore_index=0)
        # 이거 logits이 들어가서 안에서 소프트맥스 알아서 됨.
        print(f'inputs are {batch_item["input_ids"]}')
        # 마스킹 된 인풋 데이터가 뭔지 표시
        print(f'pred are {torch.argmax(F.softmax(logits),dim=1)}')
        # 소프트맥스 된 실제 결과 어떤 녀석인지 표시
        print(f'labels are {labels}')
        # 레이블이 뭔지 표시
        accuracy = torchmetrics.functional.accuracy(logits, labels,ignore_index=0)
        
        loss.backward()
        optimizer.step()

        return loss, accuracy
    else:
        print('validation step')
        model.eval()
        with torch.no_grad():
            output = model(input_ids = batch_item['input_ids'].to(device),
                          attention_mask = batch_item['attention_mask'].to(device),
                          decoder_input_ids = batch_item['decoder_input_ids'].to(device),
                          decoder_attention_mask = batch_item['decoder_attention_mask'].to(device),
                           use_cache=False, return_dict=True)
            
        labels = batch_item["labels"][:,:-1].reshape(-1).to(device)
        logits = output["logits"][:,:-1].reshape([labels.shape[0], -1])
            
        loss = F.cross_entropy(logits, labels,ignore_index=0)
        accuracy = torchmetrics.functional.accuracy(logits, labels,ignore_index=0)
        

        return loss, accuracy
    
    

def finetuning_step(batch_item, epoch, batch, training, model, optimizer):

    if training is True:
        model.train()
        model.model.encoder.config.gradient_checkpointing = True
        model.model.decoder.config.gradient_checkpointing = True
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input_ids = batch_item['input_ids'].to(device),
                          attention_mask = batch_item['attention_mask'].to(device),
                          decoder_input_ids = batch_item['decoder_input_ids'].to(device),
                          decoder_attention_mask = batch_item['decoder_attention_mask'].to(device),
                           use_cache=False, return_dict=True)
        
        labels = batch_item["labels"][:,:-1].reshape(-1).to(device)
        # 레이블 =  인풋 + eos + pad size 62 우리가 볼 건 61개
        logits = output["logits"][:,:-1].reshape([labels.shape[0], -1])
        # 생성된 녀석 우리가 볼 건 앞의 61개

        loss = F.cross_entropy(logits, labels,ignore_index=0)
        # 이거 logits이 들어가서 안에서 소프트맥스 알아서 됨.
        print(f'inputs are {batch_item["input_ids"]}')
        # 마스킹 된 인풋 데이터가 뭔지 표시
        print(f'pred are {torch.argmax(F.softmax(logits),dim=1)}')
        # 소프트맥스 된 실제 결과 어떤 녀석인지 표시
        print(f'labels are {labels}')
        # 레이블이 뭔지 표시
        accuracy = torchmetrics.functional.accuracy(logits, labels,ignore_index=0)
        
        loss.backward()
        optimizer.step()

        return loss, accuracy
    else:
        print('validation step')
        model.eval()
        with torch.no_grad():
            output = model(input_ids = batch_item['input_ids'].to(device),
                          attention_mask = batch_item['attention_mask'].to(device),
                          decoder_input_ids = batch_item['decoder_input_ids'].to(device),
                          decoder_attention_mask = batch_item['decoder_attention_mask'].to(device),
                           use_cache=False, return_dict=True)
            
        labels = batch_item["labels"][:,:-1].reshape(-1).to(device)
        logits = output["logits"][:,:-1].reshape([labels.shape[0], -1])
            
        loss = F.cross_entropy(logits, labels,ignore_index=0)
        accuracy = torchmetrics.functional.accuracy(logits, labels,ignore_index=0)
        

        return loss, accuracy