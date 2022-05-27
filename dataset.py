from torch.utils.data import Dataset

class SummaryDataset(Dataset):
    def __init__(self, dataframe, max_seq_len, tokenizer) -> None:
        self.dataframe = dataframe # dataframe = train
        self.bos_token = '<s>' #문장시작
        self.eos_token = '</s>' #문장 끝
        self.max_seq_len = max_seq_len #최대 시퀀스 길이
        
        self.tokenizer = tokenizer #토크나이저

    def __len__(self):
        return self.dataframe.shape[0] #판다스 데이터 길이

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)  # 해당 시퀀스의 ids들
        attention_mask = [1] * len(input_id) 
        if len(input_id) < self.max_seq_len:   
            while len(input_id) < self.max_seq_len:  # 최대 길이보다 작을 때 패딩해주는거
                input_id += [self.tokenizer.pad_token_id] 
                attention_mask += [0]
        else: # 최대 길이 보다 input_id가 길 경우 자르는거
            input_id = input_id[:self.max_seq_len - 1] + [   
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return input_id, attention_mask # 패딩되거나 잘라서 나온 값들 반환


    def __getitem__(self, index):
        target_row = self.dataframe.iloc[index]  # 몇번째 pd의 데이터를 볼 것인가
        context, summary = target_row['context'], target_row['summary'] # 목표는 컨텍스트랑 서머리
        context_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(context) + [self.eos_token] #컨텍스트 토큰 <s>\context</s>
        summary_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(summary) + [self.eos_token] #서머리 토큰 위와 동일
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            context_tokens, index) # 인코더 인풋 = 컨텍스트 토큰을 make_input_id_mask 나온 값 반환
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
            summary_tokens, index) # 디코더 인풋 = 서머리 토큰을 위의 함수에 넣고 나온 값 반환
        labels = self.tokenizer.convert_tokens_to_ids(
            summary_tokens[1:(self.max_seq_len + 1)]) #정답인 요약문을 ids와
        if len(labels) < self.max_seq_len: # 최대 길이보다 짧을 때
            while len(labels) < self.max_seq_len: 
                # for cross entropy loss masking
                  labels += [-100] # 

        return {'input_ids': np.array(encoder_input_id, dtype=np.int64),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id[:1024], dtype=np.int64),
                'decoder_attention_mask': np.array(decoder_attention_mask[:1024], dtype=np.float_),
                'labels': np.array(labels[:1024], dtype=np.int64)}