from torch.utils.data import Dataset
import pandas as pd

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

    def __init__(
        self,
        max_seq_len,
        masking_rate: float = 0.3,
    ):
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
        self.mask_token_id = tokenizer.mask_token_id
        self.ids, self.dialogues = self.load_dataset(paths)

    def load_dataset(self, paths: List[str]) -> Tuple[List[str], List[List[str]]]:
        dataframe = pd.read_csv('./datas/2022-03-01_to_2022-03-03.csv')
    


    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        dialogue = self.dialogues[index]

        # Permutate
        random.shuffle(dialogue)

        # Tokenize
        decoder_input_ids = dialogue_input["input_ids"][0]
        decoder_attention_mask = dialogue_input["attention_mask"][0]
        encoder_input_ids = decoder_input_ids.clone()
        encoder_attention_mask = decoder_attention_mask.clone()

        # Masking
        sequence_length = encoder_attention_mask.sum()
        num_masking = int(sequence_length * self.masking_rate)
        indices = torch.randperm(sequence_length)[:num_masking]
        encoder_input_ids[indices] = self.mask_token_id

        return {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }