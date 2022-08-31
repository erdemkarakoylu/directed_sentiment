from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from  transformers import AutoTokenizer
import pytorch_lightning as pl  
from lcf_utils import *


class LCFDataset(Dataset):
    def __init__(self, tokenizer, data:pd.DataFrame, max_seq_length:int) -> None:
        super().__init__() 
        self.data = data
        self.maxlen = max_seq_length
        self.tokenizer = tokenizer
        self.encoder = partial(
            tokenizer.encode_plus, return_token_type_ids=False,
            max_length=max_seq_length, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int):
        data_row = self.data.iloc[index]
        raw_text = data_row['text']
        target = data_row['target']
        label = data_row['label']
        tokenizer_is_roberta = 'Roberta' in type(self.tokenizer).__name__
        text_bert_indices = get_text_bert_indices(self.encoder, raw_text, target)
        bert_segments_ids = get_bert_segments_ids(
            self.encoder, raw_text, target, self.maxlen, tokenizer_type_is_Roberta=tokenizer_is_roberta)
        text_raw_bert_indices = get_text_raw_bert_indices(self.encoder, raw_text)
        target_bert_indices = get_target_bert_indices(self.encoder, target)
        dep_distance_to_target = get_synth_dep_dist_to_target(
            self.tokenizer, raw_text, target, self.maxlen)

        dataset_dict = dict(
            text=raw_text, target=target, text_bert_indices=text_bert_indices,
            bert_segments_ids=bert_segments_ids, text_raw_bert_indices=text_raw_bert_indices,
            target_bert_indices=target_bert_indices, dep_distance_to_target=dep_distance_to_target,
            polarity=label)
        return dataset_dict

class DataModule(pl.LightningDataModule):
    def __init__(
        self, model_name:str, batch_size:int, data_dir:str, train_val_split:float=0.3,
        num_workers:int=cpu_count()-1, max_seq_length:int=128
        ):
        """
        The datamodule expects a data_path argument to directory.
        This is expected to contain at minimum a df_train.json.
        If df_val.json is available it will be used as validation set otherwise 
        df_train.json will be split to create a validation set.
        It can optionally also contain df_test.json but that is not a requirement.
        
        """
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.train_val_split = train_val_split
        self.prepare_data()
        self.setup()
    
    def prepare_data(self):
        self.train_df = pd.read_json(self.data_dir / 'df_train.json')
        try:
            self.val_df = pd.read_json(self.data_dir / 'df_val.json')
        except:
            self.val_df = self.train_df.sample(frac=self.train_val_split)
            self.train_df.drop(self.val_df.index, inplace=True)
        try:
            self.test_df = pd.read_json(self.data_dir / 'df_test.json')
        except:
            self.test_df = None
      
    def setup(self, stage=None):
        self.train_dataset = LCFDataset(
            self.tokenizer, self.train_df, self.max_seq_length)
        self.val_dataset = LCFDataset(
            self.tokenizer, self.val_df, self.max_seq_length)
        if self.test_df is not None:
            self.test_dataset = LCFDataset(
                self.tokenizer, self.test_df, self.max_seq_length)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size * 2, 
            shuffle=False, num_workers=self.num_workers
            )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size * 2,
            shuffle=False, num_workers=self.num_workers
            )
      