from pathlib import Path
from random import randint

from annotated_text import annotated_text
import streamlit as st
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
#from pages.model_interpretation import MAX_SEQ_LENGTH

from packages.lcf_textore_datamodule import DataModule
from packages.lcf_pl_model import LCFS_BERT_PL

#TODO: create option for manual data input + data preprocessing.

CHKPT_PATH = Path.cwd() / 'experiments/experiment_logs/best_model_checkpoint.ckpt'
BERT_MODEL = 'bert-base-uncased'
DATA_PATH = Path.cwd()/'data/textore/ready/eval_samples_added_2_training'
TRAIN_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 48


def load_datamodule():
    dm = DataModule(
        model_name=BERT_MODEL, batch_size=TRAIN_BATCH_SIZE, num_workers=2,
        data_dir=DATA_PATH, max_seq_length=MAX_SEQ_LENGTH)
    return dm

def load_model():
    model = LCFS_BERT_PL(
    BERT_MODEL, 
    )
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    model.load_state_dict(torch.load(CHKPT_PATH, map_location=device))
    model.eval()
    return model

def get_sample_data(dm):
    num_samples = len(dm.test_dataset)
    sample_ix = randint(0, num_samples -1)
    dataset_dict = dm.test_dataset[sample_ix]
    df_row_dict = dm.test_df.iloc[sample_ix].to_dict()
    return dataset_dict, df_row_dict

def parse_text_for_annotation(data_dict):
    text_string = data_dict['orig_text']
    target_start, target_stop = data_dict['start'], data_dict['stop'] 
    context_beg = text_string[:target_start]
    target_string = text_string[target_start: target_stop]
    context_end = text_string[target_stop:]
    return dict(
        context_beg=context_beg, 
        target_string=target_string, 
        context_end=context_end)

datamodule = load_datamodule()
processed_sample_dict, raw_sample_dict = get_sample_data(datamodule)
st.title('Directed Sentiment Predictor')
st.subheader('Input context and target')
annotation_dict = parse_text_for_annotation(raw_sample_dict)
annotated_text(
    annotation_dict['context_beg'], 
    (annotation_dict['target_string'], "TARGET"),
    annotation_dict['context_end'])

model = load_model()
button = st.button("Predict sentiment toward target")
sentiment_code = {0: "Negative", 1: "Neutral", 2: "Positive"}

if button:
    dl = DataLoader([processed_sample_dict])
    trainer = pl.Trainer()
    sample_pred = trainer.predict(model, dataloaders=dl)[0]
    sample_pred_code = sample_pred.numpy()