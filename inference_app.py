from pathlib import Path

from annotated_text import annotated_text
import streamlit as st
import torch
from model_interpret_app import MAX_SEQ_LENGTH

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



"""
def create_annotation(text, target):
    text_list = text.split(" ")"""

data = load_datamodule().test_df
sample = data.sample(1).iloc[0]
st.title('Direct Sentiment Model Interpreter')
st.header('Input context and target below')
#context = st.text_input("Context:")
#target_text = st.text_input("Target:")
context_list = sample.masked_clean_text.split('<target>')
target = sample.target.upper()
cleaned_text = " ".join([
    context_list[0], target, context_list[1]])
cleaned_text = cleaned_text.replace(
    'rt @user:', '').replace('rt @user : ', '').replace(
        '@user', '').replace(' httpurl', '').strip(" ")
annotated_text(context_list[0], (target, "TARGET"), context_list[1])