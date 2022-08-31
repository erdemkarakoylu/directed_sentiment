from pathlib import Path
import psutil

from annotated_text import annotated_text
import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from lcf_textore_datamodule import DataModule
from lcf_pl_model import LCFS_BERT_PL

from loguru import logger

CKPT_PATH = ckpt_path = Path('tb_lcf_mixed_finetune_logs/lcf_mixed/version_0/checkpoints/epoch=39-step=6240.ckpt')
SPACY_MODEL = 'en_core_web_sm'
BERT_MODEL = 'bert-base-uncased'
TRAIN_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 48
DATA_PATH = Path.cwd()/'data/textore/ready/eval_samples_added_2_training'


def load_datamodule():
    dm = DataModule(
        model_name=BERT_MODEL, batch_size=TRAIN_BATCH_SIZE, num_workers=2,
        data_dir=DATA_PATH, max_seq_length=MAX_SEQ_LENGTH)
    return dm

def print_memory_usage():
    logger.info(f"RAM memory % used:{psutil.virtual_memory()[2]}")

def load_model():
    return LCFS_BERT_PL.load_from_checkpoint(CKPT_PATH)

def create_annotation(text, target):
    text_list = text.split(" ")

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
left_column, right_column = st.columns(2)
with left_column:
    st.header("Syntactic Dependency Graph")
    nlp = spacy.load(SPACY_MODEL)
    doc = nlp(cleaned_text)
    dep_svg = displacy.render(
        doc, style='dep', jupyter=False,
        options={
            'color': "#ffffff", 'bg': '#000000', 'distance':100,
            'collapse_phrases': True, 'compact': False} )
    st.image(dep_svg, use_column_width='always', )