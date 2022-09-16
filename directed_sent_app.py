from pathlib import Path
from random import randint

from annotated_text import annotated_text
import streamlit as st
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from numpy import array2string

from packages.lcf_textore_datamodule import DataModule
from packages.lcf_pl_model import LCFS_BERT_PL

#TODO: create option for manual data input + data preprocessing.

CHKPT_PATH = Path.cwd() / 'best_model_checkpoint.ckpt'
BERT_MODEL = 'bert-base-uncased'
DATA_PATH = Path.cwd()/'data/textore/ready/eval_samples_added_2_training'
TRAIN_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 48
SRD = 9
LCF='cdw'

if not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda')

def load_datamodule():
    dm = DataModule(
        model_name=BERT_MODEL, batch_size=TRAIN_BATCH_SIZE, num_workers=2,
        data_dir=DATA_PATH, max_seq_length=MAX_SEQ_LENGTH)
    return dm

@st.cache(persist=True, allow_output_mutation=True)
def load_model():
    model = LCFS_BERT_PL(
        BERT_MODEL, max_seq_length=MAX_SEQ_LENGTH,
        synthactic_distance_dependency=SRD,
        local_context_focus=LCF
    )
    
    model.load_state_dict(torch.load(CHKPT_PATH))
    model.to(DEVICE)
    model.eval()
    return model

def get_sample_data(dm):
    num_samples = len(dm.test_dataset)
    sample_ix = randint(0, num_samples -1)
    ds_dict = dm.test_dataset[sample_ix]
    dataset_dict = {k: (
        v.clone().unsqueeze(0).to(DEVICE) if 'bert' in k else v)
        for k, v in ds_dict.items()
        }
    dataset_dict['dep_distance_to_target'] = ds_dict[
        'dep_distance_to_target'].clone().unsqueeze(0).to(DEVICE)
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


st.title('Directed Sentiment Predictor')
datamodule = load_datamodule()
model = load_model()
load_sample_button = st.button("Load Sample and Run Model")
# Initialize session state
if "load_state" not in st.session_state:
    st.session_state.load_state = False
st.markdown("")
sentiment_code = {
    0: {"label": "Negative", "color": "#ff4040"}, 
    1: {"label": "Neutral", "color": "#5a5a5a"}, 
    2: {"label": "Positive", "color": "#004d4d"}
    }

if load_sample_button or st.session_state.load_state:
    st.session_state.load_state = True
    trainer = pl.Trainer()
    processed_sample_dict, raw_sample_dict = get_sample_data(datamodule)
    st.subheader('Input context and target')
    annotation_dict = parse_text_for_annotation(raw_sample_dict)
    annotated_text(
        annotation_dict['context_beg'],  
        (annotation_dict['target_string'], "TARGET"),
        annotation_dict['context_end'], )
    st.markdown("")
    st.markdown("")
    #dl = DataLoader([processed_sample_dict])
   
    #sample_pred_logits = trainer.predict(
    #    model, dataloaders=dl)[0].detach().numpy()[0]
    with torch.no_grad():
        sample_pred_logits = model(processed_sample_dict).squeeze().numpy()
    sample_pred_code = sample_pred_logits.argmax()
    st.write(
        f"Logits: {array2string(sample_pred_logits, precision=2, floatmode='fixed')}"
        )
    #st.write(f"Prediction: {sentiment_code[sample_pred_code]}")
    annotated_text(
        "Prediction:  ",
        (sentiment_code[sample_pred_code]['label'], "", sentiment_code[sample_pred_code]["color"])
    )
