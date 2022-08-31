from copy import deepcopy
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import AutoModel
from pytorch_transformers.modeling_bert import BertConfig, BertPooler
from torchmetrics import Accuracy, F1Score

from lcfs_model import PointwiseCovNet, SelfAttention

"""
opts needed :
opt.SRD
opt.max_seq_length
opt.dropout
opt.polarities_dim should_be_three
bert_dim
"""
class LCFS_BERT_PL(pl.LightningModule):
    def __init__(
        self, model_name: str, max_seq_length: int=64, polarities_dim: int=3, 
        synthactic_distance_dependency: int=3, dropout_rate: float=0.1, 
        lr: float=2e-5, weight_decay: float=0.01, local_context_focus: str='cdw'
        ):
        super().__init__()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.polarities_dim = polarities_dim
        self.SRD = synthactic_distance_dependency
        self.dropout = dropout_rate
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.local_context_focus = local_context_focus

        self.model = AutoModel.from_pretrained(self.model_name, output_attentions=True)
        self.hidden = self.model.config.hidden_size
        self.acc = Accuracy()
        self.f1 = F1Score()
        self.criterion = nn.CrossEntropyLoss()
        sa_config = BertConfig(hidden_size=self.hidden, output_attentions=True)
        # --Global Context Model---    
        self.bert_global =self.model
        # --Local Context Model---
        self.bert_local = deepcopy(self.model)

        self.dropout = nn.Dropout(self.dropout)
        self.bert_sa = SelfAttention(sa_config, self.max_seq_length)
        self.mean_pooling_double = PointwiseCovNet(
            self.hidden*2, self.hidden, self.hidden)
        self.bert_pooler = BertPooler(sa_config)
        self.dense = nn.Linear(self.hidden, self.polarities_dim)
        self.save_hyperparameters()


    def feature_dynamic_mask(
        self, text_local_indices, target_indices, distances_input=None):
        """The 'CDM' branch of the LCF model."""
        mask_len = self.SRD
        masked_text_raw_indices = torch.ones(
            (text_local_indices.size(0), self.max_seq_length, self.hidden),
            dtype=torch.float32)
        for text_i, targ_i in zip(
            range(len(text_local_indices)), range(len(target_indices))):
            if distances_input is None:
                # calculate target length
                targ_len = torch.count_nonzero(target_indices[targ_i]) 
                try:
                    targ_begin = torch.argwhere(
                        text_local_indices[text_i] == target_indices[targ_i][0])[0][0]
                except:
                    continue
                if targ_begin >= mask_len:
                    mask_begin = targ_begin - targ_len
                else:
                    mask_begin = 0
                # Masking to the left
                for i in range(mask_begin): 
                    masked_text_raw_indices[text_i][i] = torch.zeros(
                        (self.hidden), dtype=torch.float)
                # Masking to the right
                for j in range(targ_begin + targ_len + mask_len, self.max_seq_length):
                    masked_text_raw_indices[text_i][j] = torch.zeros(
                        (self.hidden), dtype=torch.float
                    )
            else:
                distances_i = distances_input[text_i]
                for i, dist in enumerate(distances_i):
                    if dist > mask_len:
                        masked_text_raw_indices[text_i][i] = torch.zeros(
                            (self.hidden), dtype=torch.float
                        )
        return masked_text_raw_indices.to(self.device)

    def feature_dynamic_weights(
        self, text_local_indices, target_indices, distances_input=None):
        """The CDW branch of the LCF model."""
        mask_len = self.SRD
        text_local_indices = text_local_indices.cpu()
        target_indices = target_indices.cpu()
        if distances_input is not None:
            distances_input = distances_input.cpu()
        masked_text_raw_indices = torch.ones((
            text_local_indices.size(0), self.max_seq_length, self.hidden),
            dtype=torch.float32)
        for text_i, targ_i in zip(
            range(len(text_local_indices)), range(len(target_indices))):
            if distances_input is None:
                targ_len = torch.count_nonzero(target_indices[targ_i]-2)
                try:
                    targ_begin = torch.argwhere(
                        [text_local_indices[text_i]] == target_indices[targ_i][2])[0][0]
                    targ_avg_index = (targ_begin*2 + targ_len) / 2 # central position
                except:
                    continue
                distances = torch.zeros(
                    torch.count_nonzero(text_local_indices[text_i]),
                    dtype=torch.float32)
                for i in range(1, torch.count_nonzero(text_local_indices[text_i])-1):
                    srd = abs(i - targ_avg_index) + targ_len / 2
                    if srd > self.SRD:
                        distances[i] = 1 - (srd - self.SRD) / \
                            torch.count_nonzero(text_local_indices[text_i])
                    else:
                        distances[i] = 1
                for i in range(len(distances)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
            else:
                distances_i = distances_input[text_i] # distances of batch i
                for i, dist in enumerate(distances_i):
                    if dist > mask_len:
                        distances_i[i] = 1 - (dist - mask_len) / torch.count_nonzero(
                            text_local_indices[text_i])
                    else:
                        distances_i[i] = 1
                for i in range(len(distances_i)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances_i[i]
        return masked_text_raw_indices.to(self.device)

    def forward(self, inputs, output_attentions=False):
        text_bert_indices = inputs['text_bert_indices']
        bert_segments_ids = inputs['bert_segments_ids']
        text_local_indices = inputs['text_raw_bert_indices'] # raw text indices w/out target term
        target_indices = inputs['target_bert_indices']
        distances = inputs['dep_distance_to_target']

        global_out = self.bert_global(text_bert_indices, bert_segments_ids)
        bert_global_out = global_out[0] # bs x bert_dim
        global_att = global_out[-1][-1] # dim = bert_dim
        # batch_size x max_seq_length x bert_dim
        bert_local_out = self.bert_local(text_local_indices)[0] 

        # Local Context Focus Computations
        if self.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, target_indices, distances)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)
        elif self.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weights(
                text_local_indices, target_indices, distances)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)
        
        out_cat = torch.cat((bert_local_out, bert_global_out), dim=-1)
        mean_pool = self.mean_pooling_double(out_cat)
        self_attention_out, local_att = self.bert_sa(mean_pool, self.device)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)
        if output_attentions:
            return (dense_out, global_att, local_att)
        return dense_out
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=3, threshold=0.01, 
            cooldown=0, min_lr=1e-8 
        )
        #return [optimizer], [lr_scheduler]
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor='val_loss')
    
    def run_step(self, batch, stage):
        y_true = batch['polarity']
        y_hat_logits = self(batch)
        loss = self.criterion(y_hat_logits, y_true)
        self.log(f'{stage}_loss', loss, on_epoch=True)
        if not stage == 'training':
            self.acc(y_hat_logits, y_true)
            self.f1(y_hat_logits, y_true)
            self.log(f'{stage}_acc', self.acc, on_epoch=True)
            self.log(f'{stage}_f1', self.f1, on_epoch=True)

    def training_step(self, batch, batch_idx):
        y_true = batch['polarity']
        y_hat = self(batch)
        loss = self.criterion(y_hat, y_true)
        self.log('loss', loss, on_step=False, on_epoch=True)
        return loss
    
    """ def on_validattion_start(self):
        self.logger.log_hyperparams(self.hparams, {'val_loss':0})"""
    

    def validation_step(self, batch, batch_idx):
        y_true = batch['polarity']
        y_hat = self(batch)
        val_loss = self.criterion(y_hat, y_true)
        self.acc(y_hat, y_true)
        self.f1(y_hat, y_true)
        self.log('val_loss', val_loss, on_epoch=True)
        self.log('val_acc', self.acc, on_epoch=True)
        self.log('val_f1', self.f1, on_epoch=True)
        """self.logger.log_hyperparams(self.hparams, {'val_loss': val_loss})"""
            
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """loads the best checkpoint automatically by default if checkpointing is enabled."""
        pred = self(batch)
        return pred
    
    def test_step(self, batch, batch_idx):
        y_true = batch['polarity']
        y_hat = self(batch)
        test_loss = self.criterion(y_hat, y_true)
        self.acc(y_hat, y_true)
        self.f1(y_hat, y_true)
        self.log('val_loss', test_loss, on_epoch=True)
        self.log('val_acc', self.acc, on_epoch=True)
        self.log('val_f1', self.f1, on_epoch=True)
