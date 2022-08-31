import copy

import numpy as np
import torch
import torch.nn as nn
#from transformers.models import BertPooler, BertSelfAttention, BertConfig
from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention, BertConfig

class PointwiseCovNet(nn.Module):
    """ A 2-layer FCN module. """
    def __init__(
        self, dim_hid, dim_inner_hid=None, dim_out=None, dropout_rate=0.0):
        super().__init__()
        if dim_inner_hid is None:
            dim_inner_hid = dim_hid
        if dim_out is None:
            dim_out = dim_inner_hid
        self.w_1 = nn.Conv1d(dim_hid, dim_inner_hid, 1)
        self.w_2 = nn.Conv1d(dim_inner_hid, dim_out, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.w_1(x.transpose(1, 2)))
        out = self.w_2(out).transpose(2, 1)
        out = self.dropout(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, config, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = nn.Tanh()

    def forward(self, x, device):
        zero_tensor = torch.zeros(
            (x.size(0), 1, 1, self.max_seq_length), 
            dtype=torch.float32).to(device)
        SA_out, att = self.SA(x, zero_tensor)
        SA_out = self.tanh(SA_out)
        return SA_out, att
    

class LCFS_BERT(nn.Module):
    def __init__(self,model, opt):
        super(LCFS_BERT, self).__init__()
        self.hidden = model.config.hidden_size
        sa_config = BertConfig(hidden_size=self.hidden, output_attetion=True)
        self.opt = opt
        # --Global Context Model---    
        self.bert_global = model
        # --Local Context Model---
        self.bert_local = copy.deepcopy(model)
        self.bert_local_sa = SelfAttention(sa_config, opt)
        self.bert_local_pct = PointwiseCovNet(self.hidden)

        self.dropout = nn.Dropout(opt.dropout)
        self.bert_sa = SelfAttention(sa_config, opt)
        self.mean_pooling_double = PointwiseCovNet(self.hidden*2, self.hidden, self.hidden)
        self.bert_pooler = BertPooler(sa_config)
        self.dense = nn.Linear(self.hidden, opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, target_indices, distances_input=None):
        mask_len = self.opt.SRD
        masked_text_raw_indices = torch.ones(
            (text_local_indices.size(0), self.opt.max_seq_length, self.hidden),
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
                for j in range(targ_begin + targ_len + mask_len, self.opt.max_seq_length):
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
        mask_len = self.opt.SRD
        masked_text_raw_indices = torch.ones((
            text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
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
                    if srd > self.opt.SRD:
                        distances[i] = 1 - (srd - self.opt.SRD) / \
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
                        distances[i] = 1
                for i in range(len(distances_i)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs, output_attentions=False):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2] # raw text indices w/out target term
        target_indices = inputs[3]
        distances = inputs[4]

        global_out = self.bert_global(text_bert_indices, bert_segments_ids)
        bert_global_out = global_out[0]
        global_att = global_out[-1][-1]

        bert_local_out = self.bert_local(text_local_indices)[0]

        # Local Context Focus Computations
        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, target_indices, distances)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)
        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weights(
                text_local_indices, target_indices, distances)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)
        
        out_cat = torch.cat((bert_local_out, bert_global_out), dim=-1)
        mean_pool = self.mean_pooling_double(out_cat)
        self_attention_out, local_att = self.bert_sa(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)
        if output_attentions:
            return (dense_out, global_att, local_att)
        return dense_out
        

                        



