import networkx as nx
import spacy
from spacy.cli import download
import torch
import numpy as np

try:
    nlp = spacy.load("en_core_web_sm")
except:
    download('en_core_web_sm')
    nlp = spacy.load("en_core_web_sm")

def calculate_dep_dist(sentence,aspect):
    terms = [a.lower() for a in aspect.split()]
    doc = nlp(sentence)
    # Load spacy's dependency tree into a networkx graph
    edges = []
    cnt = 0
    term_ids = [0] * len(terms)
    for token in doc:
        # Record the position of aspect terms
        if cnt < len(terms) and token.lower_ == terms[cnt]:
            term_ids[cnt] = token.i
            cnt += 1

        for child in token.children:
            edges.append(('{}_{}'.format(token.lower_,token.i),
                          '{}_{}'.format(child.lower_,child.i)))

    graph = nx.Graph(edges)
    dist = [0.0]*len(doc)
    text = [0]*len(doc)
    for i,word in enumerate(doc):
        source = '{}_{}'.format(word.lower_,word.i)
        sum = 0
        for term_id,term in zip(term_ids,terms):
            target = '{}_{}'.format(term, term_id)
            try:
                sum += nx.shortest_path_length(graph,source=source,target=target)
            except:
                sum += len(doc) # No connection between source and target
        dist[i] = sum/len(terms)
        text[i] = word.text
    return text, dist

def pad_and_truncate(sequence, maxlen, value=0, dtype=torch.int64):
    truncated = sequence[:maxlen]
    padded = torch.ones(maxlen, dtype=dtype) * value
    padded[:truncated.numel()] = truncated
    return padded

def get_text_bert_indices(encoder, text, target):
    return encoder(
        text, text_pair=target, add_special_tokens=True)['input_ids'].flatten()

def get_bert_segments_ids(encoder, text , target, max_seq_length, tokenizer_type_is_Roberta=False):
    raw_text_ids = encoder(text, add_special_tokens=False)['input_ids'].flatten()
    targ_ids = encoder(target, add_special_tokens=False)['input_ids'].flatten()
    targ_len = targ_ids.nonzero().numel()
    #if 'Roberta' in type(tokenizer).__name__:
    if tokenizer_type_is_Roberta:
        bert_segment_ids = torch.zeros(raw_text_ids.nonzero().numel() + 2 + targ_len + 1)
    else:
        bert_segment_ids = torch.tensor(
            [0] * (raw_text_ids.nonzero().numel()+2 ) + [1] * (targ_len + 1))
    bert_segment_ids = pad_and_truncate(bert_segment_ids, max_seq_length)
    return bert_segment_ids

def get_text_raw_bert_indices(encoder, text): 
    text_raw_bert_indices = encoder(text, add_special_tokens=False)['input_ids'].flatten()
    return text_raw_bert_indices

def get_target_bert_indices(encoder, target):
    target_bert_indices = encoder(target, add_special_tokens=True)['input_ids'].flatten()
    return target_bert_indices

def get_synth_dep_dist_to_target(tokenizer, text, target, max_seq_length):
    raw_tokens, dists = calculate_dep_dist(text, target)
    raw_tokens.insert(0, tokenizer.cls_token)
    raw_tokens.append(tokenizer.sep_token)
    dists.insert(0, 0)
    dists.append(0)
    new_dist = pad_and_truncate(torch.tensor(dists), maxlen=max_seq_length, value=max_seq_length)
    return new_dist

def get_encodings(encoder, text, target):
    text_bert_indices = get_text_bert_indices(encoder, text, target)
    bert_segments_ids = None
    text_raw_bert_indices = None
    target_bert_indices = None
    synth_dep_dist_to_target = None

    return dict(
        text_bert_indices=text_bert_indices, bert_segments_ids=bert_segments_ids,
        text_raw_bert_indices=text_raw_bert_indices, target_bert_indices=target_bert_indices,
        synth_dep_dist_to_target=synth_dep_dist_to_target)