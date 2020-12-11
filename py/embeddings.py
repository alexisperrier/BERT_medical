'''
Embeddings

Access to word and sentence vectors
https://towardsdatascience.com/beyond-classification-with-transformers-and-hugging-face-d38c75f574fb
and https://colab.research.google.com/github/nidharap/Notebooks/blob/master/Word_Embeddings_BERT.ipynb#scrollTo=YgTAIshMwZS6

'''

import os, sys, re, csv, json
import numpy as np
import pandas as pd

import transformers
import datasets

import torch
from transformers import BertForMaskedLM
from transformers import BertTokenizer, BertModel  #RobertaModel, RobertaTokenizer
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn import manifold          #use this for MDS computation

# #visualization libs
# import plotly.express as px
# import plotly.graph_objects as go
import matplotlib.pyplot as plt

#Library to calculate Relaxed-Word Movers distance
# from wmd import WMD
# from wmd import libwmdrelax

def preprocessing_for_bert(data, tokenizer_obj):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    @return   attention_masks_without_special_tok (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model excluding the special tokens (CLS/SEP)
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer_obj.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            truncation=True,              #Truncate longer seq to max_len
            return_attention_mask=True      # Return attention mask
            )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    #lets create another mask that will be useful when we want to average all word vectors later
    #we would like to average across all word vectors in a sentence, but excluding the CLS and SEP token
    #create a copy
    attention_masks_without_special_tok = attention_masks.clone().detach()

    #set the CLS token index to 0 for all sentences
    attention_masks_without_special_tok[:,0] = 0

    #get sentence lengths and use that to set those indices to 0 for each length
    #essentially, the last index for each sentence, which is the SEP token
    sent_len = attention_masks_without_special_tok.sum(1).tolist()

    #column indices to set to zero
    col_idx = torch.LongTensor(sent_len)
    #row indices for all rows
    row_idx = torch.arange(attention_masks.size(0)).long()

    #set the SEP indices for each sentence token to zero
    attention_masks_without_special_tok[row_idx, col_idx] = 0

    return input_ids, attention_masks, attention_masks_without_special_tok


if __name__ == "__main__":
    output_dir = "../models/"
    MAX_LEN = 15
    model = BertForMaskedLM.from_pretrained(output_dir, output_hidden_states=True)
    # or
    # from transformers import AutoModelForMaskedLM
    # model = AutoModelForMaskedLM.from_pretrained(output_dir)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    from transformers import BertConfig
    config = BertConfig.from_pretrained(output_dir)

    # put this in eval mode
    model.eval()

    '''
    Apply on sentences

    '''
    texts = ['final report indication 74-year-old man with bradycardia and heart block.','this presentation is suggestive of recurrent aspiration pneumonia or hap']
    input_ids, attention_masks, attention_masks_without_special_tok = preprocessing_for_bert(texts, tokenizer)

    #call the model on the sentences
    outputs = model(input_ids, attention_masks) #(tokenized_tensor, sent_tensor)
    hidden_states = outputs[2]

    '''
    Get vectors
    see also https://colab.research.google.com/drive/19loLGUDjxGKy4ulZJ1m3hALq2ozNyEGe#scrollTo=giJMGiGZhLLa
    '''
    outputs = model(input_ids, attention_masks) #(tokenized_tensor, sent_tensor)
    # attention: this will be different for Bert for classification models or other Berts (Roberta)
    hidden_states = outputs[1]
    # `hidden_states` has shape [13 x 1 x <sentence length> x 768]

    print("Total hidden layers:", len(hidden_states))
    # 13
    print("First layer : hidden_states[0].shape ", hidden_states[0].shape)
    # torch.Size([2, 15, 768])

    # Select the embeddings from the second to last layer.
    # `token_vecs` is a tensor with shape [<sent length> x 768]

    token_vecs = hidden_states[-2][0]
    # 15 x 769

    sentence_embedding = torch.mean(token_vecs, dim=0)



# --------
