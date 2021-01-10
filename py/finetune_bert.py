'''
Fine tuning Bert or equivalent on MIMIC dataset

This runs the run_mlm.py script available from
https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py

'''


import os, sys, re, csv, json
import numpy as np
import pandas as pd

# uncomment to install transformers and datasets
# !pip install datasets
# !pip install transfomers

# The train and test files are samples of the original MIMIC dataset restricted to the text column
# - train: 80000 lines
# - valid: 20000 lines

# uncomment to download the datasets
# !wget https://dataskat.s3.eu-west-3.amazonaws.com/data/columbia/MIMIC.csv

# script run_mlm.py
# !wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/run_mlm.py

# set the parameters
# TODO change for json file
model_name      = 'bert-base-uncased'
train_filepath  = "/home/alexis/data/MIMIC_train_samp100k.txt"
valid_filepath  = "/home/alexis/data/MIMIC_valid_samp100k.txt"
max_steps       = 5000 # the bigger, the longer it tales to train the model
save_steps      = 2000 # each time, a snapshot of the model is saved. Warning this can take a lot of space
output_dir      = "/home/alexis/amcp/BERT_medical/models/test_02/" # where the model snapshots and the final model is saved

# execute in the
cmd = f'''
    python run_mlm.py \
        --model_name_or_path {model_name} \
        --max_seq_length 128 \
        --line_by_line \
        --train_file "{train_filepath}" \
        --validation_file "{valid_filepath}" \
        --do_train \
        --do_eval \
        --max_steps 5000 \
        --save_steps 2000 \
        --output_dir "results/test_01"
'''
print("execute the following command in the shell")
print(cmd)

if False:
    import transformers
    import datasets


    '''
    Loading the fine tuned model
    see Appendix A1 in https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    '''
    # Load a trained model and vocabulary that you have fine-tuned
    output_dir = "../models/"
    from transformers import BertForMaskedLM
    model = BertForMaskedLM.from_pretrained(output_dir)
    # or
    # from transformers import AutoModelForMaskedLM
    # model = AutoModelForMaskedLM.from_pretrained(output_dir)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    from transformers import BertConfig
    config = BertConfig.from_pretrained(output_dir)


    '''
    Access to word and sentence vectors
    https://towardsdatascience.com/beyond-classification-with-transformers-and-hugging-face-d38c75f574fb
    '''

    from transformers import BertForMaskedLM
    model = BertForMaskedLM.from_pretrained(output_dir, output_hidden_states=True)

    # put this in eval mode
    model.eval()

    input_ids, attention_masks, attention_masks_without_special_tok = preprocessing_for_bert(texts, tokenizer)

    #call the model on the sentences
    outputs = model(input_ids, attention_masks) #(tokenized_tensor, sent_tensor)
    hidden_states = outputs[2]




# -----
