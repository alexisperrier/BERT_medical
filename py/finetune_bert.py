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
train_filepath  = "/home/alexis/data/MIMIC_train.txt"
valid_filepath  = "/home/alexis/data/MIMIC_valid.txt"
max_steps       = 5000 # the bigger, the longer it tales to train the model
save_steps      = 2000 # each time, a snapshot of the model is saved. Warning this can take a lot of space
output_dir      = "/home/alexis/amcp/BERT_medical/models/full/" # where the model snapshots and the final model is saved

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
        --output_dir "{output_dir}"
'''
print("execute the following command in the shell")
print(cmd)


# -----
