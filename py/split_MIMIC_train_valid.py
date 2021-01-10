'''
script that loads the MIMIC.csv file and splits it into 2 subset 80/20
- training
- validation
'''
import pandas as pd
import numpy as np
import re

# replace with your own file path
MIMIC_PATH = "/home/alexis/data/MIMIC.csv"
TRAIN_PATH = "/home/alexis/data/MIMIC_train_samp100k.txt"
VALID_PATH = "/home/alexis/data/MIMIC_valid_samp100k.txt"
# train, validation ratio
training_ratio = 0.8
# limit to N samples for testing
N = 100000

REGEX_PATTERNS = {
    'linereturns': {'regex': "[\r\n]+", 'sub_by': ' '},
    'multiple_underscore': {'regex': "__+", 'sub_by': ' '},
    'multiple_spaces': {'regex': "\s\s+", 'sub_by': ' '},
    'html': {'regex': "<[^>]*>", 'sub_by': ' '},
    'urls': {'regex': "http\S+", 'sub_by': ' --url-- '},
    'urlswww': {'regex': "www\S+", 'sub_by': ' --url-- '},
    'ats': {'regex': "@\S+", 'sub_by': ' '},
    'latex': {'regex': "\$[^>]*\$", 'sub_by': ' '},
    'brackets': {'regex': "\[\S+\]", 'sub_by': ' '},
    'digits': {'regex': "\d+", 'sub_by': ' '},
    'xao': {'regex': "\xa0|►|–", 'sub_by': ' '},
    'punctuation': {'regex': re.compile(f"[{re.escape('▬∞!.[]?#$%&()+►’*+/•:;<=>@[]^_`{|}~”“→→_,')}]"), 'sub_by': ' '},
}

if __name__ == "__main__":
    df = pd.read_csv(MIMIC_PATH, error_bad_lines = False, low_memory=False)
    print(f"Loaded {df.shape[0]} rows from {MIMIC_PATH}")
    '''
    The MIMIC.csv file has 11 columns and 2083176 rows
    we're only concerned about the TEXT column
    '''

    '''
    for testing purposes, only keep N samples
    '''
    if N is not None:
        print(f"subsampling {N} notes")
        df = df.sample(n = N, random_state = 88)


    '''
    Pre processing.
    The text contains a lot of dashes and line returns that we remove
    Some further clean up could be in order
        [image002.jpg]
        [**2120-6-2**]
        ____
    '''
    print("Regex - clean up")
    df['TEXT'] = df.TEXT.apply(lambda  txt : re.sub(
                                        REGEX_PATTERNS['linereturns']['regex'],
                                        REGEX_PATTERNS['linereturns']['sub_by'],
                                        txt   ))

    df['TEXT'] = df.TEXT.apply(lambda  txt : re.sub(
                                        REGEX_PATTERNS['multiple_underscore']['regex'],
                                        REGEX_PATTERNS['multiple_underscore']['sub_by'],
                                        txt   ))

    df['TEXT'] = df.TEXT.apply(lambda  txt : re.sub(
                                        REGEX_PATTERNS['multiple_spaces']['regex'],
                                        REGEX_PATTERNS['multiple_spaces']['sub_by'],
                                        txt   ))

    '''
    The data has 15 different categories:
        Nursing/other        822497
        Radiology            522275
        Nursing              223556
        ECG                  209051
        Physician            141624
        Discharge summary     59652
        Echo                  45794
        Respiratory           31739
        Nutrition              9418
        General                8301
        Rehab Services         5431
        Social Work            2670
        Case Management         967
        Pharmacy                103
        Consult                  98

    Which could be used to subset the data (for instance: Nursing/other + Nursing )
    '''

    '''
    Random split
    '''
    print("Random train valid split")

    train_index = df.sample(frac = training_ratio, random_state = 88).index
    valid_index = np.setdiff1d(df.index, train_index, assume_unique=True)

    mimic_train = df.loc[train_index].TEXT.values
    mimic_valid = df.loc[valid_index].TEXT.values

    '''
    save to file
    '''
    print(f"saving {len(mimic_train)} training rows to {TRAIN_PATH}")
    mimic_train = [txt + "\n" for txt in mimic_train ]
    f = open(TRAIN_PATH, "w")
    f.writelines(mimic_train)
    f.close()

    print(f"saving {len(mimic_valid)} validation rows to {VALID_PATH}")
    mimic_valid = [txt + "\n" for txt in mimic_valid ]
    f = open(VALID_PATH, "w")
    f.writelines(mimic_valid)
    f.close()







# -----
