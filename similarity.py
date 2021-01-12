import transformers
import datasets
import numpy as np

from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import BertConfig

import torch

# or
# from transformers import AutoModelForMaskedLM

def get_word_index(tokenizer, text, word):
    '''
    Returns the index of the tokens corresponding to `word` within `text`.
    `word` can consist of multiple words, e.g., "cell biology".
    '''
    # Tokenize the 'word'--it may be broken into multiple tokens or subwords.
    word_tokens = tokenizer.tokenize(word)

    # Create a sequence of `[MASK]` tokens to put in place of `word`.
    masks_str = ' '.join(['[MASK]']*len(word_tokens))

    # Replace the word with mask tokens.
    text_masked = text.replace(word, masks_str)

    # `encode` performs multiple functions:
    #   1. Tokenizes the text
    #   2. Maps the tokens to their IDs
    #   3. Adds the special [CLS] and [SEP] tokens.
    input_ids = tokenizer.encode(text_masked)

    # find all index of the [MASK] token.
    mask_token_index = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]

    return mask_token_index


def get_embedding(model, tokenizer, text, word='', ):

    encoded_dict = tokenizer.encode_plus(
                    text,
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    return_tensors = 'pt',     # Return pytorch tensors.
            )

    input_ids = encoded_dict['input_ids']
    model.eval()
    bert_outputs = model(input_ids)
    with torch.no_grad():
        outputs = model(input_ids)
    print("len(outputs)",len(outputs))
    hidden_states = outputs[len(outputs) -1]

    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding = sentence_embedding.detach().numpy()

    if not word == '':
        word_idx = get_word_index(tokenizer, text, word)
        # Take the average of the embeddings for the tokens in `word`.
        word_embedding = torch.mean(token_vecs[word_idx], dim=0)
        word_embedding = word_embedding.detach().numpy()

        return (sentence_embedding, word_embedding)
    else:
        return sentence_embedding

from scipy.spatial.distance import cosine
def cosn(a,b):
    return 1 - cosine(a,b)


if __name__ == "__main__":
    '''
    1. Load a model, tokenizer and config and vocabulary

    see Appendix A1 in https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    '''
    # location of the model files:
    output_dir = "/Users/alexis/amcp/Columbia/synonyms/models/full/"

    ft_model       = BertForMaskedLM.from_pretrained(output_dir, output_hidden_states=True)
    ft_tokenizer   = BertTokenizer.from_pretrained(output_dir)
    # ft_config      = BertConfig.from_pretrained(output_dir)

    from transformers import BertTokenizer, BertModel

    b_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    b_model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states = True)
    # b_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    word = 'Broccoli'
    text = '''Broccoli is an edible green plant in the cabbage family
            whose large flowering head, stalk and small associated leaves
            are eaten as a vegetable'''

    word = 'migraine'
    text = '''A migraine is a headache'''


    sent_ft, word_ft = get_embedding(ft_model, ft_tokenizer, text, word)
    sent_b, word_b = get_embedding(b_model, b_tokenizer, text, word)



    text = "The man in prison watched the animal from his cell."
    (emb_sen, emb_cell) = get_embedding(ft_model, ft_tokenizer, text, word="cell")
    (emb_sen, emb_animal) = get_embedding(ft_model, ft_tokenizer, text, word="animal")
    (emb_sen, emb_prison) = get_embedding(ft_model, ft_tokenizer, text, word="prison")

    print(f"ft:  cell animal {cosine(emb_cell, emb_animal)}")
    print(f"ft:  cell prison {cosine(emb_cell, emb_prison)}")

    (emb_sen, emb_cell) = get_embedding(b_model, b_tokenizer, text, word="cell")
    (emb_sen, emb_animal) = get_embedding(b_model, b_tokenizer, text, word="animal")
    (emb_sen, emb_prison) = get_embedding(b_model, b_tokenizer, text, word="prison")

    print(f"bert:  cell animal {cosine(emb_cell, emb_animal)}")
    print(f"bert:  cell prison {cosine(emb_cell, emb_prison)}")




# ----
