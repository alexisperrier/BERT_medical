# Fine tuning Bert on a Language Model task

There are 2 ways to train or fine tune BERT or GPT like models
- on a supervised downstream task
- in an unsupervised way on a corpus

The supervised training / fine-tuning requires a ground truth dataset.
We're going to work on the unsupervised approach

# Model fine tuning

We can leverage the [scripts](https://github.com/huggingface/transformers/tree/master/examples/language-modeling) given by huggingface:

- for Masked Language Models :
    - [run_mlm.py](https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/run_mlm.py)
    - BERT, RoBerta, DistilBert and [others](https://huggingface.co/models?filter=masked-lm)
- for Causale Language Models
    - [run_clm.py](https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/run_clm.py)
    - GPT, GPT-2
- for Permuted Language Models
    - [run_plm.py](https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/run_plm.py)
    - XL-NET

# Scripts
Most of the code in the above mentionned scripts are devoted to
- passing arguments. This is done through 3 classes:
    - **ModelArguments**: defined in the script. Arguments pertaining to which model, config and tokenizer we are going to fine-tune
    - **DataTrainingArgument**: defined in the script. Arguments pertaining to what data we are going to input our model for training and eval
    - [**TrainingArguments**](https://github.com/huggingface/transformers/blob/c95de29e31c13b7836fb55fdea57c761cc120650/src/transformers/training_args.py#L49) Imported from the huggingface lib. Arguments pertaining to the actual training / finetuning of the model


- allowing for both pytorch and tensorflow versions

The core of the script is organized along:

1. **loading the data** through the dataset module with ```load_dataset(data_args.dataset_name, data_args.dataset_config_name)```
2. Loading the appropriate **config**, **tokenizer** and **model**
    - ```config = AutoConfig.from_pretrained```
    - ```tokenizer = AutoTokenizer.from_pretrained```
    - ```model = AutoModelForMaskedLM.from_config(config)```
3. **tokenizing** the data

This returns 3 elements:
    - list of tokens
    - index of tokens in vocab
    - sequence of token mask [1,1,1,1,1,0,0,0,0]
4. The datacollator handles the **random masking** of tokens ```data_collator = DataCollatorForLanguageModeling```
5. and finally the **training / finetuning**
    - the trainer is instanciated ```trainer = Trainer()```
    - the training takes place ```trainer.train```
6. the model is saved


# fine tuning

To fine tune on a our own data specify the path to the trainign file and the validation file:


    python run_mlm.py \
        --model_name_or_path bert-base-uncased \
        --max_seq_length 128 \
        --line_by_line \
        --train_file "path_to_train_file" \
        --validation_file "path_to_validation_file" \
        --do_train \
        --do_eval \
        --max_steps 5000 \
        --save_steps 1000 \
        --output_dir "results/"


# parameters

- distilbert is smaller than BERT
- each save_steps the model is saved in a directory. This can quickly eat up all the space on the VM.
