#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import os
import logging
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# --- Configuration ---
OUTPUT_DATA_DIR = "collected_data"
INPUT_CSV = os.path.join(OUTPUT_DATA_DIR, "python_functions.csv")
TOKENIZER_TRAIN_CORPUS_FILE = os.path.join(OUTPUT_DATA_DIR, "tokenizer_train_corpus.txt")

TOKENIZER_DIR = "tokenizer_files"
TOKENIZER_JSON_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.json") # We will now save to a single .json file

# --- Best Settings for Tokenizer Training ---
VOCAB_SIZE = 40000
MIN_FREQUENCY = 3
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<if_mask>"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_tokenizer_corpus():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input CSV file not found: {INPUT_CSV}. Please run collect_data.py first.")
        return False
    logging.info(f"Loading '{INPUT_CSV}' to prepare tokenizer training corpus...")
    df = pd.read_csv(INPUT_CSV)
    with open(TOKENIZER_TRAIN_CORPUS_FILE, 'w', encoding='utf-8') as f:
        for code in df['original_code']:
            f.write(str(code) + "\n")
    logging.info(f"Corpus prepared at '{TOKENIZER_TRAIN_CORPUS_FILE}'.")
    return True

def train_custom_tokenizer():
    """
    Trains a Byte-Level BPE tokenizer and saves it to a single tokenizer.json file.
    """
    if not os.path.exists(TOKENIZER_TRAIN_CORPUS_FILE):
        logging.error(f"Corpus file not found: {TOKENIZER_TRAIN_CORPUS_FILE}.")
        return False

    # 1. Initialize a new Tokenizer with a BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2. Configure Byte-Level behavior
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 3. Initialize a trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS
    )

    # 4. Train the tokenizer
    logging.info("Starting tokenizer training...")
    tokenizer.train([TOKENIZER_TRAIN_CORPUS_FILE], trainer)
    logging.info("Tokenizer training complete.")

    # 5. Save the tokenizer to a single, robust tokenizer.json file
    if not os.path.exists(TOKENIZER_DIR):
        os.makedirs(TOKENIZER_DIR)
    tokenizer.save(TOKENIZER_JSON_PATH)
    logging.info(f"Tokenizer saved successfully to '{TOKENIZER_JSON_PATH}'.")


# In[6]:


if __name__ == "__main__":
    if prepare_tokenizer_corpus():
        train_custom_tokenizer()


# In[ ]:




