# Save this file as: generate_datasets.py

import pandas as pd
import numpy as np
import os
import re
import logging
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
import torch
from sklearn.model_selection import train_test_split
import time
import csv
import random

# --- Configuration ---
OUTPUT_DATA_DIR = "collected_data"
INPUT_CSV = os.path.join(OUTPUT_DATA_DIR, "python_functions.csv")
PRETRAIN_CSV = os.path.join(OUTPUT_DATA_DIR, "pretrain_dataset.csv")
FINETUNE_TRAIN_CSV = os.path.join(OUTPUT_DATA_DIR, "finetune_train.csv")
FINETUNE_VAL_CSV = os.path.join(OUTPUT_DATA_DIR, "finetune_val.csv")
FINETUNE_TEST_CSV = os.path.join(OUTPUT_DATA_DIR, "finetune_test.csv")

MAX_PRETRAIN = 150000
MAX_FINETUNE = 50000
MAX_SEQUENCE_LENGTH = 512
MLM_PROB = 0.15
IF_MASK_TOKEN = "<if_mask>"

TOKENIZER_JSON_PATH = os.path.join("tokenizer_files", "tokenizer.json")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_custom_tokenizer():
    if not os.path.exists(TOKENIZER_JSON_PATH):
        logging.error(f"Tokenizer file not found: '{TOKENIZER_JSON_PATH}'. Run train_tokenizer.py first.")
        return None
    try:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=TOKENIZER_JSON_PATH,
            bos_token="<s>", eos_token="</s>", pad_token="<pad>",
            unk_token="<unk>", mask_token="<mask>",
            model_max_length=MAX_SEQUENCE_LENGTH
        )
        logging.info("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        return None

def generate_pretraining_data(tokenizer, df):
    start_time = time.time()
    data_buffer = []
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB)
    
    total_to_process = min(len(df), MAX_PRETRAIN)
    logging.info(f"--- Starting Pre-training Data Generation (Target: {total_to_process}) ---")
    
    output_columns = ['input_ids', 'labels', 'original_code_snippet']
    if os.path.exists(PRETRAIN_CSV): os.remove(PRETRAIN_CSV)

    for i, row in enumerate(df.head(total_to_process).iterrows()):
        code = str(row[1]['original_code'])
        ids = tokenizer.encode(code, truncation=True, max_length=MAX_SEQUENCE_LENGTH, padding="max_length")
        
        # --- CRITICAL FIX FOR KeyError: 0 ---
        # The collator expects a LIST of dictionaries.
        example = {"input_ids": torch.tensor(ids, dtype=torch.long)}
        masked_output = collator([example])
        masked_ids = masked_output["input_ids"]
        labels = masked_output["labels"]
        # --- END FIX ---
        
        data_buffer.append({'input_ids': masked_ids.squeeze().tolist(), 'labels': labels.squeeze().tolist(), 'original_code_snippet': code})

        if (i + 1) % 10000 == 0 or (i + 1) == total_to_process:
            header = not os.path.exists(PRETRAIN_CSV)
            pd.DataFrame(data_buffer, columns=output_columns).to_csv(PRETRAIN_CSV, mode='a', header=header, index=False, quoting=csv.QUOTE_ALL)
            data_buffer = []
            logging.info(f"Processed and saved chunk {i+1}/{total_to_process}...")

    logging.info(f"Pre-training data generation complete. Total time: {time.time() - start_time:.0f}s")
    return True

def extract_and_mask_if_statement(code_snippet):
    lines = str(code_snippet).splitlines()
    if_pattern = r"^\s*(if\s*\(?.*?\)?\s*:)"
    
    matches = []
    for i, line in enumerate(lines):
        match_obj = re.match(if_pattern, line)
        if match_obj: matches.append((match_obj.group(1), i))
            
    if not matches: return None, None, False
        
    selected_if_text, line_idx = random.choice(matches)
    lines[line_idx] = lines[line_idx].replace(selected_if_text.strip(), IF_MASK_TOKEN, 1)
    
    return "\n".join(lines), selected_if_text.strip(), True

def generate_finetuning_data(tokenizer, df):
    start_time = time.time()
    data = []
    skipped_no_if = 0
    
    logging.info(f"--- Starting Fine-tuning Data Generation (Target: {MAX_FINETUNE}) ---")

    for code in df['original_code']:
        if len(data) >= MAX_FINETUNE: break
        
        masked_code, if_text, found = extract_and_mask_if_statement(code)
        if found:
            encoded = tokenizer.encode(masked_code, truncation=True, padding="max_length", max_length=MAX_SEQUENCE_LENGTH)
            data.append({'input_text': masked_code, 'target_text': if_text})
        else:
            skipped_no_if += 1

    if not data:
        logging.warning("No functions with 'if' statements found for fine-tuning.")
        return False
        
    df_ft = pd.DataFrame(data)
    train_df, temp_df = train_test_split(df_ft, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df.to_csv(FINETUNE_TRAIN_CSV, index=False)
    val_df.to_csv(FINETUNE_VAL_CSV, index=False)
    test_df.to_csv(FINETUNE_TEST_CSV, index=False)
    
    logging.info(f"Fine-tuning datasets saved. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logging.info(f"Total time for fine-tuning data: {time.time() - start_time:.0f}s")
    return True

if __name__ == "__main__":
    tokenizer = load_custom_tokenizer()
    if tokenizer and os.path.exists(INPUT_CSV):
        df = pd.read_csv(INPUT_CSV)
        if generate_pretraining_data(tokenizer, df):
            generate_finetuning_data(tokenizer, df)