# Save this file as: validate_datasets.py

import pandas as pd
import os
import logging
import json
from transformers import PreTrainedTokenizerFast

# --- Configuration (Must match generate_datasets.py) ---
OUTPUT_DATA_DIR = "collected_data"
PRETRAIN_CSV = os.path.join(OUTPUT_DATA_DIR, "pretrain_dataset.csv")
FINETUNE_TRAIN_CSV = os.path.join(OUTPUT_DATA_DIR, "finetune_train.csv")
# ... (rest of config variables) ...
TOKENIZER_JSON_PATH = os.path.join("tokenizer_files", "tokenizer.json")
MAX_SEQUENCE_LENGTH = 512
IF_MASK_TOKEN = "<if_mask>"
MASK_TOKEN = "<mask>"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tokenizer_for_validation():
    # ... (same as before) ...
    if not os.path.exists(TOKENIZER_JSON_PATH): return None
    return PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_JSON_PATH,
                                    bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                    unk_token="<unk>", mask_token="<mask>",
                                    model_max_length=MAX_SEQUENCE_LENGTH)

def validate_pretrain_csv_full(tokenizer, max_errors_display=5):
    print("\n" + "="*50)
    print("      Validating Pre-training Dataset (MLM)")
    print("="*50)

    if not os.path.exists(PRETRAIN_CSV):
        print(f"❌ FAILURE: Pre-training CSV '{PRETRAIN_CSV}' not found.")
        return False

    total_rows = 0
    total_errors = 0
    error_examples = []

    try:
        df = pd.read_csv(PRETRAIN_CSV, dtype={"input_ids": str, "labels": str, "original_code_snippet": str})
        print(f"✅ SUCCESS: Pre-training CSV loaded. Found {len(df)} instances.")

        required_cols = ['input_ids', 'labels', 'original_code_snippet']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ FAILURE: Missing required columns.")
            return False
        print(f"✅ SUCCESS: All required columns found.")

        for idx, row in df.iterrows():
            total_rows += 1
            row_errors = []

            try:
                input_ids = json.loads(row['input_ids'])
                labels = json.loads(row['labels'])
            except Exception as e:
                row_errors.append(f"JSON decode error: {e}")
                total_errors += 1; error_examples.append((total_rows, row_errors)); continue

            if len(input_ids) != MAX_SEQUENCE_LENGTH or len(labels) != MAX_SEQUENCE_LENGTH:
                row_errors.append(f"Sequence length mismatch")

            # --- CORRECTED VALIDATION LOGIC ---
            # A valid masked sequence is one where at least one label is NOT -100.
            # It does NOT have to contain a <mask> token.
            num_masked_positions = sum(1 for label_id in labels if label_id != -100)
            
            if num_masked_positions == 0:
                row_errors.append(f"No tokens were selected for masking (all labels are -100). This is a potential bug.")
            # --- END CORRECTION ---

            if row_errors:
                total_errors += 1
                if len(error_examples) < max_errors_display:
                    error_examples.append((total_rows, row_errors))

        print(f"✅ Total rows scanned: {total_rows}")
        if total_errors == 0:
            print("✅ All rows passed pre-training validation!")
        else:
            print(f"❌ Total rows with errors: {total_errors}")
            for row_num, errs in error_examples:
                print(f"Row {row_num}: {errs}")

        return total_errors == 0

    except Exception as e:
        print(f"❌ FAILURE: Error while validating pre-training CSV: {e}")
        return False
        
# ... (rest of the validation script for fine-tuning datasets remains the same) ...

if __name__ == "__main__":
    tokenizer = load_tokenizer_for_validation()
    if tokenizer:
        validate_pretrain_csv_full(tokenizer)
        # ... (rest of the main block) ...