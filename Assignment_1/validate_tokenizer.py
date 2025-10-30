#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Save this file as: validate_tokenizer.py

import os
import logging
from transformers import PreTrainedTokenizerFast

TOKENIZER_JSON_PATH = os.path.join("tokenizer_files", "tokenizer.json")
IF_MASK_TOKEN = "<if_mask>"

def validate():
    print("--- Running Tokenizer Validation ---")
    if not os.path.exists(TOKENIZER_JSON_PATH):
        print(f"❌ FAILURE: Tokenizer file '{TOKENIZER_JSON_PATH}' not found.")
        return

    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_JSON_PATH)
        print("✅ SUCCESS: Tokenizer loaded successfully.")
        
        test_code = f"def check_value(x):\n    if x > 10:\n        return True\n    {IF_MASK_TOKEN}"
        print(f"\n--- Testing with sample code ---\n{test_code}\n")
        
        tokens = tokenizer.tokenize(test_code)
        print(f"Tokens: {tokens}")
        
        # Check for keywords
        def_ok = 'def' in tokens or 'Ġdef' in tokens
        if_ok = 'if' in tokens or 'Ġif' in tokens

        if def_ok and if_ok:
            print("✅ SUCCESS: Common keywords ('def', 'if') are recognized correctly.")
        else:
            print("❌ WARNING: Common keywords are being split unexpectedly.")

        if IF_MASK_TOKEN in tokens:
            print(f"✅ SUCCESS: Special token '{IF_MASK_TOKEN}' is correctly recognized.")
        else:
            print(f"❌ FAILURE: Special token '{IF_MASK_TOKEN}' is NOT recognized.")
        
        # --- CORRECTED RECONSTRUCTION CHECK ---
        # 1. Encode WITHOUT adding special start/end tokens (bos/eos)
        encoded_ids = tokenizer.encode(test_code, add_special_tokens=False)
        
        # 2. Decode WITHOUT skipping any special tokens (like <if_mask>)
        decoded_text = tokenizer.decode(encoded_ids, skip_special_tokens=False)
        # --- END CORRECTION ---

        print("\n--- Reconstruction Test ---")
        print(f"Original Text:  {test_code}")
        print(f"Decoded Text:   {decoded_text}")

        if test_code == decoded_text:
            print("✅ SUCCESS: Code was reconstructed perfectly.")
        else:
            print("❌ FAILURE: Code reconstruction has differences.")
            
    except Exception as e:
        print(f"❌ FAILURE: An error occurred during validation: {e}")


# In[6]:


if __name__ == "__main__":
    validate()


# In[ ]:




