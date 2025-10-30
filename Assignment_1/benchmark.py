import os
import re
import random
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
from tqdm import tqdm
from difflib import SequenceMatcher
import torch.nn.functional as F

# --- Configuration ---
MODEL_DIR = "./models/codet5-finetuned/best_model"
TOKENIZER_JSON_PATH = "./tokenizer_files/tokenizer.json"
INPUT_CSV = "./testsets/benchmark_if_only.csv"
OUTPUT_CSV = "./testsets/benchmark_results.csv"

MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 64
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IF_MASK_TOKEN = "<if_mask>"

# --- Helper functions ---
def mask_if_statement(code_snippet):
    lines = code_snippet.splitlines()
    if_pattern = r"^\s*(if\s*\(?.*?\)?\s*:)"
    matches = []
    for i, line in enumerate(lines):
        match = re.match(if_pattern, line)
        if match:
            matches.append((match.group(1), i))
    if not matches:
        return code_snippet, None, False
    selected_if, line_idx = random.choice(matches)
    lines[line_idx] = lines[line_idx].replace(selected_if.strip(), IF_MASK_TOKEN, 1)
    return "\n".join(lines), selected_if.strip(), True

def compute_similarity(pred, target):
    if pred is None or target is None:
        return 0.0
    return SequenceMatcher(None, pred, target).ratio() * 100

def compute_perplexity(input_text, target_text):
    input_enc = tokenizer(input_text, truncation=True, max_length=MAX_INPUT_LENGTH, return_tensors="pt").to(DEVICE)
    target_enc = tokenizer(target_text, truncation=True, max_length=MAX_OUTPUT_LENGTH, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_enc.input_ids,
                        attention_mask=input_enc.attention_mask,
                        labels=target_enc.input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss).item()
    return ppl

# --- Load model & tokenizer ---
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_JSON_PATH,
    bos_token="<s>", eos_token="</s>", pad_token="<pad>",
    unk_token="<unk>", mask_token="<mask>"
)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# --- Load benchmark CSV and mask 'if' statements ---
df = pd.read_csv(INPUT_CSV)
input_texts, target_texts, has_if_flags = [], [], []

for code in df['code']:
    masked_code, target_if, found = mask_if_statement(code)
    input_texts.append(masked_code)
    target_texts.append(target_if)
    has_if_flags.append(found)

df['input_text'] = input_texts
df['target_text'] = target_texts
df['has_if'] = has_if_flags

# --- Generate predictions and compute scores ---
preds, sim_scores, perplexities = [], [], []

for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Predicting"):
    batch_texts = df['input_text'][i:i+BATCH_SIZE].tolist()
    encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_INPUT_LENGTH, return_tensors="pt")
    input_ids = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=10,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )

    # Decode predictions and compute scores
    for j, seq in enumerate(outputs.sequences):
        pred_text = tokenizer.decode(seq, skip_special_tokens=True)
        preds.append(pred_text)

        # Similarity score
        target_text = df['target_text'][i+j]
        sim_scores.append(compute_similarity(pred_text, target_text))

        # Perplexity
        if df['has_if'][i+j]:
            ppl = compute_perplexity(df['input_text'][i+j], target_text)
        else:
            ppl = None
        perplexities.append(ppl)

# --- Save results ---
df['predicted_if'] = preds
df['prediction_score'] = sim_scores
df['perplexity'] = perplexities

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions, similarity scores, and perplexities saved to {OUTPUT_CSV}")
