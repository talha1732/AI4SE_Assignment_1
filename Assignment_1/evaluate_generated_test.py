import os
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
from tqdm import tqdm
from difflib import SequenceMatcher
import torch.nn.functional as F

# --- Configuration ---
MODEL_DIR = "./models/codet5-finetuned/best_model"
TOKENIZER_JSON_PATH = "./tokenizer_files/tokenizer.json"
INPUT_CSV = "./collected_data/finetune_test.csv"
OUTPUT_CSV = "./testsets/finetune_test_results.csv"

MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 64
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper functions ---
def compute_similarity(pred, target):
    if pred is None or target is None:
        return 0.0
    return SequenceMatcher(None, pred.strip(), target.strip()).ratio() * 100

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

def is_prediction_correct(pred, target):
    """Returns True if the prediction exactly matches target ignoring whitespace."""
    if pred is None or target is None:
        return False
    return pred.strip() == target.strip()

# --- Load model & tokenizer ---
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_JSON_PATH,
    bos_token="<s>", eos_token="</s>", pad_token="<pad>",
    unk_token="<unk>", mask_token="<mask>"
)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# --- Load test CSV ---
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} test examples from {INPUT_CSV}")

# --- Generate predictions and compute scores ---
preds, sim_scores, perplexities, correctness_flags = [], [], [], []

for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Evaluating"):
    batch_inputs = df['input_text'][i:i+BATCH_SIZE].tolist()
    encodings = tokenizer(batch_inputs, padding=True, truncation=True, max_length=MAX_INPUT_LENGTH, return_tensors="pt")
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

    for j, seq in enumerate(outputs.sequences):
        pred_text = tokenizer.decode(seq, skip_special_tokens=True)
        preds.append(pred_text)

        target_text = df['target_text'][i+j]
        sim_scores.append(compute_similarity(pred_text, target_text))
        perplexities.append(compute_perplexity(df['input_text'][i+j], target_text))
        correctness_flags.append(is_prediction_correct(pred_text, target_text))

# --- Save results ---
df['predicted_if'] = preds
df['prediction_score'] = sim_scores
df['perplexity'] = perplexities
df['is_correct'] = correctness_flags

# --- Optional summary ---
accuracy = sum(correctness_flags) / len(correctness_flags) * 100
avg_similarity = sum(sim_scores) / len(sim_scores)
avg_perplexity = sum(p for p in perplexities if p is not None) / len([p for p in perplexities if p is not None])

print(f"\n✅ Evaluation complete!")
print(f"Results saved to: {OUTPUT_CSV}")
print(f"Average Accuracy: {accuracy:.2f}%")
print(f"Average Similarity: {avg_similarity:.2f}")
print(f"Average Perplexity: {avg_perplexity:.2f}\n")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
