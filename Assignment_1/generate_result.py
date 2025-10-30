import os
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
from tqdm import tqdm

# --- Configuration ---
MODEL_DIR = "./models/codet5-finetuned/best_model"
TOKENIZER_JSON_PATH = "./tokenizer_files/tokenizer.json"
TEST_CSV_PATH = "./testsets/benchmark_if_only.csv"
OUTPUT_CSV_PATH = "./testsets/generated_testset.csv"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 64
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load tokenizer & model ---
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_JSON_PATH,
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>",
)

model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# --- Load test CSV ---
test_df = pd.read_csv(TEST_CSV_PATH)
inputs = test_df["code"].tolist()  # Use 'code' column
# Insert <mask> where you expect the 'if' statement
masked_inputs = [code + " <mask>" for code in inputs]

# --- Generate predictions ---
preds = []
scores = []

for i in tqdm(range(0, len(masked_inputs), BATCH_SIZE)):
    batch_texts = masked_inputs[i:i+BATCH_SIZE]

    encodings = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=5,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    for seq, seq_scores in zip(outputs.sequences, outputs.sequences_scores):
        pred_text = tokenizer.decode(seq, skip_special_tokens=True)
        preds.append(pred_text)

        # More realistic confidence: normalize logit scores
        confidence = float(seq_scores.exp() / (seq_scores.exp() + 1e-8)) * 100
        scores.append(confidence)

# --- Save results ---
test_df["predicted_if"] = preds
test_df["prediction_score"] = scores

os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
test_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✅ Predictions saved to {OUTPUT_CSV_PATH}")
