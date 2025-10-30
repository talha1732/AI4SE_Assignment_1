# ============================================
# Train CodeT5 Model with GPU Progress & Loss Logging
# ============================================

import os
import time
import logging
import torch
import pandas as pd
from transformers import (
    T5ForConditionalGeneration,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import Dataset

# --- Configuration ---
OUTPUT_DATA_DIR = "collected_data"
PRETRAIN_CSV = os.path.join(OUTPUT_DATA_DIR, "pretrain_dataset.csv")
FINETUNE_TRAIN_CSV = os.path.join(OUTPUT_DATA_DIR, "finetune_train.csv")
FINETUNE_VAL_CSV = os.path.join(OUTPUT_DATA_DIR, "finetune_val.csv")
TOKENIZER_JSON_PATH = os.path.join("tokenizer_files", "tokenizer.json")
MAX_SEQUENCE_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Utility Functions ---
def get_gpu_memory():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        free = total - reserved
        return f"{allocated:.1f} GB used / {free:.1f} GB free of {total:.1f} GB total"
    return "No GPU"

# --- Callback for Logging Loss ---
class LossLoggingCallback(TrainerCallback):
    """Logs training and evaluation loss per logging step."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        train_loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        if train_loss is not None:
            logging.info(f"Step {step}: train_loss = {train_loss:.4f}")
        if eval_loss is not None:
            logging.info(f"Step {step}: eval_loss = {eval_loss:.4f}")

# --- Main Training Function ---
def main():
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Initial GPU memory: {get_gpu_memory()}")

    # --- Load Tokenizer & Model ---
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_JSON_PATH,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
        model_max_length=MAX_SEQUENCE_LENGTH,
    )

    logging.info("Loading pre-trained CodeT5 model...")
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base").to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # --- 1️⃣ Further Pre-training ---
    logging.info("Loading pre-training dataset...")
    pretrain_df = pd.read_csv(PRETRAIN_CSV)
    pretrain_df["input_ids"] = pretrain_df["input_ids"].apply(eval)
    pretrain_df["labels"] = pretrain_df["labels"].apply(eval)
    pretrain_dataset = Dataset.from_pandas(pretrain_df)

    def pretrain_tokenize(batch):
        return {"input_ids": batch["input_ids"], "labels": batch["labels"]}

    start_time = time.time()
    pretrain_dataset = pretrain_dataset.map(
        pretrain_tokenize,
        batched=True,
        remove_columns=pretrain_dataset.column_names,
        desc="Tokenizing pretrain dataset",
    )
    logging.info(f"Pre-training dataset tokenization done in {time.time() - start_time:.1f}s")

    pretrain_args = TrainingArguments(
        output_dir="./models/codet5-pretrained",
        per_device_train_batch_size=2,  # Adjusted for memory
        num_train_epochs=1,
        fp16=True,
        gradient_checkpointing=True,
        save_total_limit=1,
        logging_steps=500,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to="none",
    )

    pretrain_trainer = Trainer(
        model=model,
        args=pretrain_args,
        train_dataset=pretrain_dataset,
        callbacks=[LossLoggingCallback]
    )

    logging.info("🚀 Starting Further Pre-training on GPU...")
    torch.cuda.empty_cache()
    start_time = time.time()

    try:
        pretrain_trainer.train()
    except torch.cuda.OutOfMemoryError:
        logging.warning("⚠️ OOM during pre-training. Reducing batch size and retrying...")
        torch.cuda.empty_cache()
        pretrain_trainer.args.per_device_train_batch_size = 1
        pretrain_trainer.train()

    elapsed = time.time() - start_time
    logging.info(f"✅ Pre-training completed in {elapsed / 60:.1f} minutes.")
    logging.info(f"Post-training GPU memory: {get_gpu_memory()}")
    pretrain_trainer.save_model("./models/codet5-pretrained")

    # --- 2️⃣ Fine-tuning ---
    logging.info("Loading fine-tuning datasets...")
    train_df = pd.read_csv(FINETUNE_TRAIN_CSV)
    val_df = pd.read_csv(FINETUNE_VAL_CSV)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    def preprocess(batch):
        inputs = tokenizer(
            batch["input_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                padding="max_length",
                truncation=True,
                max_length=64,
            )
        inputs["labels"] = labels["input_ids"]
        return inputs

    start_time = time.time()
    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    val_dataset = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset",
    )
    logging.info(f"Fine-tuning dataset tokenization done in {time.time() - start_time:.1f}s")

    finetune_args = TrainingArguments(
        output_dir="./models/codet5-finetuned",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to="none",
    )

    finetune_trainer = Trainer(
        model=model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[LossLoggingCallback]
    )

    logging.info("🚀 Starting Fine-tuning...")
    torch.cuda.empty_cache()
    start_time = time.time()

    try:
        finetune_trainer.train()
    except torch.cuda.OutOfMemoryError:
        logging.warning("⚠️ OOM during fine-tuning. Reducing batch size and retrying...")
        torch.cuda.empty_cache()
        finetune_trainer.args.per_device_train_batch_size = 2
        finetune_trainer.train()

    elapsed = time.time() - start_time
    logging.info(f"✅ Fine-tuning completed in {elapsed / 60:.1f} minutes.")
    logging.info(f"Final GPU memory: {get_gpu_memory()}")
    finetune_trainer.save_model("./models/codet5-finetuned/best_model")


if __name__ == "__main__":
    main()
