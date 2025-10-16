import os
import torch
from datasets import load_dataset
# Load model directly
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate

use_cuda = torch.cuda.is_available()

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased",
    use_fast=True,
)

dataset = load_dataset("ag_news")

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding=False, max_length=128)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

num_labels = len(set(tokenized["train"]["label"]))

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./outputs_ag_news_bert_base",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to=["none"],
    fp16=use_cuda,
    bf16=use_cuda and torch.cuda.is_bf16_supported(),
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    # tokenizer=tokenizer,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())
