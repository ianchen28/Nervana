import os
import torch
from datasets import load_dataset
# Load model directly
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from datetime import datetime

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    use_fast=True,
)

dataset = load_dataset("ag_news")


def tokenize_fn(batch):
    return tokenizer(batch["text"],
                     truncation=True,
                     padding=False,
                     max_length=128)


tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized.set_format("torch")

num_labels = len(set(tokenized["train"]["label"]))

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_labels)

model.to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# parameters
train_batch_size = 16
eval_batchs_size = 16
num_epochs = 3
lr = 2e-5

# checkpoint settings
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# 可选：从checkpoint恢复训练
# 设置resume_from_checkpoint为None表示从头开始训练
# 设置为checkpoint文件路径表示从该checkpoint恢复
resume_from_checkpoint = None  # 例如: "checkpoints/checkpoint_epoch_2.pt"

train_dataloader = DataLoader(
    tokenized["train"],
    batch_size=train_batch_size,
    shuffle=True,
    collate_fn=data_collator,
)

eval_dataloader = DataLoader(
    tokenized["test"],
    batch_size=eval_batchs_size,
    collate_fn=data_collator,
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":
        accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro":
        f1.compute(predictions=preds, references=labels,
                   average="macro")["f1"],
    }


def save_checkpoint(model, optimizer, lr_scheduler, epoch, loss,
                    checkpoint_dir):
    """保存模型checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir,
                                   f"checkpoint_epoch_{epoch+1}.pt")

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler):
    """从checkpoint恢复训练状态"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    timestamp = checkpoint['timestamp']

    print(f"✅ Checkpoint loaded: {checkpoint_path}")
    print(f"   Epoch: {epoch}, Loss: {loss:.4f}, Timestamp: {timestamp}")

    return epoch, loss


print("✅ Step 1 complete: Removed Trainer and created DataLoaders.")

optimizer = AdamW(model.parameters(), lr=lr)

num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=10,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

# 检查是否需要从checkpoint恢复
start_epoch = 0
if resume_from_checkpoint:
    result = load_checkpoint(resume_from_checkpoint, model, optimizer,
                             lr_scheduler)
    if result:
        start_epoch, _ = result
        print(f"🔄 Resuming training from epoch {start_epoch}")
    else:
        print("❌ Failed to load checkpoint, starting from scratch")

for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(**batch)
        loss = output.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1

        progress_bar.update(1)
        progress_bar.set_description(
            f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 计算平均loss并保存checkpoint
    avg_epoch_loss = epoch_loss / num_batches
    save_checkpoint(model, optimizer, lr_scheduler, epoch, avg_epoch_loss,
                    checkpoint_dir)

print("\n✅ Step 2 complete: Manual training loop implemented.")
print("\nStarting evaluation...")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)

        logits = output.logits
        labels = batch["labels"]

        all_preds.append(logits.to("cpu").numpy())
        all_labels.append(labels.to("cpu").numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

eval_results = compute_metrics(eval_pred=(all_preds, all_labels))
print("\n✅ Step 3 complete: Manual evaluation loop implemented.")
print(f"Evaluation results: {eval_results}")
