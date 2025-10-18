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
import json
import matplotlib.pyplot as plt
from loguru import logger

# 设置输出和日志目录
outputs_dir = "outputs"
logs_dir = "logs"
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# 配置loguru日志系统
logger.add(
    os.path.join(logs_dir, "training.log"),
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logger.info(f"Using device: {device}")

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

num_labels = len(set(tokenized["train"]["label"]))
print(f"Number of labels: {num_labels}")

tokenized.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_labels)

model.to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# parameters
train_batch_size = 16
eval_batchs_size = 16
num_epochs = 3
lr = 2e-5

# 训练记录
training_history = {
    "train_loss": [],
    "eval_loss": [],
    "eval_accuracy": [],
    "eval_f1": [],
    "epochs": []
}

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


def save_training_history(history, outputs_dir):
    """保存训练历史记录"""
    history_path = os.path.join(outputs_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"训练历史已保存到: {history_path}")


def plot_training_curves(history, outputs_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Progress', fontsize=16)
    
    # 训练损失
    axes[0, 0].plot(history['epochs'], history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 验证损失
    axes[0, 1].plot(history['epochs'], history['eval_loss'], 'r-', label='Eval Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 验证准确率
    axes[1, 0].plot(history['epochs'], history['eval_accuracy'], 'g-', label='Accuracy')
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 验证F1分数
    axes[1, 1].plot(history['epochs'], history['eval_f1'], 'm-', label='F1 Score')
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"训练曲线已保存到: {plot_path}")


def evaluate_model(model, eval_dataloader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            
            total_loss += output.loss.item()
            
            logits = output.logits
            labels = batch["labels"]
            
            all_preds.append(logits.to("cpu").numpy())
            all_labels.append(labels.to("cpu").numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(eval_dataloader)
    
    # 计算指标
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    eval_results = compute_metrics(eval_pred=(all_preds, all_labels))
    
    return avg_loss, eval_results


logger.info("开始训练...")

optimizer = AdamW(model.parameters(), lr=lr)

num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=10,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
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

    # 计算平均训练损失
    avg_epoch_loss = epoch_loss / num_batches
    
    # 评估模型
    eval_loss, eval_results = evaluate_model(model, eval_dataloader, device)
    
    # 记录训练历史
    training_history['epochs'].append(epoch + 1)
    training_history['train_loss'].append(avg_epoch_loss)
    training_history['eval_loss'].append(eval_loss)
    training_history['eval_accuracy'].append(eval_results['accuracy'])
    training_history['eval_f1'].append(eval_results['f1_macro'])
    
    # 记录日志
    logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_epoch_loss:.4f}, "
                f"Eval Loss: {eval_loss:.4f}, "
                f"Eval Accuracy: {eval_results['accuracy']:.4f}, "
                f"Eval F1: {eval_results['f1_macro']:.4f}")
    
    # 保存训练历史
    save_training_history(training_history, outputs_dir)

# 训练完成，绘制训练曲线
logger.info("训练完成，正在生成训练曲线...")
plot_training_curves(training_history, outputs_dir)

# 最终评估
logger.info("进行最终评估...")
final_eval_loss, final_eval_results = evaluate_model(model, eval_dataloader, device)

logger.info(f"最终评估结果: {final_eval_results}")
logger.info(f"最终验证损失: {final_eval_loss:.4f}")

# 保存最终模型
final_model_path = os.path.join(outputs_dir, "final_model.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'training_history': training_history,
    'final_eval_results': final_eval_results,
    'timestamp': datetime.now().isoformat()
}, final_model_path)
logger.info(f"最终模型已保存到: {final_model_path}")

print(f"\n✅ 训练完成！")
print(f"📊 训练历史已保存到: {outputs_dir}/training_history.json")
print(f"📈 训练曲线已保存到: {outputs_dir}/training_curves.png")
print(f"💾 最终模型已保存到: {final_model_path}")
print(f"📝 训练日志已保存到: {logs_dir}/training.log")
