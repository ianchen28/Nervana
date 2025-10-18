import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy, )
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
# Load model directly
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
import evaluate
import os
from functools import partial


# 分布式训练初始化
def setup_distributed():
    """初始化分布式训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("未检测到分布式环境，使用单GPU模式")
        rank = 0
        world_size = 1
        local_rank = 0

    # 初始化进程组
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


# 初始化分布式环境
rank, world_size, local_rank = setup_distributed()
use_cuda = torch.cuda.is_available()

# 设置设备
device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    use_fast=True,
)

dataset = load_dataset("ag_news")


# FSDP 配置函数
def get_fsdp_config():
    """获取 FSDP 配置"""
    # 混合精度配置
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
        reduce_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
        buffer_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # 自动包装策略 - 针对 Transformer 层
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # BERT 的 TransformerEncoderLayer
            "BertLayer",
            "BertEncoder",
            "BertSelfAttention",
            "BertSelfOutput",
            "BertIntermediate",
            "BertOutput",
        })

    return {
        "mixed_precision": mixed_precision_policy,
        "auto_wrap_policy": auto_wrap_policy,
        "sharding_strategy":
        torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
        "device_id": local_rank,
    }


def setup_fsdp_model(model):
    """使用 FSDP 包装模型"""
    fsdp_config = get_fsdp_config()

    # 使用 FSDP 包装模型
    model = FSDP(model, **fsdp_config)

    return model


def tokenize_fn(batch):
    return tokenizer(batch["text"],
                     truncation=True,
                     padding=False,
                     max_length=128)


tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

num_labels = len(set(tokenized["train"]["label"]))

# 创建模型
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_labels)

# 使用 FSDP 包装模型
model = setup_fsdp_model(model)

# 训练配置
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
LOGGING_STEPS = 50

# 数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 创建数据加载器
def create_dataloader(dataset, is_training=True):
    """创建分布式数据加载器"""
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=is_training) if world_size > 1 else None

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None and is_training),
        collate_fn=data_collator,
        pin_memory=True,
    )


train_dataloader = create_dataloader(tokenized["train"], is_training=True)
eval_dataloader = create_dataloader(tokenized["test"], is_training=False)

# 优化器和学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 计算总步数
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 评估指标
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


# 训练函数
def train_epoch(model, dataloader, optimizer, scheduler, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        # 将数据移动到设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 计算损失
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # 日志记录
        if step % LOGGING_STEPS == 0 and rank == 0:
            print(
                f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return total_loss / num_batches


# 评估函数
def evaluate_model(model, dataloader):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # 将数据移动到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 计算损失
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            # 收集预测结果
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    if rank == 0:
        acc = accuracy.compute(predictions=all_predictions,
                               references=all_labels)["accuracy"]
        f1_score = f1.compute(predictions=all_predictions,
                              references=all_labels,
                              average="macro")["f1"]
        avg_loss = total_loss / num_batches

        print(
            f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1: {f1_score:.4f}"
        )
        return {"loss": avg_loss, "accuracy": acc, "f1": f1_score}

    return {"loss": total_loss / num_batches}


# 检查点保存和加载函数
def save_fsdp_checkpoint(model, optimizer, scheduler, epoch, output_dir):
    """保存 FSDP 检查点"""
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # 同步所有进程
    if world_size > 1:
        dist.barrier()

    # 保存检查点
    with FSDP.state_dict_type(
            model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }

        if rank == 0:
            checkpoint_path = os.path.join(output_dir,
                                           f"checkpoint-epoch-{epoch}.pt")
            torch.save(state_dict, checkpoint_path)
            print(f"检查点已保存到: {checkpoint_path}")


def load_fsdp_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载 FSDP 检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型状态
    with FSDP.state_dict_type(
            model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
        model.load_state_dict(checkpoint["model"])

    # 加载优化器和调度器状态
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)

    if rank == 0:
        print(f"检查点已加载: {checkpoint_path}, epoch: {epoch}")

    return epoch


# 主训练循环
def main():
    """主训练函数"""
    if rank == 0:
        print(f"开始 FSDP 训练，使用 {world_size} 个GPU")
        print(
            f"训练配置: batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}, lr={LEARNING_RATE}"
        )

    # 输出目录
    output_dir = "./outputs_ag_news_bert_base_fsdp"

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        if rank == 0:
            print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")

        # 设置分布式采样器的epoch
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)

        # 训练
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler,
                                 epoch + 1)

        # 评估
        if rank == 0:
            print(f"训练损失: {train_loss:.4f}")

        evaluate_model(model, eval_dataloader)

        # 保存检查点
        save_fsdp_checkpoint(model, optimizer, scheduler, epoch + 1,
                             output_dir)

        # 同步所有进程
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print("训练完成！")
        print(f"模型和检查点保存在: {output_dir}")


# 运行训练
if __name__ == "__main__":
    main()
