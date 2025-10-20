#!/usr/bin/env python3
"""
快速推理测试脚本 - 简化版本
用于快速验证训练后模型的效果
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
import random
from loguru import logger


def quick_model_test(trained_model_path=None, sample_size=50):
    """快速模型测试"""

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载数据和tokenizer
    logger.info("加载数据...")
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                              use_fast=True)

    # 准备测试数据
    test_data = dataset["test"]
    sample_indices = random.sample(range(len(test_data)),
                                   min(sample_size, len(test_data)))
    sampled_data = test_data.select(sample_indices)

    def tokenize_fn(batch):
        return tokenizer(batch["text"],
                         truncation=True,
                         padding=False,
                         max_length=128)

    tokenized_data = sampled_data.map(tokenize_fn,
                                      batched=True,
                                      remove_columns=["text"])
    tokenized_data.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(tokenized_data,
                            batch_size=16,
                            collate_fn=data_collator)

    # 加载模型
    num_labels = len(set(dataset["train"]["label"]))
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    # 原始模型
    logger.info("加载原始模型...")
    original_model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels)
    original_model.to(device)
    original_model.eval()

    # 训练后模型（如果存在）
    trained_model = None
    if trained_model_path and os.path.exists(trained_model_path):
        logger.info("加载训练后的模型...")
        checkpoint = torch.load(trained_model_path, map_location=device)
        trained_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels)
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        trained_model.to(device)
        trained_model.eval()

    def predict_with_model(model, dataloader):
        """使用模型进行预测"""
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch["labels"].cpu().numpy())

        return np.array(predictions), np.array(labels)

    # 进行预测
    logger.info("使用原始模型进行预测...")
    orig_preds, true_labels = predict_with_model(original_model, dataloader)
    orig_accuracy = np.mean(orig_preds == true_labels)

    print(f"\n{'='*50}")
    print(f"🎯 快速模型测试结果")
    print(f"{'='*50}")
    print(f"📊 测试样本数量: {sample_size}")
    print(f"📈 原始模型准确率: {orig_accuracy:.4f}")

    if trained_model is not None:
        logger.info("使用训练后的模型进行预测...")
        trained_preds, _ = predict_with_model(trained_model, dataloader)
        trained_accuracy = np.mean(trained_preds == true_labels)
        improvement = trained_accuracy - orig_accuracy

        print(f"🚀 训练后模型准确率: {trained_accuracy:.4f}")
        print(f"📈 准确率提升: {improvement:.4f}")

        # 显示一些预测示例
        print(f"\n🔍 预测示例 (前5个样本):")
        print(f"{'='*50}")

        for i in range(min(5, len(sampled_data))):
            text = sampled_data[i]["text"]
            true_label = true_labels[i]
            orig_pred = orig_preds[i]
            trained_pred = trained_preds[i]

            print(f"\n示例 {i+1}:")
            print(f"文本: {text[:80]}...")
            print(f"真实标签: {label_names[true_label]}")
            print(
                f"原始模型预测: {label_names[orig_pred]} {'✅' if orig_pred == true_label else '❌'}"
            )
            print(
                f"训练后模型预测: {label_names[trained_pred]} {'✅' if trained_pred == true_label else '❌'}"
            )

            if orig_pred != trained_pred:
                print(
                    f"🔄 预测发生变化: {label_names[orig_pred]} → {label_names[trained_pred]}"
                )
    else:
        print(f"\n⚠️  未找到训练后的模型文件: {trained_model_path}")
        print("只测试了原始模型")

    print(f"\n✅ 测试完成！")


def main():
    """主函数"""
    # 检查训练后的模型
    trained_model_path = "outputs/final_model.pt"

    # 运行快速测试
    quick_model_test(
        trained_model_path=trained_model_path,
        sample_size=100  # 可以调整样本数量
    )


if __name__ == "__main__":
    main()
