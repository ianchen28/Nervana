#!/usr/bin/env python3
"""
模型推理测试脚本
用于验证训练后模型的效果，比较原始模型和训练后模型的预测结果
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from loguru import logger
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import random

# 设置输出目录
outputs_dir = "outputs"
logs_dir = "logs"
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# 配置日志
logger.add(
    os.path.join(logs_dir, "inference_test.log"),
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

class ModelInferenceTester:
    """模型推理测试类"""
    
    def __init__(self, model_name="bert-base-uncased", device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        logger.info(f"初始化推理测试器，使用设备: {self.device}")
    
    def load_models(self, trained_model_path=None):
        """加载原始模型和训练后的模型"""
        logger.info("加载模型...")
        
        # 加载数据集以获取标签数量
        dataset = load_dataset("ag_news")
        num_labels = len(set(dataset["train"]["label"]))
        
        # 加载原始模型
        self.original_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )
        self.original_model.to(self.device)
        self.original_model.eval()
        
        # 加载训练后的模型
        if trained_model_path and os.path.exists(trained_model_path):
            logger.info(f"加载训练后的模型: {trained_model_path}")
            checkpoint = torch.load(trained_model_path, map_location=self.device)
            self.trained_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )
            self.trained_model.load_state_dict(checkpoint['model_state_dict'])
            self.trained_model.to(self.device)
            self.trained_model.eval()
            
            # 获取训练历史
            self.training_history = checkpoint.get('training_history', {})
            self.final_eval_results = checkpoint.get('final_eval_results', {})
        else:
            logger.warning(f"训练后的模型文件不存在: {trained_model_path}")
            self.trained_model = None
            self.training_history = {}
            self.final_eval_results = {}
        
        logger.info("模型加载完成")
    
    def prepare_test_data(self, sample_size=100):
        """准备测试数据"""
        logger.info(f"准备测试数据，采样数量: {sample_size}")
        
        # 加载数据集
        dataset = load_dataset("ag_news")
        
        # 对测试集进行采样
        test_data = dataset["test"]
        if sample_size < len(test_data):
            # 随机采样
            indices = random.sample(range(len(test_data)), sample_size)
            sampled_data = test_data.select(indices)
        else:
            sampled_data = test_data
        
        # 分词处理
        def tokenize_fn(batch):
            return self.tokenizer(
                batch["text"], 
                truncation=True, 
                padding=False, 
                max_length=128
            )
        
        tokenized_data = sampled_data.map(tokenize_fn, batched=True, remove_columns=["text"])
        tokenized_data.set_format("torch")
        
        # 创建数据加载器
        dataloader = DataLoader(
            tokenized_data, 
            batch_size=16, 
            collate_fn=self.data_collator
        )
        
        return dataloader, sampled_data
    
    def predict_with_model(self, model, dataloader):
        """使用指定模型进行预测"""
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="预测中"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                
                # 获取预测结果和概率
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)
    
    def compare_models(self, dataloader, test_data):
        """比较两个模型的预测结果"""
        logger.info("开始模型比较...")
        
        # 原始模型预测
        logger.info("使用原始模型进行预测...")
        orig_preds, orig_probs, labels = self.predict_with_model(self.original_model, dataloader)
        
        results = {
            "original_model": {
                "predictions": orig_preds,
                "probabilities": orig_probs,
                "labels": labels
            }
        }
        
        # 训练后模型预测（如果存在）
        if self.trained_model is not None:
            logger.info("使用训练后的模型进行预测...")
            trained_preds, trained_probs, _ = self.predict_with_model(self.trained_model, dataloader)
            results["trained_model"] = {
                "predictions": trained_preds,
                "probabilities": trained_probs,
                "labels": labels
            }
        
        # 计算准确率
        orig_accuracy = np.mean(orig_preds == labels)
        results["original_model"]["accuracy"] = orig_accuracy
        
        if self.trained_model is not None:
            trained_accuracy = np.mean(trained_preds == labels)
            results["trained_model"]["accuracy"] = trained_accuracy
            improvement = trained_accuracy - orig_accuracy
            results["improvement"] = improvement
            
            logger.info(f"原始模型准确率: {orig_accuracy:.4f}")
            logger.info(f"训练后模型准确率: {trained_accuracy:.4f}")
            logger.info(f"准确率提升: {improvement:.4f}")
        else:
            logger.info(f"原始模型准确率: {orig_accuracy:.4f}")
        
        return results, test_data
    
    def create_comparison_report(self, results, test_data, sample_size=10):
        """创建比较报告"""
        logger.info("生成比较报告...")
        
        # 选择一些样本进行详细比较
        sample_indices = random.sample(range(len(test_data)), min(sample_size, len(test_data)))
        
        report_data = []
        for idx in sample_indices:
            text = test_data[idx]["text"]
            true_label = test_data[idx]["label"]
            
            # 获取标签名称
            label_names = ["World", "Sports", "Business", "Sci/Tech"]
            true_label_name = label_names[true_label]
            
            row = {
                "index": idx,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "true_label": true_label_name,
                "true_label_id": true_label
            }
            
            # 原始模型结果
            orig_pred = results["original_model"]["predictions"][idx]
            orig_prob = results["original_model"]["probabilities"][idx]
            row.update({
                "orig_pred": label_names[orig_pred],
                "orig_pred_id": orig_pred,
                "orig_confidence": orig_prob[orig_pred],
                "orig_correct": orig_pred == true_label
            })
            
            # 训练后模型结果（如果存在）
            if "trained_model" in results:
                trained_pred = results["trained_model"]["predictions"][idx]
                trained_prob = results["trained_model"]["probabilities"][idx]
                row.update({
                    "trained_pred": label_names[trained_pred],
                    "trained_pred_id": trained_pred,
                    "trained_confidence": trained_prob[trained_pred],
                    "trained_correct": trained_pred == true_label
                })
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def plot_confusion_matrices(self, results, outputs_dir):
        """绘制混淆矩阵"""
        logger.info("生成混淆矩阵...")
        
        fig, axes = plt.subplots(1, 2 if "trained_model" in results else 1, figsize=(15, 6))
        if "trained_model" not in results:
            axes = [axes]
        
        label_names = ["World", "Sports", "Business", "Sci/Tech"]
        
        # 原始模型混淆矩阵
        orig_cm = confusion_matrix(
            results["original_model"]["labels"], 
            results["original_model"]["predictions"]
        )
        sns.heatmap(orig_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_names, yticklabels=label_names, ax=axes[0])
        axes[0].set_title(f'原始模型混淆矩阵\n准确率: {results["original_model"]["accuracy"]:.4f}')
        axes[0].set_xlabel('预测标签')
        axes[0].set_ylabel('真实标签')
        
        # 训练后模型混淆矩阵（如果存在）
        if "trained_model" in results:
            trained_cm = confusion_matrix(
                results["trained_model"]["labels"], 
                results["trained_model"]["predictions"]
            )
            sns.heatmap(trained_cm, annot=True, fmt='d', cmap='Greens',
                       xticklabels=label_names, yticklabels=label_names, ax=axes[1])
            axes[1].set_title(f'训练后模型混淆矩阵\n准确率: {results["trained_model"]["accuracy"]:.4f}')
            axes[1].set_xlabel('预测标签')
            axes[1].set_ylabel('真实标签')
        
        plt.tight_layout()
        confusion_path = os.path.join(outputs_dir, "confusion_matrices.png")
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"混淆矩阵已保存到: {confusion_path}")
        return confusion_path
    
    def plot_confidence_distribution(self, results, outputs_dir):
        """绘制置信度分布图"""
        logger.info("生成置信度分布图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('模型预测置信度分布', fontsize=16)
        
        # 原始模型置信度
        orig_confidences = np.max(results["original_model"]["probabilities"], axis=1)
        axes[0, 0].hist(orig_confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('原始模型置信度分布')
        axes[0, 0].set_xlabel('置信度')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 训练后模型置信度（如果存在）
        if "trained_model" in results:
            trained_confidences = np.max(results["trained_model"]["probabilities"], axis=1)
            axes[0, 1].hist(trained_confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('训练后模型置信度分布')
            axes[0, 1].set_xlabel('置信度')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 置信度对比
            axes[1, 0].hist(orig_confidences, bins=20, alpha=0.5, label='原始模型', color='blue')
            axes[1, 0].hist(trained_confidences, bins=20, alpha=0.5, label='训练后模型', color='green')
            axes[1, 0].set_title('置信度分布对比')
            axes[1, 0].set_xlabel('置信度')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 置信度提升
            confidence_improvement = trained_confidences - orig_confidences
            axes[1, 1].hist(confidence_improvement, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_title('置信度提升分布')
            axes[1, 1].set_xlabel('置信度提升')
            axes[1, 1].set_ylabel('频次')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # 如果没有训练后模型，隐藏多余的子图
            axes[0, 1].set_visible(False)
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        confidence_path = os.path.join(outputs_dir, "confidence_distribution.png")
        plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"置信度分布图已保存到: {confidence_path}")
        return confidence_path
    
    def save_detailed_results(self, results, report_df, outputs_dir):
        """保存详细结果"""
        logger.info("保存详细结果...")
        
        # 保存预测结果
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "model_comparison": {
                "original_model_accuracy": results["original_model"]["accuracy"]
            }
        }
        
        if "trained_model" in results:
            results_data["model_comparison"]["trained_model_accuracy"] = results["trained_model"]["accuracy"]
            results_data["model_comparison"]["improvement"] = results["improvement"]
        
        # 保存训练历史（如果存在）
        if self.training_history:
            results_data["training_history"] = self.training_history
        
        # 保存结果到JSON
        results_path = os.path.join(outputs_dir, "inference_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 保存详细报告到CSV
        report_path = os.path.join(outputs_dir, "detailed_predictions.csv")
        report_df.to_csv(report_path, index=False, encoding='utf-8')
        
        logger.info(f"详细结果已保存到: {results_path}")
        logger.info(f"预测报告已保存到: {report_path}")
        
        return results_path, report_path
    
    def run_full_test(self, trained_model_path=None, sample_size=100, report_sample_size=10):
        """运行完整的推理测试"""
        logger.info("开始完整的模型推理测试...")
        
        # 加载模型
        self.load_models(trained_model_path)
        
        # 准备测试数据
        dataloader, test_data = self.prepare_test_data(sample_size)
        
        # 比较模型
        results, _ = self.compare_models(dataloader, test_data)
        
        # 创建详细报告
        report_df = self.create_comparison_report(results, test_data, report_sample_size)
        
        # 生成可视化
        confusion_path = self.plot_confusion_matrices(results, outputs_dir)
        confidence_path = self.plot_confidence_distribution(results, outputs_dir)
        
        # 保存结果
        results_path, report_path = self.save_detailed_results(results, report_df, outputs_dir)
        
        # 打印总结
        print("\n" + "="*60)
        print("🎯 模型推理测试完成！")
        print("="*60)
        print(f"📊 测试样本数量: {sample_size}")
        print(f"📈 原始模型准确率: {results['original_model']['accuracy']:.4f}")
        
        if "trained_model" in results:
            print(f"🚀 训练后模型准确率: {results['trained_model']['accuracy']:.4f}")
            print(f"📈 准确率提升: {results['improvement']:.4f}")
        
        print(f"\n📁 输出文件:")
        print(f"  📊 混淆矩阵: {confusion_path}")
        print(f"  📈 置信度分布: {confidence_path}")
        print(f"  📋 详细结果: {results_path}")
        print(f"  📄 预测报告: {report_path}")
        print(f"  📝 测试日志: {logs_dir}/inference_test.log")
        
        return results, report_df


def main():
    """主函数"""
    logger.info("启动模型推理测试...")
    
    # 检查训练后的模型是否存在
    trained_model_path = os.path.join(outputs_dir, "final_model.pt")
    if not os.path.exists(trained_model_path):
        logger.warning(f"训练后的模型文件不存在: {trained_model_path}")
        logger.info("将只测试原始模型")
        trained_model_path = None
    
    # 创建测试器
    tester = ModelInferenceTester()
    
    # 运行测试
    results, report_df = tester.run_full_test(
        trained_model_path=trained_model_path,
        sample_size=200,  # 测试样本数量
        report_sample_size=20  # 详细报告样本数量
    )
    
    # 显示一些预测示例
    print("\n" + "="*60)
    print("🔍 预测示例:")
    print("="*60)
    
    for i in range(min(5, len(report_df))):
        row = report_df.iloc[i]
        print(f"\n示例 {i+1}:")
        print(f"文本: {row['text']}")
        print(f"真实标签: {row['true_label']}")
        print(f"原始模型预测: {row['orig_pred']} (置信度: {row['orig_confidence']:.3f})")
        if 'trained_pred' in row:
            print(f"训练后模型预测: {row['trained_pred']} (置信度: {row['trained_confidence']:.3f})")
        print(f"原始模型正确: {'✅' if row['orig_correct'] else '❌'}")
        if 'trained_correct' in row:
            print(f"训练后模型正确: {'✅' if row['trained_correct'] else '❌'}")


if __name__ == "__main__":
    main()
