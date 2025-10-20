# 模型推理测试指南

## 概述

这个项目提供了两个推理测试脚本来验证训练后模型的效果：

1. **`test_model_inference.py`** - 完整的推理测试脚本，包含详细的分析和可视化
2. **`examples/quick_inference_test.py`** - 快速测试脚本，用于快速验证模型效果

## 功能特性

### 完整测试脚本 (`test_model_inference.py`)
- ✅ 加载原始模型和训练后的模型
- ✅ 对测试集进行采样预测
- ✅ 生成混淆矩阵可视化
- ✅ 分析置信度分布
- ✅ 创建详细的预测报告
- ✅ 比较两个模型的性能差异

### 快速测试脚本 (`examples/quick_inference_test.py`)
- ✅ 快速加载和比较模型
- ✅ 显示预测示例
- ✅ 计算准确率提升
- ✅ 轻量级，适合快速验证

## 使用方法

### 1. 运行完整测试

```bash
python test_model_inference.py
```

**输出文件：**
- `outputs/confusion_matrices.png` - 混淆矩阵图
- `outputs/confidence_distribution.png` - 置信度分布图
- `outputs/inference_results.json` - 详细结果数据
- `outputs/detailed_predictions.csv` - 预测报告
- `logs/inference_test.log` - 测试日志

### 2. 运行快速测试

```bash
python examples/quick_inference_test.py
```

**输出：**
- 控制台显示测试结果
- 预测示例展示
- 准确率比较

## 测试结果解读

### 准确率指标
- **原始模型准确率** - 未训练的BERT模型在测试集上的表现
- **训练后模型准确率** - 训练后的模型在测试集上的表现
- **准确率提升** - 训练带来的性能改善

### 可视化图表

#### 混淆矩阵
- 显示每个类别的预测准确性
- 对角线元素表示正确预测的数量
- 非对角线元素表示错误预测的分布

#### 置信度分布
- 原始模型置信度分布
- 训练后模型置信度分布
- 置信度提升分布
- 帮助理解模型的预测确定性

### 预测示例
脚本会显示具体的预测示例，包括：
- 输入文本
- 真实标签
- 两个模型的预测结果
- 预测是否正确

## 配置参数

### 完整测试脚本参数
```python
# 在 test_model_inference.py 中修改
sample_size = 200          # 测试样本数量
report_sample_size = 20    # 详细报告样本数量
```

### 快速测试脚本参数
```python
# 在 examples/quick_inference_test.py 中修改
sample_size = 100          # 测试样本数量
```

## 标签说明

AG News数据集包含4个类别：
- **World** (0) - 世界新闻
- **Sports** (1) - 体育新闻  
- **Business** (2) - 商业新闻
- **Sci/Tech** (3) - 科技新闻

## 预期结果

### 正常情况下的结果
- 原始模型准确率：通常在 0.85-0.90 之间
- 训练后模型准确率：应该在 0.90-0.95 之间
- 准确率提升：通常为 0.02-0.05

### 结果文件说明

#### JSON结果文件 (`inference_results.json`)
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "model_comparison": {
    "original_model_accuracy": 0.8750,
    "trained_model_accuracy": 0.9200,
    "improvement": 0.0450
  },
  "training_history": {
    "epochs": [1, 2, 3],
    "train_loss": [0.5, 0.3, 0.2],
    "eval_accuracy": [0.85, 0.90, 0.92]
  }
}
```

#### CSV预测报告 (`detailed_predictions.csv`)
包含每个测试样本的详细信息：
- 文本内容
- 真实标签
- 两个模型的预测结果
- 预测置信度
- 预测是否正确

## 故障排除

### 常见问题

1. **找不到训练后的模型文件**
   ```
   ⚠️ 未找到训练后的模型文件: outputs/final_model.pt
   ```
   **解决方案：** 先运行 `python train_baseline.py` 进行训练

2. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   **解决方案：** 减少 `sample_size` 参数或使用CPU

3. **依赖包缺失**
   ```
   ModuleNotFoundError: No module named 'seaborn'
   ```
   **解决方案：** 运行 `uv sync` 安装依赖

### 性能优化

- 使用GPU可以显著加速推理
- 调整 `sample_size` 平衡测试速度和结果可靠性
- 对于大型测试，建议使用完整测试脚本的批处理功能

## 高级用法

### 自定义测试数据
```python
# 修改 test_model_inference.py 中的数据集
dataset = load_dataset("your_custom_dataset")
```

### 添加新的评估指标
```python
# 在 ModelInferenceTester 类中添加新的评估方法
def calculate_f1_score(self, predictions, labels):
    from sklearn.metrics import f1_score
    return f1_score(labels, predictions, average='macro')
```

### 批量测试多个模型
```python
# 测试多个训练后的模型
model_paths = ["outputs/model_v1.pt", "outputs/model_v2.pt"]
for path in model_paths:
    tester.run_full_test(trained_model_path=path)
```
