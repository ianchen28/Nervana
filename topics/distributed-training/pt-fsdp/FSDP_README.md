# PyTorch FSDP 分布式训练指南

## 概述

这个项目演示了如何使用 PyTorch 的 Fully Sharded Data Parallel (FSDP) 进行分布式训练。FSDP 是一种内存高效的分布式训练方法，特别适合训练大型模型。

## FSDP 核心概念

### 1. 模型分片 (Model Sharding)

- **参数分片**: 将模型参数分布到不同的 GPU 上
- **梯度分片**: 梯度也进行分片存储
- **优化器状态分片**: 优化器状态同样分片

### 2. 通信优化

- **All-Gather**: 在需要时收集完整的参数
- **Reduce-Scatter**: 分散梯度更新
- **动态通信**: 只在必要时进行参数通信

### 3. 内存效率

- **ZeRO 优化**: 类似 DeepSpeed ZeRO 的内存优化
- **激活检查点**: 可选的激活重计算以节省内存

## 代码结构解析

### 1. 分布式初始化

```python
def setup_distributed():
    """初始化分布式训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # 单GPU模式
        rank = 0
        world_size = 1
        local_rank = 0
```

### 2. FSDP 配置

```python
def get_fsdp_config():
    """获取 FSDP 配置"""
    # 混合精度配置
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        reduce_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        buffer_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    # 自动包装策略 - 针对 Transformer 层
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            "BertLayer", "BertEncoder", 
            "BertSelfAttention", "BertSelfOutput",
            "BertIntermediate", "BertOutput",
        }
    )
```

### 3. 模型包装

```python
def setup_fsdp_model(model):
    """使用 FSDP 包装模型"""
    fsdp_config = get_fsdp_config()
    model = FSDP(model, **fsdp_config)
    return model
```

### 4. 训练循环

- **前向传播**: 自动处理参数收集
- **反向传播**: 自动处理梯度分散
- **优化器步骤**: 在分片参数上更新

### 5. 检查点保存/加载

```python
# 保存检查点
with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
    state_dict = model.state_dict()
    torch.save(state_dict, checkpoint_path)

# 加载检查点
with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
    model.load_state_dict(checkpoint)
```

## 使用方法

### 单GPU训练

```bash
python train_fsdp.py
```

### 多GPU训练

```bash
# 使用 torchrun (推荐)
torchrun --nproc_per_node=2 train_fsdp.py

# 或使用 torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=2 train_fsdp.py
```

### 使用启动脚本

```bash
# 单GPU
bash run_fsdp.sh

# 多GPU (修改脚本中的 nproc_per_node)
torchrun --nproc_per_node=2 run_fsdp.sh
```

## 关键配置参数

### 1. 分片策略

- `FULL_SHARD`: 完全分片（推荐）
- `SHARD_GRAD_OP`: 只分片梯度和优化器状态
- `NO_SHARD`: 不分片（等同于 DDP）

### 2. 混合精度

- `param_dtype`: 参数数据类型
- `reduce_dtype`: 归约操作数据类型
- `buffer_dtype`: 缓冲区数据类型

### 3. 自动包装策略

- `transformer_auto_wrap_policy`: 针对 Transformer 模型的包装策略
- 可以自定义包装策略

## 性能优化建议

### 1. 内存优化

- 使用混合精度训练
- 启用激活检查点
- 调整批处理大小

### 2. 通信优化

- 使用 NCCL 后端
- 优化网络配置
- 考虑使用梯度累积

### 3. 调试技巧

- 设置 `NCCL_DEBUG=INFO` 查看通信信息
- 使用 `torch.distributed.barrier()` 同步进程
- 监控 GPU 内存使用

## 常见问题

### 1. 内存不足

- 减少批处理大小
- 启用激活检查点
- 使用梯度累积

### 2. 通信错误

- 检查网络配置
- 确保所有进程可以相互通信
- 检查防火墙设置

### 3. 性能问题

- 调整分片策略
- 优化数据加载
- 使用更快的存储

## 与 DDP 的对比

| 特性 | DDP | FSDP |
|------|-----|------|
| 内存使用 | 高 | 低 |
| 通信开销 | 低 | 中等 |
| 实现复杂度 | 低 | 中等 |
| 适用场景 | 小模型 | 大模型 |

## 总结

FSDP 是训练大型模型的有效方法，通过模型分片显著减少内存使用。虽然实现比 DDP 复杂，但内存效率的提升使得训练更大的模型成为可能。
