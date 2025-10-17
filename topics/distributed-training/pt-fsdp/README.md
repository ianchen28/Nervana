# 第一个实战项目——将文本分类模型改造为 FSDP 训练模式

理论学习的最终目的是实践。最好的项目就是拿一个您熟悉的、简单的任务，将其从单卡训练模式改造为多卡分布式训练模式。

项目目标： 将一个基于 Hugging Face Transformers 的标准文本分类模型，从单 GPU 训练脚本，成功改造为可以在多 GPU 环境下使用 FSDP 进行训练的脚本。

## 具体实施步骤

### 准备基础环境和代码

模型选择： 选择一个中等规模的预训练模型，例如 bert-base-uncased。

任务选择： 选择一个标准的文本分类任务，例如在 imdb 或 ag_news 数据集上进行情感分类或新闻分类。

基线脚本： 使用 Hugging Face Trainer 或标准的 PyTorch 训练循环，编写一个可以在单 GPU上成功运行并收敛的训练脚本。这是您的改造起点。

### 设置分布式环境

您需要一个多 GPU 的环境。最简单的方式是租用云服务（如 AWS、GCP、Azure）上的一个配备至少2个GPU的实例。

确保安装了与您的 CUDA 版本兼容的 PyTorch。

### 代码改造 (核心步骤)

引入分布式设置： 在您的训练脚本开头，加入分布式进程组的初始化代码 (setup 函数)，并在结尾加入清理代码 (cleanup 函数) 。

数据加载器改造： 将标准的 DataLoader 替换为使用 DistributedSampler 的 DataLoader。这是为了确保每个 GPU 进程都能获取到数据的一个不重复的子集 。

模型包装： 这是最关键的一步。在模型初始化之后，使用 FSDP 类来包装您的 Transformer 模型。您需要配置 auto_wrap_policy，告诉 FSDP 按照 Transformer Block 的粒度来进行分片 。

```python

# 示例代码片段
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.bert.modeling_bert import BertBlock # 导入您模型对应的Block

#... 初始化您的模型 model...

bert_auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={BertBlock},
)

model = FSDP(model, auto_wrap_policy=bert_auto_wrap_policy)
```

修改训练循环： 确保损失计算、反向传播 (loss.backward()) 和优化器步骤 (optimizer.step()) 都在 FSDP 包装后的模型上正确执行。

修改模型保存逻辑： 保存 FSDP 模型状态字典需要特殊的处理方式，通常需要在所有进程上获取分片的状态，然后在 rank 0 进程上统一保存 。

### 启动与验证

启动命令： 使用 torchrun 来启动您的训练脚本。这是 PyTorch 官方推荐的分布式任务启动器 。

```bash
torchrun --nproc_per_node=2 your_training_script.py
```

这里的 --nproc_per_node=2 表示您希望在当前节点上使用2个 GPU 进程。

验证成功： 如果脚本能够顺利运行，并且您通过 nvidia-smi 命令观察到两个 GPU 都有显存占用和计算负载，那么恭喜您，您已经成功完成了第一个 FSDP 项目！
