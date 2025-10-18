#!/bin/bash

# FSDP 分布式训练启动脚本
# 使用方法：
# 单GPU: bash run_fsdp.sh
# 多GPU: torchrun --nproc_per_node=2 run_fsdp.sh

echo "开始 FSDP 分布式训练..."

# 检查是否有 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "检测到 NVIDIA GPU"
    nvidia-smi
else
    echo "未检测到 NVIDIA GPU，将使用 CPU"
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用前两个GPU，可以根据需要调整
export NCCL_DEBUG=INFO  # 启用 NCCL 调试信息

# 运行训练
echo "启动训练..."
python train_fsdp.py

echo "训练完成！"
