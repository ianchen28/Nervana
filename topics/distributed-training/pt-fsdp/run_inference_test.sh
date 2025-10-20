#!/bin/bash

# 模型推理测试运行脚本

echo "🚀 启动模型推理测试..."
echo ""

# 检查是否存在训练后的模型
if [ ! -f "outputs/final_model.pt" ]; then
    echo "⚠️  未找到训练后的模型文件: outputs/final_model.pt"
    echo "请先运行训练脚本: python train_baseline.py"
    echo ""
    echo "是否只测试原始模型？(y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🔍 运行快速测试（仅原始模型）..."
        python examples/quick_inference_test.py
    else
        echo "❌ 退出测试"
        exit 1
    fi
else
    echo "✅ 找到训练后的模型文件"
    echo ""
    echo "选择测试类型："
    echo "1) 快速测试 (推荐)"
    echo "2) 完整测试 (详细分析)"
    echo "3) 两个都运行"
    echo ""
    read -p "请输入选择 (1-3): " choice
    
    case $choice in
        1)
            echo "🔍 运行快速测试..."
            python examples/quick_inference_test.py
            ;;
        2)
            echo "📊 运行完整测试..."
            python test_model_inference.py
            ;;
        3)
            echo "🔍 运行快速测试..."
            python examples/quick_inference_test.py
            echo ""
            echo "📊 运行完整测试..."
            python test_model_inference.py
            ;;
        *)
            echo "❌ 无效选择，退出"
            exit 1
            ;;
    esac
fi

echo ""
echo "✅ 测试完成！"
echo ""
echo "📁 查看结果文件："
echo "  - 输出目录: outputs/"
echo "  - 日志目录: logs/"
echo ""
echo "📖 详细说明请查看: docs/inference_test_guide.md"
