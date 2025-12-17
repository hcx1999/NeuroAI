"""
测试保存模型的示例脚本

使用方法：
1. 测试最佳模型：
   python test_model_example.py runs/best_snn_timesteps4.pt

2. 测试特定epoch的检查点：
   python test_model_example.py runs/checkpoint_epoch10_timesteps4.pt

3. 使用命令行参数：
   python train_cifar10_DVS_snn.py --test-model runs/best_snn_timesteps4.pt
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from train_cifar10_DVS_snn import test_saved_model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "用法: python test_model_example.py <checkpoint_path> [--frames-number N] [--split-by method]"
        )
        print("\n示例:")
        print("  python test_model_example.py runs/best_snn_timesteps4.pt")
        print(
            "  python test_model_example.py runs/checkpoint_epoch10_timesteps4.pt --frames-number 10"
        )
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    frames_number = 10
    split_by = "number"

    # 解析可选参数
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--frames-number" and i + 1 < len(sys.argv):
            frames_number = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--split-by" and i + 1 < len(sys.argv):
            split_by = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # 运行测试
    test_saved_model(
        checkpoint_path=checkpoint_path,
        frames_number=frames_number,
        split_by=split_by,
    )
