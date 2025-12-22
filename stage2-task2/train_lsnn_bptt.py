"""
训练LSNN模型，使用标准的BPTT算法
基于论文: A solution to the learning dilemma for recurrent networks of spiking neurons
"""

import os
import time
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb

from models import LSNN
from dataloader import get_dataloader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_grad: float = 1.0,
) -> Tuple[float, float]:
    """
    训练一个epoch

    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (sequences, labels) in enumerate(pbar):
        sequences = sequences.to(device)  # [batch_size, seq_length]
        labels = labels.to(device)  # [batch_size]

        # 添加input_size维度: [batch_size, seq_length] -> [batch_size, seq_length, 1]
        sequences = sequences.unsqueeze(-1)

        # 前向传播
        optimizer.zero_grad()
        output, _ = model(sequences)  # [batch_size, output_size]

        # 计算损失
        loss = criterion(output, labels)

        # 反向传播（BPTT）
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # 更新参数
        optimizer.step()

        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        # 更新进度条
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
        )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    评估模型

    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for sequences, labels in pbar:
            sequences = sequences.to(device)  # [batch_size, seq_length]
            labels = labels.to(device)  # [batch_size]

            # 添加input_size维度
            sequences = sequences.unsqueeze(-1)

            # 前向传播
            output, _ = model(sequences)

            # 计算损失
            loss = criterion(output, labels)

            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
            )

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train LSNN with BPTT")

    # 数据参数
    parser.add_argument(
        "--data-dir", type=str, default="data/sequential_mnist", help="数据集目录"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="数据加载器工作进程数"
    )

    # 模型参数
    parser.add_argument("--hidden-size", type=int, default=400, help="循环层神经元数量")
    parser.add_argument(
        "--use-sfa", action="store_true", help="使用Spike-Frequency Adaptation"
    )
    parser.add_argument("--tau-m", type=float, default=20.0, help="膜时间常数")
    parser.add_argument("--tau-adapt", type=float, default=200.0, help="SFA时间常数")
    parser.add_argument("--v-threshold", type=float, default=1.0, help="发放阈值")
    parser.add_argument("--v-reset", type=float, default=0.0, help="重置电位")
    parser.add_argument("--beta", type=float, default=1.8, help="SFA强度")
    parser.add_argument("--dt", type=float, default=1.0, help="时间步长")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout率")
    parser.add_argument(
        "--readout-type",
        type=str,
        default="rate",
        choices=["rate", "last"],
        help="Readout类型: rate或last",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument(
        "--clip-grad", type=float, default=1.0, help="梯度裁剪阈值（0表示不裁剪）"
    )
    parser.add_argument("--lr-scheduler", action="store_true", help="使用学习率调度器")
    parser.add_argument("--lr-step", type=int, default=20, help="学习率衰减步长")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="学习率衰减因子")

    # 其他参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备 (cuda/cpu)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints", help="模型保存目录"
    )
    parser.add_argument(
        "--save-name", type=str, default="lsnn_bptt", help="模型保存名称"
    )
    parser.add_argument(
        "--print-freq", type=int, default=1, help="打印频率（每N个epoch）"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="使用Weights & Biases记录训练"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="lsnn-bptt", help="WandB项目名称"
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None, help="WandB运行名称（默认自动生成）"
    )

    args = parser.parse_args()

    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 初始化WandB
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "model": "LSNN",
                "algorithm": "BPTT",
                "hidden_size": args.hidden_size,
                "use_sfa": args.use_sfa,
                "tau_m": args.tau_m,
                "tau_adapt": args.tau_adapt,
                "v_threshold": args.v_threshold,
                "v_reset": args.v_reset,
                "beta": args.beta,
                "dt": args.dt,
                "dropout": args.dropout,
                "readout_type": args.readout_type,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "clip_grad": args.clip_grad,
                "lr_scheduler": args.lr_scheduler,
                "lr_step": args.lr_step,
                "lr_gamma": args.lr_gamma,
                "device": str(device),
            },
        )
        print("WandB已初始化")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据加载器
    print("\n加载数据集...")
    train_loader = get_dataloader(
        data_dir=args.data_dir,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        normalize=True,
    )
    test_loader = get_dataloader(
        data_dir=args.data_dir,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        normalize=True,
    )

    # 模型
    print("\n创建模型...")
    model = LSNN(
        input_size=1,  # Sequential MNIST每个时间步输入1个像素值
        hidden_size=args.hidden_size,
        output_size=10,  # 10个类别
        use_sfa=args.use_sfa,
        tau_m=args.tau_m,
        tau_adapt=args.tau_adapt,
        v_threshold=args.v_threshold,
        v_reset=args.v_reset,
        beta=args.beta,
        dt=args.dt,
        dropout=args.dropout,
        readout_type=args.readout_type,
    ).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率调度器
    scheduler = None
    if args.lr_scheduler:
        scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # 训练循环
    print("\n开始训练...")
    best_acc = 0.0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # Epoch进度条
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs", position=0)

    for epoch in epoch_pbar:
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.clip_grad
        )

        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 学习率调度
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 计算epoch时间
        elapsed = time.time() - start_time

        # 记录到WandB
        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "learning_rate": current_lr,
            "epoch_time": elapsed,
            "best_test_acc": best_acc,
        }
        if args.use_wandb:
            wandb.log(log_dict)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, f"{args.save_name}_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                    "args": args,
                },
                save_path,
            )
            if args.use_wandb:
                wandb.log({"best_test_acc": best_acc})
            print(f"\n保存最佳模型 (准确率: {test_acc:.2f}%) -> {save_path}")

        # 更新epoch进度条
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.2f}%",
                "test_loss": f"{test_loss:.4f}",
                "test_acc": f"{test_acc:.2f}%",
                "best": f"{best_acc:.2f}%",
            }
        )

        # 打印
        if epoch % args.print_freq == 0:
            print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.2f}s)")
            print(f"  训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
            print(f"  测试: 损失={test_loss:.4f}, 准确率={test_acc:.2f}%")
            if scheduler is not None:
                print(f"  学习率: {current_lr:.6f}")
            print(f"  最佳测试准确率: {best_acc:.2f}%")

    epoch_pbar.close()

    # 保存最终模型
    final_save_path = os.path.join(args.save_dir, f"{args.save_name}_final.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc": test_acc,
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_losses": test_losses,
            "test_accs": test_accs,
            "args": args,
        },
        final_save_path,
    )
    print(f"\n保存最终模型 -> {final_save_path}")

    print(f"\n训练完成!")
    print(f"最佳测试准确率: {best_acc:.2f}%")

    # 完成WandB记录
    if args.use_wandb:
        wandb.log({"final/best_test_acc": best_acc})
        wandb.finish()
        print("WandB记录已完成")


if __name__ == "__main__":
    main()
