import os
import time
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb

# SpikingJelly
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

# Import SuperSpike surrogate function
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "surrogate_comparison", "src")
)
from modules import SuperSpike


# ---------------------
# Model: Conv SNN
# ---------------------
class ConvSNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        T_steps: int = 4,
        dropout_p: float = 0.2,
        surrogate_alpha: float = 10.0,
        in_channels: int = 2,
    ):
        super().__init__()
        self.T_steps = T_steps
        # Use SuperSpike (fast sigmoid) surrogate gradient
        sg = SuperSpike(alpha=surrogate_alpha)
        # Feature extractor
        # CIFAR10DVS typically has 2 channels (ON/OFF events), but can be converted to 3 (RGB)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.sn1 = neuron.IFNode(surrogate_function=sg, detach_reset=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.sn2 = neuron.IFNode(surrogate_function=sg, detach_reset=True)

        self.pool = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.sn3 = neuron.IFNode(surrogate_function=sg, detach_reset=True)

        # Classifier
        # CIFAR10DVS input is 128x128, after 2 pools: 128->64->32, so 256*32*32
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.dropout = nn.Dropout(dropout_p)
        self.sn_fc1 = neuron.IFNode(surrogate_function=sg, detach_reset=True)

        self.fc2 = nn.Linear(512, num_classes)
        # Last spiking layer optional; use membrane as logits via rate coding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)  # reset neuron states per sample/mini-batch
        # Handle CIFAR10DVS frame data: x shape is [B, T, C, H, W] from DataLoader
        # If x has 5 dimensions [B, T, C, H, W], use the time dimension from data
        # Otherwise, assume [B, C, H, W] and repeat for T_steps
        if x.dim() == 5:
            # [B, T, C, H, W] format from CIFAR10DVS
            B, T, C, H, W = x.shape
            time_steps = T
        else:
            # [B, C, H, W] format, repeat for T_steps
            B, C, H, W = x.shape
            time_steps = self.T_steps

        # Accumulate spikes over time and compute rate-coded logits
        logits_acc = 0.0
        for t in range(time_steps):
            if x.dim() == 5:
                # Extract frame at timestep t: [B, T, C, H, W] -> [B, C, H, W]
                x_t = x[:, t, :, :, :]
            else:
                # Use same input for each timestep
                x_t = x

            out = self.conv1(x_t)
            out = self.bn1(out)
            out = self.sn1(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.sn2(out)
            out = self.pool(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.sn3(out)
            out = self.pool(out)

            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.dropout(out)
            out = self.sn_fc1(out)

            logits = self.fc2(out)
            logits_acc = logits_acc + logits

        return logits_acc / float(time_steps)


# ---------------------
# Data
# ---------------------


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    frames_number: int = 10,
    split_by: str = "number",
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR10DVS dataset.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        split_by: How to split events into frames. Options: "number", "time"
        frames_number: Number of frames to split each event stream into
    """
    root = os.path.join(os.path.dirname(__file__), "..", "data")

    # CIFAR10DVS returns event streams, we'll use frame-based representation
    # The dataset will automatically convert events to frames
    # According to SpikingJelly docs: https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.datasets.html
    # Note: CIFAR10DVS doesn't have 'train' parameter, we need to load and split manually
    from torch.utils.data import Subset

    # Load full dataset (CIFAR10DVS typically has 9000 train + 1000 test = 10000 total)
    # But the actual split may vary, so we load all and split manually
    full_dataset = CIFAR10DVS(
        root=root,
        data_type="frame",
        frames_number=frames_number,
        split_by=split_by,
    )

    # Stratified split: split by class to ensure balanced distribution
    # First, collect all labels to group samples by class
    print("正在按类别收集样本信息...")
    from collections import defaultdict
    import random

    # Group indices by class label
    class_indices = defaultdict(list)
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        class_indices[label].append(idx)

    # Determine split ratio
    total_samples = len(full_dataset)
    if total_samples == 10000:
        # Standard split: 9000 train, 1000 test (90/10)
        train_ratio = 0.9
    elif total_samples == 9000:
        # Only training set available, use 80/20 split
        train_ratio = 0.8
    else:
        # Use 90/10 split as default
        train_ratio = 0.9

    # Split each class according to the ratio
    train_indices = []
    test_indices = []

    # Set random seed for reproducibility
    random.seed(42)

    for class_label, indices in sorted(class_indices.items()):
        # Shuffle indices for this class
        random.shuffle(indices)

        # Calculate split point for this class
        class_train_size = int(len(indices) * train_ratio)

        # Split indices
        class_train_indices = indices[:class_train_size]
        class_test_indices = indices[class_train_size:]

        train_indices.extend(class_train_indices)
        test_indices.extend(class_test_indices)

        print(
            f"  类别 {class_label}: {len(class_train_indices)} 训练, {len(class_test_indices)} 测试"
        )

    # Shuffle the final indices to mix classes
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    print(
        f"✓ 分层切分完成: {len(train_indices)} 训练样本, {len(test_indices)} 测试样本"
    )

    train_set = Subset(full_dataset, train_indices)
    test_set = Subset(full_dataset, test_indices)

    # Print dataset information
    print("\n" + "=" * 70)
    print("数据集信息 (Dataset Information)")
    print("=" * 70)
    print(f"数据集类型: CIFAR10-DVS")
    print(f"数据格式: Frame-based (事件流转换为帧)")
    print(f"帧数 (frames_number): {frames_number}")
    print(f"切分方式 (split_by): {split_by}")
    print(f"总样本数: {total_samples}")
    print(f"训练集样本数: {len(train_set)} ({len(train_set)/total_samples*100:.1f}%)")
    print(f"测试集样本数: {len(test_set)} ({len(test_set)/total_samples*100:.1f}%)")

    # Get sample shape to show data dimensions
    if len(train_set) > 0:
        sample_frame, sample_label = train_set[0]
        if isinstance(sample_frame, torch.Tensor):
            print(f"数据形状: {sample_frame.shape}")
            print(
                f"  格式: [T, C, H, W] = [{sample_frame.shape[0]}, {sample_frame.shape[1]}, {sample_frame.shape[2]}, {sample_frame.shape[3]}]"
            )
        else:
            print(
                f"数据形状: {sample_frame.shape if hasattr(sample_frame, 'shape') else type(sample_frame)}"
            )
    print(f"类别数: 10 (CIFAR-10)")

    # Print class distribution in train and test sets
    print("\n类别分布 (Class Distribution):")
    train_class_count = defaultdict(int)
    test_class_count = defaultdict(int)

    for idx in train_indices:
        _, label = full_dataset[idx]
        train_class_count[label] += 1

    for idx in test_indices:
        _, label = full_dataset[idx]
        test_class_count[label] += 1

    print(f"{'类别':<8} {'训练集':<12} {'测试集':<12} {'总计':<8}")
    print("-" * 45)
    for class_label in sorted(
        set(train_class_count.keys()) | set(test_class_count.keys())
    ):
        train_count = train_class_count.get(class_label, 0)
        test_count = test_class_count.get(class_label, 0)
        total_count = train_count + test_count
        print(f"{class_label:<8} {train_count:<12} {test_count:<12} {total_count:<8}")

    print(f"\n批次大小: {batch_size}")
    print(f"DataLoader workers: {num_workers}")
    print("=" * 70 + "\n")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------
# Train & Eval
# ---------------------


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    grad_clip: float,
    progress: bool,
    epoch: int = 0,
):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    if progress:
        # Enhanced progress bar with more details
        data_iter = tqdm(
            loader,
            desc=f"Epoch {epoch} [Train]",
            ncols=100,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix}",
        )
    else:
        data_iter = loader

    for batch_idx, (images, targets) in enumerate(data_iter):
        # print(batch_idx)
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        # print("forwarding")
        outputs = model(images)
        loss = criterion(outputs, targets)

        # print("backwarding")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # print("steping")
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

        # Update progress bar with current metrics
        if progress:
            current_loss = running_loss / total
            current_acc = correct / total
            data_iter.set_postfix_str(f"{current_loss:.4f}, Acc: {current_acc:.2%}")

    return running_loss / total, correct / total


def evaluate(
    model,
    loader,
    criterion,
    device,
    progress: bool,
    epoch: int = 0,
    split: str = "Test",
):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    if progress:
        # Enhanced progress bar with more details
        data_iter = tqdm(
            loader,
            desc=f"Epoch {epoch} [{split}]",
            ncols=100,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix}",
        )
    else:
        data_iter = loader

    with torch.no_grad():
        for images, targets in data_iter:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

            # Update progress bar with current metrics
            if progress:
                current_loss = running_loss / total
                current_acc = correct / total
                data_iter.set_postfix_str(f"{current_loss:.4f}, Acc: {current_acc:.2%}")

    return running_loss / total, correct / total


# ---------------------
# Main
# ---------------------


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10-DVS Conv SNN with SuperSpike surrogate gradient"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=4,
        help="Number of timesteps (used if frames_number doesn't match)",
    )
    parser.add_argument(
        "--frames-number", type=int, default=10, help="Number of frames for CIFAR10DVS"
    )
    parser.add_argument(
        "--surrogate-alpha",
        type=float,
        default=10.0,
        help="Alpha parameter for SuperSpike surrogate",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=2,
        help="Input channels (2 for ON/OFF events, 3 for RGB)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--step-size", type=int, default=20, help="StepLR step size (epochs)"
    )
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR decay factor")
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Clip grad norm; <=0 to disable"
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable tqdm progress bar",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Enable tqdm progress bar (default)",
    )
    parser.set_defaults(progress=True)
    parser.add_argument(
        "--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default=os.path.join("runs"))
    parser.add_argument(
        "--wandb-project", type=str, default="cifar10-dvs-snn", help="W&B project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )
    parser.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable W&B logging",
    )
    parser.set_defaults(use_wandb=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "timesteps": args.timesteps,
                "frames_number": args.frames_number,
                "surrogate_alpha": args.surrogate_alpha,
                "surrogate": "SuperSpike",
                "in_channels": args.in_channels,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "step_size": args.step_size,
                "gamma": args.gamma,
                "grad_clip": args.grad_clip,
                "num_workers": args.num_workers,
                "device": str(device),
            },
        )

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frames_number=args.frames_number,
    )

    model = ConvSNN(
        num_classes=10,
        T_steps=args.timesteps,
        surrogate_alpha=args.surrogate_alpha,
        in_channels=args.in_channels,
    ).to(device)

    # Print model information
    print("=" * 70)
    print("模型信息 (Model Information)")
    print("=" * 70)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型架构: ConvSNN")
    print(f"替代梯度函数: SuperSpike (alpha={args.surrogate_alpha})")
    print(f"输入通道数: {args.in_channels}")
    print(f"时间步数: {args.timesteps}")
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,}")
    print(f"设备: {device}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    print("=" * 70 + "\n")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Log model architecture to wandb
    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    # Print training configuration
    print("=" * 70)
    print("训练配置 (Training Configuration)")
    print("=" * 70)
    print(f"总轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"标签平滑: {args.label_smoothing}")
    print(f"梯度裁剪: {args.grad_clip if args.grad_clip > 0 else 'Disabled'}")
    print(f"学习率调度器: StepLR (step_size={args.step_size}, gamma={args.gamma})")
    print("=" * 70 + "\n")

    print("开始训练...\n")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
            args.progress,
            epoch=epoch,
        )
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            args.progress,
            epoch=epoch,
            split="Test",
        )
        dt = time.time() - start
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{args.epochs} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Test loss {test_loss:.4f} acc {test_acc:.4f} | lr {current_lr:.5f} | {dt:.1f}s"
        )

        # Log to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "learning_rate": current_lr,
                    "epoch_time": dt,
                }
            )

        scheduler.step()

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "timesteps": args.timesteps,
                "frames_number": args.frames_number,
                "surrogate_alpha": args.surrogate_alpha,
                "in_channels": args.in_channels,
                "learning_rate": current_lr,
            }
            ckpt_path = os.path.join(
                args.save_dir, f"best_snn_timesteps{args.timesteps}.pt"
            )
            torch.save(ckpt, ckpt_path)
            if args.use_wandb:
                wandb.log({"best_test_acc": best_acc, "best_epoch": epoch})
            print(f"  ✓ 保存最佳模型 (test_acc: {best_acc:.4f})")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "timesteps": args.timesteps,
                "frames_number": args.frames_number,
                "surrogate_alpha": args.surrogate_alpha,
                "learning_rate": current_lr,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            ckpt_path = os.path.join(
                args.save_dir, f"checkpoint_epoch{epoch}_timesteps{args.timesteps}.pt"
            )
            torch.save(ckpt, ckpt_path)
            print(
                f"  ✓ 保存检查点 (epoch {epoch}, test_acc: {test_acc:.4f}, test_loss: {test_loss:.4f})"
            )

    print(f"Best test acc: {best_acc:.4f}")
    if args.use_wandb:
        wandb.log({"final/best_test_acc": best_acc})
        wandb.finish()


def test_saved_model(
    checkpoint_path: str,
    device: str = "auto",
    batch_size: int = 128,
    frames_number: int = 10,
    split_by: str = "number",
):
    """
    测试保存的模型检查点

    Args:
        checkpoint_path: 检查点文件路径
        device: 设备 ('cuda', 'cpu', or 'auto')
        batch_size: 测试时的批次大小
        frames_number: 帧数（需要与训练时一致）
        split_by: 切分方式（需要与训练时一致）
    """
    print("=" * 70)
    print("测试保存的模型 (Testing Saved Model)")
    print("=" * 70)

    # 设置设备
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"设备: {device}")

    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return

    print(f"加载检查点: {checkpoint_path}")

    # 加载检查点
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        print("✓ 检查点加载成功")
    except Exception as e:
        print(f"错误: 无法加载检查点: {e}")
        return

    # 显示检查点信息
    print("\n检查点信息:")
    print(f"  - Epoch: {ckpt.get('epoch', 'N/A')}")
    print(
        f"  - Test Accuracy: {ckpt.get('test_acc', 'N/A'):.4f}"
        if "test_acc" in ckpt
        else "  - Test Accuracy: N/A"
    )
    print(
        f"  - Test Loss: {ckpt.get('test_loss', 'N/A'):.4f}"
        if "test_loss" in ckpt
        else "  - Test Loss: N/A"
    )
    print(
        f"  - Train Accuracy: {ckpt.get('train_acc', 'N/A'):.4f}"
        if "train_acc" in ckpt
        else "  - Train Accuracy: N/A"
    )
    print(
        f"  - Train Loss: {ckpt.get('train_loss', 'N/A'):.4f}"
        if "train_loss" in ckpt
        else "  - Train Loss: N/A"
    )
    print(f"  - Timesteps: {ckpt.get('timesteps', 'N/A')}")
    print(f"  - Frames Number: {ckpt.get('frames_number', 'N/A')}")
    print(f"  - Surrogate Alpha: {ckpt.get('surrogate_alpha', 'N/A')}")
    print(f"  - Learning Rate: {ckpt.get('learning_rate', 'N/A')}")

    # 获取模型参数
    timesteps = ckpt.get("timesteps", 4)
    surrogate_alpha = ckpt.get("surrogate_alpha", 10.0)
    in_channels = ckpt.get("in_channels", 2)

    # 创建模型
    print(
        f"\n创建模型 (timesteps={timesteps}, surrogate_alpha={surrogate_alpha}, in_channels={in_channels})..."
    )
    model = ConvSNN(
        num_classes=10,
        T_steps=timesteps,
        surrogate_alpha=surrogate_alpha,
        in_channels=in_channels,
    ).to(device)

    # 加载模型权重
    try:
        model.load_state_dict(ckpt["model_state"])
        print("✓ 模型权重加载成功")
    except Exception as e:
        print(f"错误: 无法加载模型权重: {e}")
        return

    # 加载测试数据
    print(f"\n加载测试数据 (frames_number={frames_number}, split_by={split_by})...")
    _, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=2,
        frames_number=frames_number,
        split_by=split_by,
    )
    print(f"✓ 测试集大小: {len(test_loader.dataset)} 样本")

    # 在测试集上评估
    print("\n在测试集上评估模型...")
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="测试中", ncols=100):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

    test_acc = correct / total
    test_loss = running_loss / total

    print("\n" + "=" * 70)
    print("测试结果 (Test Results)")
    print("=" * 70)
    print(f"测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"测试损失: {test_loss:.4f}")
    print(f"正确样本数: {correct}/{total}")

    # 与检查点中的指标对比
    if "test_acc" in ckpt:
        saved_acc = ckpt["test_acc"]
        diff = abs(test_acc - saved_acc)
        print(f"\n与保存的指标对比:")
        print(f"  保存的准确率: {saved_acc:.4f}")
        print(f"  当前测试准确率: {test_acc:.4f}")
        print(f"  差异: {diff:.4f} ({diff*100:.2f}%)")
        if diff < 0.01:  # 差异小于1%
            print("  ✓ 指标匹配（差异 < 1%）")
        else:
            print("  ⚠ 指标差异较大，可能使用了不同的数据或参数")

    print("=" * 70)
    print("测试完成！")
    print("=" * 70)

    return {
        "test_acc": test_acc,
        "test_loss": test_loss,
        "correct": correct,
        "total": total,
    }


if __name__ == "__main__":
    import sys

    # 如果提供了检查点路径作为命令行参数，则测试模型
    if len(sys.argv) > 1 and sys.argv[1] == "--test-model":
        if len(sys.argv) < 3:
            print(
                "用法: python train_cifar10_DVS_snn.py --test-model <checkpoint_path> [--frames-number N] [--split-by method]"
            )
            sys.exit(1)

        checkpoint_path = sys.argv[2]
        frames_number = 10
        split_by = "number"

        # 解析可选参数
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg == "--frames-number" and i + 1 < len(sys.argv):
                frames_number = int(sys.argv[i + 1])
            elif arg == "--split-by" and i + 1 < len(sys.argv):
                split_by = sys.argv[i + 1]

        test_saved_model(
            checkpoint_path=checkpoint_path,
            frames_number=frames_number,
            split_by=split_by,
        )
    else:
        main()
