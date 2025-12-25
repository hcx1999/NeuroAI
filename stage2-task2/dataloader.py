import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import os
from pathlib import Path


class SequentialMNIST(Dataset):
    """
    Sequential MNIST 数据集加载器

    数据集说明：
    - 数据形状: (N, 784)，其中 N 是样本数量，784 = 28*28
    - 每个样本是一个展平的 28x28 图像序列
    - 标签形状: (N,)，每个样本对应一个类别标签 (0-9)
    - 序列长度: 784 (图像按行展开)
    """

    def __init__(
        self,
        data_dir: str = "data/sequential_mnist",
        split: str = "train",
        transform: Optional[callable] = None,
        normalize: bool = True,
    ):
        """
        初始化 Sequential MNIST 数据集

        Args:
            data_dir: 数据集目录路径
            split: 数据集分割，'train' 或 'test'
            transform: 可选的变换函数
            normalize: 是否将数据归一化到 [0, 1] 范围（如果数据是 uint8）
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.normalize = normalize

        # 加载数据文件 - 使用 pathlib 处理路径，确保跨平台兼容
        data_file = Path(data_dir) / f"{split}.pt"
        # 如果路径不存在，尝试相对于当前脚本的路径
        if not data_file.exists():
            # 尝试相对于脚本所在目录的路径
            script_dir = Path(__file__).parent
            data_file = script_dir / data_dir / f"{split}.pt"

        data_file = str(data_file.resolve())  # 转换为绝对路径字符串
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        # 加载数据
        data_dict = torch.load(data_file, map_location="cpu")

        # 提取数据和标签
        self.data = data_dict["data"]  # shape: (N, 784)
        self.targets = data_dict["targets"]  # shape: (N,)
        self.seq_length = data_dict["seq_length"]  # 784
        self.image_shape = data_dict["image_shape"]  # (1, 28, 28)

        # 如果数据是 uint8 类型且需要归一化，转换为 float32 并归一化
        if self.normalize and self.data.dtype == torch.uint8:
            self.data = self.data.float() / 255.0

        print(f"加载 {split} 数据集:")
        print(f"  - 样本数量: {len(self.data)}")
        print(f"  - 数据形状: {self.data.shape}")
        print(f"  - 标签形状: {self.targets.shape}")
        print(f"  - 序列长度: {self.seq_length}")
        print(f"  - 图像形状: {self.image_shape}")
        print(f"  - 数据类型: {self.data.dtype}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (sequence, label): 序列数据和标签
            - sequence: shape (784,)，展平的图像序列
            - label: shape ()，类别标签 (0-9)
        """
        sequence = self.data[idx]  # shape: (784,)
        label = self.targets[idx]  # shape: ()

        # 应用变换（如果有）
        if self.transform is not None:
            sequence = self.transform(sequence)

        return sequence, label

    def get_image(self, idx: int) -> torch.Tensor:
        """
        获取原始图像形状的数据（用于可视化）

        Args:
            idx: 样本索引

        Returns:
            image: shape (1, 28, 28)，原始图像形状
        """
        sequence = self.data[idx]  # shape: (784,)
        image = sequence.view(self.image_shape)  # shape: (1, 28, 28)
        return image


def get_dataloader(
    data_dir: str = "data/sequential_mnist",
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    normalize: bool = True,
    transform: Optional[callable] = None,
) -> DataLoader:
    """
    获取 Sequential MNIST 数据加载器

    Args:
        data_dir: 数据集目录路径
        split: 数据集分割，'train' 或 'test'
        batch_size: 批次大小
        shuffle: 是否打乱数据（通常训练集为 True，测试集为 False）
        num_workers: 数据加载的进程数
        pin_memory: 是否将数据固定到内存（用于 GPU 加速）
        normalize: 是否归一化数据
        transform: 可选的变换函数

    Returns:
        DataLoader: PyTorch 数据加载器
    """
    dataset = SequentialMNIST(
        data_dir=data_dir,
        split=split,
        transform=transform,
        normalize=normalize,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader


if __name__ == "__main__":
    # 测试数据加载器
    print("=" * 60)
    print("测试 Sequential MNIST 数据加载器")
    print("=" * 60)

    # 创建训练集和测试集
    # 使用相对于脚本文件的路径
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "sequential_mnist"

    train_dataset = SequentialMNIST(
        data_dir=str(data_dir), split="train", normalize=True
    )
    test_dataset = SequentialMNIST(data_dir=str(data_dir), split="test", normalize=True)

    print("\n" + "=" * 60)
    print("测试单个样本")
    print("=" * 60)
    sequence, label = train_dataset[0]
    print(f"序列形状: {sequence.shape}")
    print(f"标签: {label.item()}")
    print(f"序列数据类型: {sequence.dtype}")
    print(f"序列值范围: [{sequence.min().item():.4f}, {sequence.max().item():.4f}]")

    # 获取图像形状
    image = train_dataset.get_image(0)
    print(f"图像形状: {image.shape}")

    print("\n" + "=" * 60)
    print("测试 DataLoader")
    print("=" * 60)
    train_loader = get_dataloader(
        data_dir=str(data_dir),
        split="train",
        batch_size=32,
        shuffle=True,
    )

    # 获取一个批次
    sequences, labels = next(iter(train_loader))
    print(f"批次序列形状: {sequences.shape}")  # (batch_size, 784)
    print(f"批次标签形状: {labels.shape}")  # (batch_size,)
    print(f"批次标签示例: {labels[:5].tolist()}")
