import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import torch

from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

# 修复 NumPy 兼容性问题：np.bool 在 NumPy 1.20+ 中已弃用
# 在导入 spikingjelly 之前打补丁
import spikingjelly.datasets.cifar10_dvs as cifar10_dvs_module

# 修复 parse_raw_address 函数中的 np.bool 问题
_original_parse_raw_address = cifar10_dvs_module.parse_raw_address


def parse_raw_address_fixed(addr, **kwargs):
    """修复版本的 parse_raw_address，使用 bool 替代 np.bool"""
    polarity_mask = kwargs.pop("polarity_mask", cifar10_dvs_module.polarity_mask)
    polarity_shift = kwargs.pop("polarity_shift", cifar10_dvs_module.polarity_shift)
    x_mask = kwargs.pop("x_mask", cifar10_dvs_module.x_mask)
    x_shift = kwargs.pop("x_shift", cifar10_dvs_module.x_shift)
    y_mask = kwargs.pop("y_mask", cifar10_dvs_module.y_mask)
    y_shift = kwargs.pop("y_shift", cifar10_dvs_module.y_shift)

    # 使用 bool 替代已弃用的 np.bool
    polarity = cifar10_dvs_module.read_bits(addr, polarity_mask, polarity_shift).astype(
        bool
    )
    x = cifar10_dvs_module.read_bits(addr, x_mask, x_shift)
    y = cifar10_dvs_module.read_bits(addr, y_mask, y_shift)
    return x, y, polarity


# 替换原函数
cifar10_dvs_module.parse_raw_address = parse_raw_address_fixed

from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly import configure


# 修复 create_events_np_files 方法，确保等待所有任务完成
def create_events_np_files_fixed(extract_root: str, events_np_root: str):
    """
    修复版本的 create_events_np_files，确保所有转换任务都完成
    """

    t_ckp = time.time()
    futures = []

    with ThreadPoolExecutor(
        max_workers=min(
            multiprocessing.cpu_count(),
            configure.max_threads_number_for_datasets_preprocess,
        )
    ) as tpe:
        for class_name in os.listdir(extract_root):
            aedat_dir = os.path.join(extract_root, class_name)
            np_dir = os.path.join(events_np_root, class_name)
            os.makedirs(np_dir, exist_ok=True)
            print(f"Mkdir [{np_dir}].")
            for bin_file in os.listdir(aedat_dir):
                source_file = os.path.join(aedat_dir, bin_file)
                target_file = os.path.join(
                    np_dir, os.path.splitext(bin_file)[0] + ".npz"
                )
                print(f"Start to convert [{source_file}] to [{target_file}].")
                future = tpe.submit(
                    CIFAR10DVS.read_aedat_save_to_np, source_file, target_file
                )
                futures.append(future)

        # 显式等待所有任务完成并检查错误
        print(f"等待 {len(futures)} 个转换任务完成...")
        completed = 0
        for future in as_completed(futures):
            try:
                future.result()  # 获取结果，如果有异常会抛出
                completed += 1
                if completed % 100 == 0:
                    print(f"已完成 {completed}/{len(futures)} 个文件转换")
            except Exception as e:
                print(f"转换任务失败: {e}")
                raise

    print(f"所有转换任务完成！用时 = [{round(time.time() - t_ckp, 2)}s].")


# 临时替换 CIFAR10DVS 的 create_events_np_files 方法
CIFAR10DVS.create_events_np_files = staticmethod(create_events_np_files_fixed)

root = os.path.join(os.path.dirname(__file__), "..", "data")

print("=" * 70)
print("测试 CIFAR10-DVS 数据集加载（事件数量切分）")
print("=" * 70)

# 使用事件数量切分（split_by="number"）
frames_number = 10
split_by = "number"
dataset = None

try:
    dataset = CIFAR10DVS(
        root=root,
        data_type="frame",
        frames_number=frames_number,
        split_by=split_by,
    )
    print(f"[OK] 数据集加载成功！")
    print(f"  总样本数: {len(dataset)}")

    # 检查样本形状和标签
    if len(dataset) > 0:
        sample_frame, sample_label = dataset[0]
        print(
            f"  样本形状: {sample_frame.shape if hasattr(sample_frame, 'shape') else type(sample_frame)}"
        )
        print(f"  标签类型: {type(sample_label)}, 值: {sample_label}")

        if isinstance(sample_frame, np.ndarray):
            print(f"  数据格式: numpy.ndarray")
            print(f"  数据类型: {sample_frame.dtype}")
        elif isinstance(sample_frame, torch.Tensor):
            print(f"  数据格式: torch.Tensor")
            print(f"  数据类型: {sample_frame.dtype}")

        # 显示帧数据的统计信息
        if hasattr(sample_frame, "shape") and len(sample_frame.shape) >= 2:
            if isinstance(sample_frame, np.ndarray):
                print(f"\n  帧数据统计:")
                print(f"    最小值: {sample_frame.min():.4f}")
                print(f"    最大值: {sample_frame.max():.4f}")
                print(f"    均值: {sample_frame.mean():.4f}")
                print(f"    标准差: {sample_frame.std():.4f}")

    # 统计类别分布 - 检查所有样本
    from collections import Counter

    labels = []
    print(f"\n  正在统计类别分布（所有样本）...")
    print(f"  进度: ", end="", flush=True)
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(f"{i}/{len(dataset)} ", end="", flush=True)
        _, label = dataset[i]
        labels.append(label)
    print(f"\n  完成！共检查 {len(labels)} 个样本")

    label_counts = Counter(labels)
    print(f"\n  类别分布（所有样本）:")
    print(f"    总类别数: {len(label_counts)}")
    for label in sorted(label_counts.keys()):
        print(
            f"    类别 {label}: {label_counts[label]} 样本 ({label_counts[label]/len(dataset)*100:.1f}%)"
        )

    # 打印一些样本的详细信息
    print(f"\n  样本详细信息（前10个样本）:")
    for i in range(min(10, len(dataset))):
        frame, label = dataset[i]
        print(
            f"    索引 {i}: 标签={label}, 形状={frame.shape if hasattr(frame, 'shape') else 'N/A'}"
        )

except Exception as e:
    print(f"[ERROR] 加载失败: {e}")
    import traceback

    traceback.print_exc()

# 测试分层切分功能
print("\n【测试2】测试分层切分功能")
print("-" * 70)

try:
    from collections import defaultdict
    import random
    from torch.utils.data import Subset

    if dataset is None or len(dataset) == 0:
        print("[ERROR] 数据集未加载或为空，跳过分层切分测试")
    else:
        # 使用number切分的数据集进行分层切分测试
        full_dataset = dataset
        total_samples = len(full_dataset)

        # 按类别分组 - 检查所有样本
        class_indices = defaultdict(list)
        print(f"  正在收集类别信息（总样本数: {total_samples}）...")
        print(f"  进度: ", end="", flush=True)
        for idx in range(total_samples):
            if idx % 1000 == 0:
                print(f"{idx}/{total_samples} ", end="", flush=True)
            _, label = full_dataset[idx]
            class_indices[label].append(idx)
        print(f"\n  完成！")

        print(f"  找到的类别数: {len(class_indices)}")
        print(f"  各类别样本数:")
        for class_label in sorted(class_indices.keys()):
            print(
                f"    类别 {class_label}: {len(class_indices[class_label])} 样本 ({len(class_indices[class_label])/total_samples*100:.1f}%)"
            )

        # 模拟分层切分
        train_ratio = 0.8
        train_indices = []
        test_indices = []

        random.seed(42)
        for class_label, indices in sorted(class_indices.items()):
            random.shuffle(indices)
            class_train_size = int(len(indices) * train_ratio)
            train_indices.extend(indices[:class_train_size])
            test_indices.extend(indices[class_train_size:])
            print(
                f"    类别 {class_label}: {len(indices[:class_train_size])} 训练, {len(indices[class_train_size:])} 测试"
            )

        print(f"\n  分层切分结果（所有 {total_samples} 个样本）:")
        print(
            f"    训练集: {len(train_indices)} 样本 ({len(train_indices)/total_samples*100:.1f}%)"
        )
        print(
            f"    测试集: {len(test_indices)} 样本 ({len(test_indices)/total_samples*100:.1f}%)"
        )

        # 验证分层切分后的类别分布
        train_class_count = defaultdict(int)
        test_class_count = defaultdict(int)
        for idx in train_indices:
            _, label = full_dataset[idx]
            train_class_count[label] += 1
        for idx in test_indices:
            _, label = full_dataset[idx]
            test_class_count[label] += 1

        print(f"\n  分层切分后的类别分布:")
        print(f"    {'类别':<8} {'训练集':<12} {'测试集':<12} {'总计':<8}")
        print(f"    {'-'*45}")
        for class_label in sorted(
            set(train_class_count.keys()) | set(test_class_count.keys())
        ):
            train_count = train_class_count.get(class_label, 0)
            test_count = test_class_count.get(class_label, 0)
            total_count = train_count + test_count
            print(
                f"    {class_label:<8} {train_count:<12} {test_count:<12} {total_count:<8}"
            )

except Exception as e:
    print(f"[ERROR] 分层切分测试失败: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
