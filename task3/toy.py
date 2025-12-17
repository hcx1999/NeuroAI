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


class CIFAR10DVSOrganizedDataset:
    """
    高效包装CIFAR10-DVS数据集，利用数据集已按类别顺序组织的特点
    每个类别有固定数量的样本（默认1000个），按顺序排列

    数据集组织方式：
    - 类别 0: 索引 0 到 samples_per_class-1
    - 类别 1: 索引 samples_per_class 到 2*samples_per_class-1
    - ...
    - 类别 n: 索引 n*samples_per_class 到 (n+1)*samples_per_class-1
    """

    def __init__(self, dataset, samples_per_class=1000, num_classes=10):
        """
        初始化包装类

        Args:
            dataset: CIFAR10DVS数据集对象
            samples_per_class: 每个类别的样本数（默认1000）
            num_classes: 类别总数（默认10）
        """
        self.dataset = dataset
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.total_samples = len(dataset)

        # 验证数据集大小
        expected_size = samples_per_class * num_classes
        if self.total_samples != expected_size:
            print(
                f"[警告] 数据集大小 {self.total_samples} 与预期 {expected_size} 不匹配"
            )

    def __len__(self):
        """返回数据集总大小"""
        return self.total_samples

    def __getitem__(self, idx):
        """获取单个样本"""
        return self.dataset[idx]

    def get_class_indices(self, class_label):
        """
        获取指定类别的所有样本索引（高效方法，直接计算）

        Args:
            class_label: 类别标签 (0-9)

        Returns:
            range对象，包含该类别的所有索引
        """
        start_idx = class_label * self.samples_per_class
        end_idx = start_idx + self.samples_per_class
        return range(start_idx, end_idx)

    def split_stratified(self, train_ratio=0.8, shuffle=False, seed=None):
        """
        高效的分层切分方法，直接使用索引切片，无需遍历数据集

        Args:
            train_ratio: 训练集比例（默认0.8）
            shuffle: 是否在每个类别内打乱顺序（默认False）
            seed: 随机种子（仅在shuffle=True时有效）

        Returns:
            train_indices: 训练集索引列表
            test_indices: 测试集索引列表
        """
        train_indices = []
        test_indices = []

        if shuffle and seed is not None:
            import random

            random.seed(seed)

        for class_label in range(self.num_classes):
            # 直接计算该类别的索引范围
            start_idx = class_label * self.samples_per_class
            end_idx = start_idx + self.samples_per_class

            # 创建该类别的索引列表
            class_indices = list(range(start_idx, end_idx))

            if shuffle:
                import random

                random.shuffle(class_indices)

            # 计算切分点
            train_size = int(len(class_indices) * train_ratio)

            # 分割索引
            train_indices.extend(class_indices[:train_size])
            test_indices.extend(class_indices[train_size:])

        return train_indices, test_indices

    def get_train_test_subsets(self, train_ratio=0.8, shuffle=False, seed=None):
        """
        返回训练集和测试集的Subset对象（可直接用于DataLoader）

        Args:
            train_ratio: 训练集比例（默认0.8）
            shuffle: 是否在每个类别内打乱顺序（默认False）
            seed: 随机种子（仅在shuffle=True时有效）

        Returns:
            train_subset: 训练集Subset对象
            test_subset: 测试集Subset对象
        """
        from torch.utils.data import Subset

        train_indices, test_indices = self.split_stratified(
            train_ratio=train_ratio, shuffle=shuffle, seed=seed
        )

        train_subset = Subset(self.dataset, train_indices)
        test_subset = Subset(self.dataset, test_indices)

        return train_subset, test_subset

    def get_class_subset(self, class_label):
        """
        获取指定类别的子集

        Args:
            class_label: 类别标签 (0-9)

        Returns:
            Subset对象，包含该类别的所有样本
        """
        from torch.utils.data import Subset

        indices = list(self.get_class_indices(class_label))
        return Subset(self.dataset, indices)

    def verify_organization(self):
        """
        验证数据集是否按照预期方式组织
        返回验证结果和详细信息
        """
        is_organized = True
        issues = []

        for class_label in range(self.num_classes):
            start_idx = class_label * self.samples_per_class
            end_idx = start_idx + self.samples_per_class

            # 检查前几个和后几个样本的标签
            if start_idx < self.total_samples:
                _, first_label = self.dataset[start_idx]
                if first_label != class_label:
                    is_organized = False
                    issues.append(
                        f"类别 {class_label} 起始位置 {start_idx} 的标签是 {first_label}，不是 {class_label}"
                    )

            if end_idx - 1 < self.total_samples:
                _, last_label = self.dataset[end_idx - 1]
                if last_label != class_label:
                    is_organized = False
                    issues.append(
                        f"类别 {class_label} 结束位置 {end_idx-1} 的标签是 {last_label}，不是 {class_label}"
                    )

        return is_organized, issues


root = os.path.join(os.path.dirname(__file__), "..", "data")

print("=" * 70)
print("测试 CIFAR10-DVS 数据集加载（事件数量切分）")
print("=" * 70)

# 使用事件数量切分（split_by="number"）
frames_number = 10
split_by = "number"
dataset = None

dataset = CIFAR10DVS(
    root=root,
    data_type="frame",
    frames_number=frames_number,
    split_by=split_by,
)

# 测试高效包装类
print("\n【测试3】测试高效数据集包装类")
print("-" * 70)

try:
    if dataset is None or len(dataset) == 0:
        print("[ERROR] 数据集未加载或为空，跳过包装类测试")
    else:
        # 创建包装类
        print("  创建 CIFAR10DVSOrganizedDataset 包装类...")
        wrapped_dataset = CIFAR10DVSOrganizedDataset(
            dataset, samples_per_class=1000, num_classes=10
        )
        print(f"  ✓ 包装类创建成功！总样本数: {len(wrapped_dataset)}")

        # 验证数据集组织
        print("\n  验证数据集组织方式...")
        is_organized, issues = wrapped_dataset.verify_organization()
        if is_organized:
            print("  ✓ 数据集组织验证通过！")
        else:
            print("  ✗ 数据集组织验证失败:")
            for issue in issues:
                print(f"    - {issue}")

        # 测试获取类别索引
        print("\n  测试获取类别索引（高效方法）:")
        for class_label in range(min(3, wrapped_dataset.num_classes)):
            indices = wrapped_dataset.get_class_indices(class_label)
            print(
                f"    类别 {class_label}: 索引范围 {indices.start}-{indices.stop-1} (共 {len(indices)} 个样本)"
            )

        # 测试高效分层切分（不打乱）
        print("\n  测试高效分层切分（不打乱，直接索引切片）:")
        import time

        start_time = time.time()
        train_indices, test_indices = wrapped_dataset.split_stratified(
            train_ratio=0.8, shuffle=False
        )
        elapsed_time = time.time() - start_time
        print(f"  ✓ 切分完成！用时: {elapsed_time*1000:.2f} ms")
        print(
            f"    训练集: {len(train_indices)} 样本 ({len(train_indices)/len(dataset)*100:.1f}%)"
        )
        print(
            f"    测试集: {len(test_indices)} 样本 ({len(test_indices)/len(dataset)*100:.1f}%)"
        )

        # 验证切分后的类别分布
        from collections import Counter

        train_labels = [
            dataset[idx][1] for idx in train_indices[:1000]
        ]  # 只检查前1000个以节省时间
        test_labels = [
            dataset[idx][1] for idx in test_indices[:200]
        ]  # 只检查前200个以节省时间
        print(f"\n  验证切分后的类别分布（采样检查）:")
        print(f"    训练集前1000个样本的类别分布: {dict(Counter(train_labels))}")
        print(f"    测试集前200个样本的类别分布: {dict(Counter(test_labels))}")

        # 测试高效分层切分（打乱）
        print("\n  测试高效分层切分（打乱，seed=42）:")
        start_time = time.time()
        train_indices_shuffled, test_indices_shuffled = (
            wrapped_dataset.split_stratified(train_ratio=0.8, shuffle=True, seed=42)
        )
        elapsed_time = time.time() - start_time
        print(f"  ✓ 切分完成！用时: {elapsed_time*1000:.2f} ms")
        print(f"    训练集: {len(train_indices_shuffled)} 样本")
        print(f"    测试集: {len(test_indices_shuffled)} 样本")

        # 测试获取Subset对象
        print("\n  测试获取Subset对象（可直接用于DataLoader）:")
        train_subset, test_subset = wrapped_dataset.get_train_test_subsets(
            train_ratio=0.8, shuffle=False
        )
        print(f"  ✓ Subset对象创建成功！")
        print(f"    训练集Subset大小: {len(train_subset)}")
        print(f"    测试集Subset大小: {len(test_subset)}")

        # 测试获取单个类别的子集
        print("\n  测试获取单个类别的子集:")
        class_0_subset = wrapped_dataset.get_class_subset(0)
        print(f"    类别 0 的子集大小: {len(class_0_subset)}")
        sample_frame, sample_label = class_0_subset[0]
        print(f"    第一个样本的标签: {sample_label} (应该是 0)")

except Exception as e:
    print(f"[ERROR] 包装类测试失败: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
