"""
高效包装CIFAR10-DVS数据集的工具类
利用数据集已按类别顺序组织的特点，提供高效的数据分割方法
"""

from torch.utils.data import Subset


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
