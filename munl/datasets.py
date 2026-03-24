import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class OriginDataset(Dataset):
    """为数据集添加来源标记 (0: Retain, 1: Forget)"""

    def __init__(self, dataset, origin_label):
        self.dataset = dataset
        self.origin_label = origin_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 兼容现有的 dataset 返回格式，并在最后追加一个 origin 标签
        img, target = self.dataset[idx]
        return img, target, self.origin_label


def get_discernible_retain_and_forget_loaders(retain_loader, forget_loader, shuffle=True):
    """真正地合并保留集和遗忘集，并打上 0 和 1 的标签"""
    retain_ds = OriginDataset(retain_loader.dataset, 0)
    forget_ds = OriginDataset(forget_loader.dataset, 1)

    # 将两个数据集拼接在一起
    combined_ds = ConcatDataset([retain_ds, forget_ds])

    # 继承原有的 batch_size
    batch_size = retain_loader.batch_size if retain_loader.batch_size else 128
    return DataLoader(combined_ds, batch_size=batch_size, shuffle=shuffle)


class RandomRelabelDataset(Dataset):
    """为数据集生成随机标签 (保持不变，用于 SalUn 等算法)"""

    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # 忽略原始标签
        # 返回随机生成的新标签
        return img, torch.randint(0, self.num_classes, (1,)).item()