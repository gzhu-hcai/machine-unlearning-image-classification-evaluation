import torch


class BaseUnlearner:
    def __init__(self, cfg, device, writer=None, save_steps=False, should_evaluate=False):
        self.cfg = cfg
        self.device = device

    def save_and_log(self, model, optimizer, scheduler, payload, epoch):
        pass

    def evaluate_if_needed(self, model, val_loader, criterion, device):
        # 返回一个包含 0 的 Tensor，防止 finetune 调用 .mean() 时报错
        return torch.tensor([0.0])


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    """真实的通用训练循环，供 Finetune 等算法调用"""
    model.train()
    batch_losses = []

    for batch in train_loader:
        # 兼容我们的 Dataset (可能返回 2个 或 3个元素)
        inputs = batch[0].to(device)
        targets = batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.detach().item())

    # 如果有学习率调度器，则步进
    if scheduler is not None:
        scheduler.step()

    # 返回所有 batch loss 的 tensor，适配 finetune.py 中的 train_batch_loss.mean()
    return torch.tensor(batch_losses)