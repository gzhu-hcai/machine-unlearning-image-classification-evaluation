import torch
def get_optimizer_scheduler_criterion(model, cfg):
    optimizer = torch.optim.SGD(model.parameters(), lr=getattr(cfg, 'learning_rate', 0.01))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    return optimizer, scheduler, criterion