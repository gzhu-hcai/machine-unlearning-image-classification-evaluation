import torch


class DictConfig(dict):
    # 让字典支持 obj.key 的访问方式
    def __getattr__(self, key): return self.get(key)

    def __setattr__(self, key, value): self[key] = value


def get_num_classes_from_model(model):
    """动态获取模型的分类数，自动适配 CIFAR-10 和 CIFAR-100"""
    # 针对 ResNet 架构 (分类头通常叫 fc)
    if hasattr(model, 'fc'):
        return model.fc.out_features

    # 针对 VGG / MobileNet 等架构 (分类头通常叫 classifier)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, torch.nn.Sequential):
            # 如果是个序列，取最后一层的输出维度
            return model.classifier[-1].out_features
        else:
            return model.classifier.out_features

    # 兜底默认值
    return 10